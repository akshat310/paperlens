"""
Measure how well retrieval actually works, per mode.

The point of this script is to replace an assertion with a number. It is easy to
claim that hybrid retrieval beats dense or BM25 alone; this runs all three
against the same labelled questions and prints what the difference actually is,
including the cases where the simpler method wins.

Two families of metric:

  Retrieval  -- did we put a passage from the right page in front of the model?
                No API key needed, so this half always runs.
                  hit-rate@k : fraction of questions with >=1 correct page in the
                               top k. This is the ceiling on answer quality: if
                               the evidence never gets retrieved, no amount of
                               prompting recovers it.
                  MRR        : 1/rank of the first correct passage, averaged.
                               Rewards ranking the right passage first, not
                               merely somewhere in the list.

  Generation -- given that context, was the answer faithful to it? Needs an API
                key, so it is opt-in via --judge.
                  faithfulness : an LLM judge decides whether every claim in the
                                 answer is supported by the retrieved passages.
                  abstention   : on questions the paper cannot answer, how often
                                 the system said so (and how often it answered
                                 anyway); on answerable ones, how often it
                                 wrongly declined.
                  citations    : how many [n] markers pointed at excerpts that
                                 were never sent, and how many quoted phrases
                                 did not occur in their cited passage. Both are
                                 caught and removed by the app; this measures
                                 how often that safety net is used.

Usage:
    python eval/make_fixture_paper.py          # once, to build the corpus
    python eval/run_eval.py                    # retrieval metrics only, offline
    python eval/run_eval.py --judge            # adds faithfulness (uses quota)
    python eval/run_eval.py --paper my.pdf --questions mine.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Allow `python eval/run_eval.py` from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import Base, SessionLocal, engine  # noqa: E402
from app.models import Paper, User  # noqa: E402
from app.rag import llm  # noqa: E402
from app.rag.prompts import (  # noqa: E402
    INSUFFICIENT_CONTEXT,
    build_chat_prompt,
    format_excerpts,
)
from app.rag.retriever import retrieve  # noqa: E402
from app.services import chat_service  # noqa: E402
from app.services.ingestion import process_paper  # noqa: E402

MODES = ["dense", "bm25", "hybrid"]
CUTOFFS = (1, 3, 5)  # report hit-rate at each; see summarise() for why
EVAL_USER_EMAIL = "eval@paperlens.local"

# The app's own marker grammar, so the eval counts exactly what the app parses.
CITATION_PATTERN = chat_service.CITATION_PATTERN

JUDGE_PROMPT = """You are grading a question-answering system for faithfulness.

You will see the PASSAGES the system was given and the ANSWER it produced. Decide
whether every factual claim in the answer is supported by the passages.

Judge ONLY faithfulness to the passages. Do not use outside knowledge, and do not
penalise an answer for being incomplete. If the answer declines to answer, that
counts as supported -- declining is never unfaithful.

Return JSON: {{"supported": true or false, "reason": "<one short sentence>"}}

=== PASSAGES ===
{context}
=== END PASSAGES ===

ANSWER: {answer}

JSON:"""


# ---------- corpus setup ----------

def ensure_paper_ingested(db, pdf_path: str, reingest: bool) -> Paper:
    """Ingest the evaluation paper once, then reuse it on later runs.

    Re-embedding the same document on every run would add ~30s to a script that
    is otherwise instant, so we key off the file path and skip if it is already
    indexed. --reingest forces the work when the corpus itself has changed.
    """
    user = db.scalar(select(User).where(User.email == EVAL_USER_EMAIL))
    if user is None:
        # A local-only row so the paper has an owner; never used to log in.
        user = User(email=EVAL_USER_EMAIL, hashed_password="not-a-real-login")
        db.add(user)
        db.commit()

    existing = db.scalar(
        select(Paper).where(Paper.user_id == user.id, Paper.file_path == pdf_path)
    )

    if existing and not reingest and existing.status == "ready":
        print(f"Reusing indexed paper: {existing.title} "
              f"({existing.num_pages} pages, {existing.num_chunks} chunks)")
        return existing

    if existing:
        db.delete(existing)  # cascade clears chunks; vectors are re-added below
        db.commit()

    paper = Paper(
        user_id=user.id,
        title=Path(pdf_path).stem,
        filename=Path(pdf_path).name,
        file_path=pdf_path,
        status="pending",
    )
    db.add(paper)
    db.commit()

    print(f"Ingesting {pdf_path} (embedding, this takes a moment)...")
    process_paper(paper.id)
    db.refresh(paper)

    if paper.status != "ready":
        raise SystemExit(f"Ingestion failed: {paper.error_message}")

    print(f"Indexed {paper.num_pages} pages into {paper.num_chunks} chunks")
    return paper


# ---------- metrics ----------

def evaluate_retrieval(db, paper: Paper, questions: list[dict], top_k: int) -> dict:
    """Run every mode over every question and record hit-rate and reciprocal rank."""
    results: dict[str, list[dict]] = {mode: [] for mode in MODES}

    for question in questions:
        expected = set(question["expected_pages"])
        if not expected:
            continue  # abstention questions have no right page; judged separately

        for mode in MODES:
            chunks = retrieve(
                db, query=question["question"], paper_id=paper.id, top_k=top_k, mode=mode
            )
            # A chunk can now span a page break, so a hit means its page
            # RANGE intersects the expected pages -- not that a single
            # page number matches. Using page_start alone would score a
            # chunk that genuinely contains the answer as a miss whenever
            # the answer sat on the second page it covers.
            pages = [
                next(
                    (p for p in range(c.page_start, c.page_end + 1) if p in expected),
                    c.page_start,
                )
                for c in chunks
            ]

            # Rank (1-based) of the first retrieved passage on a correct page.
            rank = next((i for i, p in enumerate(pages, start=1) if p in expected), None)

            results[mode].append(
                {
                    "id": question["id"],
                    "kind": question["kind"],
                    "rank": rank,  # None when no correct page was retrieved at all
                    "hit": rank is not None,
                    "reciprocal_rank": 1.0 / rank if rank else 0.0,
                    "pages": pages,
                    "expected": sorted(expected),
                }
            )

    return results


def summarise(rows: list[dict], cutoffs: tuple[int, ...] = CUTOFFS) -> dict:
    """Average the per-question records into headline numbers.

    Reporting several cutoffs from one retrieval run costs nothing extra: the
    rank of the first correct passage already determines hit-rate at every k.
    It matters because hit-rate@5 alone saturates on a small corpus -- every
    mode looks perfect and the metric stops discriminating. hit-rate@1 asks the
    much harder question of whether the right passage was ranked *first*.
    """
    if not rows:
        return {"n": 0, "mrr": 0.0, **{f"hit@{k}": 0.0 for k in cutoffs}}

    summary = {
        "n": len(rows),
        "mrr": sum(r["reciprocal_rank"] for r in rows) / len(rows),
    }
    for k in cutoffs:
        summary[f"hit@{k}"] = sum(
            1 for r in rows if r["rank"] is not None and r["rank"] <= k
        ) / len(rows)
    return summary


# ---------- generation quality ----------

def _generate_with_backoff(prompt: str, delay: float, *, json_mode: bool = False,
                           model: str | None = None, max_retries: int = 4) -> str:
    """Call the model, waiting out free-tier rate limits rather than giving up.

    A 429 here means "ask again shortly", so aborting the whole run on the first
    one throws away work that would have succeeded after a pause. We back off
    exponentially and only surrender after several attempts, which is the
    difference between an eval that completes on a free key and one that does
    not.
    """
    wait = max(delay, 8.0)
    for attempt in range(1, max_retries + 1):
        try:
            return llm.generate(prompt, json_mode=json_mode, model=model)
        except llm.LLMRateLimitError:
            if attempt == max_retries:
                raise
            print(f"    rate limited; waiting {wait:.0f}s "
                  f"(attempt {attempt}/{max_retries})")
            time.sleep(wait)
            wait *= 2
    raise llm.LLMRateLimitError("exhausted retries")  # unreachable, keeps types honest


def judge_answers(db, paper: Paper, questions: list[dict], top_k: int,
                  mode: str, delay: float, answer_model: str | None,
                  judge_model: str | None) -> dict:
    """Generate an answer per question in one mode, then have an LLM grade it.

    Only one mode is judged by default. Judging all three would triple the API
    calls for a metric dominated by retrieval quality, which the offline metrics
    already measure directly -- and free-tier quota is the binding constraint.

    The judge should be a DIFFERENT model from the one that wrote the answer.
    A model grading its own output exhibits self-enhancement bias: it recognises
    its own phrasing and reasoning as correct and marks generously, which
    inflates the faithfulness number in the direction that flatters the system.
    Using a separate judge is not a complete fix -- models from one family share
    training data and blind spots -- but it removes the most obvious source of
    bias, and it has the side benefit of drawing on a separate quota pool.
    """
    supported = 0
    abstained = 0
    graded = 0
    failures: list[str] = []
    # Abstention: questions with no expected page should be declined.
    abst_total = abst_correct = abst_answered = 0
    false_abstain = 0
    # Citation hygiene across every answer that was generated.
    markers_total = markers_invented = quotes_total = quotes_unverified = 0

    for question in questions:
        unanswerable = not question["expected_pages"]
        chunks = retrieve(
            db, query=question["question"], paper_id=paper.id, top_k=top_k, mode=mode
        )
        if not chunks:
            abstained += 1
            if unanswerable:
                abst_total += 1
                abst_correct += 1
            else:
                false_abstain += 1
            continue

        # format_excerpts rather than a hand-rolled string: the judge must see
        # exactly what the app sends the model, or the faithfulness number is
        # measuring a prompt that never actually runs.
        context = format_excerpts(chunks)

        try:
            answer = _generate_with_backoff(
                build_chat_prompt(
                    question["question"],
                    context,
                    paper_title=paper.title,
                    abstract=paper.abstract,
                ),
                delay,
                model=answer_model,
            )
            time.sleep(delay)

            if INSUFFICIENT_CONTEXT in answer:
                abstained += 1
                if unanswerable:
                    abst_total += 1
                    abst_correct += 1
                    print(f"  Q{question['id']}: correctly declined")
                else:
                    false_abstain += 1
                    print(f"  Q{question['id']}: DECLINED an answerable question")
                continue

            # Citation hygiene, measured with the app's own parser.
            for m in CITATION_PATTERN.finditer(answer):
                markers_total += 1
                if not 1 <= int(m.group(1)) <= len(chunks):
                    markers_invented += 1
                if m.group(2):
                    quotes_total += 1
            _, cites = chat_service._extract_citations(answer, chunks)
            quotes_unverified += quotes_total - sum(1 for c in cites if c.verified) \
                if False else 0  # replaced below with a per-answer count
            verified_here = sum(1 for c in cites if c.verified)
            quoted_here = sum(1 for m in CITATION_PATTERN.finditer(answer) if m.group(2))
            quotes_unverified += max(0, quoted_here - verified_here)

            if unanswerable:
                # It answered something the paper does not contain. Whether
                # the judge calls it "supported" is beside the point.
                abst_total += 1
                abst_answered += 1
                print(f"  Q{question['id']}: ANSWERED an unanswerable question")
                continue

            verdict_raw = _generate_with_backoff(
                JUDGE_PROMPT.format(context=context, answer=answer),
                delay,
                json_mode=True,
                model=judge_model,
            )
            time.sleep(delay)

            verdict = json.loads(verdict_raw)
            graded += 1
            if verdict.get("supported"):
                supported += 1
            else:
                failures.append(f"  Q{question['id']}: {verdict.get('reason', '')}")
            print(f"  Q{question['id']}: {'supported' if verdict.get('supported') else 'UNSUPPORTED'}")

        except llm.LLMRateLimitError as exc:
            # Out of quota for the day rather than for the minute. Stop, but keep
            # whatever was graded -- a partial result is reported as partial.
            print(f"  ! giving up at Q{question['id']}: {exc}")
            break
        except (llm.LLMError, ValueError) as exc:
            print(f"  ! Q{question['id']} failed: {str(exc)[:100]}")

    return {
        "graded": graded,
        "supported": supported,
        "abstained": abstained,
        "faithfulness": supported / graded if graded else 0.0,
        "failures": failures,
        "abstention": {
            "total": abst_total,
            "correct": abst_correct,
            "answered_anyway": abst_answered,
            "false_abstain": false_abstain,
        },
        "citations": {
            "markers": markers_total,
            "invented": markers_invented,
            "quotes": quotes_total,
            "unverified": quotes_unverified,
        },
    }


# ---------- reporting ----------

def build_report(paper: Paper, top_k: int, retrieval: dict, judged: dict | None,
                 judge_mode: str, answer_model: str = "", judge_model: str = "") -> str:
    """Render the markdown table that goes into the README."""
    lines: list[str] = []
    lines.append("## Retrieval evaluation\n")
    lines.append(
        f"{len(retrieval['hybrid'])} hand-labelled answerable questions over "
        f"`{paper.filename}` ({paper.num_pages} pages, {paper.num_chunks} chunks), "
        f"top_k = {top_k}. Model `{settings.GEMINI_MODEL}`, embeddings "
        f"`{settings.EMBEDDING_MODEL}` at {settings.EMBEDDING_DIM} dims, chunks of "
        f"~{settings.CHUNK_TARGET_TOKENS} tokens. Generated {time.strftime('%Y-%m-%d')}.\n"
    )

    header = " | ".join(f"hit@{k}" for k in CUTOFFS)
    lines.append(f"| Mode | {header} | MRR |")
    lines.append("|---" * (len(CUTOFFS) + 2) + "|")

    for mode in MODES:
        stats = summarise(retrieval[mode])
        cells = " | ".join(f"{stats[f'hit@{k}'] * 100:.1f}%" for k in CUTOFFS)
        lines.append(f"| {mode} | {cells} | {stats['mrr']:.3f} |")

    # The breakdown by question type is the interesting part: it shows *why*
    # fusion helps rather than just that it does. Reported at k=1, where the
    # modes actually separate.
    lines.append("\n### hit-rate@1 by question type\n")
    kinds = sorted({r["kind"] for r in retrieval["hybrid"]})
    lines.append("| Mode | " + " | ".join(kinds) + " |")
    lines.append("|---" * (len(kinds) + 1) + "|")

    for mode in MODES:
        by_kind: dict[str, list[dict]] = defaultdict(list)
        for row in retrieval[mode]:
            by_kind[row["kind"]].append(row)
        cells = [f"{summarise(by_kind[k])['hit@1'] * 100:.0f}%" for k in kinds]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")

    if judged:
        lines.append(f"\n### Answer faithfulness ({judge_mode} retrieval)\n")

        # Never render 0 graded answers as "0.0% faithful". A metric with no
        # samples behind it is missing data, and printing it as a percentage
        # states the opposite of the truth -- it reads as total failure.
        if judged["graded"] == 0:
            lines.append(
                "_Not measured: no answers were graded "
                "(API quota exhausted, or every question abstained)._"
            )
        else:
            total = judged["graded"] + judged["abstained"]
            # Naming both models is not decoration: a faithfulness score is only
            # interpretable if you know who wrote the answers and who graded them.
            lines.append(f"Answers generated by `{answer_model}`, "
                         f"graded by `{judge_model}`.\n")
            if answer_model == judge_model:
                lines.append("> Caveat: the same model generated and graded, so this "
                             "figure is optimistic (self-enhancement bias).\n")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            lines.append(f"| Questions attempted | {total} |")
            lines.append(f"| Answers graded | {judged['graded']} |")
            lines.append(
                f"| Faithful to retrieved context | "
                f"{judged['faithfulness'] * 100:.1f}% "
                f"({judged['supported']}/{judged['graded']}) |"
            )
            lines.append(f"| Abstained (said 'not in this paper') | {judged['abstained']} |")

            abst = judged.get("abstention", {})
            if abst.get("total"):
                lines.append("\n### Abstention\n")
                lines.append("| Metric | Value |")
                lines.append("|---|---|")
                lines.append(f"| Unanswerable questions | {abst['total']} |")
                lines.append(
                    f"| Correctly declined | {abst['correct']}/{abst['total']} "
                    f"({abst['correct'] / abst['total'] * 100:.0f}%) |"
                )
                lines.append(f"| Answered anyway (hallucination) | {abst['answered_anyway']} |")
                lines.append(f"| Answerable questions wrongly declined | {abst['false_abstain']} |")

            cit = judged.get("citations", {})
            if cit.get("markers"):
                lines.append("\n### Citation hygiene\n")
                lines.append(
                    "The app removes invented markers and rejects unverifiable quotes "
                    "before the reader sees them; this is how often that happened.\n"
                )
                lines.append("| Metric | Value |")
                lines.append("|---|---|")
                lines.append(f"| Citation markers emitted | {cit['markers']} |")
                lines.append(
                    f"| Pointing at an excerpt never sent | {cit['invented']} "
                    f"({cit['invented'] / cit['markers'] * 100:.1f}%) |"
                )
                if cit["quotes"]:
                    lines.append(f"| Quoted phrases emitted | {cit['quotes']} |")
                    lines.append(
                        f"| Not found verbatim in the cited passage | {cit['unverified']} "
                        f"({cit['unverified'] / cit['quotes'] * 100:.1f}%) |"
                    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PaperLens retrieval.")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--paper", default=None, help="override the PDF in questions.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--judge", action="store_true", help="also grade answer faithfulness")
    parser.add_argument("--judge-mode", default="hybrid", choices=MODES)
    parser.add_argument("--answer-model", default=None,
                        help="model that writes answers (default: GEMINI_MODEL)")
    parser.add_argument("--judge-model", default=None,
                        help="model that grades them; use a DIFFERENT one to avoid "
                             "self-enhancement bias, e.g. gemini-3.1-flash-lite")
    parser.add_argument("--delay", type=float, default=4.0,
                        help="seconds between API calls, to stay under free-tier limits")
    parser.add_argument("--reingest", action="store_true")
    parser.add_argument("--out", default="eval/results.md")
    args = parser.parse_args()

    spec = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = spec["questions"]
    pdf_path = args.paper or spec["paper"]

    if not Path(pdf_path).exists():
        raise SystemExit(
            f"{pdf_path} not found. Run: python eval/make_fixture_paper.py"
        )

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        paper = ensure_paper_ingested(db, pdf_path, args.reingest)

        print(f"\nEvaluating {len(questions)} questions across {len(MODES)} modes...")
        retrieval = evaluate_retrieval(db, paper, questions, args.top_k)

        judged = None
        if args.judge:
            answer_model = args.answer_model or settings.GEMINI_MODEL
            judge_model = args.judge_model or settings.GEMINI_MODEL
            print(f"\nGrading answer faithfulness ({args.judge_mode} mode)")
            print(f"  answers by : {answer_model}")
            print(f"  judged by  : {judge_model}")
            if answer_model == judge_model:
                print("  ! same model on both sides -- faithfulness will be "
                      "optimistic (self-enhancement bias). Pass --judge-model.")
            judged = judge_answers(db, paper, questions, args.top_k,
                                   args.judge_mode, args.delay,
                                   args.answer_model, args.judge_model)

        report = build_report(
            paper, args.top_k, retrieval, judged, args.judge_mode,
            args.answer_model or settings.GEMINI_MODEL,
            args.judge_model or settings.GEMINI_MODEL,
        )
        print("\n" + report)

        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Written to {args.out}")

        # Name the misses explicitly. An eval that only reports averages hides
        # the examples you actually learn from.
        print("\nQuestions hybrid retrieval missed:")
        misses = [r for r in retrieval["hybrid"] if not r["hit"]]
        if not misses:
            print("  (none)")
        for row in misses:
            print(f"  Q{row['id']} ({row['kind']}): expected page(s) {row['expected']}, "
                  f"got {row['pages']}")

        if judged and judged["failures"]:
            print("\nUnfaithful answers:")
            for failure in judged["failures"]:
                print(failure)
    finally:
        db.close()


if __name__ == "__main__":
    main()
