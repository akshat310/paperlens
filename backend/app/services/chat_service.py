"""
The generation half of RAG: retrieved chunks -> grounded, cited answer.

    question -> expand -> retrieve -> assemble context -> generate -> cite -> persist

The single most important property of this file is that **the model never
authors a citation's metadata**. It chooses which numbered excerpt supports a
claim; we look the pages and quotable text up in our own database from the chunk
id. That is the difference between a citation you can click and verify and a
plausible-looking number.

What changed, and why the old answers were thin
-----------------------------------------------
Three things, compounding:

1. The model saw 5 chunks of ~900 characters -- about 1,100 tokens of the paper,
   total. It is now 10 chunks of ~1,000 tokens each.
2. It saw *only* those excerpts. No title, no abstract, no section list. Asked
   "what is this paper about?" it could only answer if the abstract happened to
   rank in the top 5. A global overview is now always in the prompt.
3. The system prompt ended with "Be concise and factual." It now says to match
   the answer's length to the question.

Conversation memory
-------------------
Chat used to be stateless: history was stored and displayed but never fed back,
so "what about the second one?" could not work. Recent turns are now replayed
verbatim and older ones are folded into a running summary on the paper row. The
summary is what keeps a fifty-turn conversation from growing the prompt without
bound -- a sliding window alone would silently forget the beginning, and sending
everything would eventually cost more than the paper does.
"""

import json
import logging
import re
from collections.abc import Iterator

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import ChatMessage, Paper, Section
from app.rag import llm, prompts
from app.rag.retriever import RetrievedChunk, retrieve
from app.schemas import ChatMessageOut, ChatResponse, Citation, ClaimEvidence, ClaimResponse
from app.services import analysis

logger = logging.getLogger(__name__)

# Matches the inline markers the model is told to emit, in either form:
#   [3]                          -- cite excerpt 3
#   [3: "exact words from it"]   -- cite excerpt 3 and quote the supporting phrase
# The quote is optional in the grammar because a model that forgets it should
# still produce a usable (chunk-level) citation, not a dropped one.
# The quote may contain "]" -- papers cite as "Adam [20]" and the model copies
# that verbatim -- so the only forbidden character inside it is the closing
# double quote itself.
CITATION_PATTERN = re.compile(r'\[(\d+)(?::\s*"([^"]{3,400})")?\]')

# Quotes shorter than this are too generic to verify anything ("the model").
MIN_QUOTE_CHARS = 12

SNIPPET_CHARS = 300  # enough to recognise the quote, short enough for a UI chip

NOT_FOUND_MESSAGE = (
    "I could not find an answer to that in this paper. The sections I retrieved "
    "do not cover it."
)

# Section summaries included in the system context, truncated per section. The
# whole point is global structure, not detail -- detail arrives through
# retrieval. 600 characters is two or three sentences per section.
SECTION_OVERVIEW_CHARS = 600
MAX_ABSTRACT_CHARS = 3000


def _snippet(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= SNIPPET_CHARS:
        return collapsed
    return collapsed[:SNIPPET_CHARS].rstrip() + "..."


def _normalise_for_match(text: str) -> str:
    """Collapse whitespace and the punctuation PDF extraction mangles.

    Curly quotes, hyphenation and line-wrap spaces differ between what the
    model saw and what it types back; none of them change whether the words
    are the paper's words, so they are neutralised on both sides before the
    substring test. Case is preserved: "Adam" and "ADAM" are different tokens
    in a paper.
    """
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    return " ".join(text.split())


def _verify_quote(quote: str | None, chunk: RetrievedChunk) -> str | None:
    """The quote if it literally occurs in the chunk, else None.

    This is the whole verification step and it is deliberately dumb. An
    entailment model would judge *semantic* support and does not fit in the
    memory budget; substring containment judges *literal* support, which is
    stricter, free, and something nobody has to trust us on -- the phrase is
    either in the passage or it is not.
    """
    if not quote or len(quote.strip()) < MIN_QUOTE_CHARS:
        return None
    needle = _normalise_for_match(quote).strip(" .,;:")
    if needle and needle in _normalise_for_match(chunk.content):
        return needle
    return None


def _citation_for(chunk: RetrievedChunk, quote: str | None = None) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section=chunk.section,
        label=prompts.page_label(chunk),
        # A verified quote is a better snippet than the chunk's first 300
        # characters: it is the exact words the answer rests on.
        snippet=quote if quote else _snippet(chunk.content),
        quote=quote,
        verified=quote is not None,
    )


def _extract_citations(
    answer: str, chunks: list[RetrievedChunk]
) -> tuple[str, list[Citation]]:
    """Resolve the model's [n] markers into real citations.

    Returns the answer with markers renumbered, plus the citations in the order
    they first appear.

    Three things happen here, all deliberate:

    * **Validation.** A marker outside the range of what we actually sent is a
      hallucination -- the model inventing a source. We drop it from the text
      rather than rendering a citation that points nowhere.
    * **Quote verification.** A marker may carry the phrase the claim rests
      on. It is kept only if it literally occurs in that chunk (see
      `_verify_quote`); otherwise the citation degrades to chunk-level and
      the miss is counted. The quote text is removed from the answer either
      way -- the marker stays, the evidence moves to the citation.
    * **Renumbering.** If the model cites excerpts 4 and 2, the answer becomes
      [1] and [2] with `citations[0]` and `citations[1]` matching. Without this
      the frontend would need the original retrieval list to resolve a marker;
      with it, `[n]` is simply `citations[n - 1]`.
    """
    order: list[int] = []  # original 1-based indices, deduplicated, first-seen order
    quotes: dict[int, str] = {}  # original index -> first verified quote
    invented = 0
    unverified = 0
    for match in CITATION_PATTERN.finditer(answer):
        index = int(match.group(1))
        if not 1 <= index <= len(chunks):
            invented += 1
            continue
        if index not in order:
            order.append(index)
        if match.group(2) and index not in quotes:
            verified = _verify_quote(match.group(2), chunks[index - 1])
            if verified:
                quotes[index] = verified
            else:
                unverified += 1

    if unverified:
        logger.info("%d quoted phrase(s) not found in their cited passage", unverified)

    if invented:
        # Counted, not just dropped. "How often does the model invent a source?"
        # is a number worth being able to quote, and this log line is where the
        # eval harness and the usage ledger read it from.
        logger.info("Dropped %d out-of-range citation marker(s)", invented)

    renumbered = {original: new for new, original in enumerate(order, start=1)}

    def _replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        # The quoted phrase is part of the model's sentence ("trained with
        # [1: "Adam"]"), so it stays in the prose as plain text and only the
        # marker syntax is rewritten; otherwise the sentence is left with a
        # hole. Whether it was *verified* is recorded on the citation, not in
        # the text.
        phrase = match.group(2)
        prose = f"{phrase.strip()} " if phrase else ""
        if index not in renumbered:
            # Unknown marker -> drop it. Silently deleting beats a dead link;
            # the phrase, if any, is still the model's own words and stays.
            return prose.rstrip()
        return f"{prose}[{renumbered[index]}]"

    # One pass with a function, not repeated str.replace calls -- sequential
    # replacement would corrupt markers (rewriting 4->1 then 1->2 hits it twice).
    cleaned = CITATION_PATTERN.sub(_replace, answer)

    citations = [_citation_for(chunks[i - 1], quotes.get(i)) for i in order]
    return cleaned.strip(), citations


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _section_overview(db: Session, paper: Paper) -> str | None:
    """A compact map of the paper: section names, pages, and a line on each.

    Drawn from the map-stage analyses when they exist, so the model gets a real
    description of each section rather than just its heading. Falls back to
    names and page ranges alone before the analysis has run, which is still
    enough for the model to know what the paper contains and to say "that would
    be in the Results section, which I did not retrieve".
    """
    sections = list(
        db.scalars(
            select(Section)
            .where(Section.paper_id == paper.id)
            .order_by(Section.ordinal)
        )
    )
    if not sections:
        return None

    lines: list[str] = []
    for section in sections:
        pages = (
            f"p{section.page_start}"
            if section.page_start == section.page_end
            else f"pp{section.page_start}-{section.page_end}"
        )
        line = f"- {section.name} ({pages})"

        if section.analysis_json:
            try:
                text = json.loads(section.analysis_json)["text"]
                gist = " ".join(text.split())[:SECTION_OVERVIEW_CHARS]
                line += f": {gist}"
            except (ValueError, KeyError, TypeError):
                pass

        lines.append(line)

    return "\n".join(lines)


def _conversation_context(db: Session, paper: Paper) -> str | None:
    """Recent turns verbatim, older ones as a running summary.

    The summary is regenerated lazily: when the number of turns outside the
    recent window grows past the point already summarised, we fold the gap in
    with one cheap call. Doing it on write instead would put an LLM call on the
    path of every message even when nobody asks a follow-up.
    """
    messages = list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.paper_id == paper.id)
            .order_by(ChatMessage.turn_index)
        )
    )
    if not messages:
        return None

    # A "turn" is a user/assistant pair, so the window is measured in messages.
    window = settings.CHAT_RECENT_TURNS * 2
    older = messages[:-window] if len(messages) > window else []
    recent = messages[-window:] if len(messages) > window else messages

    # Fold anything older that the stored summary does not yet cover.
    unsummarised = [m for m in older if m.turn_index > paper.chat_summary_upto]
    if unsummarised:
        transcript = "\n".join(
            f"{m.role.upper()}: {' '.join(m.content.split())[:1500]}"
            for m in unsummarised
        )
        try:
            with llm.calling(paper.id, "summary"):
                paper.chat_summary = llm.generate(
                    prompts.build_conversation_summary_prompt(paper.chat_summary, transcript),
                    temperature=0.2,
                    max_output_tokens=prompts.CHAT_SUMMARY_MAX_TOKENS,
                )
            paper.chat_summary_upto = unsummarised[-1].turn_index
            db.commit()
        except llm.LLMError as exc:
            # Not fatal. Without the fold, the older turns are simply absent
            # from this prompt -- the answer loses some context but is still
            # grounded in the paper.
            logger.info("Conversation summary skipped: %s", exc)

    parts: list[str] = []
    if paper.chat_summary:
        parts.append(f"[Earlier in this conversation]\n{paper.chat_summary}")
    for message in recent:
        parts.append(f"{message.role.upper()}: {' '.join(message.content.split())}")

    return "\n\n".join(parts) if parts else None


def _build_prompt(
    db: Session, paper: Paper, question: str, chunks: list[RetrievedChunk]
) -> str:
    abstract = paper.abstract
    if abstract:
        abstract = abstract[:MAX_ABSTRACT_CHARS]
    elif (report := analysis.get_report(db, paper)) is not None:
        # No parsed abstract -- the plain-language paragraph from the report is
        # a good substitute, and it is grounded in the same paper.
        abstract = report.plain_language

    return prompts.build_chat_prompt(
        question=question,
        excerpts=prompts.format_excerpts(chunks),
        paper_title=paper.title,
        abstract=abstract,
        section_overview=_section_overview(db, paper),
        conversation=_conversation_context(db, paper),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _persist_turn(
    db: Session,
    paper_id: str,
    question: str,
    answer: str,
    citations: list[Citation],
) -> None:
    """Save both sides of the exchange in one transaction.

    Both messages are written together so history can never end up with a
    question that has no answer attached to it.
    """
    next_index = (
        db.scalar(
            select(func.count())
            .select_from(ChatMessage)
            .where(ChatMessage.paper_id == paper_id)
        )
        or 0
    )

    db.add(
        ChatMessage(
            paper_id=paper_id, role="user", content=question, turn_index=next_index
        )
    )
    db.add(
        ChatMessage(
            paper_id=paper_id,
            role="assistant",
            content=answer,
            citations_json=json.dumps([c.model_dump() for c in citations]),
            turn_index=next_index + 1,
        )
    )
    db.commit()


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------

def _retrieve_for(
    db: Session, paper: Paper, question: str, section_id: str | None = None
) -> list[RetrievedChunk]:
    if section_id and db.scalar(
        select(Section.id).where(Section.id == section_id, Section.paper_id == paper.id)
    ) is None:
        # A section id from another paper (or a made-up one) is silently
        # ignored rather than 404ing: the question is still answerable, it
        # just is not narrowed. Narrowing is a convenience, not a contract.
        section_id = None

    with llm.calling(paper.id, "retrieve"):
        return retrieve(
            db,
            query=question,
            paper_id=paper.id,
            top_k=settings.TOP_K,
            paper_title=paper.title,
            section_id=section_id,
        )


def answer_question(
    db: Session, paper: Paper, question: str, section_id: str | None = None
) -> ChatResponse:
    """Answer one question about one paper, grounded in that paper alone."""
    chunks = _retrieve_for(db, paper, question, section_id)

    if not chunks:
        # Nothing retrieved means nothing to ground an answer in. Short-circuit
        # rather than spend an API call to be told the obvious.
        _persist_turn(db, paper.id, question, NOT_FOUND_MESSAGE, [])
        return ChatResponse(answer=NOT_FOUND_MESSAGE, citations=[], grounded=False)

    with llm.calling(paper.id, "chat"):
        raw = llm.generate(
            _build_prompt(db, paper, question, chunks),
            temperature=0.2,
            max_output_tokens=prompts.CHAT_MAX_TOKENS,
        )

    if prompts.INSUFFICIENT_CONTEXT in raw:
        _persist_turn(db, paper.id, question, NOT_FOUND_MESSAGE, [])
        return ChatResponse(answer=NOT_FOUND_MESSAGE, citations=[], grounded=False)

    answer, citations = _extract_citations(raw, chunks)

    # Belt and braces. The sentinel is the model cooperating; zero citations is
    # the model answering from somewhere we cannot verify. Either way the answer
    # is not something we are willing to label as grounded.
    grounded = bool(citations)
    if not grounded:
        logger.warning("Paper %s: answer produced no valid citations", paper.id)

    _persist_turn(db, paper.id, question, answer, citations)
    return ChatResponse(answer=answer, citations=citations, grounded=grounded)


def stream_answer(
    db: Session, paper: Paper, question: str, section_id: str | None = None
) -> Iterator[dict]:
    """Answer as a stream of events, for Server-Sent Events.

    Yields dicts the router serialises: `{"type": "token", ...}` while the
    answer is being written, then one `{"type": "done", ...}` carrying the
    citations and follow-ups.

    Citations are resolved only at the end, and this is unavoidable rather than
    lazy: markers are renumbered against the set the model actually used, and
    we do not know that set until it stops writing. The client therefore renders
    raw `[4]` markers mid-stream and swaps in the renumbered text on `done`.
    That swap is a visible flicker on a slow answer; the alternative -- not
    renumbering -- would push the mapping problem into the frontend.
    """
    chunks = _retrieve_for(db, paper, question, section_id)

    if not chunks:
        _persist_turn(db, paper.id, question, NOT_FOUND_MESSAGE, [])
        yield {"type": "token", "text": NOT_FOUND_MESSAGE}
        yield {"type": "done", "answer": NOT_FOUND_MESSAGE, "citations": [],
               "grounded": False, "followups": []}
        return

    yield {"type": "status", "message": f"Reading {len(chunks)} passages"}

    pieces: list[str] = []
    with llm.calling(paper.id, "chat"):
        for piece in llm.stream(
            _build_prompt(db, paper, question, chunks),
            temperature=0.2,
            max_output_tokens=prompts.CHAT_MAX_TOKENS,
        ):
            pieces.append(piece)
            yield {"type": "token", "text": piece}

    raw = "".join(pieces).strip()

    if not raw or prompts.INSUFFICIENT_CONTEXT in raw:
        _persist_turn(db, paper.id, question, NOT_FOUND_MESSAGE, [])
        yield {"type": "done", "answer": NOT_FOUND_MESSAGE, "citations": [],
               "grounded": False, "followups": []}
        return

    answer, citations = _extract_citations(raw, chunks)
    grounded = bool(citations)
    _persist_turn(db, paper.id, question, answer, citations)

    yield {
        "type": "done",
        "answer": answer,
        "citations": [c.model_dump() for c in citations],
        "grounded": grounded,
        "followups": suggest_followups(db, paper, last_question=question),
    }


def check_claim(db: Session, paper: Paper, claim: str) -> ClaimResponse:
    """Verdict on a claim, grounded in retrieved passages.

    Reuses chat's retrieval and citation machinery wholesale; the only new
    parts are the JSON contract and the verdict vocabulary. Not persisted as a
    chat turn -- it is a lookup, not a conversation.
    """
    chunks = _retrieve_for(db, paper, claim)
    if not chunks:
        return ClaimResponse(
            claim=claim, verdict="not_addressed",
            reasoning="Nothing in the paper was retrieved for this claim.",
        )

    abstract = paper.abstract[:MAX_ABSTRACT_CHARS] if paper.abstract else None
    prompt = prompts.build_claim_prompt(
        claim, prompts.format_excerpts(chunks), paper_title=paper.title, abstract=abstract,
    )
    with llm.calling(paper.id, "claim"):
        raw = llm.generate(
            prompt, json_mode=True, temperature=0.1, max_output_tokens=prompts.CLAIM_MAX_TOKENS
        )

    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise llm.LLMError("The model returned an unreadable verdict.") from exc

    verdict = str(data.get("verdict", "not_addressed")).lower()
    if verdict not in prompts.CLAIM_VERDICTS:
        verdict = "not_addressed"

    reasoning, citations = _extract_citations(str(data.get("reasoning", "")), chunks)

    evidence: list[ClaimEvidence] = []
    for item in data.get("evidence", []) or []:
        try:
            index = int(item.get("excerpt", 0))
            quote = str(item.get("quote", "")).strip()
        except (AttributeError, TypeError, ValueError):
            continue
        if not 1 <= index <= len(chunks) or not quote:
            continue
        chunk = chunks[index - 1]
        verified = _verify_quote(quote, chunk)
        evidence.append(
            ClaimEvidence(
                citation=_citation_for(chunk, verified),
                quote=verified or quote,
                verified=verified is not None,
            )
        )

    # A "supports"/"contradicts" verdict with no verified evidence is the model
    # asserting something we cannot check. Downgrade it rather than show a
    # confident badge over nothing.
    if verdict in ("supports", "contradicts") and not any(e.verified for e in evidence):
        if not citations:
            verdict = "not_addressed"
        else:
            verdict = "partial"

    return ClaimResponse(
        claim=claim,
        verdict=verdict,
        reasoning=reasoning,
        evidence=evidence,
        caveats=str(data.get("caveats", "") or ""),
        citations=citations,
    )


def list_messages(db: Session, paper: Paper) -> list[ChatMessageOut]:
    """Full chat history for a paper, oldest first.

    Ordered by `turn_index`, not `created_at` -- a question and its answer are
    written in the same transaction and routinely share a timestamp, and the id
    is a random UUID, so the tie-break would be random too.

    Citations are stored as a JSON string in one column, so they need rehydrating
    into `Citation` objects; `model_validate` alone would leave the field empty
    because the column is named `citations_json`, not `citations`.
    """
    messages = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.paper_id == paper.id)
        .order_by(ChatMessage.turn_index)
    )

    out: list[ChatMessageOut] = []
    for message in messages:
        citations: list[Citation] = []
        if message.citations_json:
            for item in json.loads(message.citations_json):
                try:
                    citations.append(Citation(**item))
                except (TypeError, ValueError):
                    # A row written before the Citation schema gained
                    # page_start/page_end. Dropping the citation keeps old
                    # history readable rather than 500ing the whole endpoint.
                    logger.debug("Skipping unreadable stored citation")

        out.append(
            ChatMessageOut(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
                citations=citations,
            )
        )
    return out


def clear_conversation(db: Session, paper: Paper) -> None:
    db.execute(delete(ChatMessage).where(ChatMessage.paper_id == paper.id))
    paper.chat_summary = None
    paper.chat_summary_upto = -1
    db.commit()


def suggest_followups(
    db: Session, paper: Paper, last_question: str | None = None
) -> list[str]:
    """Three next questions, generated from the paper's actual content.

    Built from the report when one exists -- it is a compact, accurate digest of
    the whole paper, which is exactly what this needs and much cheaper than
    re-reading chunks. Failures return an empty list: suggestions are a
    convenience, and losing them must never cost the answer they follow.
    """
    report = analysis.get_report(db, paper)

    if report is not None:
        context = (
            f"WHAT THE PAPER DOES:\n{report.plain_language}\n\n"
            f"CONTRIBUTIONS:\n" + "\n".join(f"- {c}" for c in report.contributions[:5])
        )
        if report.key_results:
            context += "\n\nRESULTS:\n" + "\n".join(
                f"- {r}" for r in report.key_results[:5]
            )
    elif paper.abstract:
        context = f"ABSTRACT:\n{paper.abstract[:MAX_ABSTRACT_CHARS]}"
    else:
        overview = _section_overview(db, paper)
        if not overview:
            return []
        context = f"SECTIONS:\n{overview}"

    if last_question:
        context += f"\n\nThe reader just asked: {last_question}\nSuggest different angles."

    try:
        with llm.calling(paper.id, "followups"):
            raw = llm.generate(
                prompts.build_followups_prompt(paper.title, context),
                temperature=0.6,  # variety is the goal here, not determinism
                max_output_tokens=prompts.FOLLOWUP_MAX_TOKENS,
            )
    except llm.LLMError as exc:
        logger.info("Follow-up suggestions unavailable: %s", exc)
        return []

    questions: list[str] = []
    for line in raw.splitlines():
        cleaned = re.sub(r"^\s*[-*\d.)\]]+\s*", "", line).strip()
        if 10 < len(cleaned) < 300 and cleaned.endswith("?"):
            questions.append(cleaned)

    return questions[:3]
