"""
Deep analysis: map over sections, reduce into one report.

    for each section:  analyse it with a prompt chosen for its kind   (map)
    then:              synthesise those analyses into a report        (reduce)

**Why this replaced a single call.** The previous design sent the first 30,000
characters of the paper to one prompt and asked for a summary. That is not a
summary of the paper -- it is a summary of the paper's *beginning*. On anything
longer than about twelve pages the Results, Limitations and Conclusion never
reached the model at all, which is precisely why the output read as generic:
it was working from the introduction, and an introduction is where papers are
most generic.

Map-reduce fixes the coverage problem, and it fixes a subtler one too. A section
prompt can ask the right questions. "What are the datasets, splits and
baselines?" is a good question to ask of an experimental setup section and a
waste of a call against an introduction. One prompt spanning the whole paper has
to ask everything of everything, and gets shallow answers to all of it.

The cost is honest and worth naming: one API call per section instead of one per
paper -- typically 8 to 12 calls, plus the reduce. On a free tier that is real
quota. It runs once per paper and is cached against the prompt version, so the
cost is paid once.

Memory: one section's chunks are loaded, analysed, and released before the next
is read. The reduce stage holds the section analyses, which are prose summaries
-- a few KB each, tens of KB in total. Nothing here scales with the size of the
PDF.
"""

import json
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Chunk, Paper, Section
from app.rag import llm, prompts
from app.schemas import PaperReport
from app.services import jobs

logger = logging.getLogger(__name__)

# Sections we do not spend a call on. A reference list has no argument to
# analyse and an acknowledgements block has no content at all; both are still
# indexed and retrievable for chat, they simply do not earn a map-stage call
# against a free-tier quota.
SKIP_KINDS = {"acknowledgements"}

# A section shorter than this is a heading with a stray line under it -- a
# figure caption orphaned by the parser, or a page header. Analysing it wastes
# a call to produce "this section contains no substantive content".
MIN_SECTION_CHARS = 400

# Character budget for one section's excerpts. Generous: sections are the unit
# we deliberately chose to keep whole, and truncating them here would reintroduce
# exactly the bug this module exists to fix. A very long section (a 20-page
# appendix) is still capped, because a single prompt has limits.
MAX_SECTION_CHARS = 60_000

# Budget for the reduce stage's input. Section analyses are prose, ~1-2 KB each;
# a 12-section paper lands around 20 KB, well under this. The cap exists for the
# pathological case of a paper with 40 sections.
MAX_REDUCE_CHARS = 120_000


def _section_excerpts(db: Session, section: Section) -> tuple[str, int]:
    """Render one section's chunks as a numbered excerpt block."""
    chunks = list(
        db.scalars(
            select(Chunk)
            .where(Chunk.section_id == section.id)
            .order_by(Chunk.chunk_index)
        )
    )
    if not chunks:
        return "", 0

    kept: list[Chunk] = []
    used = 0
    for chunk in chunks:
        if used + len(chunk.content) > MAX_SECTION_CHARS:
            break
        kept.append(chunk)
        used += len(chunk.content)

    if len(kept) < len(chunks):
        logger.warning(
            "Section '%s' truncated for analysis: %d of %d chunks",
            section.name, len(kept), len(chunks),
        )

    return prompts.format_excerpts(kept), len(kept)


def _analyse_section(db: Session, paper: Paper, section: Section) -> str | None:
    """Run the map stage for one section. Returns its analysis, or None."""
    excerpts, count = _section_excerpts(db, section)
    if not count or section.char_count < MIN_SECTION_CHARS:
        return None

    prompt = prompts.build_section_prompt(
        kind=section.kind,
        section_name=section.name,
        paper_title=paper.title,
        excerpts=excerpts,
    )

    try:
        with llm.calling(paper.id, "map"):
            return llm.generate(
                prompt,
                temperature=0.2,
                max_output_tokens=prompts.MAP_MAX_TOKENS,
            )
    except llm.LLMError as exc:
        # One failed section must not fail the paper. An eleven-of-twelve report
        # is far more useful than an error page, and the reduce stage works from
        # whatever it is given. The gap is logged and the report simply will not
        # cite that section.
        logger.warning("Section '%s' analysis failed: %s", section.name, exc)
        return None


def _collect_analyses(db: Session, paper_id: str) -> str:
    """Assemble the stored per-section analyses into the reduce-stage input."""
    sections = db.scalars(
        select(Section)
        .where(Section.paper_id == paper_id, Section.analysis_json.is_not(None))
        .order_by(Section.ordinal)
    )

    blocks: list[str] = []
    used = 0
    for section in sections:
        try:
            analysis = json.loads(section.analysis_json)["text"]
        except (ValueError, KeyError, TypeError):
            continue

        pages = (
            f"page {section.page_start}"
            if section.page_start == section.page_end
            else f"pages {section.page_start}-{section.page_end}"
        )
        block = f"### SECTION: {section.name} ({pages})\n{analysis}"
        if used + len(block) > MAX_REDUCE_CHARS:
            break
        blocks.append(block)
        used += len(block)

    return "\n\n".join(blocks)


def _parse_report(raw: str) -> PaperReport:
    """Validate the reduce stage's JSON into the report schema.

    Pydantic is the guardrail, not trust. `json_mode` guarantees the response
    parses as JSON; only validation guarantees it has the fields we need and
    that a list field is a list rather than a string the model felt like
    writing instead.
    """
    return PaperReport.model_validate_json(raw)


def run_analysis(paper_id: str, job_id: str | None = None, force: bool = False) -> None:
    """Map-reduce the whole paper into a cached report.

    Runs as a background task. Opens its own session -- the request that started
    it is long gone.

    `force` re-runs even when a current-version report exists, for the UI's
    "regenerate" action.
    """
    db = SessionLocal()

    try:
        paper = db.get(Paper, paper_id)
        if paper is None:
            return

        # Cache check. Keyed on the prompt version, so editing a prompt
        # invalidates every stored report without anyone having to remember to
        # clear anything -- see the note in prompts.py.
        if (
            not force
            and paper.analysis_json
            and paper.analysis_prompt_version == prompts.PROMPT_VERSION
        ):
            if job_id:
                jobs.finish(db, job_id)
            return

        if job_id is None:
            job_id = jobs.create(db, paper_id, prompts.PROMPT_VERSION).id

        sections = list(
            db.scalars(
                select(Section)
                .where(Section.paper_id == paper_id)
                .order_by(Section.ordinal)
            )
        )
        analysable = [
            s for s in sections
            if s.kind not in SKIP_KINDS and s.char_count >= MIN_SECTION_CHARS
        ]

        if not analysable:
            jobs.fail(db, job_id, "The paper produced no sections long enough to analyse.")
            return

        # --- MAP ------------------------------------------------------------
        jobs.update(
            db, job_id,
            stage="analyzing",
            current_item=0,
            total=len(analysable),
            message=f"Analysing section 1 of {len(analysable)}",
        )

        succeeded = 0
        for index, section in enumerate(analysable, start=1):
            jobs.update(
                db, job_id,
                current_item=index - 1,
                message=f"Analysing section {index} of {len(analysable)}: {section.name}",
            )

            if force or not section.analysis_json:
                text = _analyse_section(db, paper, section)
                if text:
                    # Stored as JSON rather than raw text so the shape can grow
                    # (confidence, token counts) without a migration.
                    section.analysis_json = json.dumps({"text": text})
                    db.commit()

            if section.analysis_json:
                succeeded += 1

        jobs.update(db, job_id, current_item=len(analysable))

        if succeeded == 0:
            jobs.fail(
                db, job_id,
                "Every section analysis failed. This usually means the language "
                "model is unavailable or out of quota -- try again shortly.",
            )
            return

        # --- REDUCE ----------------------------------------------------------
        jobs.update(db, job_id, stage="synthesizing", message="Writing the report")

        analyses = _collect_analyses(db, paper_id)
        venue_year = " ".join(x for x in (paper.venue, paper.year) if x) or None
        prompt = prompts.build_report_prompt(
            paper_title=paper.title,
            authors=paper.authors,
            venue_year=venue_year,
            section_analyses=analyses,
        )

        # One retry, then give up. JSON mode makes malformed output rare, not
        # impossible; a second attempt at a slightly higher temperature converts
        # most of those failures. Retrying indefinitely would burn quota against
        # a prompt the model cannot satisfy.
        report: PaperReport | None = None
        last_error: Exception | None = None
        raw = ""  # bound before the loop so the logging in `except` is always safe

        for attempt in (1, 2):
            try:
                with llm.calling(paper_id, "reduce"):
                    raw = llm.generate(
                        prompt,
                        json_mode=True,
                        temperature=0.2 if attempt == 1 else 0.45,
                        max_output_tokens=prompts.REDUCE_MAX_TOKENS,
                    )
                report = _parse_report(raw)
                break
            except llm.LLMError as exc:
                last_error = exc
                logger.warning("Report generation failed on attempt %d: %s", attempt, exc)
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "Unusable report JSON on attempt %d (%d chars): %s ... TAIL: %s",
                    attempt, len(raw), raw[:400], raw[-200:],
                )

        if report is None:
            jobs.fail(db, job_id, f"Could not produce the final report: {last_error}")
            return

        paper.analysis_json = report.model_dump_json()
        paper.analysis_prompt_version = prompts.PROMPT_VERSION
        db.commit()

        jobs.finish(db, job_id)
        logger.info(
            "Analysis complete for paper %s: %d/%d sections",
            paper_id, succeeded, len(analysable),
        )

    except Exception as exc:  # noqa: BLE001 -- a background task must never die silently
        logger.exception("Unexpected analysis error")
        if job_id:
            jobs.fail(db, job_id, f"Unexpected error: {exc}")
    finally:
        db.close()


def get_report(db: Session, paper: Paper) -> PaperReport | None:
    """The cached report, or None if it is missing or stale.

    Stale means it was written by a different prompt version. Returning None
    rather than the old report is the point of storing the version at all.
    """
    if not paper.analysis_json:
        return None
    if paper.analysis_prompt_version != prompts.PROMPT_VERSION:
        logger.info(
            "Paper %s has a report from prompt %s, current is %s -- treating as absent",
            paper.id, paper.analysis_prompt_version, prompts.PROMPT_VERSION,
        )
        return None
    try:
        return PaperReport.model_validate_json(paper.analysis_json)
    except ValueError:
        # A stored report that no longer validates means the schema changed
        # under it. Same handling as stale: regenerate rather than crash.
        logger.warning("Stored report for paper %s failed validation", paper.id)
        return None
