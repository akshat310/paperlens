"""
"Papers this builds on": the reference list, enriched from OpenAlex.

The references section is chunked and searchable but never *structured*. This
turns it into entries -- title, first author, year -- and asks OpenAlex (free,
no key, ~250M works) for each one's abstract, venue, citation count and a link.

Two honest limits:

1. **Parsing a bibliography with regexes is approximate.** Reference formats
   vary wildly and pypdf's extraction does not preserve entry boundaries.
   The parser here splits on the patterns that cover the common styles
   ("[12] Author...", "12. Author...", and blank-line-separated entries) and
   pulls a title-shaped span from each. It will miss some and mangle a few.
   The OpenAlex title search is fuzzy enough to absorb most of the damage,
   and a miss is shown as "not found", not hidden.

2. **It is network-bound, and OpenAlex asks for politeness.** One request per
   reference, capped at MAX_LOOKUPS, sent with a contact address in the
   User-Agent (their "polite pool" gives a higher rate limit). The results
   are cached on the paper as JSON so it is one round of calls per paper, not
   one per page view.

Memory: nothing is held beyond the list of entries (a few KB). Zero model
calls -- this is a metadata lookup, not a generation.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Paper, Section

logger = logging.getLogger(__name__)

OPENALEX = "https://api.openalex.org/works"
USER_AGENT = "PaperLens/2.0 (mailto:paperlens@example.com)"
MAX_LOOKUPS = 25
TIMEOUT = httpx.Timeout(10.0)

# Entry boundaries. Extraction has flattened the list into a few long lines, so
# "[12]" markers are found anywhere, not only at line starts. The "12." style
# is only trusted at a line start (a year ends with "." mid-line too).
_BRACKET_MARKER = re.compile(r"\s*\[\d{1,3}\]\s+")
_LINE_NUMBER_OR_BLANK = re.compile(r"(?:^|\n)\s*\d{1,3}\.\s+(?=[A-Z])|\n\s*\n")
_YEAR = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
# A title-shaped span: after the year (APA-ish) or after the first period
# that follows the author list (IEEE-ish), up to the next period.
_TITLE_AFTER_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?[\).]?\s*[\.:]?\s*([A-Z][^.]{10,200}?)\.\s")
_TITLE_AFTER_AUTHORS = re.compile(r"^[^.]{5,200}?\.\s+([A-Z][^.]{10,200}?)\.\s")


@dataclass
class RelatedEntry:
    raw: str
    title: str | None
    year: str | None
    first_author: str | None
    # From OpenAlex; None when not found.
    openalex_id: str | None = None
    matched_title: str | None = None
    abstract: str | None = None
    venue: str | None = None
    cited_by: int | None = None
    url: str | None = None


def references_text(db: Session, paper: Paper) -> str:
    """The reference section's text, from its chunks, in order."""
    sections = list(
        db.scalars(
            select(Section).where(Section.paper_id == paper.id, Section.kind == "references")
        )
    )
    if not sections:
        return ""
    ids = [s.id for s in sections]
    chunks = db.scalars(
        select(Chunk).where(Chunk.section_id.in_(ids)).order_by(Chunk.chunk_index)
    )
    return "\n".join(c.content for c in chunks)


def parse_references(text: str) -> list[RelatedEntry]:
    entries: list[RelatedEntry] = []
    if len(_BRACKET_MARKER.findall(text)) >= 3:
        pieces = _BRACKET_MARKER.split(text)
    else:
        pieces = _LINE_NUMBER_OR_BLANK.split(text)
    for piece in pieces:
        raw = " ".join((piece or "").split())
        if len(raw) < 30 or len(raw) > 600:
            continue
        year_match = _YEAR.search(raw)
        year = year_match.group(0)[:4] if year_match else None

        title = None
        for pattern in (_TITLE_AFTER_YEAR, _TITLE_AFTER_AUTHORS):
            match = pattern.search(raw)
            if match:
                title = match.group(1).strip()
                break

        first_author = None
        # Up to the first comma, " and ", or period: "Jimmy Lei Ba", "Ba, J.",
        # "J. Ba" all come out as a usable name.
        author_match = re.match(r"([A-Z][^,.]{1,40}?(?:\.\s?[A-Z][^,.]{1,30})?)(?:,| and |\.\s)", raw)
        if author_match:
            first_author = author_match.group(1).strip()

        entries.append(RelatedEntry(raw=raw, title=title, year=year, first_author=first_author))
    return entries


def _reconstruct_abstract(inverted: dict | None) -> str | None:
    """OpenAlex stores abstracts as an inverted index {word: [positions]}."""
    if not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, places in inverted.items():
        for place in places:
            positions.append((place, word))
    positions.sort()
    text = " ".join(word for _, word in positions)
    return text[:1200] if text else None


def lookup(entry: RelatedEntry, client: httpx.Client) -> None:
    """Fill the OpenAlex fields on one entry, in place. Failures leave it empty."""
    if not entry.title:
        return
    try:
        response = client.get(
            OPENALEX,
            params={
                "search": entry.title,
                "per-page": 1,
                "select": "id,title,publication_year,cited_by_count,primary_location,"
                          "abstract_inverted_index,doi",
            },
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            return
        work = results[0]
        # Guard against a confidently wrong match: the year should agree
        # within a year when we know it.
        if entry.year and work.get("publication_year"):
            if abs(int(entry.year) - int(work["publication_year"])) > 1:
                return
        entry.openalex_id = work.get("id")
        entry.matched_title = work.get("title")
        entry.cited_by = work.get("cited_by_count")
        entry.abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        location = work.get("primary_location") or {}
        source = location.get("source") or {}
        entry.venue = source.get("display_name")
        entry.url = work.get("doi") or location.get("landing_page_url") or work.get("id")
    except Exception as exc:  # noqa: BLE001 -- one bad lookup must not sink the list
        logger.info("OpenAlex lookup failed for %r: %s", entry.title[:60], exc)


def cached(paper: Paper) -> list[dict] | None:
    if not paper.related_json:
        return None
    try:
        return json.loads(paper.related_json)
    except ValueError:
        return None


# Papers whose enrichment is queued or running, so a page that polls every
# two seconds does not enqueue the same job twenty times. In-memory is
# correct here: there is one process, and a lost entry only means one
# duplicate run after a restart.
_in_flight: set[str] = set()


def enqueue(paper_id: str) -> None:
    """Compute on the worker thread. Twenty-five OpenAlex round-trips take
    tens of seconds -- far too long to hold one of the two request threads."""
    from app.services import worker

    if paper_id in _in_flight:
        return
    _in_flight.add(paper_id)
    worker.enqueue(_compute, paper_id)


def _compute(paper_id: str) -> None:
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        paper = db.get(Paper, paper_id)
        if paper is None:
            return
        entries = parse_references(references_text(db, paper))
        with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT) as client:
            for entry in entries[:MAX_LOOKUPS]:
                lookup(entry, client)
        paper.related_json = json.dumps([asdict(e) for e in entries])
        db.commit()
    finally:
        _in_flight.discard(paper_id)
        db.close()
