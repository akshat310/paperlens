"""
Step 2: group pages into sections, and sections into semantic chunks.

Why chunk at all? Two reasons, both worth being able to say out loud:

  1. A vector is a single point in space. One vector for a whole paper averages
     every idea in it into mush -- it matches everything weakly and nothing
     strongly. Smaller units give sharper matches.
  2. The answer needs to cite something specific. A chunk is the unit a citation
     points at, so it has to be small enough to check by eye.

**What changed from the previous version, and why.** Chunks used to be 900
*characters* (~220 tokens) and were forbidden from crossing a page boundary.
That guaranteed one page number per chunk, which made citations simple. It also
meant every retrieved passage was a fragment: a method description split across
a page break arrived as two half-thoughts, and the model had to answer from
whichever half ranked higher. Chunks are now ~1000 tokens and follow *section*
boundaries instead, so a chunk is a coherent piece of argument.

The cost is that a chunk can now span pages, so `page_start` and `page_end` are
both recorded and a citation reads "pages 4-5" when it has to. That is a real
loss of precision, and it buys back a much larger gain in answer quality.

Splitting is hierarchical: paragraphs first, then sentences, then words. Never a
blind character cut except as a final backstop, because a chunk that begins
mid-word embeds badly and reads worse.
"""

import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from app.rag.pdf_parser import PageText

logger = logging.getLogger(__name__)

# Characters per token, for English prose in a research paper. Real tokenisers
# land between 3.8 and 4.3; we are sizing a budget, not billing anyone, so a
# constant is the honest tool. Using tiktoken here would add a dependency and a
# vocabulary download to make an estimate marginally less rough.
CHARS_PER_TOKEN = 4

DEFAULT_TARGET_TOKENS = 1000   # middle of the 800-1200 band
DEFAULT_OVERLAP_TOKENS = 120   # ~12%: one or two sentences of run-up
MIN_CHUNK_CHARS = 120          # below this a chunk is a page header, not content


@dataclass
class SectionSpan:
    """A logical section, accumulated across however many pages it covers."""

    kind: str
    name: str
    ordinal: int
    page_start: int
    page_end: int
    text: str = ""
    captions: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class TextChunk:
    content: str
    page_start: int
    page_end: int
    section_ordinal: int
    section_name: str | None
    chunk_index: int  # position within the paper


def iter_sections(pages: Iterable[PageText]) -> Iterator[SectionSpan]:
    """Consume the page stream and yield complete sections.

    Peak memory here is **one section**, not one document. A section is at most
    a few pages of text -- tens of kilobytes -- so buffering it is cheap, and it
    is the smallest unit that can be chunked coherently.

    Text before the first recognised heading is emitted as a `front_matter`
    section rather than discarded. On a paper whose headings are unconventional
    that can be the entire document, which is exactly the fallback we want: the
    text still gets chunked, embedded and retrieved, it simply carries a vaguer
    label. **No text is ever dropped for want of a recognised heading.**
    """
    current: SectionSpan | None = None
    ordinal = 0

    for page in pages:
        # Split the page's text at the offsets where headings were found. The
        # offsets were computed on the raw text and the text has since been
        # cleaned, so they are not usable as exact indices -- we re-find each
        # heading in the cleaned text instead, which is reliable because a
        # heading is a distinctive short string.
        segments = _split_page_at_headings(page)

        for heading, body in segments:
            if heading is not None:
                kind, name = heading
                if (
                    current is not None
                    and kind == "other"
                    and current.kind == "other"
                    and not current.text.strip()
                ):
                    # Two heading lines with nothing between them are one
                    # heading that wrapped -- a long title, or "Appendix for
                    # ..." over three lines. Join them rather than emit empty
                    # sections; the body below still attaches to it.
                    current.name = f"{current.name} {name}"[:300]
                else:
                    if current is not None:
                        yield current
                    ordinal += 1
                    current = SectionSpan(
                        kind=kind,
                        name=name,
                        ordinal=ordinal,
                        page_start=page.page_number,
                        page_end=page.page_number,
                    )

            if current is None:
                # Text before any heading. Title block, abstract on papers that
                # do not label it, arXiv stamp.
                ordinal += 1
                current = SectionSpan(
                    kind="front_matter",
                    name="Front matter",
                    ordinal=ordinal,
                    page_start=page.page_number,
                    page_end=page.page_number,
                )

            if body.strip():
                current.text = f"{current.text}\n\n{body.strip()}" if current.text else body.strip()
                current.page_end = page.page_number

        if current is not None and page.captions:
            current.captions.extend(page.captions)
            current.page_end = max(current.page_end, page.page_number)

    if current is not None:
        yield current


def _split_page_at_headings(
    page: PageText,
) -> list[tuple[tuple[str, str] | None, str]]:
    """Cut one page's cleaned text into (heading | None, body) segments."""
    if not page.headings:
        return [(None, page.text)]

    text = page.text
    segments: list[tuple[tuple[str, str] | None, str]] = []
    cursor = 0

    for kind, name, _raw_offset in page.headings:
        # Find the heading in the cleaned text, searching forward from where the
        # previous one ended so repeated words cannot match out of order.
        position = text.find(name, cursor)
        if position == -1:
            # Cleaning changed the heading's spacing. Fall back to a whitespace-
            # insensitive search rather than silently dropping the section.
            pattern = re.compile(r"\s+".join(re.escape(w) for w in name.split()))
            match = pattern.search(text, cursor)
            if match is None:
                continue
            position, end = match.start(), match.end()
        else:
            end = position + len(name)

        before = text[cursor:position]
        if before.strip():
            segments.append((None, before))
        segments.append(((kind, name), ""))
        cursor = end

    tail = text[cursor:]
    if tail.strip():
        if segments and segments[-1][0] is not None:
            # Attach the body to the heading that just opened.
            heading = segments[-1][0]
            segments[-1] = (heading, tail)
        else:
            segments.append((None, tail))

    return segments or [(None, page.text)]


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Sentence split that survives academic prose.

    The lookbehind refuses to break after a single capital letter (initials like
    "J. Doe") or a known abbreviation, which are the two things that shatter a
    naive `split('. ')` on a references-heavy paper.
    """
    protected = re.sub(
        r"\b(et al|e\.g|i\.e|cf|vs|Fig|Eq|Sec|Tab|Ref|approx|Dr|Prof)\.",
        lambda m: m.group(0).replace(".", "\x00"),
        text,
    )
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", protected)
    return [p.replace("\x00", ".").strip() for p in pieces if p.strip()]


def _pack(units: list[str], target_chars: int, joiner: str) -> list[str]:
    """Greedily fill chunks up to target_chars, never splitting a unit."""
    chunks: list[str] = []
    buffer: list[str] = []
    size = 0

    for unit in units:
        unit_len = len(unit)
        if size and size + unit_len > target_chars:
            chunks.append(joiner.join(buffer))
            buffer, size = [], 0
        buffer.append(unit)
        size += unit_len + len(joiner)

    if buffer:
        chunks.append(joiner.join(buffer))
    return chunks


def _chunk_text(text: str, target_chars: int) -> list[str]:
    """Split one section's text, preferring the largest natural boundary."""
    if len(text) <= target_chars:
        return [text]

    # Paragraphs first.
    paragraphs = _split_paragraphs(text)
    oversized = [p for p in paragraphs if len(p) > target_chars]

    if not oversized:
        return _pack(paragraphs, target_chars, "\n\n")

    # Some paragraph is longer than a whole chunk -- common in papers, where a
    # method description is one unbroken block. Fall to sentences for those.
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= target_chars:
            units.append(paragraph)
            continue
        for sentence in _split_sentences(paragraph):
            if len(sentence) <= target_chars:
                units.append(sentence)
            else:
                # A single sentence longer than the target: a table dumped as
                # prose, or a mangled equation. Word-split it. This is the
                # backstop, and it is the only place a unit is cut arbitrarily.
                words = sentence.split(" ")
                units.extend(_pack(words, target_chars, " "))

    return _pack(units, target_chars, " ")


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """The last `overlap_chars` of text, snapped forward to a sentence start.

    Snapping to a sentence rather than a raw character offset means the overlap
    reads as English. An overlap that begins mid-clause adds noise to the
    embedding of the chunk it is prepended to.
    """
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return text
    tail = text[-overlap_chars:]
    match = re.search(r"(?<=[.!?])\s+(?=[A-Z(\[])", tail)
    if match:
        return tail[match.end():]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def chunk_sections(
    sections: Iterable[SectionSpan],
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> Iterator[tuple[SectionSpan, list[TextChunk]]]:
    """Chunk each section as it arrives, yielding (section, its chunks).

    A generator over the section stream, so the caller can persist one section's
    chunks and move on. Nothing accumulates across sections.

    Reference lists are chunked but at a smaller target: a citation list has no
    argument running through it, so large chunks of it are just dead weight in
    the index, and retrieving one is rarely useful. We keep them because "which
    prior work does this build on?" is a real question a reader asks.
    """
    target_chars = target_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    index = 0

    for section in sections:
        body = section.text
        # Captions are appended to their section so a figure's finding is
        # retrievable, and labelled so the model knows it is reading a caption.
        if section.captions:
            body = f"{body}\n\n" + "\n".join(section.captions)

        if section.kind == "references":
            section_target = max(400 * CHARS_PER_TOKEN, target_chars // 2)
            section_overlap = 0  # nothing to carry over between citations
        else:
            section_target, section_overlap = target_chars, overlap_chars

        pieces = _chunk_text(body, section_target)

        chunks: list[TextChunk] = []
        previous: str | None = None
        for piece in pieces:
            content = piece.strip()
            if previous is not None and section_overlap:
                content = f"{_overlap_tail(previous, section_overlap)} {content}".strip()
            previous = piece

            collapsed = re.sub(r"[ \t]+", " ", content).strip()
            if len(collapsed) < MIN_CHUNK_CHARS:
                continue

            chunks.append(
                TextChunk(
                    content=collapsed,
                    # We know the section's page range but not which page each
                    # chunk fell on. Attributing the whole range is the honest
                    # answer: the citation says "pages 4-6" and the reader finds
                    # it, rather than a precise page number we cannot support.
                    # Single-page sections -- most of them -- are unaffected.
                    page_start=section.page_start,
                    page_end=section.page_end,
                    section_ordinal=section.ordinal,
                    section_name=section.name,
                    chunk_index=index,
                )
            )
            index += 1

        if not chunks and body.strip():
            logger.debug(
                "Section '%s' produced no chunks (%d chars)", section.name, len(body)
            )

        yield section, chunks
