"""
Step 1: turn a PDF into page-tagged text and a section structure.

Two properties drive every choice in this file.

**Pages are attached to everything.** Text is extracted per page and the page
number rides along through chunking into the citation the user eventually
clicks. Without it the assistant can say "the paper says X" but not "page 7 says
X", and an unverifiable citation is worse than none.

**Nothing whole-document is held in memory.** `iter_pages` is a generator: it
yields one page, and the caller is expected to write what it needs to storage
and let the page go. The previous version built a list of every page, then the
chunker built a second full copy, then ingestion built a third as ORM rows --
three copies of the document alive at once. Here the peak is one page.

Parser choice: pypdf, not PyMuPDF. PyMuPDF binds MuPDF, a full rendering engine
with a large native library resident for the life of the process, and it can
rasterize pages -- capability we never used and cannot afford. pypdf is pure
Python and reads one page object at a time. The honest cost is extraction
quality: pypdf follows the PDF content stream's drawing order, so a two-column
layout can interleave columns where MuPDF would have reconstructed the reading
order. The README says so plainly rather than hiding it.
"""

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Section vocabulary
#
# Each entry is (normalised kind, pattern). The kind is what selects an
# analysis prompt in Phase 2, so several surface spellings collapse onto one:
# "Methodology", "Approach" and "Our Model" are all `method`.
#
# Matching is deliberately conservative. A false heading is worse than a missed
# one, because it mislabels every chunk that follows it -- and unlike a missed
# heading, there is no fallback that recovers from it.
# --------------------------------------------------------------------------
SECTION_RULES: list[tuple[str, str]] = [
    ("abstract", r"abstract"),
    ("introduction", r"introduction"),
    ("related_work", r"(?:related work|background|prior work|literature review)"),
    ("method", r"(?:method(?:s|ology)?|approach|model|architecture|our model|"
               r"proposed (?:method|approach|model)|system design|framework)"),
    ("experiments", r"(?:experiment(?:s|al)?(?: setup| settings?| details?)?|"
                    r"setup|implementation details|training details|"
                    r"dataset(?:s)?|data|evaluation setup|baselines?)"),
    ("results", r"(?:result(?:s)?|evaluation|findings|main results|"
                r"experimental results|ablation(?: stud(?:y|ies))?|analysis)"),
    ("discussion", r"discussion"),
    ("limitations", r"(?:limitation(?:s)?|threats to validity|"
                    r"broader impact(?:s)?|ethical considerations)"),
    ("conclusion", r"(?:conclusion(?:s)?|concluding remarks|"
                   r"conclusions? and future work|future work|summary)"),
    ("references", r"(?:reference(?:s)?|bibliography|works cited)"),
    ("acknowledgements", r"acknowledg(?:e)?ment(?:s)?"),
    ("appendix", r"(?:appendix|supplementary(?: material)?)"),
]

# A heading line is: optional numbering ("3", "3.2", "IV", "A.1"), the keyword,
# an optional short qualifier, and nothing else. Anchored at both ends so a
# sentence that merely contains the word "introduction" is not mistaken for a
# heading.
#
# The trailing qualifier matters in practice: real papers write "3. Method
# Overview", "4 Experimental Results" and "5 Datasets and Baselines", none of
# which match the bare keyword. It is capped at three words and forbids
# sentence-ending punctuation, so it cannot swallow prose -- and the length and
# shape checks in `_classify_heading` still apply on top.
_NUMBERING = r"(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.|[A-Z]\.(?:\d+)?)?\s*"
_QUALIFIER = r"(?:\s+[\w&/-]+){0,3}"
_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (
        kind,
        re.compile(rf"^\s*{_NUMBERING}{pattern}{_QUALIFIER}\s*[:.]?\s*$", re.IGNORECASE),
    )
    for kind, pattern in SECTION_RULES
]

# A numbered heading whose *title* we don't recognise -- "4 Sparse Attention
# Kernels". Real structure, unknown vocabulary. We keep it as a section with
# kind="other" rather than folding it into whatever came before, because
# dropping it would merge two genuinely different parts of the paper.
_GENERIC_HEADING = re.compile(
    r"^\s*(\d+(?:\.\d+)*\.?)\s+([A-Z][^.!?]{2,60})\s*$"
)

# Figure/table captions. Captured separately because they carry a
# disproportionate amount of a paper's actual findings ("Table 2: accuracy on
# CIFAR-100") and are easy to lose in the body text.
_CAPTION = re.compile(
    r"^\s*((?:figure|fig\.?|table|algorithm)\s*\d+[.:]?\s+.{10,400})$",
    re.IGNORECASE,
)

# Lines that are mostly mathematical symbols. Kept as text (an equation the
# model can read) but marked so chunking does not treat them as prose.
_EQUATION_CHARS = set("=+-*/^_∑∏∫√≈≤≥≠∈∀∃αβγδθλμσφψΩ∇∂|⟨⟩")


@dataclass
class PageText:
    """One page, as it comes out of the PDF."""

    page_number: int  # 1-based, matches what a human sees in a PDF reader
    text: str
    # Headings that started on this page, as (kind, display name, char offset).
    headings: list[tuple[str, str, int]] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)


@dataclass
class DocumentMeta:
    title: str
    authors: str | None
    venue: str | None
    year: str | None
    num_pages: int


def _clean_text(text: str) -> str:
    """Repair the usual damage PDF extraction does to text.

    PDFs store glyph positions, not sentences, so extracted text arrives with
    hyphenated line breaks, hard-wrapped lines and stray whitespace. Feeding
    that to an embedding model measurably hurts retrieval quality.

    Note this runs *after* heading detection, not before: unwrapping newlines
    would destroy the line structure headings are recognised by.
    """
    # Re-join words split across lines: "transfor-\nmer" -> "transformer"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Ligatures pypdf sometimes passes through as single code points.
    for bad, good in (("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("ﬃ", "ffi"), ("ﬄ", "ffl")):
        text = text.replace(bad, good)
    # Turn single newlines (hard wrapping) into spaces, keep paragraph breaks.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _classify_heading(line: str) -> tuple[str, str] | None:
    """Return (kind, display name) if this line looks like a section heading."""
    stripped = line.strip()
    # Headings are short. A 200-character line is a paragraph.
    if not stripped or len(stripped) > 80:
        return None
    # A line ending in a sentence-ending period with lowercase words is prose.
    # Headings are capitalised. "approach." on its own line is the tail of a
    # wrapped sentence that happens to end in a vocabulary word.
    first_alpha = next((ch for ch in stripped if ch.isalpha()), "")
    if first_alpha.islower():
        return None
    for kind, pattern in _COMPILED:
        if pattern.match(stripped):
            return kind, stripped
    match = _GENERIC_HEADING.match(stripped)
    if match:
        # Reject if it reads like a sentence: headings are not mostly lowercase
        # function words, and they do not end in a comma.
        title = match.group(2).strip()
        if title.endswith(",") or title.count(" ") > 8:
            return None
        # "2018. Contextual string embeddings..." is a bibliography entry
        # whose year happens to look like section numbering. A four-digit
        # number is never a section number.
        if re.fullmatch(r"(?:19|20)\d{2}\.?", match.group(1)):
            return None
        return "other", stripped
    return None


# --------------------------------------------------------------------------
# Typography as a second heading signal
#
# The vocabulary above recognises "3 Method" but not "4 Sparse Attention
# Kernels", and the numbered-heading fallback cannot tell a heading from a
# numbered list item. The PDF knows more than the text does: a heading is set
# in a bolder or larger font than the body. pypdf exposes that per fragment
# through `visitor_text`, at no extra parse cost.
#
# The unconstrained answer is a layout model (GROBID, Nougat); those are a
# Java service or a torch model and neither fits the budget. Font weight and
# size are the two cheapest signals the file already carries, and they cover
# the common case -- LaTeX section headings -- well.
# --------------------------------------------------------------------------

_BOLD_MARKERS = ("bold", "medi", "heavy", "black", "semibold", "demibold", "-b,", "-bd")
_HEADING_SIZE_RATIO = 1.15  # a line this much larger than body text is a heading


@dataclass
class _LineStyle:
    text: str
    size: float
    bold: bool  # every fragment on the line is in a bold face


def _font_is_bold(font_dict) -> bool:
    name = str((font_dict or {}).get("/BaseFont", "")).lower()
    return any(marker in name for marker in _BOLD_MARKERS)


def _extract_with_styles(page) -> tuple[str, list[_LineStyle] | None]:
    """Page text plus, when the visitor output lines up with it, per-line style.

    Returns (text, styles) where `styles` is None if the fragments the visitor
    saw do not concatenate to exactly the text pypdf returned -- in that case
    line indices would not correspond and the signal is dropped rather than
    misapplied. In practice they match; the check is cheap insurance.
    """
    fragments: list[tuple[str, float, bool]] = []

    def visitor(text, cm, tm, font_dict, font_size):
        # Effective size is the nominal Tf size scaled by the text matrix and
        # the CTM; LaTeX output usually keeps both at 1 and varies Tf, but
        # some generators do the opposite.
        scale = abs(tm[0] if tm else 1.0) * abs(cm[0] if cm else 1.0)
        fragments.append((text, float(font_size or 0) * (scale or 1.0), _font_is_bold(font_dict)))

    text = page.extract_text(visitor_text=visitor) or ""
    if "".join(f[0] for f in fragments) != text:
        return text, None

    lines: list[_LineStyle] = []
    current: list[tuple[str, float, bool]] = []

    def flush() -> None:
        content = "".join(p for p, _, _ in current)
        inked = [(s, b) for p, s, b in current if p.strip()]
        lines.append(
            _LineStyle(
                text=content,
                size=max((s for s, _ in inked), default=0.0),
                bold=bool(inked) and all(b for _, b in inked),
            )
        )

    for piece, size, bold in fragments:
        parts = piece.split("\n")
        for index, part in enumerate(parts):
            if part:
                current.append((part, size, bold))
            if index < len(parts) - 1:
                flush()
                current = []
    if current:
        flush()

    # Sanity: the line count must match a plain split, or indices are off.
    if len(lines) != len(text.split("\n")):
        return text, None
    return text, lines


def _body_size(styles: list[_LineStyle]) -> float:
    """The most common font size on the page, weighted by characters."""
    weights: dict[float, int] = {}
    for line in styles:
        if line.size > 0:
            key = round(line.size, 1)
            weights[key] = weights.get(key, 0) + len(line.text.strip())
    return max(weights, key=weights.get) if weights else 0.0


_STYLED_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][^.!?]{1,70}\s*$")


def _styled_heading(line: str, style: _LineStyle | None, body: float) -> bool:
    """A short, title-shaped line set in bold or larger type than the body."""
    if style is None or body <= 0:
        return False
    emphasised = style.bold or style.size >= body * _HEADING_SIZE_RATIO
    if not emphasised:
        return False
    stripped = line.strip()
    if not _STYLED_HEADING.match(stripped) or stripped.count(" ") > 10:
        return False
    # A single bold word with no numbering is far more often a table header
    # ("Model", "Accuracy") than a section. Ask for numbering or two words.
    if " " not in stripped:
        return False
    # "Encoder: The encoder is composed..." starts bold and continues regular;
    # `style.bold` already requires the whole line, so a run-in heading that
    # shares its line with body text is correctly not a section break.
    return True


def _is_equation_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > 200:
        return False
    symbols = sum(1 for ch in stripped if ch in _EQUATION_CHARS)
    letters = sum(1 for ch in stripped if ch.isalpha())
    # Dense in operators and sparse in words.
    return symbols >= 2 and symbols * 3 >= letters


def read_metadata(file_path: str, max_pages: int | None = None) -> DocumentMeta:
    """Read title/authors/venue/year without holding the document open.

    Opens the PDF a second time, separately from `iter_pages`. That is a
    deliberate trade: one extra parse of the front matter, in exchange for
    `iter_pages` never needing to keep state about "have I done the metadata
    yet". Only pages 1-2 are read here.

    `max_pages` is enforced here because this is the first place the page
    count is known, and it is known before any page has been extracted --
    rejecting a 2,000-page file costs nothing at this point and a great deal
    one stage later.
    """
    reader = _open(file_path)
    try:
        info = reader.metadata or {}
        num_pages = len(reader.pages)
        if max_pages is not None and num_pages > max_pages:
            raise ValueError(
                f"This PDF has {num_pages} pages; the limit is {max_pages}. "
                "Split it, or upload the part you want analysed."
            )

        first = ""
        if num_pages:
            first = reader.pages[0].extract_text() or ""

        title = _pick_title(info, first)
        authors = _pick_authors(info, first, title)
        venue, year = _pick_venue_year(first)
        return DocumentMeta(
            title=title,
            authors=authors,
            venue=venue,
            year=year,
            num_pages=num_pages,
        )
    finally:
        # pypdf holds the file handle on the reader's stream.
        if hasattr(reader, "stream") and hasattr(reader.stream, "close"):
            reader.stream.close()


def _open(file_path: str) -> PdfReader:
    try:
        return PdfReader(file_path)
    except (PdfReadError, OSError, ValueError) as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc


def _pick_title(info: dict, first_page: str) -> str:
    """Prefer PDF metadata; fall back to the first substantial line of page 1."""
    meta_title = str(info.get("/Title") or "").strip()
    # Many PDFs carry junk metadata titles like "Microsoft Word - paper.doc",
    # or the LaTeX source filename.
    if (
        10 < len(meta_title) < 300
        and not meta_title.lower().endswith((".doc", ".docx", ".tex", ".dvi", ".pdf"))
        and not meta_title.lower().startswith("microsoft word")
    ):
        return meta_title

    for line in first_page.split("\n"):
        candidate = line.strip()
        # Skip arXiv stamps and similar running heads.
        if candidate.lower().startswith(("arxiv:", "preprint", "under review")):
            continue
        if 15 < len(candidate) < 300:
            return candidate
    return "Untitled Paper"


def _pick_authors(info: dict, first_page: str, title: str) -> str | None:
    meta_authors = str(info.get("/Author") or "").strip()
    if 3 < len(meta_authors) < 500:
        return meta_authors

    # Otherwise: the lines just after the title, before the abstract. Author
    # lines are comma- or 'and'-separated names, rarely longer than 200 chars.
    lines = [ln.strip() for ln in first_page.split("\n") if ln.strip()]
    try:
        start = next(i for i, ln in enumerate(lines) if title[:40] in ln) + 1
    except StopIteration:
        start = 1
    for line in lines[start : start + 6]:
        if line.lower().startswith("abstract"):
            break
        if 5 < len(line) < 200 and ("," in line or " and " in line.lower()):
            # Reject affiliation-looking lines.
            if not re.search(r"universit|institute|laborator|@|\.edu|\.com", line, re.I):
                return line
    return None


def _pick_venue_year(first_page: str) -> tuple[str | None, str | None]:
    """Best-effort venue and year from the first page's running head."""
    year = None
    match = re.search(r"\b(19|20)\d{2}\b", first_page[:1500])
    if match:
        year = match.group(0)

    venue = None
    venue_match = re.search(
        r"\b(NeurIPS|ICML|ICLR|CVPR|ECCV|ICCV|ACL|EMNLP|NAACL|AAAI|IJCAI|"
        r"KDD|SIGIR|WWW|ICSE|OSDI|SOSP|arXiv)\b",
        first_page[:2000],
        re.IGNORECASE,
    )
    if venue_match:
        venue = venue_match.group(0)
    return venue, year


def iter_pages(file_path: str) -> Iterator[PageText]:
    """Yield one page at a time. The caller must not retain what it yields.

    This is the memory-critical function in the whole ingest path. It is a
    generator on purpose: at any moment exactly one page's raw and cleaned text
    is alive, so peak usage is a function of the largest page, not of the
    document. `del` on the raw string is not ceremony -- the cleaned copy is a
    separate allocation and the raw one is often the larger of the two.
    """
    reader = _open(file_path)

    try:
        for index, page in enumerate(reader.pages):
            try:
                raw, styles = _extract_with_styles(page)
            except Exception as exc:  # noqa: BLE001 -- one bad page is not fatal
                # A malformed content stream on page 12 should cost page 12, not
                # the upload. The rest of the paper is still worth indexing.
                logger.warning("Page %d could not be extracted: %s", index + 1, exc)
                continue

            if not raw.strip():
                continue

            headings: list[tuple[str, str, int]] = []
            captions: list[str] = []
            offset = 0
            body = _body_size(styles) if styles else 0.0
            for line_index, line in enumerate(raw.split("\n")):
                style = styles[line_index] if styles else None
                classified = _classify_heading(line)
                if classified is None and _styled_heading(line, style, body):
                    # Real structure, unknown vocabulary, but the typography
                    # says heading. Kept as its own section rather than folded
                    # into whatever came before.
                    classified = ("other", line.strip())
                if classified:
                    headings.append((classified[0], classified[1], offset))
                else:
                    caption = _CAPTION.match(line.strip())
                    if caption:
                        captions.append(" ".join(caption.group(1).split()))
                offset += len(line) + 1

            cleaned = _clean_text(raw)
            del raw  # the larger of the two allocations; drop it before yielding

            if cleaned:
                yield PageText(
                    page_number=index + 1,
                    text=cleaned,
                    headings=headings,
                    captions=captions,
                )
            # The PageText the caller received is theirs; ours goes out of scope
            # on the next iteration.
    finally:
        if hasattr(reader, "stream") and hasattr(reader.stream, "close"):
            reader.stream.close()
