"""
Render a report as Markdown, and Markdown as PDF.

Markdown is generated here rather than in the frontend because it is the same
document the PDF is built from -- generating it twice in two languages would
guarantee they drift.

**The PDF is built by hand, with no dependency.** ReportLab, WeasyPrint and
anything wkhtmltopdf-based are all substantial installs, and WeasyPrint pulls
Cairo and Pango as system libraries. For a document that is headings and
paragraphs, the PDF format's own text-layout primitives are enough: a PDF is a
handful of objects, a content stream of positioned text, and a cross-reference
table. It is about 150 lines, it costs nothing at runtime, and it is a genuinely
interesting thing to be able to explain.

Limits, stated rather than hidden: WinAnsi encoding with the built-in Helvetica
font, so non-Latin-1 characters are transliterated or dropped. Greek letters in
a maths-heavy paper will not survive. The Markdown export has no such limit, and
the UI offers both.
"""

import logging
import re
from datetime import datetime, timezone

from app.models import Paper
from app.schemas import PaperReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

# (attribute on PaperReport, heading, is it a list?)
REPORT_LAYOUT: list[tuple[str, str, bool]] = [
    ("plain_language", "In plain language", False),
    ("problem", "The problem", False),
    ("contributions", "Contributions", True),
    ("method_walkthrough", "How the method works", False),
    ("experimental_setup", "Experimental setup", False),
    ("key_results", "Key results", True),
    ("results_interpretation", "What the results show", False),
    ("limitations_stated", "Limitations the authors state", True),
    ("limitations_observed", "Limitations a careful reader would notice", True),
    ("assumptions", "Assumptions", True),
    ("prior_work", "Relation to prior work", False),
    ("reproducibility", "Reproducibility", False),
    ("open_questions", "Open questions", True),
]


def report_to_markdown(paper: Paper, report: PaperReport) -> str:
    lines: list[str] = [f"# {paper.title}", ""]

    meta = [x for x in (paper.authors, paper.venue, paper.year) if x]
    if meta:
        lines += [f"*{' · '.join(meta)}*", ""]
    lines += [f"*{paper.num_pages} pages · analysed by PaperLens*", "", "---", ""]

    for attribute, heading, is_list in REPORT_LAYOUT:
        value = getattr(report, attribute, None)
        if not value:
            continue
        lines.append(f"## {heading}")
        lines.append("")
        if is_list:
            lines += [f"- {item}" for item in value]
        else:
            lines.append(str(value))
        lines.append("")

    if report.glossary:
        lines += ["## Glossary", ""]
        lines += [f"**{term.term}** — {term.definition}" + "\n" for term in report.glossary]

    lines += [
        "---",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d')}. Every claim "
        "above is drawn from the paper's own text; section and page references "
        "are the paper's.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

PAGE_WIDTH, PAGE_HEIGHT = 595, 842   # A4 in PDF points
MARGIN = 56
LINE_HEIGHT = 14
BODY_SIZE = 10
H1_SIZE, H2_SIZE = 18, 13
MAX_WIDTH = PAGE_WIDTH - 2 * MARGIN

# Helvetica advance widths per character, as a fraction of font size. A real
# implementation reads these from the font's metrics; for the built-in fonts
# they are a fixed table, and this approximation (average lowercase width, with
# the obvious narrow and wide characters corrected) is close enough that lines
# wrap where you expect.
_NARROW = set("ijlt.,;:!|'`()[]{}/\\ ")
_WIDE = set("mwMW@%")


def _text_width(text: str, size: float) -> float:
    total = 0.0
    for char in text:
        if char in _NARROW:
            total += 0.30
        elif char in _WIDE:
            total += 0.85
        elif char.isupper():
            total += 0.68
        else:
            total += 0.53
    return total * size


def _escape(text: str) -> str:
    """Escape the three characters that are syntax inside a PDF string."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _to_latin1(text: str) -> str:
    """Transliterate to something WinAnsi can represent.

    The common typographic characters are mapped rather than dropped, because
    losing every em dash and curly quote makes the output look broken. Anything
    still unrepresentable becomes '?' -- visible, so it is obvious what happened,
    rather than silently vanishing.
    """
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "--", "…": "...", " ": " ",
        "−": "-", "×": "x", "→": "->", "≤": "<=",
        "≥": ">=", "≈": "~=", "±": "+/-", "•": "-",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "replace").decode("latin-1")


def _wrap(text: str, size: float, width: float) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _text_width(candidate, size) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text


def report_to_pdf(paper: Paper, report: PaperReport) -> bytes:
    """Lay the report out as a PDF and serialise it.

    Two passes in spirit: build a list of (font, size, text) lines, paginating
    as we go, then write the file. Page content streams are held as strings
    until the end -- a 20-page report is well under a megabyte, so there is no
    reason to be cleverer.
    """
    # --- Build the line list ---------------------------------------------
    blocks: list[tuple[str, float, str]] = []  # (font key, size, text)

    blocks.append(("H1", H1_SIZE, _strip_markdown(paper.title)))
    meta = [x for x in (paper.authors, paper.venue, paper.year) if x]
    if meta:
        blocks.append(("I", BODY_SIZE, " - ".join(meta)))
    blocks.append(("SPACE", 0, ""))

    for attribute, heading, is_list in REPORT_LAYOUT:
        value = getattr(report, attribute, None)
        if not value:
            continue
        blocks.append(("SPACE", 0, ""))
        blocks.append(("H2", H2_SIZE, heading))
        if is_list:
            for item in value:
                blocks.append(("BULLET", BODY_SIZE, _strip_markdown(str(item))))
        else:
            for paragraph in str(value).split("\n"):
                if paragraph.strip():
                    blocks.append(("BODY", BODY_SIZE, _strip_markdown(paragraph.strip())))

    if report.glossary:
        blocks.append(("SPACE", 0, ""))
        blocks.append(("H2", H2_SIZE, "Glossary"))
        for term in report.glossary:
            blocks.append(("BULLET", BODY_SIZE, f"{term.term}: {term.definition}"))

    # --- Paginate ---------------------------------------------------------
    pages: list[list[str]] = []
    current: list[str] = []
    y = PAGE_HEIGHT - MARGIN

    def new_page() -> None:
        nonlocal current, y
        if current:
            pages.append(current)
        current = []
        y = PAGE_HEIGHT - MARGIN

    for kind, size, text in blocks:
        if kind == "SPACE":
            y -= LINE_HEIGHT * 0.6
            continue

        font = {"H1": "/F2", "H2": "/F2", "I": "/F3"}.get(kind, "/F1")
        indent = 12 if kind == "BULLET" else 0
        prefix = "- " if kind == "BULLET" else ""
        leading = size * 1.45

        lines = _wrap(_to_latin1(prefix + text), size, MAX_WIDTH - indent)
        for index, line in enumerate(lines):
            if y - leading < MARGIN:
                new_page()
            # Hanging indent: continuation lines of a bullet align under the text.
            x = MARGIN + indent + (10 if kind == "BULLET" and index > 0 else 0)
            y -= leading
            current.append(
                f"BT {font} {size:.1f} Tf 1 0 0 1 {x:.1f} {y:.1f} Tm "
                f"({_escape(line)}) Tj ET"
            )
        y -= leading * 0.35

    new_page()
    if not pages:
        pages = [[]]

    return _serialise(pages)


def _serialise(pages: list[list[str]]) -> bytes:
    """Assemble the PDF object graph and cross-reference table.

    A PDF is a set of numbered objects plus a table of their byte offsets. The
    offsets must be exact, which is why the objects are written into one buffer
    and measured as they go rather than assembled from parts.

    Object layout: 1 = catalog, 2 = page tree, 3-5 = fonts, then two objects per
    page (the page dict and its content stream).
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # object numbers are 1-based

    n_pages = len(pages)
    first_page_obj = 6  # after catalog, pages, and three fonts
    page_ids = [first_page_obj + i * 2 for i in range(n_pages)]

    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    add(
        f"<< /Type /Pages /Count {n_pages} /Kids [".encode()
        + b" ".join(f"{pid} 0 R".encode() for pid in page_ids)
        + b"] >>"
    )
    # The three base-14 fonts. No embedding needed: every conforming reader has
    # them, which is what keeps this file dependency-free.
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>")

    for index, lines in enumerate(pages):
        content = "\n".join(lines).encode("latin-1", "replace")
        page_id = page_ids[index]
        stream_id = page_id + 1
        add(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> "
            f"/Contents {stream_id} 0 R >>".encode()
        )
        add(
            f"<< /Length {len(content)} >>\nstream\n".encode()
            + content
            + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    ).encode()

    return bytes(out)


def safe_filename(title: str, extension: str) -> str:
    """A download filename derived from the title, safe on every OS."""
    stem = re.sub(r"[^\w\s-]", "", title).strip()
    stem = re.sub(r"[\s_]+", "-", stem)[:60].strip("-")
    return f"{stem or 'paperlens-report'}.{extension}"
