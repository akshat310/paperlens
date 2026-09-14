"""
OCR for scanned PDFs, with no OCR engine in the process.

A scanned paper is a stack of page images; pypdf finds no text and ingestion
used to stop with "OCR is not supported". The conventional fix is Tesseract
(a native library plus language data) over rasterised pages -- and rasterising
needs a renderer, which is the dependency removed to fit in 512 MB.

Gemini reads PDFs natively. So each page is cut into a one-page PDF with pypdf
(a few hundred KB for a scanned page), sent inline, and the model is asked to
transcribe it. The transcript then enters the same `iter_sections -> chunk ->
embed` chain as extracted text; nothing downstream knows the difference.

Costs, stated:
  - One generation call per page against the free quota. That is why it is off
    by default (`SCANNED_PDF_OCR`) and capped (`OCR_MAX_PAGES`).
  - Memory: one page's PDF bytes and one transcript at a time. The whole
    document is never held, exactly as in the text path.
  - Fidelity: a transcription model can mis-read a table or drop a footnote.
    The output is labelled as OCR on the paper so a reader knows.
"""

import io
import logging
from collections.abc import Iterator

from pypdf import PdfReader, PdfWriter

from app.config import settings
from app.rag import llm
from app.rag.pdf_parser import PageText, _classify_heading, _clean_text, _CAPTION

logger = logging.getLogger(__name__)

OCR_PROMPT = """Transcribe the text of this scanned page of an academic paper, exactly as
printed, in reading order. Rules:
- Output plain text only: no commentary, no markdown fences.
- Keep section headings on their own line, exactly as printed (e.g. "3.2 Attention").
- Keep paragraph breaks as blank lines. Reproduce tables as rows of text.
- Keep figure and table captions on their own line, starting "Figure N:" or "Table N:".
- If the page is blank or contains no readable text, output exactly: [EMPTY]"""

OCR_MAX_TOKENS = 6000


def _single_page_pdf(reader: PdfReader, index: int) -> bytes:
    writer = PdfWriter()
    writer.add_page(reader.pages[index])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def transcribe_page(pdf_bytes: bytes, page_number: int) -> str:
    """One call: a single-page PDF in, its text out. Isolated so tests can stub it."""
    from google.genai import types

    client = llm.get_client()
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            f"Page {page_number}.\n\n{OCR_PROMPT}",
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=OCR_MAX_TOKENS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    llm._record(
        model=settings.GEMINI_MODEL,
        prompt_tokens=getattr(response.usage_metadata, "prompt_token_count", None),
        output_tokens=getattr(response.usage_metadata, "candidates_token_count", None),
        latency_ms=0,
        ok=True,
        kind="ocr",
    )
    text = (response.text or "").strip()
    return "" if text == "[EMPTY]" else text


def iter_ocr_pages(file_path: str, max_pages: int) -> Iterator[PageText]:
    """Yield transcribed pages in the same shape `pdf_parser.iter_pages` yields.

    Same contract: one page alive at a time, headings and captions detected
    on the raw transcript before cleaning, so the section vocabulary applies
    unchanged.
    """
    reader = PdfReader(file_path)
    total = min(len(reader.pages), max_pages)
    if len(reader.pages) > max_pages:
        logger.warning(
            "OCR capped at %d of %d pages for %s", max_pages, len(reader.pages), file_path
        )

    for index in range(total):
        try:
            raw = transcribe_page(_single_page_pdf(reader, index), index + 1)
        except llm.LLMError as exc:
            # One unreadable page costs that page, not the document.
            logger.warning("OCR failed on page %d: %s", index + 1, exc)
            continue
        if not raw.strip():
            continue

        headings: list[tuple[str, str, int]] = []
        captions: list[str] = []
        offset = 0
        for line in raw.split("\n"):
            classified = _classify_heading(line)
            if classified:
                headings.append((classified[0], classified[1], offset))
            else:
                caption = _CAPTION.match(line.strip())
                if caption:
                    captions.append(" ".join(caption.group(1).split()))
            offset += len(line) + 1

        cleaned = _clean_text(raw)
        del raw
        if cleaned:
            yield PageText(
                page_number=index + 1, text=cleaned, headings=headings, captions=captions
            )
