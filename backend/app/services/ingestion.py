"""
The ingestion pipeline: PDF file -> searchable, cited knowledge.

    upload -> parse (per page) -> section -> chunk -> embed -> persist -> ready

Runs in a background task, because embedding a 20-page paper takes 10-30
seconds of network round-trips. The upload endpoint returns immediately and the
frontend polls the job row.

**The memory discipline is the design.** The pipeline is a chain of generators:
pages stream out of the parser, sections accumulate from pages, chunks come from
one section at a time, and each section's chunks are embedded and written to
SQLite before the next section is touched. At no point does the whole document,
the whole chunk list, or the whole embedding matrix exist in memory.

Concretely, peak resident data during ingest is:

    one page of text        (~4 KB)
  + one section's text      (~20 KB, the largest section)
  + one embedding batch     (64 x 768 float32 = 196 KB, plus the JSON response)

That is under a megabyte of working set regardless of whether the PDF is 5 pages
or 300. The previous version held three full copies of the document plus a
Python-list embedding matrix, and grew linearly with page count.
"""

import hashlib
import logging
import time
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import Chunk, Paper, Section, _uuid
from app.rag import llm, vector_store
from app.rag.chunker import chunk_sections, iter_sections
from app.rag.embeddings import EmbeddingError, embed_texts
from app.rag.pdf_parser import iter_pages, read_metadata
from app.services import jobs

logger = logging.getLogger(__name__)

# Read the upload in 1 MB pieces. Large enough that the syscall overhead is
# irrelevant, small enough that a rejected 500 MB upload never allocates more
# than this. The previous code did `await file.read()` -- the entire body into
# one bytes object -- and only *then* checked the size limit.
UPLOAD_CHUNK_BYTES = 1024 * 1024


def hash_and_save(source, destination: Path, max_bytes: int) -> tuple[str, int]:
    """Stream an uploaded file to disk, hashing as it goes, enforcing a cap.

    Returns (sha256 hex, bytes written). Raises ValueError if the cap is
    exceeded, having already removed the partial file -- an aborted upload must
    not leave a half-written PDF behind for the parser to choke on later.

    Hashing during the same pass costs nothing: the bytes are already in cache.
    The hash is what makes re-uploading a paper free.
    """
    digest = hashlib.sha256()
    written = 0
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with destination.open("wb") as out:
            while True:
                block = source.read(UPLOAD_CHUNK_BYTES)
                if not block:
                    break
                written += len(block)
                if written > max_bytes:
                    raise ValueError(
                        f"File exceeds the {max_bytes // (1024 * 1024)} MB limit."
                    )
                digest.update(block)
                out.write(block)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return digest.hexdigest(), written


def find_existing(db: Session, user_id: str, content_hash: str) -> Paper | None:
    """A previously ingested copy of the same file, belonging to the same user.

    Scoped to the user deliberately. A global lookup would be cheaper -- one
    popular paper ingested once for everyone -- but it would hand user B the
    parse of a file uploaded by user A, and let anyone probe whether a given PDF
    had been uploaded before by watching how fast the upload returned. Privacy
    beats the saved API call.
    """
    return db.scalar(
        select(Paper)
        .where(
            Paper.user_id == user_id,
            Paper.content_hash == content_hash,
            Paper.status == "ready",
        )
        .limit(1)
    )


def _mark_failed(db: Session, paper: Paper, message: str, job_id: str | None) -> None:
    paper.status = "failed"
    paper.error_message = message[:500]
    db.commit()
    if job_id:
        jobs.fail(db, job_id, message)
    logger.error("Ingestion failed for paper %s: %s", paper.id, message)


def _index_pages(db: Session, paper: Paper, pages, job_id: str | None):
    """Section, chunk, embed and write a stream of pages.

    Returns (total_chunks, section_count, abstract_text), or None after
    marking the paper failed. Separated from `process_paper` so the same
    chain can run over a second page source (OCR) when the first finds no
    text.

    One expression drives it, and it is the whole point of the rewrite: each
    stage is a generator pulling from the one before, so a page is parsed,
    folded into a section, chunked, embedded and written, and only then is
    the next page read.
    """
    sections = iter_sections(pages)
    batches = chunk_sections(
        sections,
        target_tokens=settings.CHUNK_TARGET_TOKENS,
        overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
    )

    total_chunks = 0
    section_count = 0
    abstract_text: str | None = None
    deadline = time.monotonic() + settings.INGEST_TIMEOUT_SECONDS

    for span, chunks in batches:
        section_count += 1

        # Embed FIRST, before any row for this section is written. A flush
        # would take SQLite's single write lock, and holding it across a
        # multi-second network call blocks every other writer in the process
        # -- a chat message, a job-progress update, the ledger -- until it
        # times out with "database is locked".
        vectors: list[list[float]] = []
        if chunks:
            try:
                with llm.calling(paper.id, "ingest"):
                    vectors = embed_texts([c.content for c in chunks])
            except EmbeddingError as exc:
                _mark_failed(db, paper, str(exc), job_id)
                return None

        if time.monotonic() > deadline:
            # Cooperative: checked between sections, because pypdf cannot be
            # interrupted from another thread mid-page. Bounds a pathological
            # file at "one section over", not at infinity.
            _mark_failed(
                db, paper,
                f"Processing exceeded {settings.INGEST_TIMEOUT_SECONDS}s and was "
                "stopped. The PDF may be malformed or unusually large.",
                job_id,
            )
            return None

        section = Section(
            id=_uuid(),  # assigned here so no flush is needed before the chunks
            paper_id=paper.id,
            name=span.name[:300],
            kind=span.kind,
            ordinal=span.ordinal,
            page_start=span.page_start,
            page_end=span.page_end,
            char_count=span.char_count,
        )
        db.add(section)

        # Capture the abstract while we have it, for the chat system prompt.
        if abstract_text is None and span.kind in ("abstract", "front_matter"):
            candidate = span.text.strip()
            if 200 < len(candidate) < 4000:
                abstract_text = candidate

        if not chunks:
            db.commit()
            continue

        for chunk, vector in zip(chunks, vectors, strict=True):
            db.add(
                Chunk(
                    paper_id=paper.id,
                    section_id=section.id,
                    content=chunk.content,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chunk_index=chunk.chunk_index,
                    section=span.name[:300],
                    embedding=vector_store.pack(vector),
                )
            )

        total_chunks += len(chunks)
        db.commit()

        # Freed explicitly. These are the two largest live objects in the loop
        # and the next iteration's parse begins immediately.
        del chunks, vectors

        if job_id:
            jobs.update(
                db,
                job_id,
                current_item=section_count,
                message=f"Indexed {total_chunks} passages across {section_count} sections",
            )

    return total_chunks, section_count, abstract_text


def process_paper(paper_id: str) -> None:
    """Run the full ingestion pipeline for one paper.

    Opens its own DB session because it runs outside the request lifecycle --
    the request's session is already closed by the time this executes.
    """
    db = SessionLocal()
    job_id: str | None = None

    try:
        paper = db.get(Paper, paper_id)
        if paper is None:
            logger.warning("process_paper: paper %s vanished", paper_id)
            return

        job = jobs.current(db, paper_id)
        job_id = job.id if job else None

        paper.status = "processing"
        db.commit()

        # --- 1. Front matter -------------------------------------------------
        if job_id:
            jobs.update(db, job_id, stage="parsing", message="Reading the PDF")
        try:
            meta = read_metadata(paper.file_path, max_pages=settings.MAX_PAGES)
        except ValueError as exc:
            _mark_failed(db, paper, str(exc), job_id)
            return

        if paper.source_url and paper.authors:
            # Metadata came from the source (arXiv API) and is exact; the
            # parser's guess from page one would only make it worse.
            paper.num_pages = meta.num_pages
        else:
            paper.title = (meta.title or paper.filename)[:500]
            paper.authors = meta.authors
            paper.venue = meta.venue
            paper.year = meta.year
            paper.num_pages = meta.num_pages
        db.commit()

        # --- 2. Stream pages -> sections -> chunks -> embeddings -------------
        if job_id:
            jobs.update(db, job_id, stage="embedding", message="Building the search index")

        outcome = _index_pages(db, paper, iter_pages(paper.file_path), job_id)
        if outcome is None:
            return  # already marked failed
        total_chunks, section_count, abstract_text = outcome

        if total_chunks == 0 and settings.SCANNED_PDF_OCR:
            # No extractable text: a scanned paper. Second pass, same chain,
            # different page source -- the model transcribes each page. See
            # app/rag/ocr.py for the cost. Sections from the empty first pass
            # are discarded first so ordinals start clean.
            from app.rag.ocr import iter_ocr_pages

            if job_id:
                jobs.update(
                    db, job_id, stage="parsing",
                    message="No text layer found -- transcribing scanned pages",
                )
            for stale in db.scalars(select(Section).where(Section.paper_id == paper.id)):
                db.delete(stale)
            db.commit()

            outcome = _index_pages(
                db, paper, iter_ocr_pages(paper.file_path, settings.OCR_MAX_PAGES), job_id
            )
            if outcome is None:
                return
            total_chunks, section_count, abstract_text = outcome
            if total_chunks:
                paper.ocr = True

        if total_chunks == 0:
            _mark_failed(
                db,
                paper,
                "No extractable text found. This is likely a scanned PDF"
                + ("; transcription produced nothing readable." if settings.SCANNED_PDF_OCR
                   else "; OCR is disabled (SCANNED_PDF_OCR=false)."),
                job_id,
            )
            return

        if not paper.abstract:  # keep an exact one from the source if present
            paper.abstract = abstract_text
        paper.num_chunks = total_chunks
        paper.status = "ready"
        paper.error_message = None
        db.commit()

        logger.info(
            "Paper %s ready: %d pages, %d sections, %d chunks",
            paper.id, paper.num_pages, section_count, total_chunks,
        )

        # Ingestion is done; the deep analysis is a separate job that starts
        # from here. Imported locally to keep the module graph acyclic --
        # analysis imports nothing from ingestion, and this is the one edge back.
        from app.services import analysis

        analysis.run_analysis(paper_id, job_id=job_id)

    except Exception as exc:  # noqa: BLE001 -- a background task must never die silently
        logger.exception("Unexpected ingestion error")
        paper = db.get(Paper, paper_id)
        if paper is not None:
            _mark_failed(db, paper, f"Unexpected error: {exc}", job_id)
    finally:
        db.close()
