"""
Paper endpoints: upload, list, get, sections, job status, delete.

Upload validation is layered deliberately -- extension, then declared MIME type,
then a streamed size cap, then actual file magic bytes. Extension and
content-type are both attacker-controlled, so the magic-byte check is the one
that matters; the earlier checks just fail fast and cheaply on honest mistakes.
"""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select

from app import ratelimit
from app.config import settings
from app.deps import CurrentUser, DbSession, OwnedPaper
from app.models import Paper, Section
from app.schemas import FromUrlRequest, JobStatus, PaperOut, SectionOut
from app.services import fetch, jobs, worker
from app.services.ingestion import find_existing, hash_and_save, process_paper
from app.rag.prompts import PROMPT_VERSION

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/papers", tags=["papers"])

PDF_MAGIC = b"%PDF-"


@router.post("", response_model=PaperOut, status_code=status.HTTP_201_CREATED)
def upload_paper(
    db: DbSession,
    user: CurrentUser,
    file: UploadFile = File(...),
) -> PaperOut:
    """Accept a PDF, save it, and kick off ingestion in the background.

    A **sync** endpoint, deliberately. The body is streamed to disk with a
    blocking read loop, which FastAPI runs on the threadpool. The previous async
    version did `await file.read()` -- pulling the entire upload into one bytes
    object -- and only then compared its length to the limit, so a 500 MB POST
    allocated 500 MB before being rejected. Now nothing larger than one 1 MB
    block is ever resident, and the cap aborts mid-stream.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")
    ratelimit.check(ratelimit.INGEST, user.id, "uploads")
    ratelimit.check_daily_calls(db, user.id)

    # Store under a generated name, never the user-supplied one. A filename like
    # "../../etc/passwd" would otherwise let an upload escape the directory.
    stored_name = f"{uuid.uuid4()}.pdf"
    destination = Path(settings.UPLOAD_DIR) / stored_name

    try:
        content_hash, size = hash_and_save(
            file.file, destination, settings.max_upload_bytes
        )
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("Could not write upload")
        raise HTTPException(status_code=500, detail="Could not save the file.") from exc

    return _register(db, user, destination, file.filename, content_hash, size)


@router.post("/from-url", response_model=PaperOut, status_code=status.HTTP_201_CREATED)
def paper_from_url(db: DbSession, user: CurrentUser, payload: FromUrlRequest) -> PaperOut:
    """Ingest a paper from an arXiv or DOI link.

    Same pipeline as an upload from the moment the bytes are on disk -- the
    same size cap, magic-byte check and per-user dedupe -- plus exact
    metadata from the arXiv API when the link is an arXiv one. The host
    allowlist in services/fetch.py is the SSRF guard; it runs before any
    connection is opened.
    """
    ratelimit.check(ratelimit.INGEST, user.id, "uploads")
    ratelimit.check_daily_calls(db, user.id)

    stored_name = f"{uuid.uuid4()}.pdf"
    destination = Path(settings.UPLOAD_DIR) / stored_name

    try:
        content_hash, size = fetch.download_pdf(
            payload.url, destination, settings.max_upload_bytes
        )
    except fetch.FetchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:  # size cap, raised by hash_and_save
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 -- network errors become a clean 502
        logger.warning("Fetch failed for %s: %s", payload.url, exc)
        raise HTTPException(
            status_code=502, detail="Could not download from that link."
        ) from exc

    arxiv_id = fetch.arxiv_id_from_url(payload.url)
    filename = f"{arxiv_id or 'paper'}.pdf"

    # Exact metadata beats the parser's guess. Looked up before the paper row
    # exists so it is written in the same transaction that creates it -- the
    # worker may start ingesting the moment the row is committed, and
    # ingestion leaves these fields alone when source_url is set.
    meta = fetch.arxiv_metadata(arxiv_id) if arxiv_id else None

    return _register(
        db, user, destination, filename, content_hash, size,
        source_url=payload.url, meta=meta,
    )


def _register(
    db,
    user,
    destination: Path,
    filename: str,
    content_hash: str,
    size: int,
    source_url: str | None = None,
    meta: fetch.ArxivMeta | None = None,
) -> PaperOut:
    """Everything after the bytes are on disk, shared by upload and from-url."""
    # The real format check: a file can be named .pdf and be anything at all.
    # Done after writing rather than before, because we never hold the body in
    # memory to inspect it -- so we read the first bytes back off disk.
    with destination.open("rb") as handle:
        if not handle.read(len(PDF_MAGIC)).startswith(PDF_MAGIC):
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="File is not a valid PDF.")

    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The file is empty.")

    # Already ingested by this user? Return it and delete the duplicate upload.
    # Parsing and embedding the same bytes twice costs real quota and produces
    # an identical index.
    existing = find_existing(db, user.id, content_hash)
    if existing is not None:
        destination.unlink(missing_ok=True)
        logger.info("Upload matched existing paper %s by hash", existing.id)
        return PaperOut.model_validate(existing)

    paper = Paper(
        user_id=user.id,
        title=Path(filename).stem[:500],  # replaced by the real title after parsing
        filename=filename[:500],
        file_path=str(destination),
        content_hash=content_hash,
        source_url=source_url,
        status="pending",
    )
    if meta is not None:
        paper.title = meta.title[:500]
        paper.authors = meta.authors or None
        paper.abstract = meta.abstract or None
        paper.year = meta.year
        paper.venue = "arXiv"
    db.add(paper)
    db.commit()
    db.refresh(paper)

    jobs.create(db, paper.id, PROMPT_VERSION)

    # Onto the worker thread, not BackgroundTasks: a background task would run
    # on the request threadpool, which is pinned to two threads, and hold one
    # of them for the minutes an ingest takes. See app/services/worker.py.
    worker.enqueue(process_paper, paper.id)

    return PaperOut.model_validate(paper)


@router.get("", response_model=list[PaperOut])
def list_papers(db: DbSession, user: CurrentUser) -> list[PaperOut]:
    """All of the caller's papers, newest first."""
    papers = db.scalars(
        select(Paper).where(Paper.user_id == user.id).order_by(Paper.created_at.desc())
    )
    return [PaperOut.model_validate(p) for p in papers]


@router.get("/{paper_id}", response_model=PaperOut)
def get_paper(paper: OwnedPaper) -> PaperOut:
    """Fetch one paper. The frontend polls this while status is 'processing'."""
    return PaperOut.model_validate(paper)


@router.get("/{paper_id}/file", include_in_schema=True)
def paper_file(paper: OwnedPaper) -> FileResponse:
    """The original PDF, for the in-browser viewer.

    Streamed from disk by Starlette's FileResponse -- the file is never read
    into memory here. Rendering happens in the browser (PDF.js): the
    unconstrained design would rasterise pages server-side with a highlight
    drawn on, but the renderer that could do that is the library removed to
    fit in 512 MB. The client already has a PDF engine; this hands it the bytes.

    `inline`, not `attachment`, so a browser that is given the URL directly
    displays it rather than downloading it.
    """
    path = Path(paper.file_path)
    if not path.is_file():
        # Ephemeral storage: the row survived a restart but the file did not.
        raise HTTPException(
            status_code=410,
            detail="The PDF is no longer on this server. Re-upload it to view pages.",
        )
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{paper.filename}"'},
    )


@router.get("/{paper_id}/sections", response_model=list[SectionOut])
def paper_sections(paper: OwnedPaper, db: DbSession) -> list[SectionOut]:
    """The paper's detected structure, for the reader view's outline."""
    sections = db.scalars(
        select(Section).where(Section.paper_id == paper.id).order_by(Section.ordinal)
    )
    return [SectionOut.model_validate(s) for s in sections]


@router.get("/{paper_id}/job", response_model=JobStatus)
def job_status(paper: OwnedPaper, db: DbSession) -> JobStatus:
    """Progress of the ingestion + analysis run.

    Polled by the frontend so the user sees "analysing section 4 of 11" rather
    than a spinner that might mean anything. Returns a synthetic done/failed
    record when no job row exists, so the client never has to special-case a
    404 -- a paper that is ready and has no job simply finished before this
    endpoint existed, or was restored from cache.
    """
    job = jobs.current(db, paper.id)

    if job is None:
        finished = paper.status in ("ready", "failed")
        return JobStatus(
            id="",
            status="done" if paper.status == "ready" else
                   "failed" if paper.status == "failed" else "queued",
            stage="done" if finished else "queued",
            stage_index=len(jobs.STAGES) - 1 if finished else 0,
            stage_count=len(jobs.STAGES),
            current=0,
            total=0,
            message=paper.error_message or paper.status,
            error_message=paper.error_message,
        )

    try:
        stage_index = jobs.STAGES.index(job.stage)
    except ValueError:
        stage_index = 0

    return JobStatus(
        id=job.id,
        status=job.status,
        stage=job.stage,
        stage_index=stage_index,
        stage_count=len(jobs.STAGES),
        current=job.current,
        total=job.total,
        message=job.message,
        error_message=job.error_message,
    )


@router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_paper(paper: OwnedPaper, db: DbSession) -> None:
    """Delete a paper: database rows (cascading to chunks, sections, messages,
    jobs) and the file on disk.

    Simpler than it used to be. Vectors live in the `chunks` rows now, so the
    cascade removes them -- the old design deleted from ChromaDB first and could
    leave the two stores inconsistent if one half failed.
    """
    file_path = Path(paper.file_path)

    db.delete(paper)
    db.commit()

    # After the commit: if unlinking fails we have an orphaned file, which is
    # recoverable. Doing it first risks deleting the file and failing the
    # commit, leaving a row pointing at nothing.
    try:
        file_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove %s: %s", file_path, exc)
