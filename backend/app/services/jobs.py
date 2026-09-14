"""
Progress reporting for background work.

The `analysis_jobs` table is the whole contract between a background task and
the UI. This module is the only thing that writes to it, so there is one place
where a stage name is defined and one place where progress is committed.

Why a table rather than an in-memory dict: a dict would be lost on restart and
would be wrong the moment there is more than one worker. A row costs a small
write per stage transition and is correct in both cases. It also means a user
who reloads the page mid-analysis sees the real state instead of a fresh
spinner.

Each update opens and commits its own short transaction. Holding one open across
a 30-second LLM call would keep a SQLite write lock for the whole call and block
every other write in the process.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AnalysisJob

logger = logging.getLogger(__name__)

# The stages a paper moves through, in order. The frontend renders a progress
# bar from the index of the current stage, so the order here is load-bearing.
STAGES = ("queued", "parsing", "embedding", "analyzing", "synthesizing", "done")

# Human-readable default for each stage, used when no specific message is given.
STAGE_LABELS = {
    "queued": "Waiting to start",
    "parsing": "Reading the PDF",
    "embedding": "Building the search index",
    "analyzing": "Analysing sections",
    "synthesizing": "Writing the report",
    "done": "Done",
}


def create(db: Session, paper_id: str, prompt_version: str) -> AnalysisJob:
    """Start a new job row, superseding any earlier one for this paper.

    Older jobs are deleted rather than kept as history: nobody asks "how did the
    analysis go three runs ago", and leaving them means the status endpoint has
    to reason about which row is current.
    """
    for stale in db.scalars(select(AnalysisJob).where(AnalysisJob.paper_id == paper_id)):
        db.delete(stale)

    job = AnalysisJob(
        paper_id=paper_id,
        status="queued",
        stage="queued",
        message=STAGE_LABELS["queued"],
        prompt_version=prompt_version,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def current(db: Session, paper_id: str) -> AnalysisJob | None:
    return db.scalar(
        select(AnalysisJob)
        .where(AnalysisJob.paper_id == paper_id)
        .order_by(AnalysisJob.created_at.desc())
        .limit(1)
    )


def update(
    db: Session,
    job_id: str,
    *,
    stage: str | None = None,
    current_item: int | None = None,
    total: int | None = None,
    message: str | None = None,
) -> None:
    """Move a job forward. Safe to call often; each call is one small commit."""
    job = db.get(AnalysisJob, job_id)
    if job is None:
        # The paper was deleted mid-run. Not an error -- the work is moot, and
        # the task's next storage write will fail cleanly on its own.
        logger.debug("Progress update for vanished job %s", job_id)
        return

    if stage is not None:
        job.stage = stage
        job.status = "running"
        job.message = message if message is not None else STAGE_LABELS.get(stage, stage)
    elif message is not None:
        job.message = message

    if current_item is not None:
        job.current = current_item
    if total is not None:
        job.total = total

    job.updated_at = datetime.now(timezone.utc)
    db.commit()


def finish(db: Session, job_id: str) -> None:
    job = db.get(AnalysisJob, job_id)
    if job is None:
        return
    job.status = "done"
    job.stage = "done"
    job.message = STAGE_LABELS["done"]
    job.current = job.total
    job.updated_at = datetime.now(timezone.utc)
    db.commit()


def fail(db: Session, job_id: str, message: str) -> None:
    job = db.get(AnalysisJob, job_id)
    if job is None:
        return
    job.status = "failed"
    # Truncated: a full traceback in the database helps nobody and wrecks the
    # UI layout. The traceback is in the logs, where it belongs.
    job.error_message = message[:500]
    job.message = "Failed"
    job.updated_at = datetime.now(timezone.utc)
    db.commit()
    logger.error("Job %s failed: %s", job_id, message)


def sweep_interrupted(db: Session) -> int:
    """Fail every job that was mid-flight when the process last died.

    Jobs run on an in-process worker thread (app/services/worker.py), so a
    restart -- a deploy, or the free instance waking from sleep -- loses
    whatever was running. Without this, that job sits at `running` forever and
    the UI polls a progress bar that will never move. Marking it failed with a
    reason turns an invisible hang into a state the user can act on: the paper
    shows "interrupted by a restart" and a "Run the analysis" button.

    A paper that never reached `ready` is failed too; one that did keeps its
    index and only loses the report, which can be regenerated.

    Returns the number of jobs swept, for the startup log.
    """
    from app.models import Paper

    stuck = list(
        db.scalars(select(AnalysisJob).where(AnalysisJob.status.in_(("queued", "running"))))
    )
    for job in stuck:
        job.status = "failed"
        job.message = "Failed"
        job.error_message = "Interrupted by a server restart. Run the analysis again."
        job.updated_at = datetime.now(timezone.utc)

        paper = db.get(Paper, job.paper_id)
        if paper is not None and paper.status in ("pending", "processing"):
            paper.status = "failed"
            paper.error_message = "Processing was interrupted by a server restart. Please re-upload."

    if stuck:
        db.commit()
        logger.warning("Swept %d job(s) interrupted by the last restart", len(stuck))
    return len(stuck)
