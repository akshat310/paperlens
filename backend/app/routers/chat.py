"""
Chat, analysis and export endpoints.

Thin by design: these functions validate that the request makes sense, delegate
to a service, and translate exceptions into HTTP status codes. All the actual
reasoning lives in the services, which keeps it testable without a client.

Every route depends on `OwnedPaper`, so the ownership check applies here for
free -- asking questions about someone else's paper 404s exactly like reading it
does.
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import StreamingResponse

from app import ratelimit
from app.deps import CurrentUser, DbSession, OwnedPaper
from app.models import Paper
from app.rag.llm import LLMError, LLMRateLimitError
from app.schemas import (
    ChatMessageOut,
    ChatRequest,
    ChatResponse,
    ClaimRequest,
    ClaimResponse,
    CompareRequest,
    CompareResponse,
    FollowupsOut,
    PaperReport,
    RelatedEntryOut,
    RelatedOut,
    UsageOut,
)
from app.services import analysis, chat_service, compare, export, jobs, ledger, related, worker
from app.rag.prompts import PROMPT_VERSION

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/papers", tags=["chat"])


def _require_ready(paper: Paper) -> None:
    """Refuse to answer until ingestion has finished.

    Without this, a question asked seconds after upload would retrieve zero
    chunks and get a confusing "not in this paper" reply, when the real answer
    is "not yet". 409 Conflict says the request is valid but the resource is in
    the wrong state -- and the message tells the UI what to do about it.
    """
    if paper.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Paper is not ready for questions yet (status: {paper.status}).",
        )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@router.post("/{paper_id}/chat", response_model=ChatResponse)
def chat(paper: OwnedPaper, db: DbSession, payload: ChatRequest) -> ChatResponse:
    """Ask a question and get an answer grounded in this paper, with citations.

    The non-streaming path. Kept alongside `/chat/stream` because it is what the
    tests and the eval harness use, and because a client that cannot consume SSE
    should still be able to ask a question.
    """
    _require_ready(paper)
    ratelimit.check(ratelimit.CHAT, paper.user_id, "questions")
    ratelimit.check_daily_calls(db, paper.user_id)
    try:
        return chat_service.answer_question(db, paper, payload.question, payload.section_id)
    except LLMRateLimitError as exc:
        # 429 before 503: except clauses are checked in order, and
        # LLMRateLimitError is a subclass of LLMError, so this must come first.
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LLMError as exc:
        # 503, not 500: nothing is broken in our code, the upstream model is
        # unavailable or unconfigured. The distinction matters when debugging.
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/{paper_id}/chat/stream")
def chat_stream(paper: OwnedPaper, db: DbSession, payload: ChatRequest) -> StreamingResponse:
    """Same answer, streamed as Server-Sent Events.

    SSE rather than WebSockets: the data flows one way, it is text, and SSE is
    plain HTTP -- it needs no protocol upgrade, no separate connection lifecycle,
    and it works through Render's proxy without configuration. A WebSocket would
    be more machinery for a strictly smaller problem.

    Errors are awkward here and worth being explicit about. Once the response
    has started, the status code is already sent, so a failure part-way cannot
    become a 503. It is emitted as an `error` event instead and the client
    renders it in place. That is why the retrieval and prompt-building work,
    which is where most failures happen, is done inside the generator before the
    first token: a failure there still arrives as a proper error event rather
    than a half-written answer.
    """
    _require_ready(paper)
    # Checked before the stream opens, so a limit is a proper 429 rather than
    # an error event after a 200.
    ratelimit.check(ratelimit.CHAT, paper.user_id, "questions")
    ratelimit.check_daily_calls(db, paper.user_id)

    def events():
        try:
            for event in chat_service.stream_answer(
                db, paper, payload.question, payload.section_id
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except LLMRateLimitError as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc), 'retryable': True})}\n\n"
        except LLMError as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc), 'retryable': False})}\n\n"
        except Exception:  # noqa: BLE001 -- never leak a traceback down the wire
            logger.exception("Streaming chat failed")
            yield (
                "data: "
                + json.dumps({"type": "error", "detail": "Something went wrong "
                              "generating that answer.", "retryable": False})
                + "\n\n"
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tells nginx-style proxies not to buffer the response. Without it a
            # proxy can hold the whole stream and deliver it at once, which
            # looks exactly like streaming being broken.
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{paper_id}/claim", response_model=ClaimResponse)
def claim(paper: OwnedPaper, db: DbSession, payload: ClaimRequest) -> ClaimResponse:
    """Does the paper support, contradict, or not address a claim?"""
    _require_ready(paper)
    ratelimit.check(ratelimit.CHAT, paper.user_id, "questions")
    ratelimit.check_daily_calls(db, paper.user_id)
    try:
        return chat_service.check_claim(db, paper, payload.claim)
    except LLMRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/compare", response_model=CompareResponse)
def compare_papers(db: DbSession, user: CurrentUser, payload: CompareRequest) -> CompareResponse:
    """One question across two or three of the caller's papers.

    Lives on the /papers prefix but not under one id. It cannot be shadowed
    by the single-paper routes: every one of those is /{paper_id}/<action>
    (two segments) or a GET/DELETE on /{paper_id}, and this is a one-segment
    POST.
    """
    ratelimit.check(ratelimit.CHAT, user.id, "questions")
    ratelimit.check_daily_calls(db, user.id)
    papers = compare.load_papers(db, user.id, payload.paper_ids)
    if len(papers) != len(dict.fromkeys(payload.paper_ids)):
        raise HTTPException(status_code=404, detail="One or more papers were not found.")
    not_ready = [p.title for p in papers if p.status != "ready"]
    if not_ready:
        raise HTTPException(
            status_code=409, detail=f"Not ready yet: {', '.join(not_ready)[:200]}"
        )
    try:
        return compare.compare(db, papers, payload.question)
    except LLMRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/{paper_id}/messages", response_model=list[ChatMessageOut])
def messages(paper: OwnedPaper, db: DbSession) -> list[ChatMessageOut]:
    """Full conversation history for this paper, oldest first."""
    return chat_service.list_messages(db, paper)


@router.delete("/{paper_id}/messages", status_code=status.HTTP_204_NO_CONTENT)
def clear_messages(paper: OwnedPaper, db: DbSession) -> None:
    """Forget the conversation: every turn and the rolled-up summary.

    Both, deliberately. Deleting the messages but leaving `chat_summary` would
    let the model keep "remembering" a conversation the reader can no longer
    see, which is exactly the kind of invisible state that makes answers look
    wrong.
    """
    chat_service.clear_conversation(db, paper)


@router.get("/{paper_id}/followups", response_model=FollowupsOut)
def followups(paper: OwnedPaper, db: DbSession) -> FollowupsOut:
    """Suggested questions, generated from this paper's content."""
    _require_ready(paper)
    return FollowupsOut(questions=chat_service.suggest_followups(db, paper))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@router.get("/{paper_id}/report", response_model=PaperReport)
def get_report(paper: OwnedPaper, db: DbSession) -> PaperReport:
    """The cached deep-analysis report.

    404 when it does not exist yet or was written by an older prompt version --
    the client polls `/job` for progress and retries this when the job is done.
    """
    report = analysis.get_report(db, paper)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="No analysis available yet for this paper.",
        )
    return report


@router.post("/{paper_id}/report", status_code=status.HTTP_202_ACCEPTED)
def start_analysis(
    paper: OwnedPaper,
    db: DbSession,
    force: bool = False,
) -> dict[str, str]:
    """Start (or restart) the deep analysis in the background.

    202 Accepted, not 200: the work has been queued, not done. The response
    carries the job id and the client polls `/job`.
    """
    _require_ready(paper)

    existing = analysis.get_report(db, paper)
    if existing is not None and not force:
        return {"status": "cached", "job_id": ""}

    # A re-run is 8-12 model calls; it counts as an ingest for limiting.
    ratelimit.check(ratelimit.INGEST, paper.user_id, "analysis runs")
    ratelimit.check_daily_calls(db, paper.user_id)

    job = jobs.create(db, paper.id, PROMPT_VERSION)
    worker.enqueue(analysis.run_analysis, paper.id, job.id, force)
    return {"status": "queued", "job_id": job.id}


# ---------------------------------------------------------------------------
# Related work
# ---------------------------------------------------------------------------

@router.get("/{paper_id}/related", response_model=RelatedOut)
def related_work(paper: OwnedPaper, db: DbSession, refresh: bool = False) -> RelatedOut:
    """The reference list, enriched from OpenAlex.

    Computed once on the worker thread and cached on the paper; until then the
    response is `pending` and the client polls. Network-bound (one lookup per
    reference, capped) but zero model calls.
    """
    _require_ready(paper)
    entries = None if refresh else related.cached(paper)
    if entries is None:
        related.enqueue(paper.id)
        return RelatedOut(status="pending", entries=[])
    return RelatedOut(status="ready", entries=[RelatedEntryOut(**e) for e in entries])


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

@router.get("/{paper_id}/usage", response_model=UsageOut)
def paper_usage(paper: OwnedPaper, db: DbSession) -> UsageOut:
    """What this paper has cost so far: calls, tokens, seconds, by purpose."""
    return ledger.for_paper(db, paper.id)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _report_or_404(db, paper: Paper) -> PaperReport:
    report = analysis.get_report(db, paper)
    if report is None:
        raise HTTPException(
            status_code=404, detail="No analysis available to export yet."
        )
    return report


@router.get("/{paper_id}/export.md")
def export_markdown(paper: OwnedPaper, db: DbSession) -> Response:
    report = _report_or_404(db, paper)
    body = export.report_to_markdown(paper, report)
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition":
                f'attachment; filename="{export.safe_filename(paper.title, "md")}"'
        },
    )


@router.get("/{paper_id}/export.pdf")
def export_pdf(paper: OwnedPaper, db: DbSession) -> Response:
    report = _report_or_404(db, paper)
    body = export.report_to_pdf(paper, report)
    return Response(
        content=body,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                f'attachment; filename="{export.safe_filename(paper.title, "pdf")}"'
        },
    )
