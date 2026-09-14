"""
The LLM call ledger: write one row per call, read totals back.

Writes open their own short session. A call happens on whichever thread is
running the request or the worker, often in the middle of someone else's
transaction; borrowing that session would either commit their half-done work
or hold a SQLite write lock across a network round-trip. A separate session
that commits one row and closes is correct and cheap.
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import LLMCall
from app.schemas import UsageOut

logger = logging.getLogger(__name__)


def record(
    *,
    paper_id: str | None,
    purpose: str,
    kind: str,
    model: str,
    prompt_tokens: int | None,
    output_tokens: int | None,
    latency_ms: int,
    ok: bool,
) -> None:
    db = SessionLocal()
    try:
        db.add(
            LLMCall(
                paper_id=paper_id,
                purpose=purpose,
                kind=kind,
                model=model,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                ok=ok,
            )
        )
        db.commit()
    finally:
        db.close()


def _summarise(db: Session, where) -> UsageOut:
    totals = db.execute(
        select(
            func.count(LLMCall.id),
            func.coalesce(func.sum(LLMCall.prompt_tokens), 0),
            func.coalesce(func.sum(LLMCall.output_tokens), 0),
            func.coalesce(func.sum(LLMCall.latency_ms), 0),
        ).where(*where)
    ).one()

    failed = db.scalar(
        select(func.count(LLMCall.id)).where(*where, LLMCall.ok.is_(False))
    )

    by_purpose = {
        purpose: int(count)
        for purpose, count in db.execute(
            select(LLMCall.purpose, func.count(LLMCall.id))
            .where(*where)
            .group_by(LLMCall.purpose)
        )
    }

    return UsageOut(
        calls=int(totals[0] or 0),
        failed=int(failed or 0),
        prompt_tokens=int(totals[1]),
        output_tokens=int(totals[2]),
        total_latency_ms=int(totals[3]),
        by_purpose=by_purpose,
    )


def for_paper(db: Session, paper_id: str) -> UsageOut:
    """Everything this paper has cost: ingest, analysis, every chat turn."""
    return _summarise(db, [LLMCall.paper_id == paper_id])


def for_period(db: Session, days: int = 1) -> UsageOut:
    """Totals across all papers for the last `days` -- the operator's view."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    return _summarise(db, [LLMCall.created_at >= since])
