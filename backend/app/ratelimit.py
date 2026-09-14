"""
Per-user rate limiting and daily quotas, in memory.

A public demo runs on one free Gemini key. Without limits, one visitor can
drain it -- upload twenty papers, or script a thousand questions -- and the
app is dead for everyone until the quota resets. So:

  - a **token bucket** per (user, action) bounds burst rate: N requests per
    window, refilling continuously;
  - a **daily cap** per user on the expensive actions (papers ingested, model
    calls), counted from the LLM ledger for calls and from the papers table
    for uploads, so the numbers are true rather than approximate.

Why in memory and not Redis: the deployment is one process by design
(WEB_CONCURRENCY=1 -- every extra worker duplicates the ~110 MB floor), so a
dict guarded by a lock is *correct*, not a shortcut. Redis would add a service
the free tier does not provide to solve a problem this deployment cannot have.
The cost is that buckets reset on restart, which for a demo is fine.

Why not `slowapi`: it is a fine library, but this is 60 lines and every line
of it can be explained. The limits themselves are settings so they can be
tuned without a deploy.
"""

import threading
import time
from dataclasses import dataclass, field

from fastapi import HTTPException

from app.config import settings


@dataclass
class _Bucket:
    tokens: float
    updated: float = field(default_factory=time.monotonic)


class TokenBucket:
    """Classic token bucket: `capacity` tokens, refilled at `capacity/window`."""

    def __init__(self, capacity: int, window_seconds: float) -> None:
        self.capacity = capacity
        self.rate = capacity / window_seconds
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def try_acquire(self, key: str) -> float:
        """Take one token. Returns 0.0 on success, else seconds until one is free."""
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.capacity), updated=now)
                self._buckets[key] = bucket
            # Refill for the time elapsed, capped at capacity.
            bucket.tokens = min(self.capacity, bucket.tokens + (now - bucket.updated) * self.rate)
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return 0.0
            return (1.0 - bucket.tokens) / self.rate

    def prune(self, older_than: float = 3600.0) -> None:
        """Drop buckets idle for an hour so the dict cannot grow without bound."""
        cutoff = time.monotonic() - older_than
        with self._lock:
            for key in [k for k, b in self._buckets.items() if b.updated < cutoff]:
                del self._buckets[key]


# One limiter per action class. Chat is the hot path; uploads and analysis
# runs are expensive; login attempts are limited per email to slow guessing.
CHAT = TokenBucket(settings.RATE_CHAT_PER_MINUTE, 60.0)
INGEST = TokenBucket(settings.RATE_INGEST_PER_HOUR, 3600.0)
LOGIN = TokenBucket(settings.RATE_LOGIN_PER_MINUTE, 60.0)

_last_prune = time.monotonic()


def check(limiter: TokenBucket, key: str, what: str) -> None:
    """Raise 429 with a Retry-After header if the bucket is empty."""
    global _last_prune
    if not settings.RATE_LIMITING:
        return
    now = time.monotonic()
    if now - _last_prune > 600:
        for bucket in (CHAT, INGEST, LOGIN):
            bucket.prune()
        _last_prune = now

    wait = limiter.try_acquire(key)
    if wait > 0:
        raise HTTPException(
            status_code=429,
            detail=f"Too many {what} -- try again in {max(1, round(wait))}s.",
            headers={"Retry-After": str(max(1, round(wait)))},
        )


def check_daily_calls(db, user_id: str) -> None:
    """Refuse model-backed actions once a user has spent their daily call budget.

    Counted from the ledger, so it is the real number. Runs one indexed
    COUNT per request; on the scale of this app that is free.
    """
    if not settings.RATE_LIMITING or settings.DAILY_CALLS_PER_USER <= 0:
        return
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import func, select

    from app.models import LLMCall, Paper

    since = datetime.now(timezone.utc) - timedelta(days=1)
    used = db.scalar(
        select(func.count(LLMCall.id))
        .join(Paper, Paper.id == LLMCall.paper_id)
        .where(Paper.user_id == user_id, LLMCall.created_at >= since)
    ) or 0
    if used >= settings.DAILY_CALLS_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily limit reached ({settings.DAILY_CALLS_PER_USER} model calls). "
                "This demo runs on a free API key; it resets in 24 hours."
            ),
        )
