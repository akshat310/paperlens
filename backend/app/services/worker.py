"""
A single background worker thread with a queue.

Why this exists
---------------
FastAPI's `BackgroundTasks` looks like a job queue but is not one. A sync task
is run through `run_in_threadpool`, which draws a token from the **same** anyio
limiter that `app/main.py` pins to 2 for the memory budget. An ingest+analysis
run holds that token for the one to three minutes it takes. Two uploads at once
and there are zero threads left for `/chat`, `/job` -- even `/health`, which is
the endpoint the polling UI and Render's health check both depend on.

The unconstrained answer is a real queue: Celery or RQ, a Redis broker, a
separate worker process. On 512 MB a worker process is a second copy of the
interpreter and every library (~130 MB of fixed floor duplicated), and Redis is
a service the free tier does not provide. So: one `threading.Thread` consuming
a `queue.Queue`. It gives the property that actually matters -- long jobs off
the request pool -- for the cost of a thread.

What it does not give, stated plainly: durability. A job in flight when the
process dies is gone. `sweep_interrupted_jobs` in jobs.py turns that into a
visible failed state on the next boot rather than an infinite spinner.

Jobs run **one at a time**. That is a feature here, not a limitation: it also
serialises Gemini calls, so two analyses can no longer race each other into the
free tier's per-minute quota.
"""

import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# A queued job: the callable and its positional arguments. Kept as a plain
# tuple rather than a dataclass so the queue holds nothing that needs importing
# at module load -- services enqueue their own functions.
_Job = tuple[Callable[..., Any], tuple[Any, ...]]

_queue: "queue.Queue[_Job | None]" = queue.Queue()
_thread: threading.Thread | None = None
_lock = threading.Lock()


def _run_forever() -> None:
    while True:
        job = _queue.get()
        if job is None:  # shutdown sentinel
            _queue.task_done()
            return
        fn, args = job
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 -- the worker must outlive any one job
            logger.exception("Background job %s failed", getattr(fn, "__name__", fn))
        finally:
            _queue.task_done()


def start() -> None:
    """Start the worker thread. Idempotent; called from the app's lifespan."""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _thread = threading.Thread(target=_run_forever, name="paperlens-worker", daemon=True)
        _thread.start()
        logger.info("Background worker started")


def stop(timeout: float = 5.0) -> None:
    """Ask the worker to finish its current job and exit.

    Daemon thread, so an unclean exit would not hang the process either; this
    just lets a job that is mid-commit finish cleanly on a graceful shutdown.
    """
    global _thread
    with _lock:
        if _thread is None:
            return
        _queue.put(None)
        _thread.join(timeout)
        _thread = None


def enqueue(fn: Callable[..., Any], *args: Any) -> None:
    """Schedule `fn(*args)` on the worker thread.

    If the worker was never started -- the test client without a lifespan, or a
    script -- the job runs inline. That keeps the call site identical in every
    environment and means a test can drive an upload end to end synchronously.
    """
    if _thread is None or not _thread.is_alive():
        fn(*args)
        return
    _queue.put((fn, args))


def pending() -> int:
    """Jobs waiting (not including the one running). Reported by /debug/memory."""
    return _queue.qsize()
