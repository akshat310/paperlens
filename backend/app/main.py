"""
Application entry point.

Run locally with:  uvicorn app.main:app --reload
Interactive docs:  http://localhost:8000/docs
"""

import gc
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import anyio.to_thread
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import SessionLocal, init_db
from app.routers import auth, chat, papers
from app.database import get_db
from app.schemas import MemoryOut, UsageOut
from app.services import jobs, ledger, worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# The instance size we are budgeting against. Used only to report a percentage
# from /api/debug/memory -- nothing enforces it, the platform does.
MEMORY_LIMIT_MB = 512

# Set by _limit_threadpool() at startup and reported by /api/debug/memory.
# 40 is anyio's default, so seeing it there means the cap did NOT apply.
APPLIED_THREAD_LIMIT = 40


def _limit_threadpool() -> int:
    """Cap the threadpool every sync endpoint runs on.

    This is load-bearing for the memory budget and is easy to miss, because
    setting THREAD_LIMIT in the environment does nothing on its own -- something
    has to apply it.

    Starlette runs every `def` (non-async) endpoint via `run_in_threadpool`,
    which uses anyio's default limiter of **40** threads. Every route in this
    app is sync, so forty concurrent requests would each be building an
    embedding batch, a BM25 index and a NumPy matrix at the same time. A single
    request's peak is small by construction; forty of them is the realistic path
    to an OOM kill on a 512MB instance.

    Two is enough to keep a background ingest running while a chat request is
    served. Requests beyond that queue rather than fail -- slower under load,
    which is the correct trade against being killed under load.
    """
    global APPLIED_THREAD_LIMIT

    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = int(os.environ.get("THREAD_LIMIT", "2"))
    # Read back off the limiter rather than trusting the value we just wrote, and
    # cache it: /api/debug/memory is a sync endpoint, so it runs in a worker
    # thread with no event loop and cannot query the limiter itself.
    APPLIED_THREAD_LIMIT = int(limiter.total_tokens)
    return APPLIED_THREAD_LIMIT


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup and shutdown hooks.

    Directories and tables are created here rather than at import time so that
    importing the module (e.g. in tests) has no side effects on disk.
    """
    threads = _limit_threadpool()
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    db_path = settings.DATABASE_URL.removeprefix("sqlite:///").removeprefix("/")
    if settings.DATABASE_URL.startswith("sqlite") and db_path:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    init_db()

    # Anything that was running when the process last died is failed now,
    # with a reason, before the worker starts picking up new work.
    db = SessionLocal()
    try:
        jobs.sweep_interrupted(db)
    finally:
        db.close()

    # Long jobs (ingest, analysis) run here, not on the request threadpool.
    worker.start()

    logger.info(
        "%s started (model=%s, threads=%d)",
        settings.APP_NAME, settings.GEMINI_MODEL, threads,
    )
    yield
    worker.stop()
    logger.info("%s shutting down", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description="Deep analysis of research papers, and chat grounded in their text.",
    version="2.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def security_headers(request, call_next):
    """Baseline browser hardening on every response.

    - nosniff: a PDF or JSON body is never re-interpreted as HTML.
    - frame-ancestors 'none': the app cannot be embedded in another site's
      iframe (clickjacking).
    - Referrer-Policy: paper ids in URLs are not leaked to third parties.
    The CSP is intentionally minimal: the bundle is same-origin and inlines
    nothing, so 'self' plus the data: URIs PDF.js uses for its worker is enough.
    """
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    else:
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'",
        )
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(papers.router, prefix="/api")
# Shares the /papers prefix but is a separate module: paper lifecycle and
# question-answering are different concerns that happen to hang off the same URL.
app.include_router(chat.router, prefix="/api")


@app.get("/api/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe -- used by Render's health check and by uptime monitors.

    Deliberately does no work: no database query, no model call. A health check
    that touches a dependency reports the dependency's health, not the app's,
    and on a free instance it would also be the thing keeping the database file
    warm for no reason. If the process can answer this, it is alive.
    """
    return {"status": "ok", "app": settings.APP_NAME}


@app.get("/api/debug/memory", response_model=MemoryOut, tags=["health"])
def debug_memory() -> MemoryOut:
    """Current resident memory, so the 512 MB budget can actually be verified.

    RSS is the number that matters: it is what the platform measures and kills
    on. VMS is reported alongside because it is usually much larger and alarming
    if you have not seen it before -- it counts address space that is reserved
    but not backed by physical pages, and it is not what gets you OOM-killed.

    Left enabled in production on purpose. It exposes no user data, and the
    whole point of a memory budget is being able to check it on the real
    instance rather than on a laptop.
    """
    import psutil

    info = psutil.Process(os.getpid()).memory_info()
    rss_mb = info.rss / (1024 * 1024)

    return MemoryOut(
        rss_mb=round(rss_mb, 1),
        vms_mb=round(info.vms / (1024 * 1024), 1),
        percent_of_limit=round(100 * rss_mb / MEMORY_LIMIT_MB, 1),
        limit_mb=MEMORY_LIMIT_MB,
        python_objects=len(gc.get_objects()),
        # Captured at startup by reading the limiter back after setting it, so
        # this reflects what actually applied rather than what was requested.
        thread_limit=APPLIED_THREAD_LIMIT,
        queued_jobs=worker.pending(),
    )


@app.get("/api/debug/usage", response_model=UsageOut, tags=["health"])
def debug_usage(days: int = 1, db=Depends(get_db)) -> UsageOut:
    """LLM calls across all papers for the last `days`.

    The operator's view of quota: how many calls, how many failed, how many
    tokens. Like /debug/memory it exposes no user data -- totals only -- and
    it exists so the free-tier quota can be checked on the live instance.
    """
    return ledger.for_period(db, days=max(1, min(days, 90)))


# ---------------------------------------------------------------------------
# Static frontend
#
# In the single-container deployment the built React app is served by this same
# process, so the browser sees one origin for the page and the API, and CORS
# never applies.
#
# Everything below is registered AFTER the routers on purpose. Starlette matches
# routes in registration order and takes the first hit, so the catch-all further
# down would shadow every /api route if it came first. The explicit prefix check
# inside it is a second line of defence for paths FastAPI owns but that are not
# routes we registered (/docs, /openapi.json).
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_RESERVED_PREFIXES = ("api/", "docs", "redoc", "openapi.json")

if STATIC_DIR.is_dir():
    # Hashed filenames, mounted directly so StaticFiles handles ETags and range
    # requests rather than the catch-all re-reading the file every time.
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        """Serve a real static file if one exists, otherwise the SPA shell.

        The index.html fallback is what makes React Router deep links survive a
        refresh: the browser asks the server for /papers/abc-123, no such file
        exists, and returning the shell lets the router read the URL and render
        the right view. Without it that is a 404.
        """
        if full_path.startswith(_RESERVED_PREFIXES):
            raise HTTPException(status_code=404, detail="Not found")

        candidate = (STATIC_DIR / full_path).resolve()
        # Containment check: full_path is attacker-controlled, and "../../etc/
        # passwd" would otherwise escape the static root.
        if candidate.is_file() and candidate.is_relative_to(STATIC_DIR):
            return FileResponse(candidate)

        return FileResponse(STATIC_DIR / "index.html")

else:

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        """Signpost for anyone who opens the API's base URL directly.

        Only registered when there is no frontend to serve. Every real route
        lives under /api, so the bare root would otherwise 404 -- which looks
        like a broken deployment in the logs when a platform health probe or a
        curious visitor hits it.
        """
        return {
            "app": settings.APP_NAME,
            "docs": "/docs",
            "health": "/api/health",
        }
