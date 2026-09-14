"""
Shared test configuration.

The background worker thread is replaced with an inline call for the whole
suite. Production enqueues ingestion and analysis onto a worker thread so they
never occupy the request threadpool; a test that uploads a paper and then
immediately asserts on its report needs that work to have *finished* by the
time the response comes back. Running the job inline gives exactly that with no
sleeps and no polling, and it is the same code path `worker.enqueue` already
takes when no worker has been started.
"""

import os
import tempfile

# Point the app at a throwaway database and upload directory BEFORE anything
# imports app.config. The test client runs the real lifespan (create tables,
# sweep interrupted jobs, start the worker) and doing that against the
# developer's storage/paperlens.db is not just untidy: the sweep marks every
# running job there as failed, which once killed an ingest on a dev server
# that happened to be running alongside the suite. Environment variables win
# over .env in pydantic-settings, so this is the whole mechanism.
_TMP = tempfile.mkdtemp(prefix="paperlens-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["UPLOAD_DIR"] = f"{_TMP}/uploads"
os.environ["DEBUG"] = "true"

import pytest  # noqa: E402

from app.services import worker  # noqa: E402


@pytest.fixture(autouse=True)
def inline_worker(monkeypatch):
    monkeypatch.setattr(worker, "enqueue", lambda fn, *args: fn(*args))
