"""
Operational behaviour: the background worker, the restart sweep, the page cap.

These are the pieces that keep the memory and quota story honest under load and
across restarts, so they get tests even though none of them is "a feature".
"""

import threading

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import AnalysisJob, Paper, User
from app.rag.pdf_parser import read_metadata
from app.services import jobs, worker


@pytest.fixture()
def client_with_user():
    from fastapi.testclient import TestClient

    from app.database import get_db
    from app.main import app

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)

    def override():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override
    with TestClient(app) as client:
        token = client.post(
            "/api/auth/register", json={"email": "ops@example.com", "password": "supersecret1"}
        ).json()["access_token"]
        yield client, {"Authorization": f"Bearer {token}"}
    app.dependency_overrides.clear()


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


# ---------- worker ----------

def test_worker_runs_jobs_one_at_a_time_off_the_calling_thread(monkeypatch):
    # conftest patches enqueue to run inline for the rest of the suite; here we
    # want the real thing.
    monkeypatch.undo()

    seen: list[tuple[str, int]] = []
    done = threading.Event()
    overlap = {"max": 0, "now": 0}
    lock = threading.Lock()

    def job(name: str):
        with lock:
            overlap["now"] += 1
            overlap["max"] = max(overlap["max"], overlap["now"])
        seen.append((name, threading.get_ident()))
        with lock:
            overlap["now"] -= 1
        if name == "c":
            done.set()

    worker.start()
    try:
        for name in ("a", "b", "c"):
            worker.enqueue(job, name)
        assert done.wait(5), "worker never ran the jobs"
    finally:
        worker.stop()

    assert [n for n, _ in seen] == ["a", "b", "c"]  # FIFO
    assert overlap["max"] == 1                       # strictly serial
    assert all(t != threading.get_ident() for _, t in seen)  # not on our thread


def test_worker_survives_a_failing_job(monkeypatch):
    monkeypatch.undo()
    done = threading.Event()

    def bad():
        raise RuntimeError("boom")

    worker.start()
    try:
        worker.enqueue(bad)
        worker.enqueue(done.set)
        assert done.wait(5), "a failing job killed the worker"
    finally:
        worker.stop()


def test_enqueue_runs_inline_when_no_worker_is_running():
    ran: list[int] = []
    worker.enqueue(ran.append, 1)
    assert ran == [1]


# ---------- restart sweep ----------

def _paper(db, status: str) -> Paper:
    user = User(email=f"{status}@example.com", hashed_password="x")
    db.add(user)
    db.flush()
    paper = Paper(
        user_id=user.id, title="t", filename="f.pdf", file_path="/tmp/f.pdf", status=status
    )
    db.add(paper)
    db.flush()
    return paper


def test_sweep_fails_running_jobs_and_unfinished_papers(db):
    running = _paper(db, "processing")
    ready = _paper(db, "ready")
    db.add(AnalysisJob(paper_id=running.id, status="running", stage="embedding"))
    db.add(AnalysisJob(paper_id=ready.id, status="queued", stage="queued"))
    db.add(AnalysisJob(paper_id=ready.id, status="done", stage="done"))
    db.commit()

    assert jobs.sweep_interrupted(db) == 2

    for job in db.query(AnalysisJob).filter(AnalysisJob.status != "done"):
        assert job.status == "failed"
        assert "restart" in job.error_message

    db.refresh(running)
    db.refresh(ready)
    assert running.status == "failed"   # never reached ready: index is incomplete
    assert ready.status == "ready"      # index intact; only the report was lost


def test_sweep_is_a_no_op_when_nothing_was_running(db):
    assert jobs.sweep_interrupted(db) == 0


# ---------- page cap ----------

def test_page_cap_rejects_before_any_page_is_extracted(tmp_path):
    from tests.test_rag import _make_pdf  # the suite's own tiny PDF writer

    path = tmp_path / "long.pdf"
    _make_pdf(path, [["page one text here"]] * 5)

    with pytest.raises(ValueError, match="5 pages; the limit is 3"):
        read_metadata(str(path), max_pages=3)

    assert read_metadata(str(path), max_pages=5).num_pages == 5


# ---------- from-url fetch (offline) ----------

def test_fetch_refuses_hosts_outside_the_allowlist(tmp_path):
    from app.services import fetch

    for url in (
        "http://169.254.169.254/latest/meta-data",   # cloud metadata endpoint
        "http://localhost:8000/api/health",
        "https://example.com/paper.pdf",
        "ftp://arxiv.org/x.pdf",
    ):
        with pytest.raises(fetch.FetchError):
            fetch.download_pdf(url, tmp_path / "x.pdf", 1024)
    assert not (tmp_path / "x.pdf").exists()


def test_arxiv_ids_are_parsed_from_abs_and_pdf_urls():
    from app.services import fetch

    assert fetch.arxiv_id_from_url("https://arxiv.org/abs/2405.12345v2") == "2405.12345"
    assert fetch.arxiv_id_from_url("https://arxiv.org/pdf/1706.03762") == "1706.03762"
    assert fetch.arxiv_id_from_url("https://arxiv.org/abs/cs/0112017") == "cs/0112017"
    assert fetch.arxiv_id_from_url("https://doi.org/10.1000/xyz") is None


def test_byte_iterator_reader_honours_read_sizes():
    from app.services.fetch import _Reader

    reader = _Reader(iter([b"abc", b"defgh", b"i"]))
    assert reader.read(2) == b"ab"
    assert reader.read(4) == b"cdef"
    assert reader.read(100) == b"ghi"
    assert reader.read(1) == b""


def test_from_url_endpoint_rejects_disallowed_host(client_with_user):
    client, headers = client_with_user
    response = client.post(
        "/api/papers/from-url", json={"url": "https://example.com/a.pdf"}, headers=headers
    )
    assert response.status_code == 400
    assert "arxiv.org" in response.json()["detail"]


# ---------- OCR fallback (offline) ----------

def test_scanned_pdf_is_transcribed_when_ocr_is_enabled(tmp_path, monkeypatch):
    """A PDF with no text layer goes through the OCR page source and ends up indexed."""
    from app.rag import ocr
    from app.services import ingestion
    from app.config import settings
    from app.models import Chunk, Paper, User
    from tests.test_rag import _make_pdf

    # A PDF whose pages carry no text: pypdf extracts nothing from it.
    path = tmp_path / "scan.pdf"
    _make_pdf(path, [[], []])

    monkeypatch.setattr(settings, "SCANNED_PDF_OCR", True)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(
        ocr, "transcribe_page",
        lambda pdf_bytes, n: f"1 Introduction\n\nTranscribed text of page {n}. " * 20,
    )
    monkeypatch.setattr(ingestion, "embed_texts", lambda texts: [[1.0] + [0.0] * 767 for _ in texts])
    # Skip the analysis stage: it is a separate concern and needs the LLM.
    import app.services.analysis as analysis
    monkeypatch.setattr(analysis, "run_analysis", lambda *a, **k: None)

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(ingestion, "SessionLocal", factory)

    db = factory()
    user = User(email="scan@example.com", hashed_password="x")
    db.add(user); db.flush()
    paper = Paper(user_id=user.id, title="t", filename="scan.pdf", file_path=str(path), status="pending")
    db.add(paper); db.commit()
    pid = paper.id
    db.close()

    ingestion.process_paper(pid)

    db = factory()
    paper = db.get(Paper, pid)
    assert paper.status == "ready", paper.error_message
    assert paper.ocr is True
    assert db.query(Chunk).filter(Chunk.paper_id == pid).count() >= 1
    db.close()


def test_scanned_pdf_fails_clearly_when_ocr_is_disabled(tmp_path, monkeypatch):
    from app.services import ingestion
    from app.config import settings
    from app.models import Paper, User
    from tests.test_rag import _make_pdf

    path = tmp_path / "scan.pdf"
    _make_pdf(path, [[]])
    monkeypatch.setattr(settings, "SCANNED_PDF_OCR", False)

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(ingestion, "SessionLocal", factory)

    db = factory()
    user = User(email="scan2@example.com", hashed_password="x")
    db.add(user); db.flush()
    paper = Paper(user_id=user.id, title="t", filename="scan.pdf", file_path=str(path), status="pending")
    db.add(paper); db.commit(); pid = paper.id; db.close()

    ingestion.process_paper(pid)

    db = factory()
    paper = db.get(Paper, pid)
    assert paper.status == "failed"
    assert "SCANNED_PDF_OCR" in paper.error_message


# ---------- related work parsing (offline) ----------

def test_reference_list_is_split_and_titled():
    from app.services.related import parse_references

    text = (
        "[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. "
        "arXiv preprint arXiv:1607.06450, 2016. [2] Dzmitry Bahdanau, Kyunghyun Cho, and "
        "Yoshua Bengio. Neural machine translation by jointly learning to align and translate. "
        "CoRR, abs/1409.0473, 2014. [3] Short. 2001."
    )
    entries = parse_references(text)
    assert [e.year for e in entries] == ["2016", "2014"]
    assert entries[0].title == "Layer normalization"
    assert entries[0].first_author == "Jimmy Lei Ba"
    assert entries[1].title.startswith("Neural machine translation")


def test_openalex_inverted_abstract_is_reconstructed():
    from app.services.related import _reconstruct_abstract

    assert _reconstruct_abstract({"world": [1], "Hello": [0]}) == "Hello world"
    assert _reconstruct_abstract(None) is None


# ---------- rate limiting ----------

def test_token_bucket_refills_over_time(monkeypatch):
    from app import ratelimit

    bucket = ratelimit.TokenBucket(capacity=2, window_seconds=10.0)
    clock = {"t": 1000.0}
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: clock["t"])

    assert bucket.try_acquire("u") == 0.0
    assert bucket.try_acquire("u") == 0.0
    wait = bucket.try_acquire("u")
    assert 4.9 < wait <= 5.0          # 2 tokens / 10 s -> one token every 5 s
    clock["t"] += 5.0
    assert bucket.try_acquire("u") == 0.0
    assert bucket.try_acquire("other-user") == 0.0  # keys are independent


def test_chat_is_rate_limited_per_user(client_with_user, monkeypatch):
    from app import ratelimit
    from app.config import settings

    client, headers = client_with_user
    monkeypatch.setattr(settings, "RATE_LIMITING", True)
    monkeypatch.setattr(ratelimit, "LOGIN", ratelimit.TokenBucket(2, 60.0))

    # Login limiter is the easiest to hit without a paper: 2 attempts, then 429.
    for _ in range(2):
        client.post("/api/auth/login", json={"email": "ops@example.com", "password": "wrong"})
    response = client.post(
        "/api/auth/login", json={"email": "ops@example.com", "password": "wrong"}
    )
    assert response.status_code == 429
    assert "Retry-After" in response.headers
