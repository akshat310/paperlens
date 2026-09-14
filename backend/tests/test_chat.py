"""
Tests for grounded generation: citation parsing, the grounded flag, persistence,
conversation memory, streaming, and the report cache.

Nothing here touches the network or an API key. `llm.generate` is monkeypatched
with a fake and retrieval is stubbed, so the whole file runs in milliseconds.
That is the practical payoff of having put the vendor behind one function: the
LLM becomes a seam we can substitute.
"""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models import ChatMessage, Chunk, Paper, Section
from app.rag import prompts
from app.rag.llm import LLMError, LLMRateLimitError
from app.rag.retriever import RetrievedChunk
from app.services import analysis, chat_service


@pytest.fixture()
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.fixture()
def client(session_factory):
    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def auth_headers(client):
    token = client.post(
        "/api/auth/register",
        json={"email": "ada@example.com", "password": "supersecret1"},
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _make_paper(
    session_factory, email: str = "ada@example.com", status: str = "ready"
) -> str:
    """Insert a ready paper with one section and three chunks."""
    from app.models import User

    db = session_factory()
    user = db.query(User).filter(User.email == email).one()
    paper = Paper(
        user_id=user.id,
        title="Attention Is Sufficient",
        filename="paper.pdf",
        file_path="/tmp/paper.pdf",
        abstract="We remove recurrence and rely entirely on attention.",
        num_pages=3,
        num_chunks=3,
        status=status,
    )
    db.add(paper)
    db.flush()

    section = Section(
        paper_id=paper.id, name="Method", kind="method", ordinal=1,
        page_start=2, page_end=4, char_count=2000,
    )
    db.add(section)
    db.flush()

    for i, (text, page) in enumerate(
        [
            ("We trained the model with AdamW for 40 epochs.", 4),
            ("Evaluation used the CIFAR-100 benchmark.", 6),
            ("The approach relies entirely on attention.", 2),
        ]
    ):
        db.add(
            Chunk(
                paper_id=paper.id, section_id=section.id, content=text,
                page_start=page, page_end=page, chunk_index=i, section="Method",
            )
        )
    db.commit()
    paper_id = paper.id
    db.close()
    return paper_id


@pytest.fixture()
def stub_retrieval(monkeypatch):
    """Fixed chunks, so tests never embed anything or hit the vector store."""
    chunks = [
        RetrievedChunk("c1", "We trained the model with AdamW for 40 epochs.", 4, 4, "Method", 0.9),
        RetrievedChunk("c2", "Evaluation used the CIFAR-100 benchmark.", 6, 7, "Results", 0.8),
        RetrievedChunk("c3", "The approach relies entirely on attention.", 2, 2, None, 0.7),
    ]
    monkeypatch.setattr(chat_service, "retrieve", lambda *a, **kw: chunks)
    return chunks


def _stub_llm(monkeypatch, reply: str) -> dict:
    """Replace the model with a canned reply; records the prompts it received."""
    calls: dict = {"prompts": []}

    def fake_generate(prompt: str, **kwargs) -> str:
        calls["prompts"].append(prompt)
        return reply

    monkeypatch.setattr(chat_service.llm, "generate", fake_generate)
    return calls


# ---------- citation parsing (pure logic) ----------

def test_citations_resolve_pages_from_our_data(stub_retrieval):
    answer, citations = chat_service._extract_citations(
        "They used AdamW [1] and evaluated on CIFAR-100 [2].", stub_retrieval
    )
    assert [c.page_start for c in citations] == [4, 6]
    assert [c.chunk_id for c in citations] == ["c1", "c2"]
    assert answer == "They used AdamW [1] and evaluated on CIFAR-100 [2]."


def test_multi_page_chunk_gets_a_range_label(stub_retrieval):
    """A chunk spanning a page break must say so rather than pick one page."""
    _, citations = chat_service._extract_citations("Evaluated on CIFAR [2].", stub_retrieval)
    assert citations[0].page_start == 6
    assert citations[0].page_end == 7
    assert "pages 6-7" in citations[0].label


def test_single_page_chunk_gets_a_singular_label(stub_retrieval):
    _, citations = chat_service._extract_citations("AdamW [1].", stub_retrieval)
    assert citations[0].label.startswith("page 4")


def test_citations_are_renumbered_to_match_their_order(stub_retrieval):
    """Model cites excerpts 3 then 1; the reader should see [1] then [2]."""
    answer, citations = chat_service._extract_citations(
        "Attention only [3], trained with AdamW [1].", stub_retrieval
    )
    assert answer == "Attention only [1], trained with AdamW [2]."
    assert [c.chunk_id for c in citations] == ["c3", "c1"]


def test_hallucinated_marker_is_dropped_not_rendered(stub_retrieval):
    """Excerpt [9] was never sent, so it cannot be cited."""
    answer, citations = chat_service._extract_citations(
        "It was trained with AdamW [1] on ImageNet [9].", stub_retrieval
    )
    assert "[9]" not in answer
    assert len(citations) == 1
    assert citations[0].chunk_id == "c1"


def test_repeated_marker_yields_one_citation(stub_retrieval):
    _, citations = chat_service._extract_citations(
        "AdamW [1]. Forty epochs [1].", stub_retrieval
    )
    assert len(citations) == 1


# ---------- quote verification ----------

def test_verified_quote_becomes_the_snippet(stub_retrieval):
    answer, citations = chat_service._extract_citations(
        'They trained [1: "with AdamW for 40 epochs"] and it worked.', stub_retrieval
    )
    assert answer == "They trained with AdamW for 40 epochs [1] and it worked."  # syntax gone, words stay
    assert citations[0].verified is True
    assert citations[0].quote == "with AdamW for 40 epochs"
    assert citations[0].snippet == "with AdamW for 40 epochs"


def test_quote_not_in_the_passage_is_rejected(stub_retrieval):
    """The model paraphrased (or invented). Chunk-level citation survives, the quote does not."""
    answer, citations = chat_service._extract_citations(
        'They used [1: "stochastic gradient descent for 40 epochs"].', stub_retrieval
    )
    assert answer == "They used stochastic gradient descent for 40 epochs [1]."
    assert citations[0].verified is False
    assert citations[0].quote is None
    assert citations[0].snippet.startswith("We trained the model")


def test_quote_matching_survives_typographic_differences(stub_retrieval):
    chunks = [RetrievedChunk("c1", "The learning–rate\nwas   fixed at 3e−4.", 1, 1, None, 0.9)]
    _, citations = chat_service._extract_citations(
        'Fixed rate [1: "learning-rate was fixed at 3e-4"].', chunks
    )
    assert citations[0].verified is True


def test_too_short_quote_is_not_treated_as_verified(stub_retrieval):
    _, citations = chat_service._extract_citations('AdamW [1: "AdamW"].', stub_retrieval)
    assert citations[0].verified is False


def test_quoted_hallucinated_marker_is_still_dropped(stub_retrieval):
    answer, citations = chat_service._extract_citations(
        'Reached [9: "90% ImageNet accuracy"] and AdamW [1].', stub_retrieval
    )
    assert "[9" not in answer
    assert answer == "Reached 90% ImageNet accuracy and AdamW [1]."
    assert len(citations) == 1


# ---------- chat endpoint ----------

def test_chat_returns_grounded_answer_and_persists_both_turns(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    _stub_llm(monkeypatch, "They trained with AdamW for 40 epochs [1].")
    paper_id = _make_paper(session_factory)

    res = client.post(
        f"/api/papers/{paper_id}/chat",
        json={"question": "How was the model trained?"},
        headers=auth_headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["grounded"] is True
    assert body["citations"][0]["page_start"] == 4

    db = session_factory()
    stored = db.query(ChatMessage).order_by(ChatMessage.turn_index).all()
    assert [m.role for m in stored] == ["user", "assistant"]
    assert [m.turn_index for m in stored] == [0, 1]
    db.close()


def test_prompt_always_carries_the_global_paper_overview(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    """The fix for "what is this paper about?" failing when the abstract did not
    happen to rank in the top k."""
    calls = _stub_llm(monkeypatch, "It is about attention [1].")
    paper_id = _make_paper(session_factory)

    client.post(
        f"/api/papers/{paper_id}/chat",
        json={"question": "What is this paper about?"},
        headers=auth_headers,
    )

    prompt = calls["prompts"][-1]
    assert "Attention Is Sufficient" in prompt          # title
    assert "rely entirely on attention" in prompt        # abstract
    assert "Method" in prompt                            # section overview


def test_insufficient_context_sets_grounded_false(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    _stub_llm(monkeypatch, prompts.INSUFFICIENT_CONTEXT)
    paper_id = _make_paper(session_factory)

    body = client.post(
        f"/api/papers/{paper_id}/chat",
        json={"question": "What is the author's favourite colour?"},
        headers=auth_headers,
    ).json()
    assert body["grounded"] is False
    assert body["citations"] == []
    assert prompts.INSUFFICIENT_CONTEXT not in body["answer"]  # sentinel is internal


def test_answer_with_no_citations_is_not_grounded(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    """An uncited answer came from somewhere we cannot verify."""
    _stub_llm(monkeypatch, "The paper is about transformers.")
    paper_id = _make_paper(session_factory)

    body = client.post(
        f"/api/papers/{paper_id}/chat",
        json={"question": "What is this about?"},
        headers=auth_headers,
    ).json()
    assert body["grounded"] is False


def test_empty_retrieval_never_calls_the_model(
    client, auth_headers, session_factory, monkeypatch
):
    monkeypatch.setattr(chat_service, "retrieve", lambda *a, **kw: [])
    calls = _stub_llm(monkeypatch, "should not be used")
    paper_id = _make_paper(session_factory)

    body = client.post(
        f"/api/papers/{paper_id}/chat",
        json={"question": "Anything?"},
        headers=auth_headers,
    ).json()
    assert body["grounded"] is False
    assert calls["prompts"] == []  # short-circuited before spending an API call


def test_chat_rejected_while_paper_is_still_processing(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    _stub_llm(monkeypatch, "irrelevant")
    paper_id = _make_paper(session_factory, status="processing")

    res = client.post(
        f"/api/papers/{paper_id}/chat",
        json={"question": "How was it trained?"},
        headers=auth_headers,
    )
    assert res.status_code == 409


def test_another_users_paper_is_not_found(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    """IDOR check: 404, not 403 -- we do not confirm the id exists."""
    _stub_llm(monkeypatch, "irrelevant")
    paper_id = _make_paper(session_factory)

    other = client.post(
        "/api/auth/register",
        json={"email": "eve@example.com", "password": "supersecret1"},
    ).json()["access_token"]

    res = client.post(
        f"/api/papers/{paper_id}/chat",
        json={"question": "How was it trained?"},
        headers={"Authorization": f"Bearer {other}"},
    )
    assert res.status_code == 404


# ---------- conversation memory ----------

def test_recent_turns_are_replayed_into_the_prompt(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    """The fix for follow-ups like "what about the second one?"."""
    calls = _stub_llm(monkeypatch, "Answer [1].")
    paper_id = _make_paper(session_factory)

    client.post(
        f"/api/papers/{paper_id}/chat",
        json={"question": "What optimiser did they use?"},
        headers=auth_headers,
    )
    client.post(
        f"/api/papers/{paper_id}/chat",
        json={"question": "And for how long?"},
        headers=auth_headers,
    )

    second_prompt = calls["prompts"][-1]
    assert "CONVERSATION SO FAR" in second_prompt
    assert "What optimiser did they use?" in second_prompt


def test_older_turns_are_rolled_into_a_summary(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    """A long chat must not grow the prompt without bound."""
    monkeypatch.setattr(chat_service.settings, "CHAT_RECENT_TURNS", 2)
    _stub_llm(monkeypatch, "Answer [1].")
    paper_id = _make_paper(session_factory)

    for i in range(5):
        client.post(
            f"/api/papers/{paper_id}/chat",
            json={"question": f"Question number {i} about the method?"},
            headers=auth_headers,
        )

    db = session_factory()
    paper = db.get(Paper, paper_id)
    assert paper.chat_summary is not None
    assert paper.chat_summary_upto >= 0
    db.close()


# ---------- history ----------

def test_messages_endpoint_hydrates_citations(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    _stub_llm(monkeypatch, "Trained with AdamW [1].")
    paper_id = _make_paper(session_factory)
    client.post(
        f"/api/papers/{paper_id}/chat",
        json={"question": "How was the model trained?"},
        headers=auth_headers,
    )

    history = client.get(f"/api/papers/{paper_id}/messages", headers=auth_headers).json()
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["citations"] == []
    assert history[1]["citations"][0]["page_start"] == 4


def test_history_keeps_order_across_several_questions(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    """Regression guard: question and answer share a timestamp, so ordering must
    not depend on created_at (or on the random UUID id as a tie-break)."""
    _stub_llm(monkeypatch, "Trained with AdamW [1].")
    paper_id = _make_paper(session_factory)

    for question in ["First question?", "Second question?", "Third question?"]:
        client.post(
            f"/api/papers/{paper_id}/chat",
            json={"question": question},
            headers=auth_headers,
        )

    history = client.get(f"/api/papers/{paper_id}/messages", headers=auth_headers).json()
    assert [m["content"] for m in history if m["role"] == "user"] == [
        "First question?",
        "Second question?",
        "Third question?",
    ]


# ---------- streaming ----------

def test_stream_emits_tokens_then_a_done_event_with_citations(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    monkeypatch.setattr(
        chat_service.llm, "stream",
        lambda *a, **kw: iter(["They trained ", "with AdamW ", "[1]."]),
    )
    _stub_llm(monkeypatch, "unused")  # follow-ups path
    paper_id = _make_paper(session_factory)

    with client.stream(
        "POST",
        f"/api/papers/{paper_id}/chat/stream",
        json={"question": "How was the model trained?"},
        headers=auth_headers,
    ) as response:
        assert response.status_code == 200
        events = [
            json.loads(line[6:])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    kinds = [e["type"] for e in events]
    assert "token" in kinds
    assert kinds[-1] == "done"

    done = events[-1]
    assert done["grounded"] is True
    assert done["citations"][0]["page_start"] == 4
    assert "".join(e["text"] for e in events if e["type"] == "token") == (
        "They trained with AdamW [1]."
    )


def test_stream_reports_a_rate_limit_as_an_error_event(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    """Once the response has started, the status code is already sent -- the
    failure has to arrive in-band."""
    def boom(*_a, **_kw):
        raise LLMRateLimitError("Rate limit reached on the free tier.")

    monkeypatch.setattr(chat_service.llm, "stream", boom)
    paper_id = _make_paper(session_factory)

    with client.stream(
        "POST",
        f"/api/papers/{paper_id}/chat/stream",
        json={"question": "How was the model trained?"},
        headers=auth_headers,
    ) as response:
        events = [
            json.loads(line[6:])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert events[-1]["type"] == "error"
    assert events[-1]["retryable"] is True


# ---------- error mapping ----------

def test_rate_limit_maps_to_429_not_503(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    """A quota error is not a server fault -- the request would succeed later."""
    def boom(*_args, **_kwargs):
        raise LLMRateLimitError("Rate limit reached on the free tier.")

    monkeypatch.setattr(chat_service.llm, "generate", boom)
    paper_id = _make_paper(session_factory)

    res = client.post(
        f"/api/papers/{paper_id}/chat",
        json={"question": "How was the model trained?"},
        headers=auth_headers,
    )
    assert res.status_code == 429
    assert "free tier" in res.json()["detail"]


def test_missing_api_key_maps_to_503(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    def boom(*_args, **_kwargs):
        raise LLMError("GEMINI_API_KEY is not set.")

    monkeypatch.setattr(chat_service.llm, "generate", boom)
    paper_id = _make_paper(session_factory)

    res = client.post(
        f"/api/papers/{paper_id}/chat",
        json={"question": "How was the model trained?"},
        headers=auth_headers,
    )
    assert res.status_code == 503


# ---------- report ----------

REPORT_JSON = json.dumps(
    {
        "plain_language": "The paper removes recurrence from sequence models.",
        "problem": "Recurrent models cannot be parallelised across a sequence.",
        "contributions": ["Removes recurrence entirely"],
        "method_walkthrough": "Tokens are projected to queries, keys and values...",
        "experimental_setup": "CIFAR-100, compared against a ResNet baseline.",
        "key_results": ["94.2% accuracy vs 91.0% for the baseline"],
        "results_interpretation": "Improves the metric; does not show generality.",
        "limitations_stated": ["Evaluated on one dataset"],
        "limitations_observed": ["Single seed, no variance reported"],
        "assumptions": ["Sequence length fits in memory"],
        "prior_work": "Builds on attention mechanisms from prior translation work.",
        "reproducibility": "No code link. Hyperparameters given, seeds not stated.",
        "open_questions": ["Does it hold at longer sequence lengths?"],
        "glossary": [{"term": "AdamW", "definition": "Adam with decoupled weight decay."}],
    }
)


def _make_analysable_paper(session_factory) -> str:
    """A paper with one section long enough for the map stage to run on."""
    paper_id = _make_paper(session_factory)
    db = session_factory()
    section = db.query(Section).filter(Section.paper_id == paper_id).one()
    section.char_count = 5000
    db.commit()
    db.close()
    return paper_id


def test_report_is_generated_then_cached(client, auth_headers, session_factory, monkeypatch):
    calls: list[str] = []

    def fake_generate(prompt: str, **kwargs) -> str:
        calls.append(prompt)
        # The map stage returns prose; the reduce stage returns JSON.
        return REPORT_JSON if kwargs.get("json_mode") else "Section analysis text."

    monkeypatch.setattr(analysis.llm, "generate", fake_generate)
    monkeypatch.setattr(analysis, "SessionLocal", session_factory)

    paper_id = _make_analysable_paper(session_factory)

    started = client.post(f"/api/papers/{paper_id}/report", headers=auth_headers)
    assert started.status_code == 202

    report = client.get(f"/api/papers/{paper_id}/report", headers=auth_headers)
    assert report.status_code == 200
    assert report.json()["key_results"] == ["94.2% accuracy vs 91.0% for the baseline"]

    before = len(calls)
    client.post(f"/api/papers/{paper_id}/report", headers=auth_headers)
    assert len(calls) == before  # served from the cache


def test_a_report_from_an_older_prompt_version_is_treated_as_absent(
    client, auth_headers, session_factory
):
    """Editing a prompt must invalidate cached reports -- that is what the
    version constant is for."""
    paper_id = _make_paper(session_factory)
    db = session_factory()
    paper = db.get(Paper, paper_id)
    paper.analysis_json = REPORT_JSON
    paper.analysis_prompt_version = "v0-ancient"
    db.commit()

    assert analysis.get_report(db, paper) is None
    db.close()

    res = client.get(f"/api/papers/{paper_id}/report", headers=auth_headers)
    assert res.status_code == 404


def test_one_failing_section_does_not_fail_the_whole_report(
    client, auth_headers, session_factory, monkeypatch
):
    """An eleven-of-twelve report beats an error page."""
    state = {"n": 0}

    def flaky(prompt: str, **kwargs) -> str:
        if kwargs.get("json_mode"):
            return REPORT_JSON
        state["n"] += 1
        if state["n"] == 1:
            raise LLMError("transient")
        return "Section analysis text."

    monkeypatch.setattr(analysis.llm, "generate", flaky)
    monkeypatch.setattr(analysis, "SessionLocal", session_factory)

    paper_id = _make_analysable_paper(session_factory)
    db = session_factory()
    paper = db.get(Paper, paper_id)
    db.add(
        Section(
            paper_id=paper.id, name="Results", kind="results", ordinal=2,
            page_start=5, page_end=6, char_count=5000,
        )
    )
    db.commit()
    section_id = db.query(Section).filter(Section.name == "Results").one().id
    db.add(
        Chunk(
            paper_id=paper.id, section_id=section_id, content="Accuracy was 94.2 percent.",
            page_start=5, page_end=5, chunk_index=9, section="Results",
        )
    )
    db.commit()
    db.close()

    client.post(f"/api/papers/{paper_id}/report", headers=auth_headers)
    res = client.get(f"/api/papers/{paper_id}/report", headers=auth_headers)
    assert res.status_code == 200


def test_malformed_report_json_is_rejected(client, auth_headers, session_factory, monkeypatch):
    """Pydantic is the guardrail: a missing required field must not reach the client."""
    def fake_generate(prompt: str, **kwargs) -> str:
        if kwargs.get("json_mode"):
            return '{"plain_language": "oops, nothing else"}'
        return "Section analysis text."

    monkeypatch.setattr(analysis.llm, "generate", fake_generate)
    monkeypatch.setattr(analysis, "SessionLocal", session_factory)

    paper_id = _make_analysable_paper(session_factory)
    client.post(f"/api/papers/{paper_id}/report", headers=auth_headers)

    res = client.get(f"/api/papers/{paper_id}/report", headers=auth_headers)
    assert res.status_code == 404  # no report was stored

    job = client.get(f"/api/papers/{paper_id}/job", headers=auth_headers).json()
    assert job["status"] == "failed"


# ---------- job progress ----------

def test_job_endpoint_reports_stage_and_position(client, auth_headers, session_factory):
    from app.services import jobs

    paper_id = _make_paper(session_factory)
    db = session_factory()
    job = jobs.create(db, paper_id, prompts.PROMPT_VERSION)
    jobs.update(db, job.id, stage="analyzing", current_item=3, total=11)
    db.close()

    body = client.get(f"/api/papers/{paper_id}/job", headers=auth_headers).json()
    assert body["stage"] == "analyzing"
    assert body["current"] == 3
    assert body["total"] == 11
    assert 0 < body["stage_index"] < body["stage_count"]


# ---------- export ----------

def test_markdown_export_contains_every_report_section(
    client, auth_headers, session_factory
):
    paper_id = _make_paper(session_factory)
    db = session_factory()
    paper = db.get(Paper, paper_id)
    paper.analysis_json = REPORT_JSON
    paper.analysis_prompt_version = prompts.PROMPT_VERSION
    db.commit()
    db.close()

    res = client.get(f"/api/papers/{paper_id}/export.md", headers=auth_headers)
    assert res.status_code == 200
    assert "attachment" in res.headers["content-disposition"]

    body = res.text
    assert "# Attention Is Sufficient" in body
    assert "## How the method works" in body
    assert "94.2% accuracy" in body
    assert "AdamW" in body  # glossary


def test_pdf_export_is_a_valid_pdf(client, auth_headers, session_factory):
    paper_id = _make_paper(session_factory)
    db = session_factory()
    paper = db.get(Paper, paper_id)
    paper.analysis_json = REPORT_JSON
    paper.analysis_prompt_version = prompts.PROMPT_VERSION
    db.commit()
    db.close()

    res = client.get(f"/api/papers/{paper_id}/export.pdf", headers=auth_headers)
    assert res.status_code == 200
    assert res.content.startswith(b"%PDF-")
    assert res.content.rstrip().endswith(b"%%EOF")
    assert b"/Type /Catalog" in res.content
    assert len(res.content) > 1000


def test_export_404s_before_an_analysis_exists(client, auth_headers, session_factory):
    paper_id = _make_paper(session_factory)
    assert client.get(
        f"/api/papers/{paper_id}/export.md", headers=auth_headers
    ).status_code == 404


# ---------- claim check ----------

def test_claim_supported_with_verified_evidence(client, auth_headers, session_factory, stub_retrieval, monkeypatch):
    _stub_llm(monkeypatch, json.dumps({
        "verdict": "supports",
        "reasoning": "The paper trains with AdamW [1].",
        "evidence": [{"excerpt": 1, "quote": "trained the model with AdamW"}],
        "caveats": "Only one optimiser is reported.",
    }))
    paper_id = _make_paper(session_factory)

    response = client.post(
        f"/api/papers/{paper_id}/claim", json={"claim": "The model is trained with AdamW."},
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "supports"
    assert body["evidence"][0]["verified"] is True
    assert body["evidence"][0]["citation"]["page_start"] == 4
    assert body["citations"][0]["chunk_id"] == "c1"


def test_supports_verdict_without_verifiable_evidence_is_downgraded(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    """The model says 'supports' but its quote is not in the passage: partial at best."""
    _stub_llm(monkeypatch, json.dumps({
        "verdict": "supports",
        "reasoning": "It clearly uses SGD [1].",
        "evidence": [{"excerpt": 1, "quote": "trained with SGD"}],
    }))
    paper_id = _make_paper(session_factory)
    body = client.post(
        f"/api/papers/{paper_id}/claim", json={"claim": "Trained with SGD."}, headers=auth_headers
    ).json()
    assert body["verdict"] == "partial"
    assert body["evidence"][0]["verified"] is False


def test_unknown_verdict_becomes_not_addressed(
    client, auth_headers, session_factory, stub_retrieval, monkeypatch
):
    _stub_llm(monkeypatch, json.dumps({"verdict": "maybe", "reasoning": "Unclear.", "evidence": []}))
    paper_id = _make_paper(session_factory)
    body = client.post(
        f"/api/papers/{paper_id}/claim", json={"claim": "Something else."}, headers=auth_headers
    ).json()
    assert body["verdict"] == "not_addressed"


# ---------- compare ----------

def test_compare_labels_citations_with_their_paper(client, auth_headers, session_factory, monkeypatch):
    from app.services import compare as compare_service

    a = _make_paper(session_factory)
    b = _make_paper(session_factory)

    def fake_retrieve(db, query, paper_id, **kwargs):
        # Paper A yields one chunk, paper B another; ids distinguish them.
        return [RetrievedChunk(f"chunk-{paper_id[:4]}", f"Passage from {paper_id}.", 2, 2, "Method", 0.9)]

    monkeypatch.setattr(compare_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        compare_service.llm, "generate",
        lambda prompt, **kw: "Paper A does X [1]; Paper B does Y [2].",
    )

    response = client.post(
        "/api/papers/compare",
        json={"paper_ids": [a, b], "question": "How do they differ?"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert [c["paper_label"] for c in body["citations"]] == ["A", "B"]
    assert body["citations"][0]["paper_id"] == a
    assert body["citations"][1]["paper_id"] == b
    assert body["citations"][1]["label"].startswith("Paper B, page 2")
    assert [p["label"] for p in body["papers"]] == ["A", "B"]


def test_compare_refuses_another_users_paper(client, auth_headers, session_factory):
    mine = _make_paper(session_factory)
    other_token = client.post(
        "/api/auth/register", json={"email": "bob@example.com", "password": "supersecret1"}
    ).json()["access_token"]
    theirs = _make_paper(session_factory, email="bob@example.com")

    response = client.post(
        "/api/papers/compare",
        json={"paper_ids": [mine, theirs], "question": "Compare them."},
        headers=auth_headers,
    )
    assert response.status_code == 404
    assert other_token  # silence unused warning; the point is the 404 above


def test_compare_requires_at_least_two_papers(client, auth_headers, session_factory):
    only = _make_paper(session_factory)
    response = client.post(
        "/api/papers/compare", json={"paper_ids": [only], "question": "?"}, headers=auth_headers
    )
    assert response.status_code == 422


def test_clearing_the_conversation_also_drops_the_summary(client, auth_headers, session_factory, stub_retrieval, monkeypatch):
    _stub_llm(monkeypatch, "Answer [1].")
    paper_id = _make_paper(session_factory)
    client.post(f"/api/papers/{paper_id}/chat", json={"question": "First question?"}, headers=auth_headers)

    db = session_factory()
    paper = db.get(Paper, paper_id)
    paper.chat_summary = "Earlier they discussed AdamW."
    paper.chat_summary_upto = 1
    db.commit()
    db.close()

    assert client.delete(f"/api/papers/{paper_id}/messages", headers=auth_headers).status_code == 204
    assert client.get(f"/api/papers/{paper_id}/messages", headers=auth_headers).json() == []

    db = session_factory()
    paper = db.get(Paper, paper_id)
    assert paper.chat_summary is None and paper.chat_summary_upto == -1
    db.close()
