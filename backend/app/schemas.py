"""
Pydantic schemas -- the contract between the API and the outside world.

Why these exist separately from the SQLAlchemy models: models describe how data
is *stored*, schemas describe how it is *sent and received*. Keeping them apart
means a `hashed_password` column can never accidentally end up in an API
response, because no response schema has that field.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ---------- Auth ----------

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)
    full_name: str | None = Field(default=None, max_length=255)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # allows UserOut.model_validate(orm_obj)

    id: str
    email: EmailStr
    full_name: str | None
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ---------- Papers ----------

class PaperOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    filename: str
    source_url: str | None = None
    authors: str | None = None
    venue: str | None = None
    year: str | None = None
    abstract: str | None = None
    num_pages: int
    num_chunks: int
    ocr: bool = False
    status: str
    error_message: str | None
    created_at: datetime


class FromUrlRequest(BaseModel):
    url: str = Field(min_length=10, max_length=1000)


class SectionOut(BaseModel):
    """One section of the paper, for the reader view's outline."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    kind: str
    ordinal: int
    page_start: int
    page_end: int


class JobStatus(BaseModel):
    """Progress of the background analysis, polled by the frontend.

    `stage` and `stage_index` are separate because the UI needs both: the label
    to display, and a position to size a progress bar from without hardcoding
    the stage list in TypeScript as well as Python.
    """

    id: str
    status: str          # queued | running | done | failed
    stage: str           # queued | parsing | embedding | analyzing | synthesizing | done
    stage_index: int
    stage_count: int
    current: int
    total: int
    message: str
    error_message: str | None = None


# ---------- Deep analysis report ----------

class GlossaryTerm(BaseModel):
    term: str
    definition: str


class PaperReport(BaseModel):
    """The paper-level report produced by the reduce stage.

    Every field is addressable on its own so the UI can render each as a
    separate collapsible block and the exporter can lay them out in order.

    List fields default to empty rather than being required: the prompt tells
    the model to return `[]` when the paper genuinely does not cover something,
    and a model that omits the key entirely means the same thing. Failing
    validation over that would throw away an otherwise good report.
    """

    plain_language: str
    problem: str
    contributions: list[str] = []
    method_walkthrough: str
    experimental_setup: str = ""
    key_results: list[str] = []
    results_interpretation: str = ""
    limitations_stated: list[str] = []
    limitations_observed: list[str] = []
    assumptions: list[str] = []
    prior_work: str = ""
    reproducibility: str = ""
    open_questions: list[str] = []
    glossary: list[GlossaryTerm] = []


# ---------- Chat ----------

class Citation(BaseModel):
    """A pointer back into the paper, resolved from OUR data.

    `page_start` and `page_end` replaced the old single `page_number`: chunks now
    follow section boundaries and can span a page break. They are equal for most
    chunks. `label` is the rendered form ("page 4", "pages 4-6, section: Method")
    so the frontend does not reimplement that formatting.
    """

    chunk_id: str
    page_start: int
    page_end: int
    section: str | None = None
    label: str
    snippet: str
    # The phrase the model said supports the claim, kept only if it literally
    # occurs in the chunk. `verified` is True exactly when `quote` is set; it is
    # a separate field so the UI does not have to know that rule.
    quote: str | None = None
    verified: bool = False


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    # Restrict retrieval to one section. Optional; the server checks it
    # belongs to the paper rather than trusting the client.
    section_id: str | None = Field(default=None, max_length=36)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool  # False when the model had to say "not in this paper"
    followups: list[str] = []


class ClaimRequest(BaseModel):
    claim: str = Field(min_length=5, max_length=1000)


class ClaimEvidence(BaseModel):
    citation: Citation
    quote: str
    verified: bool


class ClaimResponse(BaseModel):
    """A structured verdict on one claim.

    `verdict` is a fixed vocabulary rather than prose so the UI can render it
    as a badge and a reader can scan it. `evidence` quotes are checked against
    the chunk text exactly like chat citations; an unverifiable quote is kept
    (it is the model's reasoning) but marked, never presented as the paper's
    words.
    """

    claim: str
    verdict: str  # supports | contradicts | partial | not_addressed
    reasoning: str
    evidence: list[ClaimEvidence] = []
    caveats: str = ""
    citations: list[Citation] = []


class CompareRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=3)
    question: str = Field(min_length=3, max_length=2000)


class CompareCitation(Citation):
    """A chat citation plus which paper it belongs to."""

    paper_id: str
    paper_label: str  # "A", "B", "C"
    paper_title: str


class CompareResponse(BaseModel):
    answer: str
    citations: list[CompareCitation]
    grounded: bool
    papers: list[dict]  # [{label, id, title}] in the order they were labelled


class RelatedEntryOut(BaseModel):
    raw: str
    title: str | None = None
    year: str | None = None
    first_author: str | None = None
    openalex_id: str | None = None
    matched_title: str | None = None
    abstract: str | None = None
    venue: str | None = None
    cited_by: int | None = None
    url: str | None = None


class RelatedOut(BaseModel):
    status: str  # ready | pending
    entries: list[RelatedEntryOut] = []


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    created_at: datetime
    citations: list[Citation] = []


class FollowupsOut(BaseModel):
    questions: list[str]


# ---------- Usage ----------

class UsageOut(BaseModel):
    """Totals from the LLM call ledger, for one paper or for a period."""

    calls: int
    failed: int
    prompt_tokens: int
    output_tokens: int
    total_latency_ms: int
    by_purpose: dict[str, int] = {}


# ---------- Ops ----------

class MemoryOut(BaseModel):
    """Reported by /api/debug/memory so the 512 MB budget can be verified."""

    rss_mb: float
    vms_mb: float
    percent_of_limit: float
    limit_mb: int
    python_objects: int
    # The applied threadpool cap, read back from the live limiter rather than
    # from the environment. Setting THREAD_LIMIT does nothing on its own -- this
    # is how you confirm on the running instance that it took effect, which is
    # the difference between a bounded worst case and a 40-thread one.
    thread_limit: int
    # Jobs waiting on the background worker. Non-zero under load is expected;
    # growing without bound means the worker is stuck.
    queued_jobs: int = 0
