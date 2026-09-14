> **Historical snapshot.** This is the read-only survey of the codebase *before* the v2 rewrite (ChromaDB, PyMuPDF, 900-character chunks, one-shot summary). It is kept because it is the baseline the redesign was planned against; every finding in section 8 was subsequently acted on. The current architecture is described in the top-level README.

# PaperLens — Phase 0 Analysis

Read-only survey of the repository as it stands at commit `b4d837c`. No code was
changed. Every claim below is from the current source, not from prior notes.

---

## 1. Stack

| Layer | Choice |
|---|---|
| Backend language | Python 3.12 (pinned; `backend/requirements.txt` header says so) |
| Backend framework | FastAPI 0.115.6, **sync** endpoints on the threadpool |
| ASGI server | uvicorn 0.34.0 (`[standard]` extra) |
| ORM / DB | SQLAlchemy 2.0.36 → SQLite (`storage/paperlens.db`) |
| Vector DB | **ChromaDB 1.5.9, in-process PersistentClient** |
| PDF | **PyMuPDF (`fitz`) 1.25.1** |
| Keyword search | `rank-bm25` 0.2.2 (pure Python, rebuilt per query) |
| LLM + embeddings | `google-generativeai` 0.8.3 (Gemini) |
| Auth | PyJWT + bcrypt, HTTPBearer |
| Frontend | React 19 + TypeScript 6 + Vite 8 + Tailwind 4 + react-router 8 + axios |
| Frontend pkg mgr | npm (`package-lock.json`, `npm ci` in Docker) |
| Python pkg mgr | pip / uv (`uv venv --python 3.12 && uv pip install -r requirements.txt`) |
| Tests | pytest 8.3.4 + httpx; `backend/tests/` (auth, chat, rag) |

### Entry points

- Backend: [backend/app/main.py](backend/app/main.py) → `app` (FastAPI).
  `uvicorn app.main:app --reload`, port 8000 in dev.
- Frontend: [frontend/src/main.tsx](frontend/src/main.tsx) → [frontend/src/App.tsx](frontend/src/App.tsx).
  `npm run dev`, port 5173, proxying `/api` → `localhost:8000`
  ([frontend/vite.config.ts:14-18](frontend/vite.config.ts#L14-L18)).
- Deployed: one container. Vite output is baked into `backend/static/` and
  [main.py:104-135](backend/app/main.py#L104-L135) mounts `/assets` plus an
  SPA catch-all. Same origin, `CORS_ORIGINS=""`.
- Startup side effects: `lifespan` creates `UPLOAD_DIR`, `CHROMA_DIR`, and runs
  `Base.metadata.create_all` ([main.py:28-40](backend/app/main.py#L28-L40)).
  No Alembic.

### Endpoint inventory (current public contract)

```
POST   /api/auth/register            → Token
POST   /api/auth/login               → Token
GET    /api/auth/me                  → UserOut
POST   /api/papers                   → PaperOut (201)   multipart file
GET    /api/papers                   → PaperOut[]
GET    /api/papers/{id}              → PaperOut          ← frontend polls this
DELETE /api/papers/{id}              → 204
POST   /api/papers/{id}/chat         → ChatResponse
GET    /api/papers/{id}/messages     → ChatMessageOut[]
POST   /api/papers/{id}/summary      → PaperSummary      ?refresh=bool
GET    /api/health                   → {status, app}
```

There is **no** `/debug/memory` and no job-status endpoint. Progress is inferred
purely from `Paper.status` (`pending → processing → ready | failed`).

---

## 2. Lifecycle A — upload / ingest / analyze / render

**1. Upload** — [routers/papers.py:31-76](backend/app/routers/papers.py#L31-L76)

`async def upload_paper` validates in four layers: `.pdf` extension → `await
file.read()` → 25 MB cap → `%PDF-` magic bytes. Stores under a generated UUID
filename in `UPLOAD_DIR`. Inserts a `Paper` row with `status="pending"`, then
`background_tasks.add_task(process_paper, paper.id)` and returns 201 immediately.

Two problems here, both memory-relevant:
- `await file.read()` pulls **the entire file into RAM as one `bytes` object
  before the size check runs**. A 25 MB PDF is 25 MB resident; the cap rejects
  it only *after* it is already allocated. A 200 MB upload allocates 200 MB
  before being refused.
- There is **no content hash**. Re-uploading the same PDF re-parses and
  re-embeds from scratch and creates a duplicate `Paper` row.

**2. Ingest** — [services/ingestion.py:37-119](backend/app/services/ingestion.py#L37-L119)

Runs in a Starlette `BackgroundTask` — i.e. in the **same process, same event
loop's threadpool**, after the response is flushed. Opens its own `SessionLocal`.
Steps: `status="processing"` → `parse_pdf` → `chunk_document` → bulk-insert
`Chunk` rows → `db.flush()` for ids → `embed_texts(all chunks)` → 
`vector_store.add_chunks(...)` → `status="ready"`. Any exception →
`_mark_failed` writes `error_message[:500]`.

**3. Parse** — [rag/pdf_parser.py:99-146](backend/app/rag/pdf_parser.py#L99-L146)

`fitz.open()`, then iterate pages. For each page: `page.get_text("text")`, scan
lines against 12 heading regexes to track `current_section`, run `_clean_text`
(de-hyphenate, unwrap hard line breaks, collapse whitespace), and **append a
`Page` dataclass to a list that is retained for the whole document**. Returns
`ParsedPDF(title, num_pages, pages=[...all pages...])`.

Captured today: title (metadata, else first substantial line of page 1),
page count, per-page text, a coarse section label. **Not captured:** authors,
venue/year, abstract as a distinct field, figure/table captions, equations,
reference list.

Section detection is page-order-only and single-pass; `current_section` leaks
across pages, which is right, but any heading that does not match one of the 12
regexes silently leaves the previous label attached.

**4. Chunk** — [rag/chunker.py:87-122](backend/app/rag/chunker.py#L87-L122)

Recursive character splitter over `SEPARATORS = ["\n\n", "\n", ". ", ...]`,
`CHUNK_SIZE=900` **characters** (~200–250 tokens), `CHUNK_OVERLAP=150` chars
(~17%). Chunking never crosses a page boundary — deliberate, so every chunk maps
to exactly one page. Fragments < 50 chars are dropped.

So the target chunk size is roughly **4–5× smaller** than the 800–1200 *token*
target in the brief.

**5. Analyze (the "summary")** — [services/chat_service.py:238-289](backend/app/services/chat_service.py#L238-L289)

Not part of ingest. Lazy, on-demand, triggered by a button click in
[SummaryPanel.tsx](frontend/src/components/SummaryPanel.tsx). One LLM call over
the **first 30,000 characters of the paper in document order** (`_summary_context`
walks chunks by `chunk_index` and stops at the budget — it does not select, it
truncates). Returns a 7-field JSON object validated by `PaperSummary`. Retried
once at temperature 0.4 on validation failure. Cached in `Paper.summary_json`
with **no prompt version key**, so a prompt change does not invalidate the cache.

This single truncated call is the root cause of "the analysis is shallow."

**6. Render** — [DashboardPage.tsx](frontend/src/pages/DashboardPage.tsx) polls
`GET /api/papers/{id}` while status is `processing`; [SummaryPanel.tsx](frontend/src/components/SummaryPanel.tsx)
renders the 7 fields as flat, always-expanded sections in an 80-unit-wide
`aside`, hidden below `lg:`. No collapsing, no export, no progress detail.

---

## 3. Lifecycle B — chat with paper

[routers/chat.py:41-55](backend/app/routers/chat.py#L41-L55) → `_require_ready`
(409 unless `status=="ready"`) → `chat_service.answer_question`.

**Retrieval** — [rag/retriever.py:134-164](backend/app/rag/retriever.py#L134-L164),
`mode="hybrid"`, `top_k = settings.TOP_K = 5`, `fetch_k = 10`:

- *Dense*: `embed_query(question)` (`task_type=retrieval_query`) → 
  `vector_store.search()` → Chroma `.query(where={"paper_id": ...})`, cosine.
- *BM25*: loads **every chunk row for the paper into a Python list**, tokenises
  all of them, builds a fresh `BM25Okapi` index, scores, discards it. Per query.
- *Fusion*: Reciprocal Rank Fusion, `RRF_K=60`, ranks only (scores are on
  incomparable scales). Top 5 survive.

**What is actually sent to the model** — [rag/prompts.py:78-90](backend/app/rag/prompts.py#L78-L90):

```
CHAT_SYSTEM_RULES (6 numbered rules)
=== BEGIN EXCERPTS ===
[1] (page 4, section: Methods)
<chunk text ~900 chars>
[2] ...            ← exactly 5 of these
=== END EXCERPTS ===
QUESTION: <question>
ANSWER:
```

So the model sees roughly **5 × 900 ≈ 4,500 characters ≈ 1,100 tokens** of the
paper. Nothing else. Specifically **not** sent: the title, the abstract, the
section list, any summary, and — critically — **any prior conversation turn**.
`answer_question`'s own docstring states chat is stateless by design; history is
persisted and displayed but never fed back. Rule 6 explicitly instructs "Be
concise and factual."

That combination — ~1.1k tokens of context, no global structure, no history, and
an instruction to be concise — is the direct cause of "short and generic."

**Post-processing** — `_extract_citations`
([chat_service.py:59-115](backend/app/services/chat_service.py#L59-L115)):
`[n]` markers outside `1..len(chunks)` are deleted as hallucinations; surviving
markers are renumbered so `[n] == citations[n-1]`; page number and 240-char
snippet are looked up from **our** `Chunk` rows, never authored by the model.
`grounded=False` on the `INSUFFICIENT_CONTEXT` sentinel or zero valid citations.
Both turns persisted in one transaction with an explicit `turn_index`.

No streaming anywhere — `POST /chat` is a single blocking JSON response.
No query expansion. No follow-up suggestions.

---

## 4. Every LLM / embedding call in the codebase

All generation funnels through `llm.generate()`
([rag/llm.py:60-142](backend/app/rag/llm.py#L60-L142)). Defaults:
`temperature=0.1`, `max_output_tokens=8192`, `json_mode=False`.
Model from `settings.GEMINI_MODEL`.

**Note a live inconsistency:** `config.py` defaults to `gemini-3.5-flash`,
while both `Dockerfile` and `.env.example` set `gemini-3.1-flash-lite`. The
deployed model is `flash-lite`; the code default is not. Worth reconciling.

| # | Call site | Model | Prompt | max_output_tokens | temp | Paper text passed in |
|---|---|---|---|---|---|---|
| 1 | `chat_service.answer_question` → `llm.generate(build_chat_prompt(...))` [chat_service.py:212](backend/app/services/chat_service.py#L212) | `settings.GEMINI_MODEL` | `CHAT_SYSTEM_RULES` + 5 numbered excerpts + question ([prompts.py:36-90](backend/app/rag/prompts.py#L36-L90)) | 8192 (default, never overridden) | 0.1 | **5 chunks ≈ 4,500 chars ≈ 1.1k tokens.** No title, no abstract, no history. |
| 2 | `chat_service.generate_summary` attempt 1 [chat_service.py:275](backend/app/services/chat_service.py#L275) | `settings.GEMINI_MODEL` | `SUMMARY_INSTRUCTIONS` + title + context ([prompts.py:93-112](backend/app/rag/prompts.py#L93-L112)), `json_mode=True` | 8192 | 0.1 | **First 30,000 chars in document order** (`SUMMARY_CONTEXT_CHARS`). Tail of a long paper — often Results, Discussion, Limitations — is silently cut. |
| 3 | `chat_service.generate_summary` retry | same | same prompt | 8192 | **0.4** | same |
| 4 | `eval/run_eval.py --judge` | separate `model=` override arg | judge prompt in the eval script | 8192 | 0.1 | eval-only, not on the request path |

**Embedding calls** — [rag/embeddings.py](backend/app/rag/embeddings.py):

| Call | Model | Dim | task_type | Batching |
|---|---|---|---|---|
| `embed_texts` (ingest) | `models/gemini-embedding-001` | 768 (truncated from native 3072) | `retrieval_document` | 100 per request (`_MAX_BATCH`) |
| `embed_query` (chat) | same | 768 | `retrieval_query` | 1 |

Asymmetric on purpose. Truncated vectors arrive **un-normalised** (norm ≈ 0.58),
so `_normalise` rescales them — removing that is a silent correctness bug.
Hosted, so **no local ML model runs in-process**. That part of the 512 MB
constraint is already satisfied.

---

## 5. Where the paper text lives, and what survives a restart

| Artifact | Store | Survives process restart? | Survives redeploy? |
|---|---|---|---|
| Original PDF | file, `UPLOAD_DIR` (`/home/user/storage/uploads`) | Yes | **No** — ephemeral container disk |
| Chunk **text** + page + section | SQLite `chunks` table | Yes | **No** |
| Chunk **vectors** | ChromaDB on disk (`CHROMA_DIR`) | Yes | **No** |
| Summary JSON | SQLite `papers.summary_json` | Yes | **No** |
| Chat history | SQLite `chat_messages` | Yes | **No** |
| `ParsedPDF` / `Page` objects | RAM only, inside `process_paper` | No — never persisted | No |
| BM25 index | RAM only, **rebuilt on every single query** | No | No |

Join key: a `Chunk.id` (UUID) is the same id used for the Chroma vector. Text in
SQL, vectors in Chroma.

So: everything survives a restart, **nothing survives a redeploy**, because the
Render free tier gives no persistent disk and all state lives under
`/home/user/storage`. Today that means every deploy wipes all users, papers and
vectors — silently. Losing the *cache* is acceptable; losing *accounts* is not,
and the README does not currently say this happens.

---

## 6. Memory profile — where whole objects are held in RAM

Ordered by size, worst first.

1. **Whole PDF as `bytes`** — [papers.py:44](backend/app/routers/papers.py#L44).
   `contents = await file.read()` before the size check. Up to 25 MB (or more,
   transiently, since the cap is checked after allocation). Then
   `destination.write_bytes(contents)` — the buffer is live across the write.
2. **All pages of the parsed document at once** —
   `ParsedPDF.pages: list[Page]` ([pdf_parser.py:113](backend/app/rag/pdf_parser.py#L113))
   holds every page's cleaned text for the whole run, then `chunk_document`
   builds a *second* full copy as `list[TextChunk]`, then `ingestion` builds a
   *third* as `list[Chunk]` ORM objects — all three alive simultaneously at
   [ingestion.py:88](backend/app/services/ingestion.py#L88).
3. **The whole embedding matrix as Python lists** —
   `embeddings = embed_texts([...])` ([ingestion.py:87](backend/app/services/ingestion.py#L87)).
   Python `list[list[float]]`, not NumPy. A CPython `float` is 24 bytes plus an
   8-byte pointer, so ≈ **32 bytes per dimension**: 900 chunks × 768 dims ×
   32 B ≈ **22 MB** — versus 2.6 MB for the same data as `float32`. Chroma then
   copies it again during `add`.
4. **Every chunk of the paper, per chat query** — `_bm25_search`
   ([retriever.py:64-72](backend/app/rag/retriever.py#L64-L72)) materialises all
   chunk rows, builds a tokenised corpus (`list[list[str]]` — Python string
   objects, several × the text size), constructs `BM25Okapi`, throws it away.
   Transient but repeated on every question, and it is pure GC pressure.
5. **ChromaDB resident cost** — a full vector DB with a Rust core, plus numpy,
   plus its HNSW index held in memory per collection. This is the single largest
   *fixed* library cost in the process and the brief rules it out outright.
6. **`google.generativeai` → gRPC** — imported lazily but permanently resident
   after the first call. Tens of MB.

No DataFrames anywhere (pandas is not a dependency). No local ML model — already
correct.

**Current measured peak: 147 MB** during ingest + chat under a hard 512 MB cap
(recorded during the earlier Render work). That is the anchor number; it is
measured, not derived, and it is what a redesign has to beat or match. I have
not re-measured it in this session and it predates nothing relevant, but it
should be re-verified before it is quoted in a README.

---

## 7. Deployment config as it exists

**Present:** root [Dockerfile](Dockerfile) (two-stage: `node:20-slim` builds the
frontend → `python:3.12-slim` runtime), `.dockerignore`, `backend/.env.example`,
`GET /api/health`, a Docker `HEALTHCHECK`.

**Absent:** `render.yaml`, `Procfile`, any `/debug/memory`, any worker/thread
pinning, any load test.

Runtime specifics:
- `CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]` —
  `sh -c` so Render's injected `$PORT` expands. **No `--workers`, no
  `--limit-concurrency`, no thread-pool cap.** Defaults today: 1 uvicorn
  process, but Starlette's `run_in_threadpool` defaults to **40 threads**, and
  every sync endpoint runs there. Forty concurrent BM25 rebuilds would be an OOM.
- Runs as UID 1000 (`user`), `HOME=/home/user`, all state under
  `/home/user/storage`, chowned at build.
- `DEBUG=false` in the image, which arms the `SECRET_KEY` guard in
  `Settings.model_post_init` — the app refuses to boot with the placeholder key.
- Comments still reference Hugging Face Spaces; the actual target is Render
  (HF now requires a paid plan for Docker Spaces).

Env vars in play: `APP_NAME`, `DEBUG`, `SECRET_KEY`, `ALGORITHM`,
`ACCESS_TOKEN_EXPIRE_MINUTES`, `DATABASE_URL`, `CHROMA_DIR`, `UPLOAD_DIR`,
`GEMINI_API_KEY`, `GEMINI_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`,
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `CORS_ORIGINS`, `PORT`.
`config.py` is the only module that reads the environment.

---

## 8. Conflicts between the brief and the current code

Flagging these now rather than discovering them mid-Phase-1. Each one is a
deliberate decision to reverse, not an accident.

| # | Brief requires | Code does | Notes |
|---|---|---|---|
| 1 | No Chroma in-process; vectors as blobs in SQLite, cosine in NumPy | ChromaDB 1.5.9 PersistentClient | Removes the largest fixed library cost. A few hundred vectors per paper — NumPy dot product is genuinely trivial. Requires a schema change and a new `vector_store` implementation; existing indexes are discarded. |
| 2 | Prefer `pypdf`/`pdfminer.six` over anything that rasterizes | PyMuPDF (`fitz`) | PyMuPDF is a MuPDF binding and *can* rasterize; we only call `get_text`. Swapping to `pypdf` costs some extraction quality on multi-column layouts. Worth doing for the constraint, but it is a real quality trade and I'd want to eyeball the output on a two-column paper. |
| 3 | Parse streaming, free each page | Retains all pages, then all chunks, then all ORM rows | Requires writing chunks to SQLite per-page and dropping the page. |
| 4 | Chunks 800–1200 tokens | 900 **characters** (~200–250 tokens) | ~4× increase; changes chunk counts, embedding cost, and every eval number. |
| 5 | Hash the file, reuse existing work | No hash column; duplicates re-ingest | New `Paper.content_hash` column. |
| 6 | Analysis as a polled background **job** with progress | `BackgroundTask` with a 4-value `status` string; summary is a synchronous blocking POST | Needs a job table + status endpoint. Staying in-process (no Celery) is consistent with the project's no-infrastructure rule. |
| 7 | Chat: 8–12 chunks, long answers, history, citations | 5 chunks, "be concise", stateless | Straightforward prompt + retrieval changes. |
| 8 | Stream responses token by token | Single blocking JSON POST | SSE endpoint; the citation-resolution step currently runs *after* the full text arrives, so streaming needs markers resolved at the end of the stream. Contract change. |
| 9 | Cap upload size, reject oversized | Cap exists but is checked **after** reading the whole body into RAM | Must stream to disk with a running byte count, or check `Content-Length` first. |
| 10 | Pin worker/thread counts | Neither set; threadpool defaults to 40 | One-line start-command change. |
| 11 | Persistent storage decision | SQLite on ephemeral disk, wiped every redeploy, undocumented | Needs an explicit choice: Render free Postgres, or accept loss and say so loudly. Free Postgres also expires after 30 days on Render's free plan — that belongs in the decision. |
| 12 | Every prompt in one module with a version constant | Prompts are in `rag/prompts.py` already ✓ but there is **no version constant**, and `summary_json` is cached without one | Small addition; makes cache invalidation correct. |

**Contract changes that would be visible to the frontend** if the brief is
implemented as written: `POST /papers/{id}/summary` (shape and sync/async
behaviour), `POST /papers/{id}/chat` (JSON → SSE), plus new endpoints for job
status and export. I will not change any of these without saying so explicitly.

---

## 9. Expected peak RSS for the heaviest path — after the redesign

The brief asks for this before finalizing. Estimate for **ingesting a 25 MB,
40-page PDF** under the proposed architecture, on Python 3.12 / linux slim:

| Component | MB | Basis |
|---|---:|---|
| Python 3.12 interpreter + stdlib | 15 | baseline `python -c pass` RSS |
| FastAPI + Starlette + Pydantic v2 + uvicorn | 35 | pydantic-core is a compiled Rust ext |
| SQLAlchemy 2.0 + SQLite driver | 20 | ORM metadata is the bulk |
| NumPy | 25 | imported once, stays |
| `google-generativeai` + gRPC | 45 | the largest single library; gRPC core is heavy |
| **Fixed floor** | **~140** | everything above is resident forever |
| Upload streamed to disk, 1 MB chunks | 1 | not `read()` — this is the fix for item 9 |
| One page's text + its chunks, transient | 2 | freed before the next page |
| One embedding batch (100 × 768 float32 + JSON response) | 4 | `float32` NumPy, not Python lists |
| SQLite page cache + write buffers | 10 | default 2 MB cache, headroom |
| Peak fragmentation / GC slack | 20 | glibc malloc does not return freed arenas promptly |
| **Estimated peak RSS** | **~175 MB** | |

Headroom against 512 MB: **~337 MB**, i.e. peak is ~34% of the cap.

How I got there: the fixed floor dominates and is the part I am most confident
about — it is library import cost, measurable directly with
`python -c "import ...; print(rss)"`, and I intend to measure each line rather
than ship the estimate. The per-request terms are small *by construction*
because the redesign never holds the whole document: streaming upload bounds
term 6 at the buffer size, per-page parsing bounds term 7 at one page, and
batched `float32` embeddings bound term 8 at 100 vectors.

Two honest caveats:
1. Dropping ChromaDB is worth roughly 60–90 MB of the current fixed floor, but I
   have not isolated it by measurement, only by reasoning about what it pulls in.
   That number needs verifying before it goes in a README.
2. The current *measured* 147 MB peak is lower than this 175 MB estimate. That is
   not a contradiction — the estimate is deliberately conservative on the gRPC
   and fragmentation lines, and the measured figure was taken on a specific
   paper. If the redesigned pipeline measures above ~200 MB I would treat that as
   a signal that something is retaining more than it should, and investigate
   rather than raise the budget.

The binding risk is **concurrency, not any single request**. With 40 threadpool
threads, forty simultaneous requests each holding a batch and a page would blow
the cap easily. Pinning to a single worker and 2 threads, as the brief requires,
is what actually keeps the number above meaningful.

---

## 10. What I recommend for Phase 1, pending your go-ahead

Two things I'd want your call on before writing code:

- **Item 2 (PyMuPDF → pypdf).** This is the one constraint where I think the
  cost is real: `pypdf`'s text extraction is noticeably worse than MuPDF's on
  two-column academic layouts, and layout quality feeds directly into chunk
  quality and therefore retrieval quality. PyMuPDF's *memory* cost when only
  calling `get_text()` page-by-page is modest. I'll swap it if you want the
  constraint held literally, but I'd suggest measuring PyMuPDF's actual per-page
  RSS first and keeping it if it fits.
- **Item 11 (persistence).** Render's free Postgres expires after 30 days,
  which for a portfolio project that needs to still work at an interview six
  months from now is arguably worse than documented cache loss. My inclination
  is ephemeral SQLite with loss made explicit and non-fatal, plus a seeded demo
  account — but this is your project's shop window, so it's your call.

Everything else in the brief I'd implement as written.

**Nothing has been changed. Awaiting your confirmation before Phase 1.**
