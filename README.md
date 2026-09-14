# PaperLens

Upload a research paper — or paste an arXiv link. Get a section-by-section
analysis of it, and a chat that answers from the paper's actual text, with
citations that open the PDF at the cited page and highlight the quoted sentence,
or an explicit "the paper doesn't address that" when the answer isn't there.

The point of the project is not the chat interface. It's that **the citations
are real**: page numbers are looked up from the database, quoted phrases are
kept only if they occur verbatim in the passage they point at, and the
highlight in the PDF is a string match against extracted text — none of it is
written by the language model.

The second point is that it runs in **512 MB of RAM** on a free-tier instance.
That constraint drove most of the architecture, and every place it forced a
different design than the obvious one is written down here rather than left
implicit — see [The constraint ledger](#the-constraint-ledger).

```
┌──────────────┐     ┌──────────────────────────────────────────────────────┐
│  React + TS  │     │                     FastAPI  (one process)           │
│  PDF.js      │────▶│  auth ── short-lived JWT + httpOnly refresh cookie    │
│  (static)    │ /api│  rate limits ── per-user token buckets, in memory     │
└──────────────┘     │                                                      │
                     │  upload / arXiv link ──▶ streamed to disk, hashed    │
                     │      │                     (dedupe: same file = free) │
                     │      ▼                                               │
                     │  ┌── worker thread (queue) ─────────────────────┐    │
                     │  │ pypdf, one page at a time (+ font-size cues) │    │
                     │  │ ── or Gemini OCR per page, if no text layer  │    │
                     │  │ group into sections ─▶ chunk on boundaries   │    │
                     │  │ embed per section (Gemini, 768-dim, REST)    │    │
                     │  │ MAP-REDUCE analysis: one call per section,   │    │
                     │  │   one to synthesise a 14-field report        │    │
                     │  │ OpenAlex lookups for the reference list      │    │
                     │  └──────────────────────────────────────────────┘    │
                     │                     │                                │
                     │                     ▼                                │
                     │            ┌──────────────────┐   ┌──────────────┐   │
                     │            │      SQLite      │   │  llm_calls   │   │
                     │            │ text + float32   │   │  ledger: one │   │
                     │            │ vectors + report │   │  row per call│   │
                     │            └──────────────────┘   └──────────────┘   │
                     │                                                      │
                     │  chat ──▶ expand query into 3 variants               │
                     │       └─▶ dense (NumPy) + BM25, optionally 1 section │
                     │           └─▶ RRF fusion → 10 chunks                 │
                     │               └─▶ Gemini (streamed, SSE)             │
                     │                   └─▶ [n: "quote"] markers resolved  │
                     │                       from OUR data; quotes verified │
                     │  claim ──▶ same retrieval → supports/contradicts/... │
                     │  compare ▶ same retrieval × 2–3 papers, labelled A/B │
                     └──────────────────────────────────────────────────────┘
```

---

## Quick start

**Docker (one container, exactly what gets deployed):**

```bash
cp backend/.env.example backend/.env    # add your Gemini API key
docker build -t paperlens .
docker run -p 7860:7860 \
  -e GEMINI_API_KEY=... \
  -e SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))") \
  paperlens
# open http://localhost:7860
```

**Local development:**

```bash
# Terminal 1 — backend (Python 3.12)
cd backend
uv venv --python 3.12 && uv pip install -r requirements.txt
cp .env.example .env          # add GEMINI_API_KEY
uvicorn app.main:app --reload # http://localhost:8000/docs

# Terminal 2 — frontend
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

Get a free Gemini API key at <https://aistudio.google.com/apikey>. It is used
for generation, embeddings and (optionally) OCR.

---

## The memory budget

The target is Render's free tier: **512 MB RAM, one instance**. Exceeding it is
not a slow app, it is an OOM kill. So the budget is treated as a build
constraint, and it is measured rather than assumed:

```bash
cd backend && python eval/memory_probe.py --pages=300
```

Measured on Python 3.12, after the SDK migration described below:

| | RSS |
|---|---:|
| Bare interpreter | 17.9 MB |
| + SQLAlchemy | 42.9 MB |
| + FastAPI / uvicorn | 58.8 MB |
| + NumPy | 67.0 MB |
| + pypdf | 74.1 MB |
| + rank_bm25 | 74.1 MB |
| + app code | 85.0 MB |
| + google-genai (REST, no gRPC) | **111.2 MB** ← fixed floor |
| **Peak during a 300-page ingest** | **119.8 MB** (23% of budget) |

Before the migration from the deprecated `google-generativeai` SDK (gRPC) the
floor was 126.5 MB and the peak 133 MB. Swapping the SDK removed the gRPC core
from the process and saved ~15 MB — the kind of change that actually moves the
number, because **the floor dominates**: ~90% of peak is libraries sitting in
memory, which is why every design decision that mattered was about *not
loading* something.

Ingest adds ~8 MB on top of the floor and that number barely moves with
document size: 50 / 300 / 1200 pages → 130 / 133 / 145 MB on the old SDK.
Sublinear, not flat; the residual is pypdf's internal object cache.

**Under load.** `eval/load_test.py` fires 20 concurrent clients at the running
server for 30 s while polling `/api/debug/memory`:

```
20 clients for 30s: 10,759 requests (359/s), all 200
  health  p50=32 ms   p95=50 ms
  list    p50=68 ms   p95=94 ms
RSS during load: min 128.3  mean 132.9  peak 133.9 MB  (limit 512)
```

That is the `THREAD_LIMIT=2` argument as a measurement rather than a claim:
the pool is pinned, so forty clients queue rather than forty working sets
existing at once.

The load test also found a real bug on its second run, with another process
writing to the same database: FastAPI runs the `get_db` dependency and the
endpoint body as *separate* threadpool jobs, so under a burst every request's
dependency could check out a pool connection before any body ran; the 16th
then blocked inside `get_db` — holding one of the two threads — waiting on a
pool the bodies could no longer drain. Two 500s and 30-second latencies. The
fix is to stop the connection pool being the scarce resource (`max_overflow`
above uvicorn's `--limit-concurrency`, short `pool_timeout`); re-run under the
same contention: 1,551 requests, all 200, peak 102 MB.

### The decisions the budget forced

**1. No local ML models.** A sentence-transformer needs ~300 MB resident before
embedding anything; torch needs several times that. Embeddings come from a
hosted API. The cost: ingestion needs network access and burns free-tier quota.

**2. No vector database.** ChromaDB kept a Rust core and an HNSW index resident
for the process's whole life. Every search is scoped to **one paper** — a few
hundred vectors — and exhaustively scoring 400 × 768 floats is ~0.2 ms in
NumPy, *exact* where an ANN index is approximate. Vectors are `float32` blobs
in the `chunks` table, searched with one `np.dot`.

**3. pypdf instead of PyMuPDF.** MuPDF is a full rendering engine resident
throughout. pypdf follows the content stream's drawing order, so a two-column
layout can interleave — the one place the constraint bought a worse product.
Two things now soften it: pypdf's per-fragment font data is used as a second
heading signal (a bold or larger line is a heading even with unknown words),
and rendering moved to the browser (below), where it costs the server nothing.

**4. Streaming, everywhere.** Uploads and link downloads are read in 1 MB
blocks and hashed as they go. Parsing yields one page at a time; each
section's chunks are embedded and written before the next is read. The whole
document never exists in memory.

**5. Pinned concurrency.** Starlette runs every `def` endpoint on an anyio
threadpool that defaults to **40 threads**. `THREAD_LIMIT=2` is applied in
`app/main.py` and read back at `/api/debug/memory` so it can be verified on the
live instance.

**6. One worker thread, not a job queue.** `BackgroundTasks` runs on that same
2-thread pool — an ingest would hold one of the two threads for minutes, and
two uploads left none for `/health`. Long jobs now go through a
`queue.Queue` consumed by a single `threading.Thread`
(`app/services/worker.py`). Celery would have needed a worker *process* (a
second copy of the 111 MB floor) and a Redis the free tier doesn't provide.
The trade: jobs die with the process, so a startup sweep marks anything left
`running` as failed with a reason rather than an infinite spinner.

---

## How it works

### Ingestion

Pages stream out of pypdf with their numbers attached. Headings are recognised
two ways: a conservative vocabulary (Abstract, Method, Results …) and
typography — a short, title-shaped line set in bold or in a larger font than
the page's body text is a heading even when its words are unknown ("3.2.1
Scaled Dot-Product Attention"). Wrapped heading lines are merged; bibliography
entries that start with a year are not mistaken for numbered sections. Front
matter and unrecognised text become sections too: **no text is ever dropped
for want of a recognised heading.**

Chunks are ~1000 tokens and follow section and paragraph boundaries.

**Scanned PDFs.** With no text layer, pypdf finds nothing. If
`SCANNED_PDF_OCR=true`, each page is cut into a one-page PDF and sent to Gemini
to transcribe, and the transcript enters the same chain. No OCR engine, no
rasteriser in the process — but one call per page, so it is off by default and
capped, and the paper is badged "OCR" in the UI.

**arXiv links.** `POST /papers/from-url` fetches from an allowlist
(arxiv.org, doi.org — an SSRF guard, checked before any socket opens, on every
redirect hop) through the same size cap and dedupe, and takes the title,
authors and abstract from the arXiv API instead of guessing them from page one.

Re-uploading the same file is free: the SHA-256 computed during upload finds
the existing paper. Scoped per user, deliberately.

### Analysis: map-reduce, not one call

Each section gets its own call with a prompt written for its *kind* — the
Methods prompt asks how the mechanism works step by step; the Results prompt
asks for the numbers with their baselines, including the comparisons where the
paper loses. A reduce call synthesises those into a report with fourteen
addressable fields. 8–12 calls per paper, run once, cached against
`PROMPT_VERSION`. A failed section doesn't fail the paper.

### Chat

Retrieval is hybrid: dense vectors (NumPy) + BM25, each run over three
phrasings of the question, all fused with Reciprocal Rank Fusion (k=60, ranks
only — cosine and BM25 scores aren't on comparable scales). The reader can
narrow retrieval to one section. Title, abstract and section summaries are
always in the prompt; recent turns are replayed verbatim and older ones rolled
into a running summary. Answers stream over SSE.

### Citations are the whole product

The model cites `[n]` markers against numbered excerpts, and for any claim
carrying a number or a name it must use the form `[n: "verbatim phrase"]`.
Then, in our code:

- markers outside the range we sent are **deleted** (the model invented a source);
- a quoted phrase is kept **only if it is a literal substring of chunk n**
  (whitespace and typographic punctuation normalised, case preserved) — it
  gets a "verified quote" badge and becomes the citation's snippet; otherwise
  the citation degrades to chunk level and the miss is counted;
- surviving markers are renumbered so `[n]` is exactly `citations[n-1]`.

Click a citation and **PDF.js opens the page in the browser** with the verified
phrase highlighted, using PDF.js's own text geometry. The unconstrained design
was server-side rendering with PyMuPDF — the library removed to fit the budget.
The browser already has a PDF engine; the server just streams the bytes.

`grounded=false` on the `INSUFFICIENT_CONTEXT` sentinel or zero valid citations.

### Claims and comparisons

**Check claim** asks the same retrieval a different question: does the paper
*support*, *contradict*, *partly support* or *not address* this statement?
The verdict is a fixed vocabulary rendered as a badge; evidence quotes go
through the same verbatim check, and a "supports" with no verifiable evidence
is downgraded rather than shown.

**Compare** runs retrieval once per paper (2–3), labels every excerpt A/B/C,
and answers in one call. Per-paper vector scoping — a memory decision — is
exactly the shape that keeps papers from being mixed up; citations read
"Paper B, pages 4–5".

### The ledger

Every generation, embedding and OCR call writes one row to `llm_calls`:
purpose, model, tokens in/out, latency, success. `GET /papers/{id}/usage`
answers "what did this paper cost?"; `/api/debug/usage` gives daily totals.
The unconstrained answer is OpenTelemetry and a tracing backend; a table in the
database we already have answers the same questions for zero memory.

### Related work

The reference list is parsed into entries and each is looked up on OpenAlex
(free, no key) for its abstract, venue and citation count, on the worker
thread, cached on the paper. No model involved.

---

## Security

- **Sessions.** Access tokens are short-lived (30 min) and held in the
  browser's memory only — never `localStorage`, which any injected script can
  read. The session lives in an **httpOnly, SameSite=Lax refresh cookie**
  scoped to `/api/auth`, rotated on every use, carrying a per-user token
  version so "log out everywhere" revokes all of them by bumping one integer.
  On a 401 the client refreshes once and replays the request.
- **Rate limits.** Per-user token buckets on chat (12/min), ingest (6/h) and
  login (10/min per email), plus a daily model-call cap counted from the
  ledger. In memory, which is *correct* here: `WEB_CONCURRENCY=1` by design.
- **Uploads.** Extension, then streamed size cap, then magic bytes; stored
  under a generated name; page cap and a cooperative ingest deadline.
- **Outbound fetches.** Host allowlist before any connection; redirects
  re-checked per hop.
- **Headers.** `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`,
  `Cache-Control: no-store` on the API, and a minimal CSP on the bundle.
- **Ownership.** Every paper route goes through `OwnedPaper`; another user's
  paper is a 404, not a 403.

---

## API

```
POST   /api/auth/register            → Token  (+ refresh cookie)
POST   /api/auth/login               → Token  (+ refresh cookie)
POST   /api/auth/refresh             → Token  (cookie → new access token, rotated)
POST   /api/auth/logout              → 204    ?everywhere=bool
GET    /api/auth/me                  → User

POST   /api/papers                   → Paper (201)   multipart PDF
POST   /api/papers/from-url          → Paper (201)   {url}  arxiv.org / doi.org
GET    /api/papers                   → Paper[]
GET    /api/papers/{id}              → Paper
GET    /api/papers/{id}/file         → application/pdf (streamed)
GET    /api/papers/{id}/sections     → Section[]
GET    /api/papers/{id}/job          → JobStatus     ← polled for progress
DELETE /api/papers/{id}              → 204

POST   /api/papers/{id}/chat         → ChatResponse  {question, section_id?}
POST   /api/papers/{id}/chat/stream  → text/event-stream
POST   /api/papers/{id}/claim        → ClaimResponse {claim}
POST   /api/papers/compare           → CompareResponse {paper_ids[2..3], question}
GET    /api/papers/{id}/messages     → ChatMessage[]
DELETE /api/papers/{id}/messages     → 204   (also drops the rolling summary)
GET    /api/papers/{id}/followups    → {questions}
GET    /api/papers/{id}/report       → PaperReport   404 until analysed
POST   /api/papers/{id}/report       → 202 Accepted  ?force=bool
GET    /api/papers/{id}/related      → {status, entries}   OpenAlex-enriched refs
GET    /api/papers/{id}/usage        → Usage         from the call ledger
GET    /api/papers/{id}/export.md    → Markdown
GET    /api/papers/{id}/export.pdf   → PDF

GET    /api/health                   → liveness
GET    /api/debug/memory             → RSS, thread limit, queued jobs
GET    /api/debug/usage              → ledger totals  ?days=1
```

---

## Deployment

Render, from `render.yaml`:

1. Push to GitHub.
2. Render → **New → Blueprint** → point at the repo.
3. Set `GEMINI_API_KEY` when prompted. `SECRET_KEY` is generated automatically.

`DEBUG=false` arms a guard in `app/config.py` that **refuses to boot** with the
placeholder signing key.

### Storage: nothing survives a redeploy, and that's deliberate

The free plan has no persistent disk. SQLite lives on the ephemeral filesystem
and is wiped on every deploy and cold start. The alternative was Render's free
Postgres — which **expires 30 days after creation**. For a portfolio project
that has to still work at an interview months from now, a database that deletes
itself is worse than one that resets. Loss is made non-fatal (tables are
recreated, missing columns are added, interrupted jobs are marked as such) and
stated in the UI.

To make it durable: set `DATABASE_URL` to a Postgres that does not expire
(Neon's free tier persists). Uploaded PDFs would still need object storage.

### Cold starts

A free instance sleeps after 15 minutes and takes 30–60 seconds to wake.
`WakeUpGate` probes `/api/health` on mount and shows a "waking up" panel with an
elapsed counter rather than letting the first visit look broken.

---

## Testing

```bash
cd backend && pytest                 # 94 tests, no network, no API key
cd frontend && npm test              # Vitest: SSE frame parser, citation chips
cd backend && python eval/run_eval.py         # retrieval metrics, offline
BASE_URL=http://localhost:8000 npx playwright test   # browser smoke, uses quota
```

Nothing in the unit suites touches the network. `llm.generate` is
monkeypatched, embeddings are synthetic, the PDF fixtures are built by a
~40-line PDF writer in the test file, and the tests run against a throwaway
database so a dev server can keep running alongside them. CI
(`.github/workflows/ci.yml`) runs both suites, the memory probe, and builds and
boots the Docker image.

The Playwright test drives the real thing end to end: register → upload →
analysis → **reload through the refresh cookie** → question → citation chip →
PDF viewer at the cited page with the quote highlighted → claim verdict → logout.

### Evaluation

`backend/eval/` holds a generated fixture paper with ground-truth pages known
by construction, 15 answerable questions tagged lexical/paraphrase/mixed, and 6
**unanswerable** ones the paper genuinely does not cover. `run_eval.py` measures
retrieval offline; `--judge` grades faithfulness with a *different* model,
scores abstention, and counts how often the app's citation safety net fired.

Current numbers (`eval/results.md`, `gemini-3.1-flash-lite`, 768-dim
`gemini-embedding-001`, ~1000-token chunks):

| Mode | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| dense | 93.3% | 100.0% | 100.0% | 0.956 |
| bm25 | 66.7% | 93.3% | 93.3% | 0.778 |
| hybrid | 93.3% | 100.0% | 100.0% | **0.967** |

On the previous chunking, dense beat hybrid (0.867 vs 0.856); with
section-aligned ~1000-token chunks, hybrid edges ahead. Fifteen questions over
twelve chunks cannot meaningfully separate 0.967 from 0.956 — the honest
reading is "both are near the ceiling on this corpus", and BM25 stays for the
reason it always did: it is the fallback when the embedding API fails.

The judged pass (`--judge`, answers by `gemini-3.1-flash-lite`, graded by
`gemini-3.6-flash` — a different model, to avoid self-enhancement bias):

| | |
|---|---|
| Faithful to retrieved context | 14/15 (93%) — the miss: "multi-hop reasoning" inferred from "combining evidence from multiple passages" |
| Unanswerable questions correctly declined | 5/6 (83%); 1 answered anyway |
| Answerable questions wrongly declined | 0 |
| Markers pointing at an excerpt never sent | 0 of 43 |
| Quoted phrases rejected as not verbatim | 12 of 28 (43%) |

That last line is the most useful number in the table. Asked to copy phrases
verbatim, a flash-tier model paraphrases inside the quotes almost half the
time — and every one of those would have been shown to the reader as the
paper's own words without the check. The rejected quotes degrade to
chunk-level citations, which are still real; nothing invented gets a badge.

---

## The constraint ledger

What the unconstrained design would be, what 512 MB / the free tier made me
build instead, and why the shipped version holds up.

| Wanted | Built instead | Why it holds up |
|---|---|---|
| ChromaDB / pgvector | float32 blobs in SQLite + one `np.dot` | Search is scoped to one paper (~400 vectors); exhaustive is 0.2 ms and *exact* |
| Local sentence-transformer | Hosted Gemini embeddings | ~300 MB saved; cost is network + quota, stated |
| PyMuPDF | pypdf, streaming, with font-size heading cues | Rendering engine gone; two-column cost stated, softened by typography signal |
| gRPC SDK | `google-genai` over REST | Same API, −15 MB floor, and the supported SDK |
| Celery + Redis | One worker thread + `queue.Queue` | Same property (jobs off the request pool), zero memory; restart sweep covers the loss |
| 40-thread default pool | `THREAD_LIMIT=2`, verifiable at `/api/debug/memory` | Load test: 20 clients, peak 134 MB |
| Multiple uvicorn workers | `WEB_CONCURRENCY=1` | Each worker duplicates the ~111 MB floor for a network-bound workload |
| Render Postgres | Ephemeral SQLite, loss visible, self-repairing schema | Free Postgres expires in 30 days; a reset beats a deletion |
| Server-rendered PDF highlights | PDF.js in the browser | The browser has a renderer; server cost is zero |
| NLI / cross-encoder verification | Verbatim substring check on model-emitted quotes | Stricter and free; literal support, not semantic |
| Cross-encoder reranker | RRF over dense + BM25 + query variants | Near ceiling on the eval; a reranker would not fit |
| Layout model (GROBID, Nougat) | Vocabulary + font weight/size from pypdf | Torch or Java; the PDF's own font data is free |
| Tesseract OCR | Gemini page-by-page transcription, gated | No native library; quota instead of memory |
| OpenTelemetry + tracing backend | `llm_calls` table + `/usage` | Nowhere free to send traces; one table answers the questions |
| Redis rate limiting | In-memory token buckets | One process by design, so a dict is correct |
| Shared cross-paper ANN index | Per-paper retrieval, fused and labelled | Never mixes papers; bounded N |
| RAGAS / eval framework | 300-line harness | Must be explainable line by line |
| WebSockets | SSE over plain HTTP | One-directional text; works through Render's proxy unconfigured |
| LangChain / LlamaIndex | Hand-written retrieval path | The whole point is being able to explain it |

---

## Things I'd flag in a review

- **Prompts live in one module** (`app/rag/prompts.py`) with a `PROMPT_VERSION`
  constant; reports are cached against it.
- **Sync dependencies and the threadpool.** Every `def` dependency is its own
  threadpool job, so a dependency that holds a resource (a DB session) can
  starve the endpoint that would release it. The pool is sized so this cannot
  deadlock; the principled fix is async endpoints with an async driver, which
  is a bigger change than this project needs.
- **BM25 is rebuilt on every query.** A few milliseconds for a few hundred
  chunks and no invalidation logic. At real scale: Postgres full-text.
- **Schema repair is additive only.** `init_db` adds missing columns with
  `ALTER TABLE ADD COLUMN`; it never renames, drops or retypes. Alembic is the
  real answer once there is a Postgres.
- **Reference parsing is regexes.** ~80% of entries get a usable title on
  IEEE/NeurIPS-style lists; misses are listed as unmatched, not hidden.
- **The PDF highlight is per text item.** PDF.js hands back lines or line
  fragments, so the highlight covers the line containing the phrase rather
  than the exact glyph run. Good enough to find it; not typographically exact.
- **The PDF exporter is hand-written** (~150 lines, base-14 fonts, WinAnsi).
  Greek letters don't survive; the Markdown export has no such limit.
