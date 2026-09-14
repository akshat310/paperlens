"""
Step 3: turn text into vectors.

An embedding maps text to a fixed-length list of numbers such that texts with
similar *meaning* land close together. That is why searching "how was the model
trained?" can retrieve a chunk that never uses the word "trained" but says "we
optimised using AdamW" -- keyword search would miss it entirely.

Model choice: gemini-embedding-001, hosted.
------------------------------------------
The instance has 512 MB of RAM. A local sentence-transformer needs roughly
300 MB resident on top of Python, FastAPI and SQLAlchemy before it has embedded
anything, and torch needs several times that. It does not fit. A hosted model
moves that cost off the box entirely, in exchange for a network call and an API
key.

That trade has a real downside, stated plainly: ingestion requires internet
access and burns free-tier quota, where a local model would work offline
forever. The same key already needed for answer generation covers it, so it adds
no new account, but "clone it and it just works" is no longer true without a key.

Asymmetric embeddings.
----------------------
Gemini exposes a `task_type`, and we use two: passages are embedded as
RETRIEVAL_DOCUMENT, queries as RETRIEVAL_QUERY. The model then puts a *question*
and the *passage that answers it* near each other, which is not the same
objective as putting two similar sentences near each other. Getting this wrong
is silent -- retrieval still works, just worse -- so it is worth being explicit.
"""

import logging
import time

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Hard ceiling imposed by the API. settings.EMBED_BATCH_SIZE may be lower; it is
# clamped to this, so a misconfigured value degrades rather than 400s.
_MAX_BATCH = 100

# Retries for a transient failure (rate limit, dropped connection) during
# ingestion. Ingestion is a background job that has already told the user it is
# working, so waiting a couple of seconds is far better than failing the paper.
_RETRIES = 3
_BACKOFF_SECONDS = 2.0


class EmbeddingError(RuntimeError):
    """Any failure producing embeddings.

    Mirrors LLMError in llm.py: callers catch this rather than vendor-specific
    exception types, so provider details never leak past this module.
    """


def _client():
    """The shared Gemini client from llm.py.

    One client for generation and embeddings: the model is a per-call argument
    in this SDK. Raising through `llm.get_client` keeps the "no API key" message
    in one place; it is re-raised as EmbeddingError so callers here still catch
    a single type.
    """
    from app.rag import llm

    try:
        return llm.get_client()
    except llm.LLMError as exc:
        raise EmbeddingError(str(exc)) from exc


def _normalise(vector: list[float]) -> list[float]:
    """Scale a vector to unit length.

    Necessary, and easy to miss. gemini-embedding-001 returns unit-length
    vectors at its native 3072 dimensions, but when you ask for fewer it
    truncates the (Matryoshka) representation *without* re-normalising -- the
    768-dim vectors arrive with a norm around 0.58, not 1.0.

    vector_store treats these as unit vectors and uses a bare dot product as
    cosine similarity. If this function were removed, that assumption would be
    quietly wrong and every ranking would be subtly off, with nothing failing.
    """
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:  # defensive: an all-zero embedding would be a provider bug
        return list(vector)
    return (array / norm).tolist()


def _embed_batch(client, batch: list[str], task_type: str) -> list[list[float]]:
    from google.genai import types

    from app.rag import llm

    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=settings.EMBEDDING_DIM,
    )
    last_error: Exception | None = None

    for attempt in range(1, _RETRIES + 1):
        started = time.monotonic()
        try:
            response = client.models.embed_content(
                model=settings.EMBEDDING_MODEL, contents=batch, config=config
            )
        except Exception as exc:  # noqa: BLE001 -- vendor exceptions stop here
            last_error = exc
            message = str(exc)
            code = getattr(exc, "code", None)
            retryable = code in (429, 503, 504) or any(
                token in message
                for token in ("429", "RESOURCE_EXHAUSTED", "503", "504", "deadline")
            )
            llm._record(
                model=settings.EMBEDDING_MODEL, prompt_tokens=None, output_tokens=None,
                latency_ms=int((time.monotonic() - started) * 1000), ok=False, kind="embed",
            )
            if not retryable or attempt == _RETRIES:
                break
            # Linear, not exponential. The free tier's limit is per minute, so
            # the useful wait is short and fixed; exponential backoff would
            # mostly just make a recoverable batch take longer.
            delay = _BACKOFF_SECONDS * attempt
            logger.warning(
                "Embedding batch failed (attempt %d/%d), retrying in %.0fs: %s",
                attempt, _RETRIES, delay, message[:200],
            )
            time.sleep(delay)
            continue

        # Token counts are not reported per embedding call by the API; the
        # ledger records the call itself and the number of texts as "prompt".
        llm._record(
            model=settings.EMBEDDING_MODEL, prompt_tokens=len(batch), output_tokens=None,
            latency_ms=int((time.monotonic() - started) * 1000), ok=True, kind="embed",
        )
        return [_normalise(list(e.values)) for e in response.embeddings]

    raise EmbeddingError(f"Embedding request failed: {last_error}") from last_error


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    client = _client()
    batch_size = max(1, min(settings.EMBED_BATCH_SIZE, _MAX_BATCH))
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        vectors.extend(_embed_batch(client, texts[start : start + batch_size], task_type))

    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of passages for indexing.

    Batching matters: one request carrying 64 chunks is dramatically faster than
    64 requests, and costs far less of the per-minute request quota. It also
    bounds memory -- ingestion calls this per section, so only one batch of
    vectors is ever in flight rather than the whole document's matrix.
    """
    if not texts:
        return []
    return _embed(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embed a single search query.

    RETRIEVAL_QUERY, not RETRIEVAL_DOCUMENT -- see the module docstring. Indexing
    and searching with the same task type is the bug this signature exists to
    prevent.
    """
    return _embed([text], task_type="RETRIEVAL_QUERY")[0]


def embed_queries(texts: list[str]) -> list[list[float]]:
    """Embed several query variants in one request (see retriever query expansion)."""
    if not texts:
        return []
    return _embed(texts, task_type="RETRIEVAL_QUERY")
