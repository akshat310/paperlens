"""
The only file in the project that knows Gemini exists (embeddings.py shares the
client). Everything else calls `generate(...)` or `stream(...)` and gets text.

That boundary is deliberate: swapping providers means rewriting this file and
nothing else. It is also what makes the rest of the codebase testable -- tests
monkeypatch `generate` and never touch the network or need an API key.

SDK: `google-genai`, not `google-generativeai`.
-------------------------------------------------
The older SDK is deprecated, and it spoke gRPC. Measured in-process (on top of
the pydantic/httpx/anyio that FastAPI already loads), the old SDK added ~43 MB
resident and the new one ~30 MB, with no gRPC core in the process at all. On a
512 MB instance where ~90% of peak is libraries sitting in memory, a
dependency swap that removes 13 MB is the kind of change that actually moves
the number. The eval numbers in the README were re-checked after the swap.

Every call is recorded.
-----------------------
`_record` writes one row per call -- purpose, model, tokens in and out,
latency, success -- through app/services/ledger.py. That is the entire
observability story, and it is enough: "what did this paper cost?" is a SQL
query. The purpose and paper id come from a context variable set by the caller
(`with llm.calling(paper_id, "map"):`), so the call sites stay one line.
"""

import contextlib
import contextvars
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Any failure talking to the language model.

    Callers catch this one exception rather than vendor-specific error types,
    so provider details never leak past this module.
    """


class LLMRateLimitError(LLMError):
    """The provider refused because of quota or rate limiting.

    Separate from LLMError because it means something different to the caller:
    the request was well-formed and would succeed later. It maps to HTTP 429,
    not 503, so the UI can say "wait a moment" rather than "something broke".
    """


# ---------------------------------------------------------------------------
# Call context, for the ledger
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CallContext:
    paper_id: str | None = None
    purpose: str = "other"


_context: contextvars.ContextVar[CallContext] = contextvars.ContextVar(
    "llm_call_context", default=CallContext()
)


@contextlib.contextmanager
def calling(paper_id: str | None, purpose: str):
    """Label every LLM/embedding call made inside the block.

    A context variable rather than extra parameters on `generate`: the calls
    happen several layers below the code that knows which paper and which
    stage they belong to, and threading two arguments through retrieval,
    prompt-building and chat would touch every signature for a bookkeeping
    concern. Context variables are per-thread, so the worker thread and a
    request thread never see each other's labels.
    """
    previous = _context.get()
    token = _context.set(CallContext(paper_id=paper_id, purpose=purpose))
    try:
        yield
    finally:
        try:
            _context.reset(token)
        except ValueError:
            # A generator consumed by StreamingResponse has each `next()` run
            # in a fresh context, so the token from `__enter__` is not valid
            # at `__exit__`. Restoring the previous value directly is the
            # same outcome without the bookkeeping check.
            _context.set(previous)


def _record(
    *,
    model: str,
    prompt_tokens: int | None,
    output_tokens: int | None,
    latency_ms: int,
    ok: bool,
    kind: str = "generate",
) -> None:
    # Imported here: the ledger needs the database, and this module must stay
    # importable by scripts (the memory probe, the eval) with nothing else.
    from app.services import ledger

    ctx = _context.get()
    try:
        ledger.record(
            paper_id=ctx.paper_id,
            purpose=ctx.purpose,
            kind=kind,
            model=model,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            ok=ok,
        )
    except Exception:  # noqa: BLE001 -- bookkeeping must never fail a call
        logger.debug("Ledger write failed", exc_info=True)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_client():
    """Build and cache the Gemini client.

    Lazy, so importing this module without an API key still works -- auth and
    upload need no LLM, and the memory probe imports the app without one.
    One client is enough for every model: the model is a per-call argument in
    this SDK, which is also what lets the eval harness grade with a different
    model than the one that answered.
    """
    if not settings.GEMINI_API_KEY:
        raise LLMError(
            "GEMINI_API_KEY is not set. Add it to backend/.env to enable chat and analysis."
        )

    from google import genai  # imported lazily: the SDK is the heaviest import we have

    logger.info("Gemini client ready")
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def _translate(exc: Exception, model_name: str) -> LLMError:
    """Map a vendor exception onto our two error types.

    Shared by `generate` and `stream` so both classify failures identically --
    the streaming path used to be the one place a 429 could surface as a
    generic 500.
    """
    message = str(exc)
    code = getattr(exc, "code", None)

    # Quota errors need different advice depending on why: "limit: 0" means
    # this model has no free tier at all (waiting will never help), anything
    # else 429 is ordinary rate limiting and will pass.
    if code == 429 or "429" in message or "RESOURCE_EXHAUSTED" in message:
        logger.warning("Gemini quota hit for model %s", model_name)
        if "limit: 0" in message:
            return LLMRateLimitError(
                f"The model '{model_name}' has no free-tier quota on this API key. "
                "Set GEMINI_MODEL to a model your key can use."
            )
        return LLMRateLimitError(
            "Rate limit reached on the free tier. Wait a few seconds and try again."
        )

    if code in (503, 504) or "UNAVAILABLE" in message:
        # Expected under load and already retried; a full traceback per
        # occurrence buries real errors in the log.
        logger.warning("Gemini unavailable after retries for model %s", model_name)
    else:
        logger.exception("Gemini request failed")
    return LLMError(f"Language model request failed: {message}")


def _usage(response) -> tuple[int | None, int | None]:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None, None
    return (
        getattr(meta, "prompt_token_count", None),
        getattr(meta, "candidates_token_count", None),
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    *,
    json_mode: bool = False,
    temperature: float = 0.1,
    max_output_tokens: int = 8192,
    model: str | None = None,
) -> str:
    """Send one prompt, return the model's text.

    temperature=0.1 by default: this is an extraction task, not a creative one.
    We want the model to repeat what the paper says as faithfully as possible,
    and low temperature makes the same question give the same answer -- which
    also makes the evaluation numbers reproducible.

    json_mode asks Gemini to emit syntactically valid JSON. It guarantees the
    *shape* is parseable, not that the fields are right -- Pydantic still
    validates the result at the call site.

    max_output_tokens is set explicitly. Current Gemini models spend tokens on
    internal "thinking" before answering, and if the budget runs out mid-
    response the API returns partial text with finish_reason=MAX_TOKENS. For
    prose that is a clipped sentence; for JSON it is an unparseable fragment.
    Truncation is treated as an explicit error below.

    model overrides settings.GEMINI_MODEL for one call. The app never passes
    it; the evaluation harness does, to grade with a different model.
    """
    from google.genai import types

    model_name = model or settings.GEMINI_MODEL
    client = get_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json" if json_mode else None,
        # We pass no tools, so there is nothing to auto-call; disabling it
        # silences the SDK's warning about it on every request.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    started = time.monotonic()
    prompt_tokens = output_tokens = None
    try:
        response = _generate_with_retry(client, model_name, prompt, config)
        prompt_tokens, output_tokens = _usage(response)

        candidate = response.candidates[0] if response.candidates else None
        reason = getattr(candidate, "finish_reason", None)
        if getattr(reason, "name", str(reason)) == "MAX_TOKENS":
            raise LLMError(
                "The model's response was cut off before it finished. "
                "Try again, or reduce the amount of context sent."
            )

        text = response.text
    except LLMError:
        _record(model=model_name, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                latency_ms=int((time.monotonic() - started) * 1000), ok=False)
        raise
    except Exception as exc:  # noqa: BLE001 -- deliberately collapsing vendor errors
        _record(model=model_name, prompt_tokens=None, output_tokens=None,
                latency_ms=int((time.monotonic() - started) * 1000), ok=False)
        raise _translate(exc, model_name) from exc

    _record(model=model_name, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            latency_ms=int((time.monotonic() - started) * 1000), ok=True)

    if not text or not text.strip():
        # A blocked or empty response is a failure, not an answer. Returning ""
        # here would surface as a blank chat bubble with no explanation.
        raise LLMError("Language model returned an empty response.")

    return text.strip()


# A 503 from the provider means "high demand, try again shortly" -- the request
# was fine. One or two short waits convert most of them into a success, which
# is far better than surfacing an error for a map-stage call that would then
# leave a hole in the report. 429 is NOT retried here: quota errors carry
# advice the user needs to see, and the eval harness backs off on its own.
_TRANSIENT_RETRIES = 3
_TRANSIENT_WAIT_SECONDS = 4.0  # 4, 8, 12 s: 'high demand' spikes usually clear within a minute


def _is_transient(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    return code in (503, 504) or "UNAVAILABLE" in str(exc)


def _generate_with_retry(client, model_name: str, prompt: str, config):
    for attempt in range(1, _TRANSIENT_RETRIES + 2):
        try:
            return client.models.generate_content(
                model=model_name, contents=prompt, config=config
            )
        except Exception as exc:  # noqa: BLE001 -- only transient ones are retried
            if not _is_transient(exc) or attempt > _TRANSIENT_RETRIES:
                raise
            logger.warning(
                "Gemini unavailable (attempt %d/%d), retrying in %.0fs",
                attempt, _TRANSIENT_RETRIES + 1, _TRANSIENT_WAIT_SECONDS * attempt,
            )
            time.sleep(_TRANSIENT_WAIT_SECONDS * attempt)
    raise AssertionError("unreachable")


def stream(
    prompt: str,
    *,
    temperature: float = 0.2,
    max_output_tokens: int = 4000,
    model: str | None = None,
) -> Iterator[str]:
    """Yield the model's answer in pieces as it is produced.

    Streaming exists for one reason: a substantive answer runs to several
    paragraphs, and waiting ten seconds at a spinner for it feels broken in a
    way that watching it arrive does not. The total time is the same.

    A *sync* generator over the SDK's sync streaming call. The route that
    consumes it is a normal `def` endpoint, so FastAPI runs it on the
    threadpool and it never blocks the event loop.

    Errors raised part-way through are the awkward case: some text has already
    reached the client. The caller handles that; here we translate and
    propagate. Token usage arrives on the final chunk, so it is recorded when
    the stream is exhausted.
    """
    from google.genai import types

    model_name = model or settings.GEMINI_MODEL
    client = get_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    started = time.monotonic()
    prompt_tokens = output_tokens = None
    try:
        for piece in client.models.generate_content_stream(
            model=model_name, contents=prompt, config=config
        ):
            usage = _usage(piece)
            if usage[0] is not None:
                prompt_tokens, output_tokens = usage
            # A chunk can legitimately carry no text (a safety annotation, or
            # a thinking step), so `.text` can be None.
            text = getattr(piece, "text", None)
            if text:
                yield text
    except (LLMError, LLMRateLimitError):
        raise
    except Exception as exc:  # noqa: BLE001 -- deliberately collapsing vendor errors
        _record(model=model_name, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                latency_ms=int((time.monotonic() - started) * 1000), ok=False, kind="stream")
        raise _translate(exc, model_name) from exc

    _record(model=model_name, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            latency_ms=int((time.monotonic() - started) * 1000), ok=True, kind="stream")
