"""
Central configuration.

Everything the app needs to know about its environment lives here, loaded from
a .env file. Nothing else in the codebase reads os.environ directly -- that way
there is exactly one place to look when something is misconfigured.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# The value shipped in .env.example and used as the default below. Named so the
# production check further down can recognise it rather than repeating a string
# literal that could drift out of sync.
DEV_SECRET_KEY = "dev-only-insecure-key"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "PaperLens"
    DEBUG: bool = True

    # --- Auth ---
    SECRET_KEY: str = DEV_SECRET_KEY
    ALGORITHM: str = "HS256"
    # Short-lived, and held only in the browser's memory. A stolen access
    # token is useful for minutes, not a day. The session itself lives in the
    # refresh token below.
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    # The refresh token is an httpOnly cookie: JavaScript cannot read it, so
    # an injected script cannot steal the session. It is rotated on every use.
    REFRESH_TOKEN_EXPIRE_DAYS: int = 14

    # --- Storage ---
    DATABASE_URL: str = "sqlite:///./storage/paperlens.db"
    UPLOAD_DIR: str = "./storage/uploads"

    # --- LLM ---
    GEMINI_API_KEY: str = ""
    # Pinned deliberately, not an alias like "gemini-flash-latest": evaluation
    # numbers are only meaningful if the model that produced them is
    # identifiable. This default must match the Dockerfile and .env.example --
    # they disagreed before, so the deployed app was quietly running a different
    # model than the code default claimed.
    GEMINI_MODEL: str = "gemini-3.1-flash-lite"

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    # 768 rather than the model's native 3072. Matryoshka training means the
    # leading dimensions carry most of the signal, so truncating costs little
    # accuracy while making each stored vector 4x smaller. Shortened vectors come
    # back un-normalised -- app/rag/embeddings.py handles that. Changing this
    # invalidates every stored embedding; vector_store raises rather than
    # silently ranking against mismatched dimensions.
    EMBEDDING_DIM: int = 768

    # --- Ingestion ---
    MAX_UPLOAD_MB: int = 25
    # The size cap bounds bytes; this bounds pages. They are different
    # resources: a 25 MB PDF can be 2,000 pages, which is 2,000 extractions and
    # a few thousand embedding calls against a free quota, on the one worker
    # thread everyone shares. Checked before ingestion starts.
    MAX_PAGES: int = 400
    # Wall-clock budget for one ingest, checked between pages. pypdf cannot be
    # interrupted mid-page from another thread, so this is a cooperative
    # deadline rather than a hard timeout -- it bounds a pathological file at
    # "one page past the limit", not at zero.
    INGEST_TIMEOUT_SECONDS: int = 600
    # Transcribe scanned PDFs (no text layer) with the model, one call per
    # page. Off by default because it spends quota; see app/rag/ocr.py.
    SCANNED_PDF_OCR: bool = False
    OCR_MAX_PAGES: int = 30
    # ~1000 tokens per chunk, ~120 overlapping. Chunks follow section and
    # paragraph boundaries; these are budgets, not hard cuts. See rag/chunker.py
    # for why this is 4x the previous character-based size.
    CHUNK_TARGET_TOKENS: int = 1000
    CHUNK_OVERLAP_TOKENS: int = 120
    # Chunks per embedding API request. The provider rejects more than 100.
    EMBED_BATCH_SIZE: int = 64

    # --- Retrieval ---
    # 10, not 5. The single largest cause of thin answers was that the model saw
    # about 1,100 tokens of the paper. Ten ~1000-token chunks is ~10k tokens of
    # context, which a modern model handles comfortably.
    TOP_K: int = 10
    # Query variants generated per question before retrieval. 3 total (the
    # original plus 2 rewrites): enough to catch a different phrasing, few
    # enough that the extra latency is one cheap LLM call.
    QUERY_VARIANTS: int = 3
    # Chat turns replayed verbatim before older ones are rolled into a summary.
    CHAT_RECENT_TURNS: int = 6

    # --- Rate limiting (app/ratelimit.py) ---
    # In-memory per-user token buckets. Correct because there is exactly one
    # process (WEB_CONCURRENCY=1); they reset on restart, which is fine for
    # a demo. Set RATE_LIMITING=false to disable in development.
    RATE_LIMITING: bool = True
    RATE_CHAT_PER_MINUTE: int = 12
    RATE_INGEST_PER_HOUR: int = 6
    RATE_LOGIN_PER_MINUTE: int = 10
    # Model calls per user per rolling day, counted from the ledger. 0 = off.
    DAILY_CALLS_PER_USER: int = 300

    # Empty by default because the shipped deployment serves the frontend and
    # the API from the same origin, where CORS does not apply at all. The
    # setting stays so a split deployment (separate frontend host) can opt back
    # in by listing its origins.
    CORS_ORIGINS: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    def model_post_init(self, _context) -> None:
        """Refuse to start in production with a publicly known signing key.

        SECRET_KEY signs every JWT. The default below is committed to a public
        repo, so anyone could mint a token for any user with it. Booting anyway
        would produce an app that looks fine and is silently wide open, which is
        strictly worse than not booting -- so this raises instead of warning.

        Gated on DEBUG rather than on some separate ENV flag: the app already
        uses DEBUG to mean "this is a development run", and adding a second
        notion of environment would just create a way for the two to disagree.
        """
        if self.DEBUG:
            return
        if not self.SECRET_KEY or self.SECRET_KEY == DEV_SECRET_KEY:
            raise RuntimeError(
                "SECRET_KEY is unset or still the development placeholder while "
                "DEBUG=false. Set it to a random value, e.g.\n"
                "  python -c \"import secrets; print(secrets.token_hex(32))\"\n"
                "On Render, add it under the service's Environment tab."
            )


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is parsed once, not on every import."""
    return Settings()


settings = get_settings()
