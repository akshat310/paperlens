# syntax=docker/dockerfile:1

# Single-container build.
#
# Node compiles the frontend to static files and FastAPI serves both those files
# and /api from one process. Same origin, so no CORS and no proxy in between --
# one service to deploy, one thing to explain.

# ---------- Stage 1: build the frontend ----------
FROM node:20-slim AS build

WORKDIR /app

# Copied before the source so the dependency layer is cached independently of
# code changes. `npm ci` (not `npm install`) installs exactly what the lockfile
# pins, so the image is reproducible.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

# Deliberately left empty. src/api/client.ts falls back to a relative "/api"
# when this is unset, which is exactly what a same-origin deployment needs --
# an absolute URL here would reintroduce CORS. The arg exists so a split
# deployment (frontend on a separate host) can override it at build time.
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# ---------- Stage 2: runtime ----------
# Python 3.12 to match the development environment exactly. "slim" rather than
# "alpine": alpine uses musl instead of glibc, which forces several packages
# (numpy above all) to compile from source instead of using prebuilt wheels.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Run as a non-root user. Creating it up front means everything written below
# lands in a directory that user already owns -- doing it as root and switching
# afterwards leaves root-owned files that fail to open at runtime.
RUN useradd --create-home --uid 1000 user

# HOME must point at a directory this user owns. A read-only home is a common
# cause of a container that builds fine and then fails on first request.
ENV HOME=/home/user \
    XDG_CACHE_HOME=/home/user/.cache

# All mutable state lives under the one directory this user owns, rather than
# under /app. On the free tier this is ephemeral and resets on restart, which is
# documented rather than hidden -- the app recreates these directories in its
# lifespan hook on boot. See the storage note in render.yaml.
ENV DATABASE_URL=sqlite:////home/user/storage/paperlens.db \
    UPLOAD_DIR=/home/user/storage/uploads

# DEBUG=false is what arms the SECRET_KEY check in app/config.py. Without this
# the service would happily boot signing JWTs with the placeholder key committed
# to a public repo.
ENV DEBUG=false

# Empty because the frontend is served from this same origin. Explicit rather
# than relying on the default, so a stray CORS_ORIGINS in the build environment
# cannot leak through.
ENV CORS_ORIGINS=""

# Pinned to match config.py and .env.example. These three disagreed before, so
# the deployed service was quietly running a different model than the code
# default claimed -- which makes any measured number meaningless.
ENV GEMINI_MODEL=gemini-3.1-flash-lite

WORKDIR /app

COPY backend/requirements.txt .
# --no-compile skips writing .pyc files at install time. They would be written
# again on first import anyway, and skipping them keeps the image smaller.
RUN pip install --no-cache-dir --no-compile -r requirements.txt

RUN mkdir -p /home/user/storage/uploads /home/user/.cache \
    && chown -R user:user /home/user /app

USER user

# No ML model is baked into the image. Embeddings come from a hosted API -- that
# is what lets this run in 512MB of RAM at all. See app/rag/embeddings.py.

COPY --chown=user:user backend/ .

# app/main.py looks for backend/static and mounts it if present, so this is the
# path that turns on frontend serving. Nothing else changes between the local
# (Vite) and deployed (baked-in) setups.
COPY --from=build --chown=user:user /app/dist ./static

# Render assigns a port at runtime and injects it as $PORT; 7860 is only the
# local default so `docker run -p 7860:7860` in the README works unchanged.
ENV PORT=7860
EXPOSE 7860

# --- Concurrency ------------------------------------------------------------
#
# Both defaults are overridden by render.yaml; they are set here so a plain
# `docker run` gets the same bounded behaviour as the deployment.
#
# One worker: each extra one duplicates ~140MB of interpreter, FastAPI,
# SQLAlchemy, NumPy and gRPC client for a workload that is almost entirely
# waiting on a network call.
#
# Two threads: Starlette's threadpool defaults to 40, and every `def` endpoint
# in this app runs there. Forty concurrent requests each holding an embedding
# batch is the realistic path to an OOM kill. Two keeps a background ingest
# running alongside a chat request and bounds the worst case at two working sets.
ENV WEB_CONCURRENCY=1 \
    THREAD_LIMIT=2

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; \
    urllib.request.urlopen('http://localhost:' + os.environ['PORT'] + '/api/health').read()"

# `sh -c` so ${PORT} and the concurrency variables are expanded by the shell at
# start-up. The exec form would pass the literal string "${PORT}" to uvicorn,
# which fails to parse as an int.
#
# No --reload in production: it watches the filesystem and spawns a reloader
# process, both pure overhead outside development.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --limit-concurrency 32 --timeout-keep-alive 65"]
