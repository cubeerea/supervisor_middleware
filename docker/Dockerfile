# One image for both the supervisor daemon and the executor/experiment runner.
# Build context is the repo root:  docker compose build   (or: docker build -f docker/Dockerfile .)
FROM python:3.12-slim

# Pinned uv binary (matches the host's uv; bump together with uv.lock regenerations).
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && mkdir -p /app/results \
    && chown -R app:app /app

WORKDIR /app
USER app

# 1) Dependencies only: this layer is cached until pyproject.toml / uv.lock change.
COPY --chown=app:app pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/app/.cache/uv,uid=10001,gid=10001 \
    UV_CACHE_DIR=/app/.cache/uv uv sync --frozen --no-dev --no-install-project

# 2) The project itself (hatchling package `supervisor_middleware`) plus the repo-root entry points.
COPY --chown=app:app . .
RUN --mount=type=cache,target=/app/.cache/uv,uid=10001,gid=10001 \
    UV_CACHE_DIR=/app/.cache/uv uv sync --frozen --no-dev

ENV PATH=/app/.venv/bin:$PATH

# Overridden per service in docker-compose.yml.
CMD ["python", "supervisor_daemon.py"]
