# syntax=docker/dockerfile:1.7
# ── Stage 1: build the venv ─────────────────────────────────────────
# Anything that can produce a C-extension wheel (gcc, libpq-dev) lives
# here. The runtime stage never sees them, so the final image is
# smaller AND a compromised runtime can't compile arbitrary code.

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Copy ONLY the lockfile first so the dependency layer is cache-stable.
# Source changes don't bust the uv sync layer.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-cache

# Now copy the rest of the source. Anything that's package-data
# (prompts, static files, etc.) needs to be present at install time.
COPY . .


# ── Stage 2: runtime ────────────────────────────────────────────────
# python:3.12-slim, non-root, no compilers. Just the venv + sources.

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app"

# Runtime system deps only. asyncpg's wheel is self-contained, but
# libpq5 is kept as a cheap insurance policy in case a future
# dependency (psycopg, etc.) needs the system libpq.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Dedicated non-root user. UID 10001 to avoid collision with host
# users on bind-mount setups. No login shell, no home dir (the
# process doesn't need one).
RUN groupadd --system --gid 10001 app \
    && useradd  --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app \
    && chown -R app:app /app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

USER app

EXPOSE 8000

# Drop the `uv run` wrapper — the PATH above already points at the
# venv, so the venv's uvicorn is the one that runs. Faster startup,
# one fewer process to keep alive.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
