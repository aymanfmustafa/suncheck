# Suncheck on any container host (built for Google Cloud Run). The server is standard library only; the image carries
# the package and its locked dependencies. The key comes in as the META_API_KEY environment variable, never in the image.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# opencv-python-headless links against glib
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencies first, so a code change does not reinstall them
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY suncheck ./suncheck
RUN uv sync --frozen --no-dev

# results go to /app/out/web (ephemeral on Cloud Run: they last as long as the instance)
EXPOSE 8080
CMD ["sh", "-c", "exec /app/.venv/bin/python -u -m suncheck web --port ${PORT:-8080}"]
