FROM ghcr.io/astral-sh/uv:0.11.32-python3.12-trixie-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

WORKDIR /app/backend

COPY README.md /app/README.md
COPY backend/pyproject.toml backend/uv.lock ./

ARG INSTALL_SENSEVOICE=false
RUN if [ "$INSTALL_SENSEVOICE" = "true" ]; then \
        uv sync --frozen --no-dev --extra sensevoice; \
    else \
        uv sync --frozen --no-dev; \
    fi

# The bundled static ARM64 FFmpeg cannot resolve DNS in Docker Desktop.
# Use Debian's dynamically linked FFmpeg for live EZVIZ media ingestion.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY backend/app ./app
COPY backend/alembic ./alembic
COPY backend/alembic.ini ./alembic.ini
COPY docker/backend-entrypoint.sh /usr/local/bin/backend-entrypoint

RUN chmod +x /usr/local/bin/backend-entrypoint \
    && mkdir -p /app/backend/models /app/backend/storage/ys7/raw

EXPOSE 8000

ENTRYPOINT ["backend-entrypoint"]
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
