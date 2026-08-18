#!/bin/sh
set -eu

# Keep database preparation in the existing backend startup path.
uv run --no-sync alembic upgrade head

if [ "${APP_SEED_DEMO:-true}" = "true" ]; then
    uv run --no-sync python -m app.scripts.seed_demo
fi

exec "$@"
