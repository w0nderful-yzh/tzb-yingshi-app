#!/bin/sh
set -eu

uv run alembic upgrade head

if [ "${APP_SEED_DEMO:-true}" = "true" ]; then
    uv run python -m app.scripts.seed_demo
fi

exec "$@"
