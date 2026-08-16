"""Export FALSE_ALARM risk events as a desensitized review dataset.

Never appends to train.jsonl automatically. This script produces a human
review artifact for the evaluation/private/ directory (not committed to Git),
where a reviewer confirms every evidence label before any data is merged into
the versioned training set. FALSE_ALARM marks the whole event as a negative;
it does not mean every evidence label in it is a negative sample.

Usage (backend directory, database must be enabled):

    uv run python -m app.scripts.export_fraud_feedback \
        --output backend/evaluation/private/false_alarm_feedback_2026-08-14.jsonl

Removed by default: user/device/family identifiers, phone numbers, ID cards,
bank card numbers, verification codes, image URLs, and any field unrelated to
model training. Event IDs are replaced with irreversible digests.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.core.config import get_settings
from app.infrastructure.database.models import EventActionModel, RiskEventModel
from app.infrastructure.database.session import Database

_PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_ID_CARD = re.compile(r"(?<!\d)[1-9]\d{16}[\dX](?!\d)")
_BANK_CARD = re.compile(r"(?<!\d)(?:6[0-9]{14,18}|[4-9]\d{14,18})(?!\d)")
_VERIFY_CODE = re.compile(r"(?:验证码|短信(?:里的)?密码)\s*[是为：:]\s*(\d{4,8})")
_URL = re.compile(r"https?://[^\s\"'，。；！？]+")


def mask_text(text: str) -> str:
    masked = str(text or "")
    masked = _URL.sub("[URL]", masked)
    masked = _PHONE.sub("138****0000", masked)
    masked = _ID_CARD.sub("110101********001X", masked)
    masked = _BANK_CARD.sub("6222************1234", masked)
    masked = _VERIFY_CODE.sub(lambda match: match.group(0)[:-6] + "******", masked)
    return masked


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _mask_evidence(chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in chain if isinstance(chain, list) else []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "kind": str(item.get("kind", "")),
                "stage": str(item.get("stage", "")),
                "strength": str(item.get("strength", "")),
                "text": mask_text(item.get("text", "")),
                "reason": mask_text(item.get("reason", "")),
                "used_for_transition": bool(item.get("used_for_transition", False)),
                "decayed_confidence": item.get("decayed_confidence"),
            }
        )
    return rows


async def export(output: Path, database: Database) -> int:
    async with database.session_factory() as session:
        rows = (
            await session.execute(
                select(RiskEventModel, EventActionModel)
                .join(
                    EventActionModel,
                    EventActionModel.risk_event_id == RiskEventModel.id,
                )
                .where(
                    EventActionModel.action_type == "FALSE_ALARM",
                    RiskEventModel.event_type == "FRAUD_SUSPECTED",
                )
                .order_by(RiskEventModel.occurred_at)
            )
        ).all()

    output.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    with output.open("w", encoding="utf-8") as target:
        for event, _action in rows:
            evidence = event.evidence or {}
            record = {
                "event_digest": _digest(str(event.id)),
                "occurred_at": event.occurred_at.isoformat(),
                "risk_level": event.risk_level,
                "alert_level": event.alert_level,
                "summary": mask_text(event.summary),
                "state": str(evidence.get("state", "")),
                "state_index": evidence.get("state_index"),
                "evidence": _mask_evidence(evidence.get("evidence_chain") or []),
                "reviewed": False,
                "review_notes": "",
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1
    return exported


def main() -> int:
    parser = argparse.ArgumentParser(description="Export desensitized FALSE_ALARM feedback")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/evaluation/private/false_alarm_feedback.jsonl"),
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.database_enabled:
        print("database is disabled; set APP_DATABASE_ENABLED=true", file=sys.stderr)
        return 2

    async def _run() -> int:
        database = Database(settings.database_url.get_secret_value())
        try:
            return await export(args.output, database)
        finally:
            await database.dispose()

    count = asyncio.run(_run())
    print(f"exported {count} desensitized FALSE_ALARM events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
