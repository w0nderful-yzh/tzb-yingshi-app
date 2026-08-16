from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.infrastructure.database.models import RiskEventModel
from app.infrastructure.database.session import Database
from app.modules.fraud.ports import RecentFraudContext


class RecentFraudRiskRepository:
    """PostgreSQL-backed RecentFraudRiskStore over the risk_events table.

    Only reads state, risk level, evidence kinds and occurrence time — never
    full transcripts or raw payloads — so the profile stays desensitized.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    async def load_recent_context(
        self,
        *,
        device_id: str,
        session_id: str,
        lookback_hours: int = 24,
    ) -> RecentFraudContext | None:
        since = datetime.now(UTC) - timedelta(hours=lookback_hours)
        async with self._database.session_factory() as session:
            rows = list(
                (
                    await session.scalars(
                        select(RiskEventModel)
                        .where(
                            RiskEventModel.external_device_id == device_id,
                            RiskEventModel.occurred_at >= since,
                        )
                        .order_by(RiskEventModel.occurred_at.desc())
                        .limit(20)
                    )
                ).all()
            )
        if not rows:
            return None
        kinds: list[str] = []
        for row in rows:
            chain = (row.evidence or {}).get("evidence_chain")
            if isinstance(chain, list):
                for item in chain[:3]:
                    if isinstance(item, dict) and item.get("kind"):
                        kinds.append(str(item["kind"]))
        last_risk_level = next(
            (row.risk_level for row in rows if row.risk_level),
            None,
        )
        return RecentFraudContext(
            device_id=device_id,
            session_id=session_id,
            recent_risk_events=len(rows),
            last_risk_level=last_risk_level,
            last_kinds=tuple(dict.fromkeys(kinds))[:6],
            last_occurred_at=rows[0].occurred_at,
        )
