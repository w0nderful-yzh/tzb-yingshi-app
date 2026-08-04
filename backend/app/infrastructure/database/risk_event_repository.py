from sqlalchemy.dialects.postgresql import insert

from app.infrastructure.database.models import RiskEventModel
from app.infrastructure.database.session import Database
from app.modules.fraud.ports import FraudRiskEventWrite


class RiskEventRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def upsert(self, event: FraudRiskEventWrite) -> None:
        statement = insert(RiskEventModel).values(
            source_event_id=event.source_event_id,
            external_device_id=event.external_device_id,
            event_type="FRAUD_SUSPECTED",
            risk_level=event.risk_level,
            status="PENDING",
            confidence=event.confidence,
            summary=event.summary,
            occurred_at=event.occurred_at,
            received_at=event.received_at,
            evidence=event.evidence,
            model_name=event.model_name,
            model_version=event.model_version,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[RiskEventModel.source_event_id],
            set_={
                "risk_level": statement.excluded.risk_level,
                "confidence": statement.excluded.confidence,
                "summary": statement.excluded.summary,
                "occurred_at": statement.excluded.occurred_at,
                "received_at": statement.excluded.received_at,
                "evidence": statement.excluded.evidence,
                "model_name": statement.excluded.model_name,
                "model_version": statement.excluded.model_version,
                "updated_at": statement.excluded.updated_at,
            },
        )
        async with self._database.session_factory() as session, session.begin():
            await session.execute(statement)
