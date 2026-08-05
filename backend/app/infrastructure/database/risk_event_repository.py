from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from app.infrastructure.database.models import DeviceModel, RiskEventModel
from app.infrastructure.database.session import Database
from app.modules.fraud.ports import FraudRiskEventWrite


class RiskEventRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def upsert(self, event: FraudRiskEventWrite) -> None:
        state = str(event.evidence.get("state", "")).upper()
        alert_level = {
            "S1": "REMINDER",
            "S2": "REMINDER",
            "S3": "WARNING",
            "S4": "WARNING",
            "S5": "EMERGENCY",
        }.get(
            state,
            {
                "LOW": "REMINDER",
                "MEDIUM": "REMINDER",
                "HIGH": "WARNING",
                "CRITICAL": "EMERGENCY",
            }.get(event.risk_level, "REMINDER"),
        )
        device_id = (
            select(DeviceModel.id)
            .where(DeviceModel.external_device_id == event.external_device_id)
            .scalar_subquery()
        )
        elder_user_id = (
            select(DeviceModel.elder_user_id)
            .where(DeviceModel.external_device_id == event.external_device_id)
            .scalar_subquery()
        )
        statement = insert(RiskEventModel).values(
            source="FRAUD_ENGINE",
            source_event_id=event.source_event_id,
            elder_user_id=elder_user_id,
            external_device_id=event.external_device_id,
            device_id=device_id,
            event_type="FRAUD_SUSPECTED",
            risk_level=event.risk_level,
            alert_level=alert_level,
            status="OPEN",
            confidence=event.confidence,
            summary=event.summary,
            occurred_at=event.occurred_at,
            received_at=event.received_at,
            evidence=event.evidence,
            model_name=event.model_name,
            model_version=event.model_version,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[RiskEventModel.source, RiskEventModel.source_event_id],
            set_={
                "elder_user_id": func.coalesce(
                    statement.excluded.elder_user_id,
                    RiskEventModel.elder_user_id,
                ),
                "device_id": func.coalesce(
                    statement.excluded.device_id,
                    RiskEventModel.device_id,
                ),
                "risk_level": statement.excluded.risk_level,
                "alert_level": statement.excluded.alert_level,
                "confidence": statement.excluded.confidence,
                "summary": statement.excluded.summary,
                "occurred_at": statement.excluded.occurred_at,
                "received_at": statement.excluded.received_at,
                "evidence": statement.excluded.evidence,
                "model_name": statement.excluded.model_name,
                "model_version": statement.excluded.model_version,
                "version": RiskEventModel.version + 1,
                "updated_at": statement.excluded.updated_at,
            },
        )
        async with self._database.session_factory() as session, session.begin():
            await session.execute(statement)
