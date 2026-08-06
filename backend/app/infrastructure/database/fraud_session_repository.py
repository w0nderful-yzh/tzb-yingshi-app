from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert

from app.infrastructure.database.models import FraudSessionModel
from app.infrastructure.database.session import Database
from app.modules.fraud.ports import FraudSessionRecord


class FraudSessionRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def load(self, *, device_id: str, session_id: str) -> FraudSessionRecord | None:
        async with self._database.session_factory() as session:
            model = await session.scalar(
                select(FraudSessionModel)
                .where(FraudSessionModel.external_device_id == device_id)
                .where(FraudSessionModel.session_id == session_id)
            )
        if model is None:
            return None
        return FraudSessionRecord(
            session_id=model.session_id,
            device_id=model.external_device_id,
            elder_alone=model.elder_alone,
            status=model.status,
            started_at=model.started_at,
            last_activity_at=model.last_activity_at,
            ended_at=model.ended_at,
            speech_events=dict(model.speech_events),
            llm_evidence=dict(model.llm_evidence),
            last_llm_review_id=model.last_llm_review_id,
        )

    async def upsert(self, record: FraudSessionRecord) -> None:
        statement = insert(FraudSessionModel).values(
            session_id=record.session_id,
            external_device_id=record.device_id,
            elder_alone=record.elder_alone,
            status=record.status,
            started_at=record.started_at,
            last_activity_at=record.last_activity_at,
            ended_at=record.ended_at,
            speech_events=record.speech_events,
            llm_evidence=record.llm_evidence,
            last_llm_review_id=record.last_llm_review_id,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[
                FraudSessionModel.external_device_id,
                FraudSessionModel.session_id,
            ],
            set_={
                "elder_alone": statement.excluded.elder_alone,
                "status": statement.excluded.status,
                "started_at": statement.excluded.started_at,
                "last_activity_at": statement.excluded.last_activity_at,
                "ended_at": statement.excluded.ended_at,
                "speech_events": statement.excluded.speech_events,
                "llm_evidence": statement.excluded.llm_evidence,
                "last_llm_review_id": statement.excluded.last_llm_review_id,
                "version": FraudSessionModel.version + 1,
                "updated_at": statement.excluded.updated_at,
            },
        )
        async with self._database.session_factory() as session, session.begin():
            await session.execute(statement)

    async def close_other_active(
        self,
        *,
        device_id: str,
        active_session_id: str,
        ended_at: datetime,
    ) -> None:
        statement = (
            update(FraudSessionModel)
            .where(FraudSessionModel.external_device_id == device_id)
            .where(FraudSessionModel.status == "ACTIVE")
            .where(FraudSessionModel.session_id != active_session_id)
            .values(
                status="CLOSED",
                ended_at=func.greatest(FraudSessionModel.started_at, ended_at),
            )
        )
        async with self._database.session_factory() as session, session.begin():
            await session.execute(statement)
