import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert

from app.infrastructure.database.models import (
    DeviceModel,
    EventActionModel,
    RiskEventModel,
)
from app.infrastructure.database.session import Database
from app.infrastructure.realtime_events import RealtimeEventBroker, RealtimeRiskEvent
from app.modules.fall.ports import FallRiskEventWrite
from app.modules.fraud.latency import latency_stage
from app.modules.fraud.ports import FraudRiskEventWrite

SYSTEM_RETRACT_REASON = "final_transcript_retracted_preliminary"


class RiskEventRepository:
    def __init__(
        self,
        database: Database,
        *,
        realtime_broker: RealtimeEventBroker | None = None,
    ) -> None:
        self._database = database
        self._realtime_broker = realtime_broker

    async def upsert(self, event: FraudRiskEventWrite) -> None:
        state = str(event.evidence.get("state", "")).upper()
        evidence = dict(event.evidence)
        if event.verification_status is not None:
            evidence["verification_status"] = event.verification_status
            evidence["preliminary_source_event_id"] = event.source_event_id
        is_preliminary = event.verification_status == "PRELIMINARY"
        if is_preliminary:
            state = "S2_TRUST_BUILDING"
            evidence["state"] = state
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
        if is_preliminary:
            alert_level = "REMINDER"
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
            evidence=evidence,
            model_name=event.model_name,
            model_version=event.model_version,
        )
        returning_statement = statement.on_conflict_do_update(
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
        ).returning(
            RiskEventModel.id,
            RiskEventModel.elder_user_id,
            RiskEventModel.external_device_id,
            RiskEventModel.event_type,
            RiskEventModel.alert_level,
            RiskEventModel.status,
            RiskEventModel.summary,
            RiskEventModel.occurred_at,
        )
        with latency_stage("event_commit"):
            async with self._database.session_factory() as session, session.begin():
                persisted = (await session.execute(returning_statement)).one()
        if self._realtime_broker is None or persisted.elder_user_id is None:
            return
        title = str(event.evidence.get("state_label") or "发现新的风险事件")
        with latency_stage("broker_publish"):
            await self._realtime_broker.publish(
                RealtimeRiskEvent(
                    event_id=str(persisted.id),
                    elder_user_id=persisted.elder_user_id,
                    event_type=persisted.event_type,
                    level=persisted.alert_level,
                    title=title,
                    summary=persisted.summary,
                    device_id=persisted.external_device_id or "",
                    occurred_at=persisted.occurred_at,
                    status=persisted.status,
                    verification_status=event.verification_status,
                )
            )

    async def retract_preliminary(
        self,
        *,
        source_event_id: str,
        reason: str,
    ) -> None:
        """System-retract a PRELIMINARY event when FINAL falls back to S0/S1.

        Updates the same risk_events row to RESOLVED and records a system
        RESOLVE action (no actor) so the retraction is auditable. Broadcasts
        after the transaction commits.
        """
        retracted_event_id: str | None = None
        elder_user_id = None
        with latency_stage("event_commit"):
            async with self._database.session_factory() as session, session.begin():
                event = await session.scalar(
                    select(RiskEventModel).where(
                        RiskEventModel.source == "FRAUD_ENGINE",
                        RiskEventModel.source_event_id == source_event_id,
                    )
                )
                if event is None:
                    return
                retracted_event_id = str(event.id)
                elder_user_id = event.elder_user_id
                evidence = dict(event.evidence or {})
                evidence["verification_status"] = "RETRACTED"
                evidence["preliminary_source_event_id"] = source_event_id
                previous_status = event.status
                await session.execute(
                    update(RiskEventModel)
                    .where(RiskEventModel.id == event.id)
                    .values(
                        status="RESOLVED",
                        evidence=evidence,
                        version=RiskEventModel.version + 1,
                        updated_at=datetime.now(UTC),
                    )
                )
                session.add(
                    EventActionModel(
                        risk_event_id=event.id,
                        actor_user_id=None,
                        action_type="RESOLVE",
                        previous_status=previous_status,
                        new_status="RESOLVED",
                        action_metadata={
                            "reason": reason,
                            "source": "FRAUD_ENGINE",
                        },
                    )
                )
        if self._realtime_broker is None or elder_user_id is None or retracted_event_id is None:
            return
        with latency_stage("broker_publish"):
            await self._realtime_broker.publish(
                RealtimeRiskEvent(
                    event_id=retracted_event_id,
                    elder_user_id=elder_user_id,
                    event_type="FRAUD_SUSPECTED",
                    level="REMINDER",
                    title="预警已撤回",
                    summary=f"系统已撤回预警：{reason}",
                    device_id="",
                    occurred_at=datetime.now(UTC),
                    status="resolved",
                    verification_status="RETRACTED",
                )
            )

    async def upsert_fall_event(self, event: FallRiskEventWrite) -> None:
        """Persist a FALL_SUSPECTED episode event and broadcast it to the App."""
        alert_level = {
            "MEDIUM": "REMINDER",
            "HIGH": "WARNING",
            "CRITICAL": "EMERGENCY",
        }.get(event.risk_level, "REMINDER")
        elder_user_id = uuid.UUID(event.elder_user_id) if event.elder_user_id else None
        statement = (
            insert(RiskEventModel)
            .values(
                source="FALL_ENGINE",
                source_event_id=event.source_event_id,
                elder_user_id=elder_user_id,
                external_device_id=None,
                device_id=None,
                event_type="FALL_SUSPECTED",
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
            .on_conflict_do_nothing(
                index_elements=[RiskEventModel.source, RiskEventModel.source_event_id],
            )
            .returning(
                RiskEventModel.id,
                RiskEventModel.elder_user_id,
                RiskEventModel.external_device_id,
                RiskEventModel.event_type,
                RiskEventModel.alert_level,
                RiskEventModel.status,
                RiskEventModel.summary,
                RiskEventModel.occurred_at,
            )
        )
        async with self._database.session_factory() as session, session.begin():
            persisted = (await session.execute(statement)).first()
        if persisted is None:
            return
        if self._realtime_broker is None or persisted.elder_user_id is None:
            return
        await self._realtime_broker.publish(
            RealtimeRiskEvent(
                event_id=str(persisted.id),
                elder_user_id=persisted.elder_user_id,
                event_type=persisted.event_type,
                level=persisted.alert_level,
                title=str(event.evidence.get("state_label") or "疑似跌倒"),
                summary=event.summary,
                device_id=persisted.external_device_id or "",
                occurred_at=persisted.occurred_at,
                status=persisted.status,
                verification_status=None,
            )
        )
