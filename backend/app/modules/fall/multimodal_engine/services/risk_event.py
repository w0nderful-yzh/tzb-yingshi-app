from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.database.models import RiskEvent
from app.modules.fall.multimodal_engine.repositories import MonitoringRepository, RiskEventRepository
from app.modules.fall.multimodal_engine.schemas.risk_event import (
    RiskEventInput,
    RiskLevel,
    RiskModule,
    RiskEventStatus,
    RiskEventStatusUpdate,
)


class DuplicateRiskEventError(ValueError):
    pass


class RiskEventNotFoundError(LookupError):
    pass


class RiskEventSessionNotFoundError(LookupError):
    pass


def _to_utc_naive(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class RiskEventService:
    def __init__(
        self,
        session: Session,
        repository: RiskEventRepository | None = None,
        monitoring_repository: MonitoringRepository | None = None,
    ) -> None:
        self.session = session
        self.repository = repository or RiskEventRepository(session)
        self.monitoring_repository = monitoring_repository or MonitoringRepository(session)

    def save_event(self, payload: RiskEventInput) -> RiskEvent:
        if self.repository.exists_by_event_id(payload.event_id):
            raise DuplicateRiskEventError(
                f"Risk event already exists: {payload.event_id}"
            )

        if self.monitoring_repository.get_by_id(payload.session_id) is None:
            raise RiskEventSessionNotFoundError(
                f"Monitoring session does not exist: {payload.session_id}"
            )

        risk_event = RiskEvent(
            event_id=payload.event_id,
            session_id=payload.session_id,
            device_id=payload.device_id,
            module=payload.module.value,
            event_type=payload.event_type,
            risk_score=Decimal(str(payload.risk_score)).quantize(Decimal("0.0001")),
            risk_level=payload.risk_level.value,
            summary=payload.summary,
            evidence_json=[item.model_dump(mode="json") for item in payload.evidence],
            recommended_action=payload.recommended_action,
            snapshot_path=payload.snapshot_path,
            clip_path=payload.clip_path,
            model_version=payload.model_version,
            source=payload.source.value,
            status=RiskEventStatus.PENDING.value,
            occurred_at=_to_utc_naive(payload.occurred_at),
            handled_at=None,
            handling_note=None,
        )

        try:
            self.repository.add(risk_event)
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            if self.repository.exists_by_event_id(payload.event_id):
                raise DuplicateRiskEventError(
                    f"Risk event already exists: {payload.event_id}"
                ) from exc
            raise
        except SQLAlchemyError:
            self.session.rollback()
            raise

        self.session.refresh(risk_event)
        return risk_event

    def update_event_status(
        self,
        event_id: str,
        payload: RiskEventStatusUpdate,
    ) -> RiskEvent:
        risk_event = self.repository.get_by_event_id(event_id)
        if risk_event is None:
            raise RiskEventNotFoundError(f"Risk event does not exist: {event_id}")

        handled_at = (
            None
            if payload.status is RiskEventStatus.PENDING
            else _utc_now_naive()
        )
        try:
            self.repository.update_status(
                risk_event,
                status=payload.status.value,
                handled_at=handled_at,
                handling_note=payload.handling_note,
            )
            self.session.commit()
        except SQLAlchemyError:
            self.session.rollback()
            raise
        self.session.refresh(risk_event)
        return risk_event

    def get_event(self, event_id: str) -> RiskEvent:
        risk_event = self.repository.get_by_event_id(event_id)
        if risk_event is None:
            raise RiskEventNotFoundError(f"Risk event does not exist: {event_id}")
        return risk_event

    def delete_event(self, event_id: str) -> None:
        risk_event = self.repository.get_by_event_id(event_id)
        if risk_event is None:
            raise RiskEventNotFoundError(f"Risk event does not exist: {event_id}")
        try:
            self.repository.delete(risk_event)
            self.session.commit()
        except SQLAlchemyError:
            self.session.rollback()
            raise

    def delete_events(self, event_ids: list[str]) -> int:
        try:
            deleted_count = self.repository.delete_by_event_ids(event_ids)
            self.session.commit()
        except SQLAlchemyError:
            self.session.rollback()
            raise
        return deleted_count

    def delete_all_events(self) -> int:
        try:
            deleted_count = self.repository.delete_all()
            self.session.commit()
        except SQLAlchemyError:
            self.session.rollback()
            raise
        return deleted_count

    def list_events(
        self,
        *,
        module: RiskModule | None = None,
        risk_level: RiskLevel | None = None,
        status: RiskEventStatus | None = None,
        limit: int = 100,
    ) -> list[RiskEvent]:
        return self.repository.list_events(
            module=module.value if module is not None else None,
            risk_level=risk_level.value if risk_level is not None else None,
            status=status.value if status is not None else None,
            limit=limit,
        )
