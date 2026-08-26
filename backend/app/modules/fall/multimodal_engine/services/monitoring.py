from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.database.models import MonitoringSession
from app.modules.fall.multimodal_engine.repositories import MonitoringRepository
from app.modules.fall.multimodal_engine.schemas.monitoring import MonitoringSessionCreate, MonitoringStatus


class MonitoringSessionAlreadyExistsError(ValueError):
    pass


class MonitoringSessionNotFoundError(LookupError):
    pass


def _to_utc_naive(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None)


class MonitoringService:
    def __init__(
        self,
        session: Session,
        repository: MonitoringRepository | None = None,
    ) -> None:
        self.session = session
        self.repository = repository or MonitoringRepository(session)

    def create_session(self, payload: MonitoringSessionCreate) -> MonitoringSession:
        session_id = payload.session_id or f"session-{uuid4().hex}"
        if self.repository.get_by_id(session_id) is not None:
            raise MonitoringSessionAlreadyExistsError(
                f"Monitoring session already exists: {session_id}"
            )

        monitoring_session = MonitoringSession(
            id=session_id,
            mode=payload.mode.value,
            status=MonitoringStatus.RUNNING.value,
            device_id=payload.device_id,
            enabled_modules=[module.value for module in payload.enabled_modules],
            started_at=_to_utc_naive(payload.started_at),
            ended_at=None,
        )

        try:
            self.repository.add(monitoring_session)
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            if self.repository.get_by_id(session_id) is not None:
                raise MonitoringSessionAlreadyExistsError(
                    f"Monitoring session already exists: {session_id}"
                ) from exc
            raise
        except SQLAlchemyError:
            self.session.rollback()
            raise

        self.session.refresh(monitoring_session)
        return monitoring_session

    def get_current_session(self, device_id: str | None = None) -> MonitoringSession | None:
        return self.repository.get_current(device_id=device_id)

    def stop_session(self, session_id: str) -> MonitoringSession:
        monitoring_session = self.repository.get_by_id(session_id)
        if monitoring_session is None:
            raise MonitoringSessionNotFoundError(
                f"Monitoring session does not exist: {session_id}"
            )

        if monitoring_session.status == MonitoringStatus.STOPPED.value:
            return monitoring_session

        try:
            self.repository.update_status(
                monitoring_session,
                status=MonitoringStatus.STOPPED.value,
                ended_at=_to_utc_naive(datetime.now(timezone.utc)),
            )
            self.session.commit()
        except SQLAlchemyError:
            self.session.rollback()
            raise

        self.session.refresh(monitoring_session)
        return monitoring_session
