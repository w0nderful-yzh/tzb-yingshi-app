from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.database.models import MonitoringSession


class MonitoringRepository:
    """只封装monitoring_sessions表的SQLAlchemy读写。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, monitoring_session: MonitoringSession) -> MonitoringSession:
        self.session.add(monitoring_session)
        self.session.flush()
        return monitoring_session

    def get_by_id(self, session_id: str) -> MonitoringSession | None:
        return self.session.get(MonitoringSession, session_id)

    def get_current(self, device_id: str | None = None) -> MonitoringSession | None:
        statement = select(MonitoringSession).where(
            MonitoringSession.status == "RUNNING"
        )
        if device_id is not None:
            statement = statement.where(MonitoringSession.device_id == device_id)
        statement = statement.order_by(MonitoringSession.started_at.desc()).limit(1)
        return self.session.scalars(statement).first()

    def list_sessions(self, *, offset: int = 0, limit: int = 100) -> list[MonitoringSession]:
        statement = (
            select(MonitoringSession)
            .order_by(MonitoringSession.started_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(self.session.scalars(statement))

    def delete(self, monitoring_session: MonitoringSession) -> None:
        self.session.delete(monitoring_session)
        self.session.flush()

    def update_status(
        self,
        monitoring_session: MonitoringSession,
        *,
        status: str,
        ended_at: datetime | None,
    ) -> MonitoringSession:
        monitoring_session.status = status
        monitoring_session.ended_at = ended_at
        self.session.flush()
        return monitoring_session
