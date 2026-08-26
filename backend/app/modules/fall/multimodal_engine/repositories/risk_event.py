from datetime import datetime

from sqlalchemy import delete as sql_delete
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.database.models import RiskEvent


class RiskEventRepository:
    """只封装risk_events表的SQLAlchemy读写。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, risk_event: RiskEvent) -> RiskEvent:
        self.session.add(risk_event)
        self.session.flush()
        return risk_event

    def get_by_id(self, record_id: int) -> RiskEvent | None:
        return self.session.get(RiskEvent, record_id)

    def get_by_event_id(self, event_id: str) -> RiskEvent | None:
        statement = select(RiskEvent).where(RiskEvent.event_id == event_id).limit(1)
        return self.session.scalars(statement).first()

    def exists_by_event_id(self, event_id: str) -> bool:
        statement = select(RiskEvent.event_id).where(RiskEvent.event_id == event_id).limit(1)
        return self.session.scalar(statement) is not None

    def list_events(
        self,
        *,
        module: str | None = None,
        risk_level: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[RiskEvent]:
        statement = select(RiskEvent)
        if module is not None:
            statement = statement.where(RiskEvent.module == module)
        if risk_level is not None:
            statement = statement.where(RiskEvent.risk_level == risk_level)
        if status is not None:
            statement = statement.where(RiskEvent.status == status)
        statement = statement.order_by(RiskEvent.occurred_at.desc()).limit(limit)
        return list(self.session.scalars(statement))

    def get_latest_for_module(
        self,
        *,
        session_id: str,
        module: str,
    ) -> RiskEvent | None:
        statement = (
            select(RiskEvent)
            .where(
                RiskEvent.session_id == session_id,
                RiskEvent.module == module,
            )
            .order_by(RiskEvent.occurred_at.desc())
            .limit(1)
        )
        return self.session.scalars(statement).first()

    def count_by_session(self, session_id: str) -> int:
        statement = select(func.count(RiskEvent.id)).where(
            RiskEvent.session_id == session_id
        )
        return int(self.session.scalar(statement) or 0)

    def update_status(
        self,
        risk_event: RiskEvent,
        *,
        status: str,
        handled_at: datetime | None,
        handling_note: str | None,
    ) -> RiskEvent:
        risk_event.status = status
        risk_event.handled_at = handled_at
        risk_event.handling_note = handling_note
        self.session.flush()
        return risk_event

    def delete(self, risk_event: RiskEvent) -> None:
        self.session.delete(risk_event)
        self.session.flush()

    def delete_by_event_ids(self, event_ids: list[str]) -> int:
        result = self.session.execute(
            sql_delete(RiskEvent).where(RiskEvent.event_id.in_(event_ids))
        )
        return int(result.rowcount or 0)

    def delete_all(self) -> int:
        result = self.session.execute(sql_delete(RiskEvent))
        return int(result.rowcount or 0)
