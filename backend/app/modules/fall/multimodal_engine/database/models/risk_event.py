from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, String, UniqueConstraint, text
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, DECIMAL, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.modules.fall.multimodal_engine.database.base import Base


class RiskEvent(Base):
    __tablename__ = "risk_events"
    __table_args__ = (
        UniqueConstraint("event_id", name="uk_risk_event_id"),
        CheckConstraint(
            "module IN ('FALL', 'MENTAL_STATE', 'FRAUD', 'DEVICE')",
            name="chk_risk_module",
        ),
        CheckConstraint(
            "risk_score >= 0 AND risk_score <= 1",
            name="chk_risk_score",
        ),
        CheckConstraint(
            "risk_level IN ('LOW', 'MEDIUM', 'HIGH')",
            name="chk_risk_level",
        ),
        CheckConstraint(
            "source IN ('SIMULATION', 'ALGORITHM', 'EZVIZ')",
            name="chk_risk_source",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'ACKNOWLEDGED', 'FALSE_ALARM')",
            name="chk_risk_status",
        ),
        Index("idx_event_session_time", "session_id", "occurred_at"),
        Index("idx_event_module_time", "module", "occurred_at"),
        Index("idx_event_status_level", "status", "risk_level"),
        Index("idx_event_occurred_at", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True,
    )
    event_id: Mapped[str] = mapped_column(String(80), nullable=False)
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "monitoring_sessions.id",
            name="fk_risk_event_session",
            onupdate="CASCADE",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    device_id: Mapped[str] = mapped_column(String(64), nullable=False)
    module: Mapped[str] = mapped_column(String(32), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_score: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    summary: Mapped[str] = mapped_column(String(500), nullable=False)
    evidence_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    recommended_action: Mapped[str | None] = mapped_column(String(500), nullable=True)
    snapshot_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    clip_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        server_default=text("'PENDING'"),
    )
    occurred_at: Mapped[datetime] = mapped_column(DATETIME(fsp=3), nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DATETIME(fsp=3),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(3)"),
    )
    handled_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=3), nullable=True)
    handling_note: Mapped[str | None] = mapped_column(String(500), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DATETIME(fsp=3),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(3)"),
        server_onupdate=text("CURRENT_TIMESTAMP(3)"),
    )
