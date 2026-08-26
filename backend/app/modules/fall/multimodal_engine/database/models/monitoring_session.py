from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, Index, String, text
from sqlalchemy.dialects.mysql import DATETIME, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.modules.fall.multimodal_engine.database.base import Base


class MonitoringSession(Base):
    __tablename__ = "monitoring_sessions"
    __table_args__ = (
        CheckConstraint(
            "mode IN ('SIMULATION', 'FILE', 'LIVE')",
            name="chk_monitoring_mode",
        ),
        CheckConstraint(
            "status IN ('RUNNING', 'STOPPED', 'ERROR')",
            name="chk_monitoring_status",
        ),
        CheckConstraint(
            "JSON_TYPE(enabled_modules) = 'ARRAY'",
            name="chk_enabled_modules_array",
        ),
        Index("idx_session_status", "status"),
        Index("idx_session_started_at", "started_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    device_id: Mapped[str] = mapped_column(String(64), nullable=False)
    enabled_modules: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DATETIME(fsp=3), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=3), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DATETIME(fsp=3),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(3)"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DATETIME(fsp=3),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(3)"),
        server_onupdate=text("CURRENT_TIMESTAMP(3)"),
    )
