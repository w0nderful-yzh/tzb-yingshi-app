from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class UserModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "role IN ('ELDER', 'GUARDIAN', 'ADMIN')",
            name="role_values",
        ),
    )

    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    external_subject: Mapped[str | None] = mapped_column(String(255), unique=True)
    phone_masked: Mapped[str | None] = mapped_column(String(32))
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )


class DeviceModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "devices"
    __table_args__ = (
        CheckConstraint(
            "status IN ('UNKNOWN', 'ONLINE', 'OFFLINE', 'DISABLED')",
            name="status_values",
        ),
        Index("ix_devices_elder_user_id", "elder_user_id"),
    )

    external_device_id: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    provider: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="ys7",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="UNKNOWN",
    )
    elder_user_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )


class Ys7SignalInboxModel(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "ys7_signal_inbox"
    __table_args__ = (
        CheckConstraint(
            "processing_status IN ('PENDING', 'PROCESSING', 'PROCESSED', 'FAILED')",
            name="processing_status_values",
        ),
        UniqueConstraint("dedup_key", name="uq_ys7_signal_inbox_dedup_key"),
        Index(
            "ix_ys7_signal_inbox_status_received",
            "processing_status",
            "received_at",
        ),
        Index(
            "ix_ys7_signal_inbox_device_occurred",
            "external_device_id",
            "occurred_at",
        ),
    )

    dedup_key: Mapped[str] = mapped_column(String(512), nullable=False)
    message_id: Mapped[str | None] = mapped_column(String(256))
    request_id: Mapped[str | None] = mapped_column(String(256))
    source_event_id: Mapped[str] = mapped_column(String(256), nullable=False)
    external_device_id: Mapped[str] = mapped_column(String(256), nullable=False)
    device_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("devices.id", ondelete="SET NULL"),
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    processing_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="PENDING",
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="0",
    )
    last_error: Mapped[str | None] = mapped_column(Text)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class VisualEventModel(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "visual_events"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('phone_call', 'people_count', 'person_detected')",
            name="event_type_values",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
        CheckConstraint(
            "people_count IS NULL OR people_count >= 0",
            name="people_count_non_negative",
        ),
        UniqueConstraint(
            "source",
            "source_event_id",
            name="uq_visual_events_source_source_event_id",
        ),
        Index("ix_visual_events_device_occurred", "external_device_id", "occurred_at"),
    )

    source_event_id: Mapped[str] = mapped_column(String(256), nullable=False)
    message_id: Mapped[str | None] = mapped_column(String(256))
    request_id: Mapped[str | None] = mapped_column(String(256))
    external_device_id: Mapped[str] = mapped_column(String(256), nullable=False)
    device_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("devices.id", ondelete="SET NULL"),
    )
    raw_signal_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("ys7_signal_inbox.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 4, asdecimal=False))
    people_count: Mapped[int | None] = mapped_column(Integer)
    boxes: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    image_url: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )


class ModelRunModel(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "model_runs"
    __table_args__ = (
        CheckConstraint("module IN ('FRAUD', 'FALL')", name="module_values"),
        CheckConstraint(
            "status IN ('RUNNING', 'SUCCEEDED', 'FAILED')",
            name="status_values",
        ),
        Index("ix_model_runs_device_started", "device_id", "started_at"),
    )

    device_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("devices.id", ondelete="SET NULL"),
    )
    module: Mapped[str] = mapped_column(String(20), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="RUNNING")
    input_refs: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    output_summary: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    error_code: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RiskEventModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "risk_events"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('FRAUD_SUSPECTED', 'FALL_SUSPECTED')",
            name="event_type_values",
        ),
        CheckConstraint(
            "risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
            name="risk_level_values",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'CONFIRMED', 'FALSE_ALARM', 'RESOLVED')",
            name="status_values",
        ),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
        UniqueConstraint("source_event_id", name="uq_risk_events_source_event_id"),
        Index("ix_risk_events_device_occurred", "external_device_id", "occurred_at"),
        Index("ix_risk_events_status_occurred", "status", "occurred_at"),
    )

    source_event_id: Mapped[str] = mapped_column(String(256), nullable=False)
    external_device_id: Mapped[str] = mapped_column(String(256), nullable=False)
    device_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("devices.id", ondelete="SET NULL"),
    )
    model_run_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("model_runs.id", ondelete="SET NULL"),
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="PENDING")
    confidence: Mapped[float] = mapped_column(Numeric(5, 4, asdecimal=False), nullable=False)
    summary: Mapped[str] = mapped_column(String(500), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)


class EventActionModel(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "event_actions"
    __table_args__ = (
        CheckConstraint(
            "action_type IN ('CONFIRM', 'FALSE_ALARM', 'RESOLVE', 'CONTACT_ELDER', 'NOTE')",
            name="action_type_values",
        ),
        CheckConstraint(
            "previous_status IS NULL OR previous_status IN "
            "('PENDING', 'CONFIRMED', 'FALSE_ALARM', 'RESOLVED')",
            name="previous_status_values",
        ),
        CheckConstraint(
            "new_status IS NULL OR new_status IN "
            "('PENDING', 'CONFIRMED', 'FALSE_ALARM', 'RESOLVED')",
            name="new_status_values",
        ),
        Index("ix_event_actions_risk_created", "risk_event_id", "created_at"),
    )

    risk_event_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("risk_events.id", ondelete="RESTRICT"),
        nullable=False,
    )
    actor_user_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    action_type: Mapped[str] = mapped_column(String(32), nullable=False)
    previous_status: Mapped[str | None] = mapped_column(String(20))
    new_status: Mapped[str | None] = mapped_column(String(20))
    note: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
