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
    preferences: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )


class FamilyBindingModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "family_bindings"
    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'ACTIVE', 'REVOKED')",
            name="status_values",
        ),
        CheckConstraint(
            "guardian_user_id <> elder_user_id",
            name="different_users",
        ),
        UniqueConstraint(
            "guardian_user_id",
            "elder_user_id",
            name="uq_family_bindings_guardian_elder",
        ),
        Index("ix_family_bindings_guardian_status", "guardian_user_id", "status"),
        Index("ix_family_bindings_elder_status", "elder_user_id", "status"),
    )

    guardian_user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    elder_user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    relation: Mapped[str] = mapped_column(String(32), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="PENDING")
    elder_confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class BindingCodeModel(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "binding_codes"
    __table_args__ = (
        CheckConstraint(
            "status IN ('ACTIVE', 'CONSUMED', 'EXPIRED', 'REVOKED')",
            name="status_values",
        ),
        UniqueConstraint("code_hash", name="uq_binding_codes_code_hash"),
        Index("ix_binding_codes_status_expires", "status", "expires_at"),
    )

    elder_user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    code_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="ACTIVE")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_by_guardian_user_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )


class UserPushEndpointModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "user_push_endpoints"
    __table_args__ = (
        CheckConstraint(
            "platform IN ('ANDROID', 'IOS')",
            name="platform_values",
        ),
        CheckConstraint(
            "provider IN ('FCM', 'APNS', 'HUAWEI')",
            name="provider_values",
        ),
        UniqueConstraint("token_fingerprint", name="uq_user_push_endpoints_token_fingerprint"),
        UniqueConstraint(
            "user_id",
            "install_id",
            name="uq_user_push_endpoints_user_install",
        ),
        Index("ix_user_push_endpoints_user_active", "user_id", "is_active"),
    )

    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    install_id: Mapped[str] = mapped_column(String(128), nullable=False)
    platform: Mapped[str] = mapped_column(String(16), nullable=False)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    token_ciphertext: Mapped[str] = mapped_column(Text, nullable=False)
    token_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class IdempotencyRecordModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        CheckConstraint(
            "status IN ('IN_PROGRESS', 'COMPLETED')",
            name="status_values",
        ),
        UniqueConstraint(
            "user_id",
            "scope",
            "idempotency_key",
            name="uq_idempotency_records_user_scope_key",
        ),
        Index("ix_idempotency_records_expires_at", "expires_at"),
    )

    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    scope: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="IN_PROGRESS",
    )
    response_status: Mapped[int | None] = mapped_column(Integer)
    response_body: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ElderSafetySettingModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "elder_safety_settings"
    __table_args__ = (
        CheckConstraint(
            "fall_sensitivity IN ('LOW', 'MEDIUM', 'HIGH')",
            name="fall_sensitivity_values",
        ),
        CheckConstraint(
            "inactivity_threshold_hours > 0 AND inactivity_threshold_hours <= 168",
            name="inactivity_threshold_range",
        ),
        CheckConstraint("version > 0", name="version_positive"),
        UniqueConstraint("elder_user_id", name="uq_elder_safety_settings_elder_user_id"),
    )

    elder_user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    fraud_monitor_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )
    fall_detect_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )
    night_leave_bed_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )
    voice_broadcast_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )
    fall_sensitivity: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default="MEDIUM",
    )
    inactivity_threshold_hours: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="6",
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    updated_by_user_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )


class EmergencyContactModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "emergency_contacts"
    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'ACTIVE', 'DISABLED')",
            name="status_values",
        ),
        CheckConstraint(
            "pending_action IS NULL OR pending_action IN ('CREATE', 'UPDATE', 'DELETE')",
            name="pending_action_values",
        ),
        CheckConstraint("priority_order > 0", name="priority_order_positive"),
        Index(
            "ix_emergency_contacts_elder_status_order",
            "elder_user_id",
            "status",
            "priority_order",
        ),
    )

    elder_user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    linked_user_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    relation: Mapped[str] = mapped_column(String(32), nullable=False)
    phone_ciphertext: Mapped[str] = mapped_column(Text, nullable=False)
    phone_last4: Mapped[str] = mapped_column(String(4), nullable=False)
    priority_order: Mapped[int] = mapped_column(Integer, nullable=False)
    channels: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="PENDING")
    pending_action: Mapped[str | None] = mapped_column(String(16))
    pending_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DeviceModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "devices"
    __table_args__ = (
        CheckConstraint(
            "status IN ('UNKNOWN', 'ONLINE', 'OFFLINE', 'DISABLED')",
            name="status_values",
        ),
        CheckConstraint("channel_no > 0", name="channel_no_positive"),
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
    channel_no: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    room: Mapped[str | None] = mapped_column(String(64))
    monitoring_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
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


class FraudSessionModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "fraud_sessions"
    __table_args__ = (
        CheckConstraint("status IN ('ACTIVE', 'CLOSED')", name="status_values"),
        CheckConstraint("last_activity_at >= started_at", name="activity_after_start"),
        CheckConstraint(
            "ended_at IS NULL OR ended_at >= started_at",
            name="end_after_start",
        ),
        UniqueConstraint(
            "external_device_id",
            "session_id",
            name="uq_fraud_sessions_device_session",
        ),
        Index(
            "ix_fraud_sessions_device_status_activity",
            "external_device_id",
            "status",
            "last_activity_at",
        ),
    )

    session_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_device_id: Mapped[str] = mapped_column(String(256), nullable=False)
    elder_alone: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="ACTIVE")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_activity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    speech_events: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    llm_evidence: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    last_llm_review_id: Mapped[str | None] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")


class RiskEventModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "risk_events"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('FRAUD_SUSPECTED', 'FALL_SUSPECTED', 'STRANGER', "
            "'INACTIVITY', 'SOS', 'DEVICE_OFFLINE', 'NIGHT_LEAVE_BED', 'SEDENTARY')",
            name="event_type_values",
        ),
        CheckConstraint(
            "risk_level IS NULL OR risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
            name="risk_level_values",
        ),
        CheckConstraint(
            "alert_level IN ('REMINDER', 'WARNING', 'EMERGENCY')",
            name="alert_level_values",
        ),
        CheckConstraint(
            "status IN ('OPEN', 'ACKNOWLEDGED', 'FALSE_ALARM', 'RESOLVED')",
            name="status_values",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
        CheckConstraint(
            "source IN ('YS7', 'FRAUD_ENGINE', 'FALL_ENGINE', 'APP_SOS', 'SYSTEM')",
            name="source_values",
        ),
        CheckConstraint("version > 0", name="version_positive"),
        UniqueConstraint(
            "source",
            "source_event_id",
            name="uq_risk_events_source_source_event_id",
        ),
        Index("ix_risk_events_device_occurred", "external_device_id", "occurred_at"),
        Index("ix_risk_events_status_occurred", "status", "occurred_at"),
        Index(
            "ix_risk_events_elder_status_occurred",
            "elder_user_id",
            "status",
            "occurred_at",
            "id",
        ),
        Index(
            "ix_risk_events_elder_type_occurred",
            "elder_user_id",
            "event_type",
            "occurred_at",
        ),
    )

    source_event_id: Mapped[str] = mapped_column(String(256), nullable=False)
    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="FRAUD_ENGINE",
    )
    elder_user_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
    )
    external_device_id: Mapped[str | None] = mapped_column(String(256))
    device_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("devices.id", ondelete="SET NULL"),
    )
    model_run_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("model_runs.id", ondelete="SET NULL"),
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    risk_level: Mapped[str | None] = mapped_column(String(16))
    alert_level: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="OPEN")
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 4, asdecimal=False))
    summary: Mapped[str] = mapped_column(String(500), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(128))
    model_version: Mapped[str | None] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")


class EventActionModel(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "event_actions"
    __table_args__ = (
        CheckConstraint(
            "action_type IN ('CONFIRM', 'ACKNOWLEDGE', 'FALSE_ALARM', 'RESOLVE', "
            "'NEED_HELP', 'CONTACT_ELDER', 'CALL', 'ESCALATE', 'NOTE')",
            name="action_type_values",
        ),
        CheckConstraint(
            "previous_status IS NULL OR previous_status IN "
            "('OPEN', 'ACKNOWLEDGED', 'FALSE_ALARM', 'RESOLVED')",
            name="previous_status_values",
        ),
        CheckConstraint(
            "new_status IS NULL OR new_status IN "
            "('OPEN', 'ACKNOWLEDGED', 'FALSE_ALARM', 'RESOLVED')",
            name="new_status_values",
        ),
        UniqueConstraint("idempotency_key", name="uq_event_actions_idempotency_key"),
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
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    action_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )


class EventDeliveryModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "event_deliveries"
    __table_args__ = (
        CheckConstraint(
            "channel IN ('PUSH', 'SMS', 'CALL')",
            name="channel_values",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'SENDING', 'SENT', 'DELIVERED', 'FAILED', 'CANCELLED')",
            name="status_values",
        ),
        CheckConstraint("attempt_count >= 0", name="attempt_count_non_negative"),
        CheckConstraint(
            "target_user_id IS NOT NULL OR contact_id IS NOT NULL",
            name="target_present",
        ),
        UniqueConstraint("dedup_key", name="uq_event_deliveries_dedup_key"),
        Index("ix_event_deliveries_status_next_attempt", "status", "next_attempt_at"),
        Index("ix_event_deliveries_event_created", "risk_event_id", "created_at"),
    )

    risk_event_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("risk_events.id", ondelete="RESTRICT"),
        nullable=False,
    )
    target_user_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
    )
    contact_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("emergency_contacts.id", ondelete="RESTRICT"),
    )
    channel: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="PENDING")
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    provider_message_id: Mapped[str | None] = mapped_column(String(256))
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    dedup_key: Mapped[str] = mapped_column(String(256), nullable=False)
