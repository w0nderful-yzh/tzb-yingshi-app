"""add app phase one schema

Revision ID: 7f2c8d4e9a10
Revises: 0bf3027fa7ee
Create Date: 2026-08-05 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "7f2c8d4e9a10"
down_revision: str | None = "0bf3027fa7ee"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "preferences",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "devices",
        sa.Column("channel_no", sa.Integer(), server_default="1", nullable=False),
    )
    op.add_column("devices", sa.Column("room", sa.String(length=64), nullable=True))
    op.add_column(
        "devices",
        sa.Column(
            "monitoring_enabled",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        op.f("ck_devices_channel_no_positive"),
        "devices",
        "channel_no > 0",
    )

    op.create_table(
        "family_bindings",
        sa.Column("guardian_user_id", sa.UUID(), nullable=False),
        sa.Column("elder_user_id", sa.UUID(), nullable=False),
        sa.Column("relation", sa.String(length=32), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=True),
        sa.Column("status", sa.String(length=20), server_default="PENDING", nullable=False),
        sa.Column("elder_confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "guardian_user_id <> elder_user_id",
            name=op.f("ck_family_bindings_different_users"),
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'ACTIVE', 'REVOKED')",
            name=op.f("ck_family_bindings_status_values"),
        ),
        sa.ForeignKeyConstraint(
            ["elder_user_id"],
            ["users.id"],
            name=op.f("fk_family_bindings_elder_user_id_users"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["guardian_user_id"],
            ["users.id"],
            name=op.f("fk_family_bindings_guardian_user_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_family_bindings")),
        sa.UniqueConstraint(
            "guardian_user_id",
            "elder_user_id",
            name="uq_family_bindings_guardian_elder",
        ),
    )
    op.create_index(
        "ix_family_bindings_elder_status",
        "family_bindings",
        ["elder_user_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_family_bindings_guardian_status",
        "family_bindings",
        ["guardian_user_id", "status"],
        unique=False,
    )

    op.create_table(
        "binding_codes",
        sa.Column("elder_user_id", sa.UUID(), nullable=False),
        sa.Column("code_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="ACTIVE", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_by_guardian_user_id", sa.UUID(), nullable=True),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.CheckConstraint(
            "status IN ('ACTIVE', 'CONSUMED', 'EXPIRED', 'REVOKED')",
            name=op.f("ck_binding_codes_status_values"),
        ),
        sa.ForeignKeyConstraint(
            ["consumed_by_guardian_user_id"],
            ["users.id"],
            name=op.f("fk_binding_codes_consumed_by_guardian_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["elder_user_id"],
            ["users.id"],
            name=op.f("fk_binding_codes_elder_user_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_binding_codes")),
        sa.UniqueConstraint("code_hash", name="uq_binding_codes_code_hash"),
    )
    op.create_index(
        "ix_binding_codes_status_expires",
        "binding_codes",
        ["status", "expires_at"],
        unique=False,
    )

    op.create_table(
        "user_push_endpoints",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("install_id", sa.String(length=128), nullable=False),
        sa.Column("platform", sa.String(length=16), nullable=False),
        sa.Column("provider", sa.String(length=16), nullable=False),
        sa.Column("token_ciphertext", sa.Text(), nullable=False),
        sa.Column("token_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "platform IN ('ANDROID', 'IOS')",
            name=op.f("ck_user_push_endpoints_platform_values"),
        ),
        sa.CheckConstraint(
            "provider IN ('FCM', 'APNS', 'HUAWEI')",
            name=op.f("ck_user_push_endpoints_provider_values"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_user_push_endpoints_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_user_push_endpoints")),
        sa.UniqueConstraint(
            "token_fingerprint",
            name="uq_user_push_endpoints_token_fingerprint",
        ),
        sa.UniqueConstraint(
            "user_id",
            "install_id",
            name="uq_user_push_endpoints_user_install",
        ),
    )
    op.create_index(
        "ix_user_push_endpoints_user_active",
        "user_push_endpoints",
        ["user_id", "is_active"],
        unique=False,
    )

    op.create_table(
        "elder_safety_settings",
        sa.Column("elder_user_id", sa.UUID(), nullable=False),
        sa.Column(
            "fraud_monitor_enabled",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "fall_detect_enabled",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "night_leave_bed_enabled",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "voice_broadcast_enabled",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "fall_sensitivity",
            sa.String(length=16),
            server_default="MEDIUM",
            nullable=False,
        ),
        sa.Column(
            "inactivity_threshold_hours",
            sa.Integer(),
            server_default="6",
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("updated_by_user_id", sa.UUID(), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "fall_sensitivity IN ('LOW', 'MEDIUM', 'HIGH')",
            name=op.f("ck_elder_safety_settings_fall_sensitivity_values"),
        ),
        sa.CheckConstraint(
            "inactivity_threshold_hours > 0 AND inactivity_threshold_hours <= 168",
            name=op.f("ck_elder_safety_settings_inactivity_threshold_range"),
        ),
        sa.CheckConstraint(
            "version > 0",
            name=op.f("ck_elder_safety_settings_version_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["elder_user_id"],
            ["users.id"],
            name=op.f("fk_elder_safety_settings_elder_user_id_users"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["updated_by_user_id"],
            ["users.id"],
            name=op.f("fk_elder_safety_settings_updated_by_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_elder_safety_settings")),
        sa.UniqueConstraint(
            "elder_user_id",
            name="uq_elder_safety_settings_elder_user_id",
        ),
    )

    op.create_table(
        "emergency_contacts",
        sa.Column("elder_user_id", sa.UUID(), nullable=False),
        sa.Column("linked_user_id", sa.UUID(), nullable=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("relation", sa.String(length=32), nullable=False),
        sa.Column("phone_ciphertext", sa.Text(), nullable=False),
        sa.Column("phone_last4", sa.String(length=4), nullable=False),
        sa.Column("priority_order", sa.Integer(), nullable=False),
        sa.Column(
            "channels",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=20), server_default="PENDING", nullable=False),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "priority_order > 0",
            name=op.f("ck_emergency_contacts_priority_order_positive"),
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'ACTIVE', 'DISABLED')",
            name=op.f("ck_emergency_contacts_status_values"),
        ),
        sa.ForeignKeyConstraint(
            ["elder_user_id"],
            ["users.id"],
            name=op.f("fk_emergency_contacts_elder_user_id_users"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["linked_user_id"],
            ["users.id"],
            name=op.f("fk_emergency_contacts_linked_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_emergency_contacts")),
    )
    op.create_index(
        "ix_emergency_contacts_elder_status_order",
        "emergency_contacts",
        ["elder_user_id", "status", "priority_order"],
        unique=False,
    )

    op.add_column(
        "risk_events",
        sa.Column(
            "source",
            sa.String(length=32),
            server_default="FRAUD_ENGINE",
            nullable=False,
        ),
    )
    op.add_column("risk_events", sa.Column("elder_user_id", sa.UUID(), nullable=True))
    op.add_column(
        "risk_events",
        sa.Column(
            "alert_level",
            sa.String(length=16),
            server_default="REMINDER",
            nullable=False,
        ),
    )
    op.add_column(
        "risk_events",
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
    )
    op.create_foreign_key(
        op.f("fk_risk_events_elder_user_id_users"),
        "risk_events",
        "users",
        ["elder_user_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.execute(
        """
        UPDATE risk_events AS risk
        SET elder_user_id = device.elder_user_id,
            device_id = COALESCE(risk.device_id, device.id)
        FROM devices AS device
        WHERE device.external_device_id = risk.external_device_id
        """
    )

    op.drop_constraint(op.f("ck_risk_events_event_type_values"), "risk_events", type_="check")
    op.drop_constraint(op.f("ck_risk_events_risk_level_values"), "risk_events", type_="check")
    op.drop_constraint(op.f("ck_risk_events_status_values"), "risk_events", type_="check")
    op.drop_constraint(op.f("ck_risk_events_confidence_range"), "risk_events", type_="check")
    op.drop_constraint("uq_risk_events_source_event_id", "risk_events", type_="unique")
    op.execute(
        """
        UPDATE risk_events
        SET status = CASE status
                WHEN 'PENDING' THEN 'OPEN'
                WHEN 'CONFIRMED' THEN 'ACKNOWLEDGED'
                ELSE status
            END,
            alert_level = CASE risk_level
                WHEN 'CRITICAL' THEN 'EMERGENCY'
                WHEN 'HIGH' THEN 'WARNING'
                ELSE 'REMINDER'
            END
        """
    )
    op.alter_column(
        "risk_events",
        "external_device_id",
        existing_type=sa.String(length=256),
        nullable=True,
    )
    op.alter_column(
        "risk_events",
        "risk_level",
        existing_type=sa.String(length=16),
        nullable=True,
    )
    op.alter_column(
        "risk_events",
        "status",
        existing_type=sa.String(length=20),
        server_default="OPEN",
        existing_nullable=False,
    )
    op.alter_column(
        "risk_events",
        "confidence",
        existing_type=sa.Numeric(precision=5, scale=4, asdecimal=False),
        nullable=True,
    )
    op.alter_column(
        "risk_events",
        "model_name",
        existing_type=sa.String(length=128),
        nullable=True,
    )
    op.alter_column(
        "risk_events",
        "model_version",
        existing_type=sa.String(length=64),
        nullable=True,
    )
    op.alter_column(
        "risk_events",
        "alert_level",
        existing_type=sa.String(length=16),
        server_default=None,
        existing_nullable=False,
    )
    op.create_check_constraint(
        op.f("ck_risk_events_event_type_values"),
        "risk_events",
        "event_type IN ('FRAUD_SUSPECTED', 'FALL_SUSPECTED', 'STRANGER', "
        "'INACTIVITY', 'SOS', 'DEVICE_OFFLINE', 'NIGHT_LEAVE_BED', 'SEDENTARY')",
    )
    op.create_check_constraint(
        op.f("ck_risk_events_risk_level_values"),
        "risk_events",
        "risk_level IS NULL OR risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
    )
    op.create_check_constraint(
        op.f("ck_risk_events_alert_level_values"),
        "risk_events",
        "alert_level IN ('REMINDER', 'WARNING', 'EMERGENCY')",
    )
    op.create_check_constraint(
        op.f("ck_risk_events_status_values"),
        "risk_events",
        "status IN ('OPEN', 'ACKNOWLEDGED', 'FALSE_ALARM', 'RESOLVED')",
    )
    op.create_check_constraint(
        op.f("ck_risk_events_confidence_range"),
        "risk_events",
        "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
    )
    op.create_check_constraint(
        op.f("ck_risk_events_source_values"),
        "risk_events",
        "source IN ('YS7', 'FRAUD_ENGINE', 'FALL_ENGINE', 'APP_SOS', 'SYSTEM')",
    )
    op.create_check_constraint(
        op.f("ck_risk_events_version_positive"),
        "risk_events",
        "version > 0",
    )
    op.create_unique_constraint(
        "uq_risk_events_source_source_event_id",
        "risk_events",
        ["source", "source_event_id"],
    )
    op.create_index(
        "ix_risk_events_elder_status_occurred",
        "risk_events",
        ["elder_user_id", "status", "occurred_at", "id"],
        unique=False,
    )
    op.create_index(
        "ix_risk_events_elder_type_occurred",
        "risk_events",
        ["elder_user_id", "event_type", "occurred_at"],
        unique=False,
    )

    op.drop_constraint(op.f("ck_event_actions_action_type_values"), "event_actions", type_="check")
    op.drop_constraint(op.f("ck_event_actions_new_status_values"), "event_actions", type_="check")
    op.drop_constraint(
        op.f("ck_event_actions_previous_status_values"),
        "event_actions",
        type_="check",
    )
    op.execute(
        """
        UPDATE event_actions
        SET previous_status = CASE previous_status
                WHEN 'PENDING' THEN 'OPEN'
                WHEN 'CONFIRMED' THEN 'ACKNOWLEDGED'
                ELSE previous_status
            END,
            new_status = CASE new_status
                WHEN 'PENDING' THEN 'OPEN'
                WHEN 'CONFIRMED' THEN 'ACKNOWLEDGED'
                ELSE new_status
            END
        """
    )
    op.add_column(
        "event_actions",
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "event_actions",
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.create_unique_constraint(
        "uq_event_actions_idempotency_key",
        "event_actions",
        ["idempotency_key"],
    )
    op.create_check_constraint(
        op.f("ck_event_actions_action_type_values"),
        "event_actions",
        "action_type IN ('CONFIRM', 'ACKNOWLEDGE', 'FALSE_ALARM', 'RESOLVE', "
        "'NEED_HELP', 'CONTACT_ELDER', 'CALL', 'ESCALATE', 'NOTE')",
    )
    op.create_check_constraint(
        op.f("ck_event_actions_new_status_values"),
        "event_actions",
        "new_status IS NULL OR new_status IN ('OPEN', 'ACKNOWLEDGED', 'FALSE_ALARM', 'RESOLVED')",
    )
    op.create_check_constraint(
        op.f("ck_event_actions_previous_status_values"),
        "event_actions",
        "previous_status IS NULL OR previous_status IN "
        "('OPEN', 'ACKNOWLEDGED', 'FALSE_ALARM', 'RESOLVED')",
    )

    op.create_table(
        "event_deliveries",
        sa.Column("risk_event_id", sa.UUID(), nullable=False),
        sa.Column("target_user_id", sa.UUID(), nullable=True),
        sa.Column("contact_id", sa.UUID(), nullable=True),
        sa.Column("channel", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="PENDING", nullable=False),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("provider_message_id", sa.String(length=256), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("dedup_key", sa.String(length=256), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "attempt_count >= 0",
            name=op.f("ck_event_deliveries_attempt_count_non_negative"),
        ),
        sa.CheckConstraint(
            "channel IN ('PUSH', 'SMS', 'CALL')",
            name=op.f("ck_event_deliveries_channel_values"),
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'SENDING', 'SENT', 'DELIVERED', 'FAILED', 'CANCELLED')",
            name=op.f("ck_event_deliveries_status_values"),
        ),
        sa.CheckConstraint(
            "target_user_id IS NOT NULL OR contact_id IS NOT NULL",
            name=op.f("ck_event_deliveries_target_present"),
        ),
        sa.ForeignKeyConstraint(
            ["contact_id"],
            ["emergency_contacts.id"],
            name=op.f("fk_event_deliveries_contact_id_emergency_contacts"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["risk_event_id"],
            ["risk_events.id"],
            name=op.f("fk_event_deliveries_risk_event_id_risk_events"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["target_user_id"],
            ["users.id"],
            name=op.f("fk_event_deliveries_target_user_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_event_deliveries")),
        sa.UniqueConstraint("dedup_key", name="uq_event_deliveries_dedup_key"),
    )
    op.create_index(
        "ix_event_deliveries_event_created",
        "event_deliveries",
        ["risk_event_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_event_deliveries_status_next_attempt",
        "event_deliveries",
        ["status", "next_attempt_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_event_deliveries_status_next_attempt", table_name="event_deliveries")
    op.drop_index("ix_event_deliveries_event_created", table_name="event_deliveries")
    op.drop_table("event_deliveries")

    op.drop_constraint(
        op.f("ck_event_actions_previous_status_values"),
        "event_actions",
        type_="check",
    )
    op.drop_constraint(op.f("ck_event_actions_new_status_values"), "event_actions", type_="check")
    op.drop_constraint(op.f("ck_event_actions_action_type_values"), "event_actions", type_="check")
    op.drop_constraint("uq_event_actions_idempotency_key", "event_actions", type_="unique")
    op.execute(
        """
        UPDATE event_actions
        SET action_type = CASE action_type
                WHEN 'ACKNOWLEDGE' THEN 'CONFIRM'
                WHEN 'NEED_HELP' THEN 'CONTACT_ELDER'
                WHEN 'CALL' THEN 'CONTACT_ELDER'
                WHEN 'ESCALATE' THEN 'NOTE'
                ELSE action_type
            END,
            previous_status = CASE previous_status
                WHEN 'OPEN' THEN 'PENDING'
                WHEN 'ACKNOWLEDGED' THEN 'CONFIRMED'
                ELSE previous_status
            END,
            new_status = CASE new_status
                WHEN 'OPEN' THEN 'PENDING'
                WHEN 'ACKNOWLEDGED' THEN 'CONFIRMED'
                ELSE new_status
            END
        """
    )
    op.drop_column("event_actions", "metadata")
    op.drop_column("event_actions", "idempotency_key")
    op.create_check_constraint(
        op.f("ck_event_actions_action_type_values"),
        "event_actions",
        "action_type IN ('CONFIRM', 'FALSE_ALARM', 'RESOLVE', 'CONTACT_ELDER', 'NOTE')",
    )
    op.create_check_constraint(
        op.f("ck_event_actions_new_status_values"),
        "event_actions",
        "new_status IS NULL OR new_status IN ('PENDING', 'CONFIRMED', 'FALSE_ALARM', 'RESOLVED')",
    )
    op.create_check_constraint(
        op.f("ck_event_actions_previous_status_values"),
        "event_actions",
        "previous_status IS NULL OR previous_status IN "
        "('PENDING', 'CONFIRMED', 'FALSE_ALARM', 'RESOLVED')",
    )

    op.drop_index("ix_risk_events_elder_type_occurred", table_name="risk_events")
    op.drop_index("ix_risk_events_elder_status_occurred", table_name="risk_events")
    op.drop_constraint("uq_risk_events_source_source_event_id", "risk_events", type_="unique")
    op.drop_constraint(op.f("ck_risk_events_version_positive"), "risk_events", type_="check")
    op.drop_constraint(op.f("ck_risk_events_source_values"), "risk_events", type_="check")
    op.drop_constraint(op.f("ck_risk_events_confidence_range"), "risk_events", type_="check")
    op.drop_constraint(op.f("ck_risk_events_status_values"), "risk_events", type_="check")
    op.drop_constraint(op.f("ck_risk_events_alert_level_values"), "risk_events", type_="check")
    op.drop_constraint(op.f("ck_risk_events_risk_level_values"), "risk_events", type_="check")
    op.drop_constraint(op.f("ck_risk_events_event_type_values"), "risk_events", type_="check")
    op.execute(
        """
        DELETE FROM event_actions
        WHERE risk_event_id IN (
            SELECT id
            FROM risk_events
            WHERE event_type NOT IN ('FRAUD_SUSPECTED', 'FALL_SUSPECTED')
               OR risk_level IS NULL
               OR confidence IS NULL
               OR model_name IS NULL
               OR model_version IS NULL
               OR external_device_id IS NULL
        )
        """
    )
    op.execute(
        """
        DELETE FROM risk_events
        WHERE event_type NOT IN ('FRAUD_SUSPECTED', 'FALL_SUSPECTED')
           OR risk_level IS NULL
           OR confidence IS NULL
           OR model_name IS NULL
           OR model_version IS NULL
           OR external_device_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE risk_events
        SET status = CASE status
            WHEN 'OPEN' THEN 'PENDING'
            WHEN 'ACKNOWLEDGED' THEN 'CONFIRMED'
            ELSE status
        END
        """
    )
    op.alter_column(
        "risk_events",
        "model_version",
        existing_type=sa.String(length=64),
        nullable=False,
    )
    op.alter_column(
        "risk_events",
        "model_name",
        existing_type=sa.String(length=128),
        nullable=False,
    )
    op.alter_column(
        "risk_events",
        "confidence",
        existing_type=sa.Numeric(precision=5, scale=4, asdecimal=False),
        nullable=False,
    )
    op.alter_column(
        "risk_events",
        "status",
        existing_type=sa.String(length=20),
        server_default="PENDING",
        existing_nullable=False,
    )
    op.alter_column(
        "risk_events",
        "risk_level",
        existing_type=sa.String(length=16),
        nullable=False,
    )
    op.alter_column(
        "risk_events",
        "external_device_id",
        existing_type=sa.String(length=256),
        nullable=False,
    )
    op.create_check_constraint(
        op.f("ck_risk_events_event_type_values"),
        "risk_events",
        "event_type IN ('FRAUD_SUSPECTED', 'FALL_SUSPECTED')",
    )
    op.create_check_constraint(
        op.f("ck_risk_events_risk_level_values"),
        "risk_events",
        "risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
    )
    op.create_check_constraint(
        op.f("ck_risk_events_status_values"),
        "risk_events",
        "status IN ('PENDING', 'CONFIRMED', 'FALSE_ALARM', 'RESOLVED')",
    )
    op.create_check_constraint(
        op.f("ck_risk_events_confidence_range"),
        "risk_events",
        "confidence >= 0 AND confidence <= 1",
    )
    op.create_unique_constraint(
        "uq_risk_events_source_event_id",
        "risk_events",
        ["source_event_id"],
    )
    op.drop_constraint(
        op.f("fk_risk_events_elder_user_id_users"),
        "risk_events",
        type_="foreignkey",
    )
    op.drop_column("risk_events", "version")
    op.drop_column("risk_events", "alert_level")
    op.drop_column("risk_events", "elder_user_id")
    op.drop_column("risk_events", "source")

    op.drop_index("ix_emergency_contacts_elder_status_order", table_name="emergency_contacts")
    op.drop_table("emergency_contacts")
    op.drop_table("elder_safety_settings")
    op.drop_index("ix_user_push_endpoints_user_active", table_name="user_push_endpoints")
    op.drop_table("user_push_endpoints")
    op.drop_index("ix_binding_codes_status_expires", table_name="binding_codes")
    op.drop_table("binding_codes")
    op.drop_index("ix_family_bindings_guardian_status", table_name="family_bindings")
    op.drop_index("ix_family_bindings_elder_status", table_name="family_bindings")
    op.drop_table("family_bindings")

    op.drop_constraint(op.f("ck_devices_channel_no_positive"), "devices", type_="check")
    op.drop_column("devices", "monitoring_enabled")
    op.drop_column("devices", "room")
    op.drop_column("devices", "channel_no")
    op.drop_column("users", "preferences")
