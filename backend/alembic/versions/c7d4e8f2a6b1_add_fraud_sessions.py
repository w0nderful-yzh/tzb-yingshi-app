"""add fraud sessions

Revision ID: c7d4e8f2a6b1
Revises: 9b4e2f7a1c32
Create Date: 2026-08-06 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "c7d4e8f2a6b1"
down_revision: str | None = "9b4e2f7a1c32"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "fraud_sessions",
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("external_device_id", sa.String(length=256), nullable=False),
        sa.Column("elder_alone", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="ACTIVE", nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "speech_events",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "llm_evidence",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("last_llm_review_id", sa.String(length=64), nullable=True),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
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
            "status IN ('ACTIVE', 'CLOSED')",
            name=op.f("ck_fraud_sessions_status_values"),
        ),
        sa.CheckConstraint(
            "last_activity_at >= started_at",
            name=op.f("ck_fraud_sessions_activity_after_start"),
        ),
        sa.CheckConstraint(
            "ended_at IS NULL OR ended_at >= started_at",
            name=op.f("ck_fraud_sessions_end_after_start"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fraud_sessions")),
        sa.UniqueConstraint(
            "external_device_id",
            "session_id",
            name="uq_fraud_sessions_device_session",
        ),
    )
    op.create_index(
        "ix_fraud_sessions_device_status_activity",
        "fraud_sessions",
        ["external_device_id", "status", "last_activity_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_fraud_sessions_device_status_activity",
        table_name="fraud_sessions",
    )
    op.drop_table("fraud_sessions")
