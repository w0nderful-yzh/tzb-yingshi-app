"""add contact confirmation payload

Revision ID: 8a3d1e6f0b21
Revises: 7f2c8d4e9a10
Create Date: 2026-08-05 12:10:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "8a3d1e6f0b21"
down_revision: str | None = "7f2c8d4e9a10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "emergency_contacts",
        sa.Column("pending_action", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "emergency_contacts",
        sa.Column(
            "pending_payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        op.f("ck_emergency_contacts_pending_action_values"),
        "emergency_contacts",
        "pending_action IS NULL OR pending_action IN ('CREATE', 'UPDATE', 'DELETE')",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_emergency_contacts_pending_action_values"),
        "emergency_contacts",
        type_="check",
    )
    op.drop_column("emergency_contacts", "pending_payload")
    op.drop_column("emergency_contacts", "pending_action")
