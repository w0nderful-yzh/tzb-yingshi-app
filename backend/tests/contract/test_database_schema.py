from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

from app.core.config import Settings
from app.infrastructure.database import models as database_models  # noqa: F401
from app.infrastructure.database.base import Base


def test_initial_schema_contains_required_tables() -> None:
    assert set(Base.metadata.tables) == {
        "users",
        "devices",
        "ys7_signal_inbox",
        "visual_events",
        "model_runs",
        "risk_events",
        "event_actions",
    }


def test_event_tables_preserve_occurrence_and_reception_time() -> None:
    for table_name in ("ys7_signal_inbox", "visual_events", "risk_events"):
        table = Base.metadata.tables[table_name]
        assert table.c.occurred_at.type.timezone is True
        assert table.c.received_at.type.timezone is True


def test_idempotency_and_risk_constraints_are_declared() -> None:
    inbox = Base.metadata.tables["ys7_signal_inbox"]
    risk_events = Base.metadata.tables["risk_events"]

    inbox_unique_columns = {
        tuple(constraint.columns.keys())
        for constraint in inbox.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    risk_unique_columns = {
        tuple(constraint.columns.keys())
        for constraint in risk_events.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    risk_checks = {
        constraint.name
        for constraint in risk_events.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert ("dedup_key",) in inbox_unique_columns
    assert ("source_event_id",) in risk_unique_columns
    assert "ck_risk_events_confidence_range" in risk_checks
    assert "ck_risk_events_risk_level_values" in risk_checks
    assert isinstance(risk_events.c.evidence.type, JSONB)


def test_database_url_is_redacted_in_settings_representation() -> None:
    settings = Settings(
        database_url="postgresql+psycopg://user:private-password@localhost/db",
        _env_file=None,
    )

    assert "private-password" not in repr(settings)
