from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

from app.core.config import Settings
from app.infrastructure.database import models as database_models  # noqa: F401
from app.infrastructure.database.base import Base


def test_initial_schema_contains_required_tables() -> None:
    assert set(Base.metadata.tables) == {
        "users",
        "auth_sessions",
        "family_bindings",
        "binding_codes",
        "user_push_endpoints",
        "idempotency_records",
        "elder_safety_settings",
        "emergency_contacts",
        "devices",
        "ys7_signal_inbox",
        "visual_events",
        "model_runs",
        "fraud_sessions",
        "risk_events",
        "event_actions",
        "event_deliveries",
    }


def test_auth_schema_stores_password_and_session_tokens_as_hashes() -> None:
    users = Base.metadata.tables["users"]
    sessions = Base.metadata.tables["auth_sessions"]

    assert {"login_name", "password_hash"} <= set(users.c.keys())
    assert "password" not in users.c
    assert {"user_id", "token_hash", "expires_at", "revoked_at"} <= set(sessions.c.keys())
    assert "access_token" not in sessions.c


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
    assert ("source", "source_event_id") in risk_unique_columns
    assert "ck_risk_events_confidence_range" in risk_checks
    assert "ck_risk_events_risk_level_values" in risk_checks
    assert "ck_risk_events_alert_level_values" in risk_checks
    assert isinstance(risk_events.c.evidence.type, JSONB)


def test_fraud_sessions_persist_replayable_active_state() -> None:
    sessions = Base.metadata.tables["fraud_sessions"]
    unique_columns = {
        tuple(constraint.columns.keys())
        for constraint in sessions.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert ("external_device_id", "session_id") in unique_columns
    assert {"speech_events", "llm_evidence", "started_at", "last_activity_at"} <= set(
        sessions.c.keys()
    )
    assert isinstance(sessions.c.speech_events.type, JSONB)


def test_app_phase_one_tables_support_authorization_and_delivery() -> None:
    family_bindings = Base.metadata.tables["family_bindings"]
    contacts = Base.metadata.tables["emergency_contacts"]
    deliveries = Base.metadata.tables["event_deliveries"]
    idempotency_records = Base.metadata.tables["idempotency_records"]
    risk_events = Base.metadata.tables["risk_events"]

    assert {"guardian_user_id", "elder_user_id", "status"} <= set(family_bindings.c.keys())
    assert {"phone_ciphertext", "phone_last4", "priority_order"} <= set(contacts.c.keys())
    assert {"channel", "status", "scheduled_at", "dedup_key"} <= set(deliveries.c.keys())
    assert {"elder_user_id", "alert_level", "version"} <= set(risk_events.c.keys())
    assert {"scope", "idempotency_key", "request_hash", "response_body"} <= set(
        idempotency_records.c.keys()
    )


def test_database_url_is_redacted_in_settings_representation() -> None:
    settings = Settings(
        database_url="postgresql+psycopg://user:private-password@localhost/db",
        _env_file=None,
    )

    assert "private-password" not in repr(settings)
