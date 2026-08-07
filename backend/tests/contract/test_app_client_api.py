from app.core.config import Settings
from app.main import create_app


def test_app_phase_one_routes_are_exposed() -> None:
    app = create_app(Settings(environment="test", _env_file=None))
    paths = app.openapi()["paths"]

    assert {
        "/api/v1/auth/login",
        "/api/v1/auth/me",
        "/api/v1/auth/logout",
        "/api/v1/users/me",
        "/api/v1/safety/status",
        "/api/v1/sos",
        "/api/v1/events",
        "/api/v1/events/{event_id}",
        "/api/v1/events/{event_id}/confirm",
        "/api/v1/events/{event_id}/status",
        "/api/v1/events/{event_id}/intervention-reminder",
        "/api/v1/devices",
        "/api/v1/devices/{device_id}/live-url",
        "/api/v1/devices/{device_id}/live-sdk-session",
        "/api/v1/devices/{device_id}/audio-pcm",
        "/api/v1/devices/{device_id}/history-playback",
        "/api/v1/ws/tickets",
        "/api/v1/contacts",
        "/api/v1/family/elders",
        "/api/v1/stats/events",
        "/api/v1/stats/activity",
    } <= set(paths)


def test_app_business_routes_require_bearer_authentication() -> None:
    app = create_app(Settings(environment="test", _env_file=None))
    schema = app.openapi()

    assert schema["components"]["securitySchemes"]["HTTPBearer"] == {
        "type": "http",
        "scheme": "bearer",
    }
    assert schema["paths"]["/api/v1/auth/login"]["post"].get("security") is None
    assert schema["paths"]["/api/v1/users/me"]["get"]["security"] == [{"HTTPBearer": []}]
    assert schema["paths"]["/api/v1/ws/tickets"]["post"]["security"] == [{"HTTPBearer": []}]
    parameters = schema["paths"]["/api/v1/users/me"]["get"].get("parameters", [])
    assert all(item["name"] != "X-Demo-Role" for item in parameters)


def test_app_mutations_require_idempotency_key() -> None:
    app = create_app(Settings(environment="test", _env_file=None))
    paths = app.openapi()["paths"]

    for path, method in (
        ("/api/v1/sos", "post"),
        ("/api/v1/events/{event_id}/confirm", "post"),
        ("/api/v1/events/{event_id}/status", "patch"),
        ("/api/v1/events/{event_id}/intervention-reminder", "post"),
    ):
        parameters = paths[path][method]["parameters"]
        idempotency = next(item for item in parameters if item["name"] == "Idempotency-Key")
        assert idempotency["in"] == "header"
        assert idempotency["required"] is True


def test_fraud_context_is_exposed_to_app_clients() -> None:
    app = create_app(Settings(environment="test", _env_file=None))
    schemas = app.openapi()["components"]["schemas"]

    assert {
        "fraud_scene",
        "fraud_state",
        "fraud_state_index",
        "fraud_state_label",
        "fraud_decision",
    } <= set(schemas["RiskEventItem"]["properties"])
    assert "fraud" in schemas["EventDetailData"]["properties"]
    assert "evidence_frames" in schemas["RiskEventItem"]["properties"]
    assert "evidence_frames" in schemas["EventDetailData"]["properties"]
    assert {"captured_at", "image_url"} <= set(schemas["EvidenceFrameData"]["properties"])
    expected = {
        "scene",
        "state",
        "state_index",
        "state_label",
        "decision",
        "transition_reason",
    }
    assert expected <= set(schemas["FraudContextData"]["properties"])
