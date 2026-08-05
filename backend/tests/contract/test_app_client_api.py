from app.core.config import Settings
from app.main import create_app


def test_app_phase_one_routes_are_exposed() -> None:
    app = create_app(Settings(environment="test", _env_file=None))
    paths = app.openapi()["paths"]

    assert {
        "/api/v1/users/me",
        "/api/v1/safety/status",
        "/api/v1/sos",
        "/api/v1/events",
        "/api/v1/events/{event_id}",
        "/api/v1/events/{event_id}/confirm",
        "/api/v1/events/{event_id}/status",
        "/api/v1/devices",
        "/api/v1/devices/{device_id}/live-url",
        "/api/v1/devices/{device_id}/live-sdk-session",
        "/api/v1/contacts",
        "/api/v1/family/elders",
        "/api/v1/stats/events",
        "/api/v1/stats/activity",
    } <= set(paths)


def test_app_mutations_require_idempotency_key() -> None:
    app = create_app(Settings(environment="test", _env_file=None))
    paths = app.openapi()["paths"]

    for path, method in (
        ("/api/v1/sos", "post"),
        ("/api/v1/events/{event_id}/confirm", "post"),
        ("/api/v1/events/{event_id}/status", "patch"),
    ):
        parameters = paths[path][method]["parameters"]
        idempotency = next(item for item in parameters if item["name"] == "Idempotency-Key")
        assert idempotency["in"] == "header"
        assert idempotency["required"] is True
