import re

from fastapi.testclient import TestClient


def test_health_returns_unified_response(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["message"] == "success"
    assert body["data"] == {
        "status": "ok",
        "service": "老年安全监测后端",
        "version": "0.1.0",
        "environment": "test",
    }
    assert re.fullmatch(r"req_[0-9a-f]{32}", body["request_id"])
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_health_preserves_valid_request_id(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers={"X-Request-ID": "android-123"})

    assert response.status_code == 200
    assert response.json()["request_id"] == "android-123"
    assert response.headers["X-Request-ID"] == "android-123"
