from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_not_found_uses_unified_response(client: TestClient) -> None:
    response = client.get("/api/v1/not-found")

    assert response.status_code == 404
    assert response.json() == {
        "code": 10002,
        "message": "Not Found",
        "data": None,
        "request_id": response.headers["X-Request-ID"],
    }


def test_validation_error_does_not_echo_invalid_input(
    test_app: FastAPI,
) -> None:
    @test_app.get("/_test/validation")
    async def validation_probe(limit: int) -> dict[str, int]:
        return {"limit": limit}

    with TestClient(test_app) as client:
        response = client.get("/_test/validation", params={"limit": "private-value"})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == 10001
    assert body["message"] == "request validation failed"
    assert "private-value" not in response.text
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_unhandled_error_hides_internal_details(test_app: FastAPI) -> None:
    @test_app.get("/_test/error")
    async def error_probe() -> None:
        raise RuntimeError("private implementation detail")

    with TestClient(test_app, raise_server_exceptions=False) as client:
        response = client.get("/_test/error")

    assert response.status_code == 500
    assert response.json() == {
        "code": 10003,
        "message": "internal server error",
        "data": None,
        "request_id": response.headers["X-Request-ID"],
    }
    assert "private implementation detail" not in response.text
