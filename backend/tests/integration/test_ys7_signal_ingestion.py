import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.infrastructure.event_deduplicator import EventDeduplicator
from app.infrastructure.event_queue import SignalQueueFullError, Ys7EventQueue
from app.infrastructure.external.ys7.event_parser import Ys7EventParser
from app.infrastructure.external.ys7.signal_listener import Ys7SignalListener
from app.infrastructure.raw_signal_store import RawSignalStore
from app.main import create_app

WEBHOOK_TOKEN = "test-webhook-token"


def signal_payload(
    message_id: str = "msg-001",
    *,
    timestamp: str = "2026-08-04T12:00:00+08:00",
) -> dict[str, object]:
    return {
        "messageId": message_id,
        "requestId": f"request-{message_id}",
        "eventId": f"event-{message_id}",
        "deviceId": "camera-01",
        "timestamp": timestamp,
        "eventType": "phone_call",
        "confidence": 0.91,
        "peopleCount": 1,
        "boxes": [],
        "imageUrl": "https://example.invalid/evidence.jpg",
    }


def create_ys7_test_app(raw_event_dir: Path):
    settings = Settings(
        environment="test",
        ys7_signal_enabled=True,
        ys7_webhook_token=WEBHOOK_TOKEN,
        ys7_raw_event_dir=raw_event_dir,
        _env_file=None,
    )
    return create_app(settings)


def wait_for_visual_events(client: TestClient, expected_count: int) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for _ in range(50):
        response = client.get("/api/v1/fraud/visual-events")
        events = response.json()["data"]
        if len(events) >= expected_count:
            return events
        time.sleep(0.01)
    return events


def test_signal_requires_enabled_authenticated_receiver(tmp_path: Path) -> None:
    disabled_settings = Settings(environment="test", _env_file=None)
    with TestClient(create_app(disabled_settings)) as client:
        disabled = client.post("/api/v1/integrations/ys7/events", json=signal_payload())
    assert disabled.status_code == 503

    with TestClient(create_ys7_test_app(tmp_path)) as client:
        unauthorized = client.post(
            "/api/v1/integrations/ys7/events",
            json=signal_payload(),
        )
    assert unauthorized.status_code == 401


def test_signal_is_persisted_queued_and_adapted(tmp_path: Path) -> None:
    with TestClient(create_ys7_test_app(tmp_path)) as client:
        status = client.get("/api/v1/integrations/ys7/status")
        accepted = client.post(
            "/api/v1/integrations/ys7/events",
            headers={"X-YS7-Webhook-Token": WEBHOOK_TOKEN},
            json=signal_payload(),
        )
        events = wait_for_visual_events(client, 1)

    assert status.status_code == 200
    assert status.json()["data"]["worker_running"] is True
    assert accepted.status_code == 202
    assert accepted.json()["data"]["status"] == "accepted"
    assert events[0]["source_event_id"] == "event-msg-001"
    assert events[0]["event_type"] == "phone_call"
    assert events[0]["people_count"] == 1
    raw_reference = accepted.json()["data"]["raw_event_ref"]
    assert raw_reference is not None
    assert (tmp_path / raw_reference).is_file()


def test_duplicate_signal_does_not_create_second_event_or_raw_file(tmp_path: Path) -> None:
    with TestClient(create_ys7_test_app(tmp_path)) as client:
        first = client.post(
            "/api/v1/integrations/ys7/events",
            headers={"X-YS7-Webhook-Token": WEBHOOK_TOKEN},
            json=signal_payload(),
        )
        duplicate = client.post(
            "/api/v1/integrations/ys7/events",
            headers={"X-YS7-Webhook-Token": WEBHOOK_TOKEN},
            json=signal_payload(),
        )
        events = wait_for_visual_events(client, 1)

    assert first.json()["data"]["status"] == "accepted"
    assert duplicate.status_code == 202
    assert duplicate.json()["data"]["status"] == "duplicate"
    assert len(events) == 1
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_visual_events_are_ordered_by_occurrence_not_arrival(tmp_path: Path) -> None:
    with TestClient(create_ys7_test_app(tmp_path)) as client:
        for payload in (
            signal_payload("newer", timestamp="2026-08-04T12:00:10+08:00"),
            signal_payload("older", timestamp="2026-08-04T12:00:01+08:00"),
        ):
            response = client.post(
                "/api/v1/integrations/ys7/events",
                headers={"X-YS7-Webhook-Token": WEBHOOK_TOKEN},
                json=payload,
            )
            assert response.status_code == 202
        events = wait_for_visual_events(client, 2)

    assert [event["source_event_id"] for event in events] == [
        "event-newer",
        "event-older",
    ]


def test_invalid_payload_returns_422_without_raw_file(tmp_path: Path) -> None:
    with TestClient(create_ys7_test_app(tmp_path)) as client:
        response = client.post(
            "/api/v1/integrations/ys7/events",
            headers={"X-YS7-Webhook-Token": WEBHOOK_TOKEN},
            json={"messageId": "missing-required-fields"},
        )

    assert response.status_code == 422
    assert list(tmp_path.rglob("*.json")) == []


def test_full_queue_releases_dedup_reservation_for_retry(tmp_path: Path) -> None:
    async def exercise() -> None:
        event_queue = Ys7EventQueue(maxsize=1)
        listener = Ys7SignalListener(
            parser=Ys7EventParser(),
            deduplicator=EventDeduplicator(),
            raw_store=RawSignalStore(tmp_path),
            event_queue=event_queue,
        )
        first = await listener.receive(signal_payload("first"))
        assert first.status == "accepted"

        with pytest.raises(SignalQueueFullError):
            await listener.receive(signal_payload("retryable"))

        queued = await event_queue.get()
        assert queued.signal.source_event_id == "event-first"
        event_queue.task_done()

        retried = await listener.receive(signal_payload("retryable"))
        assert retried.status == "accepted"

    asyncio.run(exercise())
