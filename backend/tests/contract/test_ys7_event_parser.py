from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.infrastructure.event_queue import QueuedYs7Signal
from app.infrastructure.external.ys7.event_adapter import Ys7EventAdapter
from app.infrastructure.external.ys7.event_parser import Ys7EventParser


def test_parser_accepts_camel_case_mock_contract() -> None:
    signal = Ys7EventParser().parse(
        {
            "messageId": "msg-001",
            "requestId": "req-001",
            "eventId": "event-001",
            "deviceId": "camera-01",
            "timestamp": 1_785_123_456,
            "eventType": "phone_call",
            "confidence": 0.91,
            "peopleCount": 1,
            "boxes": [],
            "imageUrl": "https://example.invalid/evidence.jpg",
        },
        received_at=datetime(2026, 8, 4, 12, 0, tzinfo=UTC),
    )

    assert signal.source_event_id == "event-001"
    assert signal.device_id == "camera-01"
    assert signal.occurred_at == datetime.fromtimestamp(1_785_123_456, tz=UTC)
    assert signal.received_at == datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
    assert signal.event_type.value == "phone_call"


def test_parser_accepts_json_body_envelope() -> None:
    signal = Ys7EventParser().parse(
        {
            "messageId": "msg-envelope",
            "body": (
                '{"deviceId":"camera-01","timestamp":1785123456000,'
                '"eventType":"people_count","peopleCount":2}'
            ),
        }
    )

    assert signal.source_event_id == "msg-envelope"
    assert signal.people_count == 2
    assert signal.occurred_at == datetime.fromtimestamp(1_785_123_456, tz=UTC)


@pytest.mark.parametrize(
    "changes",
    [
        {"eventType": "unsupported_vendor_topic"},
        {"timestamp": "2026-08-04T12:00:00"},
        {"confidence": 1.1},
        {"boxes": [{"label": "person"}]},
    ],
)
def test_parser_rejects_invalid_contract(changes: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "messageId": "msg-001",
        "deviceId": "camera-01",
        "timestamp": "2026-08-04T12:00:00+08:00",
        "eventType": "phone_call",
        "confidence": 0.9,
    }
    payload.update(changes)

    with pytest.raises(ValidationError):
        Ys7EventParser().parse(payload)


def test_adapter_converts_xyxy_box_to_unified_event() -> None:
    signal = Ys7EventParser().parse(
        {
            "messageId": "msg-box",
            "deviceId": "camera-01",
            "timestamp": "2026-08-04T12:00:00+08:00",
            "eventType": "person_detected",
            "boxes": [
                {
                    "xyxy": [10, 20, 110, 220],
                    "label": "person",
                    "confidence": 0.88,
                }
            ],
        }
    )

    event = Ys7EventAdapter().adapt(
        QueuedYs7Signal(
            signal=signal,
            dedup_key="message:msg-box",
            raw_event_ref="2026-08-04/raw.json",
        )
    )

    assert event.source == "ys7"
    assert event.event_type == "person_detected"
    assert event.boxes[0].model_dump() == {
        "x1": 10.0,
        "y1": 20.0,
        "x2": 110.0,
        "y2": 220.0,
        "label": "person",
        "confidence": 0.88,
    }
