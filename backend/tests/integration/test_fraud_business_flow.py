from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.modules.fraud.ports import FraudRiskEventWrite
from app.modules.fraud.schemas import FraudAnalyzeRequest, VisualEvent
from app.modules.fraud.service import FraudSessionService
from app.modules.fraud.visual_event_store import VisualEventStore


class CapturingRiskSink:
    def __init__(self) -> None:
        self.events: list[FraudRiskEventWrite] = []

    async def upsert(self, event: FraudRiskEventWrite) -> None:
        self.events.append(event)


def _request(
    *,
    session_id: str,
    source_event_id: str,
    text: str,
    at: datetime,
    elder_alone: bool = False,
) -> FraudAnalyzeRequest:
    return FraudAnalyzeRequest(
        session_id=session_id,
        source_event_id=source_event_id,
        device_id="camera-01",
        occurred_at=at,
        ended_at=at + timedelta(seconds=2),
        text=text,
        elder_alone=elder_alone,
    )


@pytest.mark.asyncio
async def test_phone_call_and_identity_claim_enter_trust_building() -> None:
    at = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    visual_store = VisualEventStore()
    await visual_store.add(
        VisualEvent(
            source_event_id="ys7-phone-1",
            message_id="message-1",
            device_id="camera-01",
            occurred_at=at,
            received_at=at,
            source="ys7",
            event_type="phone_call",
            confidence=0.91,
            raw_event_ref="raw://message-1",
        )
    )
    sink = CapturingRiskSink()
    service = FraudSessionService(
        visual_event_store=visual_store,
        risk_event_sink=sink,
    )

    result = await service.analyze(
        _request(
            session_id="session-1",
            source_event_id="speech-1",
            text="我是银行客服",
            at=at + timedelta(seconds=5),
        )
    )

    assert result.status == "accepted"
    assert result.risk.state == "S2_TRUST_BUILDING"
    assert {item["kind"] for item in result.risk.evidence_chain} >= {
        "phone_call_active",
        "identity_claim",
    }
    assert sink.events[-1].risk_level == "MEDIUM"


@pytest.mark.asyncio
async def test_duplicate_speech_event_is_idempotent() -> None:
    at = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    service = FraudSessionService(visual_event_store=VisualEventStore())
    payload = _request(
        session_id="session-2",
        source_event_id="speech-duplicate",
        text="把短信验证码告诉我",
        at=at,
        elder_alone=True,
    )

    first = await service.analyze(payload)
    second = await service.analyze(payload)

    assert first.status == "accepted"
    assert second.status == "duplicate"
    assert second.risk.state == "S5_CRITICAL_CONTROL"
    speech_ids = {
        item.get("speech_event_id")
        for item in second.risk.evidence_chain
        if item.get("source") == "speech"
    }
    assert speech_ids == {"speech-001"}


def test_fraud_api_analyzes_and_returns_session(client: Any) -> None:
    payload = {
        "session_id": "api-session",
        "source_event_id": "api-speech-1",
        "device_id": "camera-api",
        "occurred_at": "2026-08-04T10:00:00+08:00",
        "ended_at": "2026-08-04T10:00:02+08:00",
        "text": "把短信验证码告诉我，不要告诉家人",
        "elder_alone": True,
    }

    response = client.post("/api/v1/fraud/analyze", json=payload)
    session_response = client.get(
        "/api/v1/fraud/sessions/api-session",
        params={"device_id": "camera-api"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["risk"]["state"] == "S5_CRITICAL_CONTROL"
    assert session_response.status_code == 200
    assert session_response.json()["data"]["session_id"] == "api-session"


def test_fraud_api_rejects_naive_time(client: Any) -> None:
    response = client.post(
        "/api/v1/fraud/analyze",
        json={
            "session_id": "api-session",
            "source_event_id": "api-speech-2",
            "device_id": "camera-api",
            "occurred_at": "2026-08-04T10:00:00",
            "ended_at": "2026-08-04T10:00:02",
            "text": "测试",
        },
    )

    assert response.status_code == 422
