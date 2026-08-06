from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import pytest

from app.modules.fraud.ports import FraudRiskEventWrite, FraudSessionRecord
from app.modules.fraud.schemas import FraudAnalyzeRequest, VisualEvent
from app.modules.fraud.service import FraudSessionService
from app.modules.fraud.visual_event_store import VisualEventStore


class CapturingRiskSink:
    def __init__(self) -> None:
        self.events: list[FraudRiskEventWrite] = []

    async def upsert(self, event: FraudRiskEventWrite) -> None:
        self.events.append(event)


class CapturingSessionStore:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str], FraudSessionRecord] = {}
        self.closed: list[tuple[str, str]] = []

    async def load(self, *, device_id: str, session_id: str) -> FraudSessionRecord | None:
        return self.records.get((device_id, session_id))

    async def upsert(self, record: FraudSessionRecord) -> None:
        self.records[(record.device_id, record.session_id)] = record

    async def close_other_active(
        self,
        *,
        device_id: str,
        active_session_id: str,
        ended_at: datetime,
    ) -> None:
        for key, record in list(self.records.items()):
            if record.device_id == device_id and record.session_id != active_session_id:
                self.closed.append(key)
                self.records[key] = FraudSessionRecord(
                    session_id=record.session_id,
                    device_id=record.device_id,
                    elder_alone=record.elder_alone,
                    status="CLOSED",
                    started_at=record.started_at,
                    last_activity_at=record.last_activity_at,
                    ended_at=max(record.started_at, ended_at),
                    speech_events=record.speech_events,
                    llm_evidence=record.llm_evidence,
                    last_llm_review_id=record.last_llm_review_id,
                )


def _request(
    *,
    session_id: str,
    source_event_id: str,
    text: str,
    at: datetime,
    elder_alone: bool = False,
    transcript_status: Literal["PARTIAL", "FINAL"] = "FINAL",
) -> FraudAnalyzeRequest:
    return FraudAnalyzeRequest(
        session_id=session_id,
        source_event_id=source_event_id,
        device_id="camera-01",
        occurred_at=at,
        ended_at=at + timedelta(seconds=2),
        text=text,
        elder_alone=elder_alone,
        transcript_status=transcript_status,
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


@pytest.mark.asyncio
async def test_partial_transcript_is_capped_at_s2_and_final_revision_can_escalate() -> None:
    at = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    sink = CapturingRiskSink()
    service = FraudSessionService(
        visual_event_store=VisualEventStore(),
        risk_event_sink=sink,
    )
    partial = _request(
        session_id="stream-session",
        source_event_id="utterance-1",
        text="把短信验证码告诉",
        at=at,
        elder_alone=True,
        transcript_status="PARTIAL",
    )
    final = partial.model_copy(
        update={
            "text": "把短信验证码告诉我，不要告诉家人",
            "transcript_status": "FINAL",
        }
    )

    partial_result = await service.analyze(partial)
    assert partial_result.risk.state == "S2_TRUST_BUILDING"
    assert sink.events == []

    final_result = await service.analyze(final)

    assert final_result.status == "updated"
    assert final_result.risk.state == "S5_CRITICAL_CONTROL"
    assert len(sink.events) == 1
    speech_events = {
        item.get("speech_event_id")
        for item in final_result.risk.evidence_chain
        if item.get("source") == "speech"
    }
    assert speech_events == {"speech-001"}


@pytest.mark.asyncio
async def test_session_state_can_be_restored_and_new_session_closes_previous() -> None:
    at = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    store = CapturingSessionStore()
    first_service = FraudSessionService(
        visual_event_store=VisualEventStore(),
        session_store=store,
    )
    await first_service.analyze(
        _request(
            session_id="persisted-session",
            source_event_id="speech-1",
            text="我是银行客服",
            at=at,
        )
    )

    restored_service = FraudSessionService(
        visual_event_store=VisualEventStore(),
        session_store=store,
    )
    restored = await restored_service.get_session(
        device_id="camera-01",
        session_id="persisted-session",
    )
    await restored_service.analyze(
        _request(
            session_id="next-session",
            source_event_id="speech-2",
            text="今天天气很好",
            at=at + timedelta(minutes=1),
        )
    )

    assert restored is not None
    assert any(item.get("kind") == "identity_claim" for item in restored.evidence_chain)
    assert ("camera-01", "persisted-session") in store.closed
    closed = store.records[("camera-01", "persisted-session")]
    assert closed.status == "CLOSED"
    assert closed.ended_at is not None


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
