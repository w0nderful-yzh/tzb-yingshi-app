from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import pytest

from app.modules.fraud.ports import FraudRiskEventWrite, FraudSessionRecord
from app.modules.fraud.schemas import FraudAnalyzeRequest
from app.modules.fraud.service import SYSTEM_RETRACT_REASON, FraudSessionService
from app.modules.fraud.visual_event_store import VisualEventStore


class CapturingRiskSink:
    def __init__(self) -> None:
        self.events: list[FraudRiskEventWrite] = []
        self.retracted: list[tuple[str, str]] = []

    async def upsert(self, event: FraudRiskEventWrite) -> None:
        self.events.append(event)

    async def retract_preliminary(
        self,
        *,
        source_event_id: str,
        reason: str,
    ) -> None:
        self.retracted.append((source_event_id, reason))


class CapturingSessionStore:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str], FraudSessionRecord] = {}

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


def _service(
    **overrides: Any,
) -> tuple[FraudSessionService, CapturingRiskSink, CapturingSessionStore]:
    sink = CapturingRiskSink()
    store = CapturingSessionStore()
    defaults: dict[str, Any] = {
        "visual_event_store": VisualEventStore(),
        "risk_event_sink": sink,
        "session_store": store,
        "preliminary_alert_enabled": True,
    }
    defaults.update(overrides)
    return FraudSessionService(**defaults), sink, store


def _request(
    *,
    source_event_id: str,
    text: str,
    at: datetime,
    transcript_status: Literal["PARTIAL", "FINAL"] = "FINAL",
    session_id: str = "session-prelim",
) -> FraudAnalyzeRequest:
    return FraudAnalyzeRequest(
        session_id=session_id,
        source_event_id=source_event_id,
        device_id="camera-01",
        occurred_at=at,
        ended_at=at + timedelta(seconds=2),
        text=text,
        elder_alone=True,
        transcript_status=transcript_status,
    )


_STRONG_ACTION = "把短信里的验证码念给我听"
_START = datetime(2026, 8, 10, 9, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_single_unstable_partial_does_not_alert() -> None:
    service, sink, _ = _service()
    result = await service.analyze(
        _request(
            source_event_id="turn-1",
            text=_STRONG_ACTION,
            at=_START,
            transcript_status="PARTIAL",
        )
    )
    assert result.status == "accepted"
    assert sink.events == []
    assert sink.retracted == []


@pytest.mark.asyncio
async def test_two_consecutive_same_strong_actions_create_one_preliminary() -> None:
    service, sink, _ = _service()
    await service.analyze(
        _request(
            source_event_id="turn-1",
            text=_STRONG_ACTION,
            at=_START,
            transcript_status="PARTIAL",
        )
    )
    second = await service.analyze(
        _request(
            source_event_id="turn-1",
            text=_STRONG_ACTION + "，不要挂电话",
            at=_START + timedelta(seconds=3),
            transcript_status="PARTIAL",
        )
    )
    assert second.status == "updated"
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.source_event_id == "turn-1"
    assert event.verification_status == "PRELIMINARY"
    assert event.evidence["verification_status"] == "PRELIMINARY"
    assert event.evidence["preliminary_kind"] == "credential_request"


@pytest.mark.asyncio
async def test_repeated_partial_does_not_create_additional_events() -> None:
    service, sink, _ = _service()
    for revision, text in enumerate(
        [_STRONG_ACTION, _STRONG_ACTION + "，不要挂电话", _STRONG_ACTION + "，不要告诉家人"],
        start=1,
    ):
        await service.analyze(
            _request(
                source_event_id="turn-2",
                text=text,
                at=_START + timedelta(seconds=revision * 3),
                transcript_status="PARTIAL",
            )
        )
    assert len(sink.events) == 1
    assert sink.events[0].source_event_id == "turn-2"


@pytest.mark.asyncio
async def test_final_confirms_same_event() -> None:
    service, sink, _ = _service()
    await service.analyze(
        _request(
            source_event_id="turn-3",
            text=_STRONG_ACTION,
            at=_START,
            transcript_status="PARTIAL",
        )
    )
    await service.analyze(
        _request(
            source_event_id="turn-3",
            text=_STRONG_ACTION,
            at=_START + timedelta(seconds=3),
            transcript_status="PARTIAL",
        )
    )
    final = await service.analyze(
        _request(
            source_event_id="turn-3",
            text=_STRONG_ACTION,
            at=_START + timedelta(seconds=6),
            transcript_status="FINAL",
        )
    )
    assert final.risk.state == "S5_CRITICAL_CONTROL"
    preliminary = sink.events[0]
    confirmed = sink.events[-1]
    assert confirmed.verification_status == "CONFIRMED"
    assert confirmed.source_event_id == "turn-3"
    assert preliminary.source_event_id == confirmed.source_event_id
    assert len(sink.events) == 2


@pytest.mark.asyncio
async def test_final_fallback_retracts_same_event() -> None:
    service, sink, _ = _service()
    await service.analyze(
        _request(
            source_event_id="turn-4",
            text=_STRONG_ACTION,
            at=_START,
            transcript_status="PARTIAL",
        )
    )
    await service.analyze(
        _request(
            source_event_id="turn-4",
            text=_STRONG_ACTION + "，不是要转账",
            at=_START + timedelta(seconds=3),
            transcript_status="PARTIAL",
        )
    )
    await service.analyze(
        _request(
            source_event_id="turn-4",
            text="天气不错，今天去公园散步",
            at=_START + timedelta(seconds=6),
            transcript_status="FINAL",
        )
    )
    assert sink.retracted == [("turn-4", SYSTEM_RETRACT_REASON)]
    assert [event.verification_status for event in sink.events] == ["PRELIMINARY"]


@pytest.mark.asyncio
async def test_protective_warning_blocks_preliminary() -> None:
    service, sink, _ = _service()
    await service.analyze(
        _request(
            source_event_id="turn-5",
            text="这是银行反诈提醒，请勿向陌生人转账汇款",
            at=_START,
            transcript_status="PARTIAL",
        )
    )
    await service.analyze(
        _request(
            source_event_id="turn-5",
            text="这是银行反诈提醒，请勿向陌生人转账汇款，不要提供验证码",
            at=_START + timedelta(seconds=3),
            transcript_status="PARTIAL",
        )
    )
    assert sink.events == []


@pytest.mark.asyncio
async def test_llm_review_is_not_submitted_for_partial_first_alert() -> None:
    service, sink, _ = _service()

    class CapturingReviewQueue:
        def __init__(self) -> None:
            self.submissions: list[object] = []

        def submit_nowait(self, request: object) -> bool:
            self.submissions.append(request)
            return True

    queue = CapturingReviewQueue()
    service._llm_review_queue = queue  # type: ignore[attr-defined]
    await service.analyze(
        _request(
            source_event_id="turn-6",
            text=_STRONG_ACTION,
            at=_START,
            transcript_status="PARTIAL",
        )
    )
    await service.analyze(
        _request(
            source_event_id="turn-6",
            text=_STRONG_ACTION,
            at=_START + timedelta(seconds=3),
            transcript_status="PARTIAL",
        )
    )
    assert sink.events, "preliminary must exist"
    assert queue.submissions == []


@pytest.mark.asyncio
async def test_restart_restores_session_without_duplicate_preliminary() -> None:
    sink = CapturingRiskSink()
    store = CapturingSessionStore()
    shared: dict[str, Any] = {
        "visual_event_store": VisualEventStore(),
        "risk_event_sink": sink,
        "session_store": store,
        "preliminary_alert_enabled": True,
    }
    service = FraudSessionService(**shared)
    await service.analyze(
        _request(
            source_event_id="turn-7",
            text=_STRONG_ACTION,
            at=_START,
            transcript_status="PARTIAL",
        )
    )
    await service.analyze(
        _request(
            source_event_id="turn-7",
            text=_STRONG_ACTION,
            at=_START + timedelta(seconds=3),
            transcript_status="PARTIAL",
        )
    )
    first_count = len(sink.events)
    assert first_count == 1

    restarted = FraudSessionService(**shared)
    await restarted.analyze(
        _request(
            source_event_id="turn-7",
            text=_STRONG_ACTION + "，马上",
            at=_START + timedelta(seconds=6),
            transcript_status="PARTIAL",
        )
    )
    assert len(sink.events) == 1
    await restarted.analyze(
        _request(
            source_event_id="turn-7",
            text=_STRONG_ACTION + "，马上",
            at=_START + timedelta(seconds=9),
            transcript_status="FINAL",
        )
    )
    assert sink.events[-1].verification_status == "CONFIRMED"


@pytest.mark.asyncio
async def test_feature_flag_disabled_keeps_final_only_flow() -> None:
    service, sink, _ = _service(preliminary_alert_enabled=False)
    await service.analyze(
        _request(
            source_event_id="turn-8",
            text=_STRONG_ACTION,
            at=_START,
            transcript_status="PARTIAL",
        )
    )
    await service.analyze(
        _request(
            source_event_id="turn-8",
            text=_STRONG_ACTION,
            at=_START + timedelta(seconds=3),
            transcript_status="PARTIAL",
        )
    )
    assert sink.events == []
    final = await service.analyze(
        _request(
            source_event_id="turn-8",
            text=_STRONG_ACTION,
            at=_START + timedelta(seconds=6),
            transcript_status="FINAL",
        )
    )
    assert final.risk.state == "S5_CRITICAL_CONTROL"
    assert len(sink.events) == 1
    assert sink.events[0].verification_status is None
