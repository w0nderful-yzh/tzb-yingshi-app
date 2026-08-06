from datetime import UTC, datetime, timedelta

from app.modules.fraud.risk_engine import build_risk_snapshot, to_epoch_ms
from app.modules.fraud.schemas import VisualEvent


def _person_event(event_id: str, occurred_at: datetime) -> VisualEvent:
    return VisualEvent(
        source_event_id=event_id,
        device_id="camera-1",
        occurred_at=occurred_at,
        received_at=occurred_at,
        source="ys7",
        event_type="person_detected",
        confidence=0.9,
        people_count=2,
        image_url=f"https://images.invalid/{event_id}.jpg",
        raw_event_ref=f"raw/{event_id}.json",
    )


def _money_instruction(at: datetime) -> dict[str, object]:
    at_ms = to_epoch_ms(at)
    return {
        "event_id": "speech-001",
        "start_ms": at_ms - 500,
        "end_ms": at_ms,
        "text": "现在马上转账",
        "evidence_observations": [
            {
                "evidence_id": "ev-money",
                "kind": "money_instruction",
                "stage": "action",
                "strength": "strong",
                "polarity": "supporting",
                "source": "speech",
                "start_ms": at_ms - 500,
                "end_ms": at_ms,
                "text": "现在马上转账",
                "reason": "明确转账指令。",
                "confidence": 0.95,
            }
        ],
    }


def test_single_person_event_is_context_only() -> None:
    started_at = datetime(2026, 8, 6, 10, 0, tzinfo=UTC)

    risk = build_risk_snapshot(
        session_id="session-1",
        device_id="camera-1",
        speech_events=[_money_instruction(started_at + timedelta(seconds=6))],
        visual_events=[_person_event("person-1", started_at + timedelta(seconds=2))],
        elder_alone=False,
    )

    assert risk.state == "S4_ACTION_INDUCEMENT"
    visitor = next(item for item in risk.evidence_chain if item["kind"] == "visitor_presence")
    assert visitor["used_for_transition"] is False


def test_sustained_person_events_can_escalate_existing_action_risk() -> None:
    started_at = datetime(2026, 8, 6, 10, 0, tzinfo=UTC)

    risk = build_risk_snapshot(
        session_id="session-1",
        device_id="camera-1",
        speech_events=[_money_instruction(started_at + timedelta(seconds=6))],
        visual_events=[
            _person_event("person-1", started_at + timedelta(seconds=1)),
            _person_event("person-2", started_at + timedelta(seconds=5)),
        ],
        elder_alone=False,
    )

    assert risk.state == "S5_CRITICAL_CONTROL"
    visitor = next(item for item in risk.evidence_chain if item["kind"] == "visitor_presence")
    assert visitor["escalation_eligible"] is True
    assert visitor["image_url"].endswith("person-2.jpg")
