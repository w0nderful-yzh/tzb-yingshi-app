from datetime import UTC, datetime
from typing import Any, Literal, cast

from app.modules.fraud.evidence_decay import MAX_WINDOW_MS, apply_stage_windows
from app.modules.fraud.fraud_evidence import extract_speech_evidence
from app.modules.fraud.fraud_state_machine import FraudProcessStateMachine
from app.modules.fraud.schemas import FraudRiskSnapshot, VisualEvent

RISK_MODEL_NAME = "fraud_process_state_machine"
RISK_MODEL_VERSION = "3.0"


def to_epoch_ms(value: datetime) -> int:
    return round(value.timestamp() * 1000)


def _visual_evidence(
    visual_events: list[VisualEvent],
    *,
    at_ms: int,
    memory_ms: int,
    elder_alone: bool,
) -> list[dict[str, Any]]:
    lower_bound = at_ms - memory_ms
    recent = [
        event for event in visual_events if lower_bound <= to_epoch_ms(event.occurred_at) <= at_ms
    ]
    evidence: list[dict[str, Any]] = []
    if elder_alone:
        evidence.append(
            {
                "evidence_id": "ev-context-elder-alone",
                "kind": "elder_alone",
                "stage": "context",
                "strength": "weak",
                "polarity": "supporting",
                "source": "context",
                "start_ms": lower_bound,
                "end_ms": at_ms,
                "text": "当前会话明确标记为老人独处。",
                "reason": "老人独处是敏感操作指令升级为高危干预的脆弱性语境。",
                "confidence": 1.0,
            }
        )

    phone_calls = [event for event in recent if event.event_type == "phone_call"]
    if phone_calls:
        strongest = max(phone_calls, key=lambda event: event.confidence or 0.0)
        occurred_ms = to_epoch_ms(strongest.occurred_at)
        evidence.append(
            {
                "evidence_id": f"ev-visual-{strongest.source_event_id}-phone-call",
                "kind": "phone_call_active",
                "stage": "context",
                "strength": "weak",
                "polarity": "supporting",
                "source": "video",
                "start_ms": occurred_ms,
                "end_ms": occurred_ms,
                "text": "萤石视觉事件检测到打电话场景。",
                "reason": "通话场景用于增强身份接触和操作诱导的时间关联。",
                "confidence": strongest.confidence or 0.5,
                "visual_event_id": strongest.source_event_id,
            }
        )

    people_events = [event for event in recent if event.people_count is not None]
    if people_events:
        latest = max(people_events, key=lambda event: event.occurred_at)
        occurred_ms = to_epoch_ms(latest.occurred_at)
        evidence.append(
            {
                "evidence_id": f"ev-visual-{latest.source_event_id}-people-count",
                "kind": "people_count_context",
                "stage": "context",
                "strength": "weak",
                "polarity": "supporting",
                "source": "video",
                "start_ms": occurred_ms,
                "end_ms": occurred_ms,
                "text": f"当前视觉事件识别到 {latest.people_count} 人。",
                "reason": "人数仅作为场景事实，不单独触发诈骗风险。",
                "confidence": latest.confidence or 0.5,
                "people_count": latest.people_count,
                "visual_event_id": latest.source_event_id,
                "used_for_transition": False,
            }
        )

    person_events = [event for event in recent if event.event_type == "person_detected"]
    if person_events:
        ordered_people = sorted(person_events, key=lambda event: event.occurred_at)
        first = ordered_people[0]
        latest = ordered_people[-1]
        first_ms = to_epoch_ms(first.occurred_at)
        latest_ms = to_epoch_ms(latest.occurred_at)
        sustained = len(ordered_people) >= 2 and latest_ms - first_ms >= 3_000
        detected_people = latest.people_count
        if detected_people is None:
            detected_people = (
                sum(
                    1
                    for box in latest.boxes
                    if (box.label or "").lower() in {"person", "people", "human"}
                )
                or 1
            )
        evidence.append(
            {
                "evidence_id": f"ev-visual-{latest.source_event_id}-visitor",
                "kind": "visitor_presence",
                "stage": "context",
                "strength": "weak",
                "polarity": "supporting",
                "source": "video",
                "start_ms": first_ms,
                "end_ms": latest_ms,
                "text": f"连续视觉事件检测到人员出现，最近人数为 {detected_people}。",
                "reason": (
                    "连续人员事件可作为入户诈骗的访客情境。"
                    if sustained
                    else "单次人员事件只展示为场景事实，不推动风险升级。"
                ),
                "confidence": max(event.confidence or 0.5 for event in ordered_people),
                "people_count": detected_people,
                "visual_event_id": latest.source_event_id,
                "image_url": latest.image_url,
                "escalation_eligible": sustained,
                "used_for_transition": sustained,
            }
        )
    return evidence


def build_risk_snapshot(
    *,
    session_id: str,
    device_id: str,
    speech_events: list[dict[str, Any]],
    visual_events: list[VisualEvent],
    elder_alone: bool,
    memory_ms: int = 120_000,
    extra_evidence: list[dict[str, Any]] | None = None,
) -> FraudRiskSnapshot:
    if not speech_events:
        now = datetime.now(UTC)
        return FraudRiskSnapshot(
            session_id=session_id,
            device_id=device_id,
            state="S0_NORMAL",
            state_index=0,
            state_label="正常交流",
            score=0,
            risk_level="LOW",
            decision="normal",
            confidence=0.0,
            occurred_at=now,
            transition_reason="尚未收到语音证据。",
            next_stage_conditions=["第二个独立弱证据", "亲属出事等中等证据"],
            evidence_chain=[],
            state_history=[],
        )

    ordered_speech = sorted(speech_events, key=lambda item: int(item["start_ms"]))
    at_ms = max(int(item["end_ms"]) for item in ordered_speech)
    lower_bound = at_ms - MAX_WINDOW_MS
    recent_speech = [event for event in ordered_speech if int(event["end_ms"]) >= lower_bound]
    evidence = _visual_evidence(
        visual_events,
        at_ms=at_ms,
        memory_ms=memory_ms,
        elder_alone=elder_alone,
    )
    for speech_event in recent_speech:
        evidence.extend(
            speech_event.get("evidence_observations") or extract_speech_evidence(speech_event)
        )
    evidence.extend(
        item for item in (extra_evidence or []) if lower_bound <= int(item["end_ms"]) <= at_ms
    )
    evidence.sort(
        key=lambda item: (
            int(item["end_ms"]),
            0 if item.get("source") in {"context", "video"} else 1,
            str(item["evidence_id"]),
        )
    )
    evidence = apply_stage_windows(evidence, at_ms=at_ms)

    machine = FraudProcessStateMachine(elder_alone=elder_alone, memory_ms=MAX_WINDOW_MS)
    for item in evidence:
        machine.consume(item)
    snapshot = machine.snapshot()
    supporting_confidences = [
        float(item.get("confidence", 0.0))
        for item in snapshot["evidence_chain"]
        if item.get("used_for_transition")
    ]
    return FraudRiskSnapshot(
        session_id=session_id,
        device_id=device_id,
        state=str(snapshot["state"]),
        state_index=int(snapshot["state_index"]),
        state_label=str(snapshot["state_label"]),
        score=int(snapshot["score"]),
        risk_level=cast(
            Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            str(snapshot["level"]).upper(),
        ),
        decision=str(snapshot["decision"]),
        confidence=max(supporting_confidences, default=0.0),
        occurred_at=datetime.fromtimestamp(at_ms / 1000, tz=UTC),
        transition_reason=str(snapshot["transition_reason"]),
        next_stage_conditions=[str(item) for item in snapshot["next_stage_conditions"]],
        evidence_chain=list(snapshot["evidence_chain"]),
        state_history=list(snapshot["state_history"]),
    )
