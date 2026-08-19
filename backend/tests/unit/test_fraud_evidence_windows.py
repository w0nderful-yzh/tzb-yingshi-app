from app.modules.fraud.evidence_decay import apply_stage_windows
from app.modules.fraud.risk_engine import build_risk_snapshot


def _evidence(kind: str, stage: str, strength: str, end_ms: int, confidence: float = 0.95):
    return {
        "evidence_id": f"ev-{kind}-{end_ms}",
        "kind": kind,
        "stage": stage,
        "strength": strength,
        "polarity": "supporting",
        "source": "speech",
        "start_ms": max(0, end_ms - 1_000),
        "end_ms": end_ms,
        "text": "测试",
        "reason": "测试",
        "confidence": confidence,
        "transcript_status": "FINAL",
    }


def test_contact_evidence_expires_after_300_seconds() -> None:
    at_ms = 600_000
    evidence = apply_stage_windows(
        [_evidence("identity_claim", "contact", "weak", end_ms=200_000)],
        at_ms=at_ms,
    )
    assert evidence[0]["expired"] is True
    assert evidence[0]["used_for_transition"] is False
    assert evidence[0]["decayed_confidence"] == 0.0


def test_action_evidence_stays_fresh_inside_120_seconds() -> None:
    at_ms = 600_000
    evidence = apply_stage_windows(
        [_evidence("money_instruction", "action", "strong", end_ms=500_000)],
        at_ms=at_ms,
    )
    assert evidence[0]["expired"] is False
    assert evidence[0]["decayed_confidence"] > 0.7


def test_old_contact_decays_but_recent_action_remains_strong() -> None:
    at_ms = 600_000
    evidence = apply_stage_windows(
        [
            _evidence("identity_claim", "contact", "weak", end_ms=200_000, confidence=0.95),
            _evidence("money_instruction", "action", "strong", end_ms=590_000, confidence=0.96),
        ],
        at_ms=at_ms,
    )
    contact = evidence[0]
    action = evidence[1]
    assert contact["expired"] is True
    assert action["expired"] is False
    assert action["decayed_confidence"] > 0.9
    assert contact["confidence"] == 0.95
    assert contact["window_ms"] == 300_000
    assert action["window_ms"] == 120_000


def test_protective_window_suppresses_within_same_context() -> None:
    at_ms = 600_000
    evidence = apply_stage_windows(
        [
            _evidence(
                "protective_warning",
                "protective",
                "protective",
                end_ms=400_000,
                confidence=0.9,
            )
        ],
        at_ms=at_ms,
    )
    assert evidence[0]["expired"] is False
    assert evidence[0]["decayed_confidence"] == 0.9


def test_five_minute_contact_plus_recent_action_combines() -> None:
    base_ms = 10_000
    snapshot = build_risk_snapshot(
        session_id="session-decay",
        device_id="camera-01",
        speech_events=[
            {
                "event_id": "speech-001",
                "start_ms": base_ms,
                "end_ms": base_ms + 2_000,
                "text": "我是银行客服，你有未完结的贷款问题",
                "transcript_status": "FINAL",
                "risk_tags": ["identity_impersonation"],
                "matched_terms": {},
                "context_adjustments": [],
                "classifier_model": "test",
                "classifier_predictions": [],
                "evidence_observations": [
                    _evidence(
                        "identity_claim",
                        "contact",
                        "weak",
                        end_ms=base_ms + 2_000,
                        confidence=0.95,
                    )
                ],
            },
            {
                "event_id": "speech-002",
                "start_ms": base_ms + 290_000,
                "end_ms": base_ms + 292_000,
                "text": "马上把验证码告诉我",
                "transcript_status": "FINAL",
                "risk_tags": ["credential"],
                "matched_terms": {},
                "context_adjustments": [],
                "classifier_model": "test",
                "classifier_predictions": [],
                "evidence_observations": [
                    _evidence(
                        "credential_request",
                        "action",
                        "strong",
                        end_ms=base_ms + 292_000,
                        confidence=0.96,
                    )
                ],
            },
        ],
        visual_events=[],
        elder_alone=True,
        memory_ms=120_000,
    )
    assert snapshot.state in {"S4_ACTION_INDUCEMENT", "S5_CRITICAL_CONTROL"}


def test_stale_contact_alone_keeps_session_normal() -> None:
    base_ms = 10_000
    snapshot = build_risk_snapshot(
        session_id="session-stale",
        device_id="camera-01",
        speech_events=[
            {
                "event_id": "speech-001",
                "start_ms": base_ms,
                "end_ms": base_ms + 2_000,
                "text": "我是银行客服",
                "transcript_status": "FINAL",
                "risk_tags": ["identity_impersonation"],
                "matched_terms": {},
                "context_adjustments": [],
                "classifier_model": "test",
                "classifier_predictions": [],
                "evidence_observations": [
                    _evidence(
                        "identity_claim",
                        "contact",
                        "weak",
                        end_ms=base_ms + 2_000,
                        confidence=0.95,
                    )
                ],
            }
        ],
        visual_events=[],
        elder_alone=True,
    )
    assert snapshot.state == "S0_NORMAL"
