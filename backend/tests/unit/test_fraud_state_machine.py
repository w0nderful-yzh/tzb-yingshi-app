from app.modules.fraud.fraud_evidence import extract_speech_evidence
from app.modules.fraud.fraud_state_machine import FraudProcessStateMachine


def _speech(
    event_id: str,
    start_ms: int,
    text: str,
    tags: list[str],
    *,
    adjustments: list[str] | None = None,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "start_ms": start_ms,
        "end_ms": start_ms + 2_000,
        "text": text,
        "risk_tags": tags,
        "matched_terms": {},
        "context_adjustments": adjustments or [],
    }


def _run(*events: dict[str, object], elder_alone: bool = False) -> dict[str, object]:
    machine = FraudProcessStateMachine(elder_alone=elder_alone)
    for event in events:
        for evidence in extract_speech_evidence(event):
            machine.consume(evidence)
    return machine.snapshot()


def test_single_weak_keyword_does_not_raise_state() -> None:
    snapshot = _run(_speech("speech-001", 0, "银行客服", ["identity_impersonation"]))

    assert snapshot["state"] == "S0_NORMAL"


def test_process_history_preserves_evidence_order() -> None:
    snapshot = _run(
        _speech("speech-001", 0, "你儿子撞到人了", ["family_emergency"]),
        _speech("speech-002", 5_000, "需要十万块", ["amount_expression"]),
        _speech(
            "speech-003",
            10_000,
            "赶紧到银行取钱",
            ["money_operation", "urgency"],
        ),
    )

    assert snapshot["state"] == "S4_ACTION_INDUCEMENT"
    assert [item["to"] for item in snapshot["state_history"]] == [
        "S1_OBSERVING",
        "S2_TRUST_BUILDING",
        "S4_ACTION_INDUCEMENT",
    ]


def test_strong_action_and_elder_alone_trigger_intervention() -> None:
    snapshot = _run(
        _speech("speech-001", 0, "把短信验证码告诉我", ["credential"]),
        elder_alone=True,
    )

    assert snapshot["state"] == "S5_CRITICAL_CONTROL"
    assert snapshot["decision"] == "intervene"


def test_protective_warning_suppresses_risk_keywords() -> None:
    snapshot = _run(
        _speech(
            "speech-001",
            0,
            "警方提醒，验证码不要告诉任何人，也不要转账",
            ["identity_impersonation", "credential", "money_operation"],
            adjustments=["anti_fraud_warning"],
        )
    )

    assert snapshot["state"] == "S0_NORMAL"
    assert snapshot["evidence_chain"][0]["kind"] == "protective_warning"


def test_explicit_non_transition_context_is_honored() -> None:
    machine = FraudProcessStateMachine()
    machine.consume(
        {
            "evidence_id": "people-1",
            "kind": "people_count_context",
            "stage": "context",
            "strength": "weak",
            "polarity": "supporting",
            "confidence": 1.0,
            "end_ms": 1,
            "used_for_transition": False,
        }
    )

    assert machine.snapshot()["state"] == "S0_NORMAL"
    assert machine.snapshot()["evidence_chain"][0]["used_for_transition"] is False
