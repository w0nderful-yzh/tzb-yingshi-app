from datetime import UTC, datetime

import pytest

from app.modules.fraud.ports import RecentFraudContext
from app.modules.fraud.risk_profile import recent_context_evidence
from app.modules.fraud.semantic_retriever import DisabledSemanticEvidenceRetriever
from app.scripts.export_fraud_feedback import mask_text


@pytest.mark.asyncio
async def test_disabled_semantic_retriever_returns_nothing() -> None:
    retriever = DisabledSemanticEvidenceRetriever(enabled=False)
    assert retriever.available is False
    assert await retriever.retrieve(text="把验证码告诉我", session_id="s1") == []


def test_recent_context_evidence_is_weak_and_never_transitions() -> None:
    context = RecentFraudContext(
        device_id="camera-01",
        session_id="s1",
        recent_risk_events=3,
        last_risk_level="HIGH",
        last_kinds=("credential_request",),
        last_occurred_at=datetime(2026, 8, 14, 8, 0, tzinfo=UTC),
    )
    evidence = recent_context_evidence(context, at_ms=1_000_000)
    assert len(evidence) == 1
    item = evidence[0]
    assert item["stage"] == "context"
    assert item["strength"] == "weak"
    assert item["used_for_transition"] is False
    assert item["source"] == "risk_profile"
    assert "credential_request" in item["text"]


def test_recent_context_evidence_empty_when_no_events() -> None:
    context = RecentFraudContext(
        device_id="camera-01",
        session_id="s1",
        recent_risk_events=0,
        last_risk_level=None,
        last_kinds=(),
        last_occurred_at=None,
    )
    assert recent_context_evidence(context, at_ms=1_000_000) == []


def test_feedback_masking_hides_sensitive_entities() -> None:
    text = (
        "我的手机号是13812345678，身份证110101199001011234，"
        "银行卡6222020202020202020，短信验证码是123456，"
        "图片地址 https://cdn.example/frame-1.jpg"
    )
    masked = mask_text(text)
    assert "13812345678" not in masked
    assert "110101199001011234" not in masked
    assert "6222020202020202020" not in masked
    assert "123456" not in masked
    assert "cdn.example" not in masked
    assert "[URL]" in masked
