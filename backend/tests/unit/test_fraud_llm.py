import json

import httpx
import pytest

from app.infrastructure.external.llm.openai_compatible import (
    OpenAiCompatibleFraudLlmJudge,
)
from app.modules.fraud.llm import (
    FraudLlmFinding,
    FraudLlmReview,
    FraudLlmReviewRequest,
    review_to_evidence,
)


def _request() -> FraudLlmReviewRequest:
    return FraudLlmReviewRequest(
        review_id="a" * 64,
        session_id="session-1",
        device_id="camera-1",
        current_state="S2_TRUST_BUILDING",
        at_ms=1_000,
        transcript_segments=(
            {
                "start_ms": 100,
                "end_ms": 900,
                "text": "我是银行客服，请把验证码告诉我",
                "language": "zh",
                "emotion": "NEUTRAL",
                "audio_events": ["speech"],
            },
        ),
        evidence_chain=(),
    )


def test_review_to_evidence_requires_verbatim_transcript_quote() -> None:
    review = FraudLlmReview(
        verdict="HIGH_RISK",
        confidence=0.96,
        summary="检测到凭证索要",
        findings=[
            FraudLlmFinding(
                kind="credential_request",
                quote="请把验证码告诉我",
                reason="明确索要验证码",
                confidence=0.98,
            ),
            FraudLlmFinding(
                kind="money_instruction",
                quote="立即转账十万元",
                reason="原文中不存在的内容",
                confidence=0.99,
            ),
        ],
    )

    evidence = review_to_evidence(_request(), review, model_name="test-model")

    assert len(evidence) == 1
    assert evidence[0]["kind"] == "credential_request"
    assert evidence[0]["strength"] == "medium"
    assert evidence[0]["confidence"] == 0.85
    assert evidence[0]["source"] == "llm"


@pytest.mark.asyncio
async def test_openai_compatible_judge_parses_strict_json() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://llm.invalid/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret"
        body = json.loads(request.content)
        assert isinstance(body["messages"][1]["content"], str)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": """```json
                            {"verdict":"SUSPICIOUS","confidence":0.8,
                             "summary":"疑似冒充客服","findings":[
                             {"kind":"identity_claim","quote":"我是银行客服",
                              "reason":"自称银行客服","confidence":0.8}]}
                            ```"""
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    judge = OpenAiCompatibleFraudLlmJudge(
        base_url="https://llm.invalid/v1",
        api_key="secret",
        model="test-model",
        timeout_seconds=1,
        client=client,
    )

    result = await judge.review(_request())

    assert result.verdict == "SUSPICIOUS"
    assert result.findings[0].kind == "identity_claim"
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_judge_sends_ys7_snapshots_as_image_inputs() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][1]["content"]
        assert body["enable_thinking"] is False
        assert content[0]["type"] == "text"
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": "https://ys7.invalid/snapshot-1.jpg"},
        }
        assert "visual-1" in content[0]["text"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"verdict":"INCONCLUSIVE","confidence":0.4,'
                                '"summary":"画面仅用于场景核对","findings":[]}'
                            )
                        }
                    }
                ]
            },
        )

    request = _request()
    request = FraudLlmReviewRequest(
        review_id=request.review_id,
        session_id=request.session_id,
        device_id=request.device_id,
        current_state=request.current_state,
        at_ms=request.at_ms,
        transcript_segments=request.transcript_segments,
        evidence_chain=request.evidence_chain,
        visual_inputs=(
            {
                "source_event_id": "visual-event-1",
                "occurred_ms": 500,
                "event_type": "phone_call",
                "confidence": 0.9,
                "people_count": 1,
                "image_url": "https://ys7.invalid/snapshot-1.jpg",
            },
        ),
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    judge = OpenAiCompatibleFraudLlmJudge(
        base_url="https://llm.invalid/v1",
        api_key="secret",
        model="qwen3.5-plus",
        timeout_seconds=1,
        enable_thinking=False,
        client=client,
    )

    result = await judge.review(request)

    assert result.verdict == "INCONCLUSIVE"
    await client.aclose()
