import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.modules.fraud.llm import (
    FraudLlmFinding,
    FraudLlmReview,
    FraudLlmReviewQueue,
    FraudLlmReviewRequest,
)
from app.modules.fraud.schemas import FraudAnalyzeRequest, VisualEvent
from app.modules.fraud.service import FraudSessionService
from app.modules.fraud.visual_event_store import VisualEventStore
from app.workers.fraud_llm_review_worker import FraudLlmReviewWorker


class BlockingJudge:
    model_name = "fake-text-llm"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.requests: list[FraudLlmReviewRequest] = []

    async def review(self, request: FraudLlmReviewRequest) -> FraudLlmReview:
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        return FraudLlmReview(
            verdict="SUSPICIOUS",
            confidence=0.8,
            summary="身份接触后试探账户信息",
            findings=[
                FraudLlmFinding(
                    kind="sensitive_info_request",
                    quote="我是银行客服",
                    reason="复核为可疑账户信息接触",
                    confidence=0.8,
                )
            ],
        )


class FailingJudge:
    model_name = "failing-llm"

    async def review(self, request: FraudLlmReviewRequest) -> FraudLlmReview:
        raise RuntimeError("provider unavailable")


async def _build_service(
    judge: BlockingJudge | FailingJudge,
) -> tuple[FraudSessionService, FraudLlmReviewQueue, FraudLlmReviewWorker]:
    at = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    visual_store = VisualEventStore()
    await visual_store.add(
        VisualEvent(
            source_event_id="phone-1",
            device_id="camera-1",
            occurred_at=at,
            received_at=at,
            source="ys7",
            event_type="phone_call",
            confidence=0.9,
            raw_event_ref="raw://phone-1",
        )
    )
    queue = FraudLlmReviewQueue(maxsize=4)
    service = FraudSessionService(
        visual_event_store=visual_store,
        llm_review_queue=queue,
    )
    worker = FraudLlmReviewWorker(
        queue=queue,
        judge=judge,
        fraud_session_service=service,
        timeout_seconds=1,
    )
    return service, queue, worker


def _request() -> FraudAnalyzeRequest:
    at = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    return FraudAnalyzeRequest(
        session_id="session-1",
        source_event_id="speech-1",
        device_id="camera-1",
        occurred_at=at + timedelta(seconds=2),
        ended_at=at + timedelta(seconds=4),
        text="我是银行客服",
    )


@pytest.mark.asyncio
async def test_llm_review_runs_in_background_and_can_add_medium_evidence() -> None:
    judge = BlockingJudge()
    service, queue, worker = await _build_service(judge)
    await worker.start()
    try:
        local_result = await service.analyze(_request())
        await asyncio.wait_for(judge.started.wait(), timeout=1)

        assert local_result.risk.state == "S2_TRUST_BUILDING"
        assert worker.reviews_processed == 0

        judge.release.set()
        await asyncio.wait_for(queue.join(), timeout=1)
        enhanced = await service.get_session(device_id="camera-1", session_id="session-1")

        assert enhanced is not None
        assert enhanced.state == "S3_INFORMATION_PROBING"
        assert any(item["source"] == "llm" for item in enhanced.evidence_chain)
        assert worker.reviews_processed == 1
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_llm_failure_keeps_local_state_machine_result() -> None:
    service, queue, worker = await _build_service(FailingJudge())
    await worker.start()
    try:
        local_result = await service.analyze(_request())
        await asyncio.wait_for(queue.join(), timeout=1)
        current = await service.get_session(device_id="camera-1", session_id="session-1")

        assert local_result.risk.state == "S2_TRUST_BUILDING"
        assert current is not None
        assert current.state == "S2_TRUST_BUILDING"
        assert worker.reviews_failed == 1
        assert worker.last_error == "provider unavailable"
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_llm_review_selects_recent_ys7_snapshots_in_time_order() -> None:
    at = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    visual_store = VisualEventStore()
    for source_event_id, seconds, image_url in (
        ("old", -121, "https://ys7.invalid/old.jpg"),
        ("newer", 1, "https://ys7.invalid/newer.jpg"),
        ("newest", 2, "https://ys7.invalid/newest.jpg"),
    ):
        await visual_store.add(
            VisualEvent(
                source_event_id=source_event_id,
                device_id="camera-1",
                occurred_at=at + timedelta(seconds=seconds),
                received_at=at + timedelta(seconds=seconds),
                source="ys7",
                event_type="phone_call",
                confidence=0.9,
                image_url=image_url,
                raw_event_ref=f"raw://{source_event_id}",
            )
        )
    queue = FraudLlmReviewQueue(maxsize=4)
    service = FraudSessionService(
        visual_event_store=visual_store,
        llm_review_queue=queue,
        llm_vision_enabled=True,
        llm_max_images=2,
    )
    request = _request().model_copy(
        update={
            "occurred_at": at + timedelta(seconds=2),
            "ended_at": at + timedelta(seconds=4),
        }
    )

    result = await service.analyze(request)
    queued = await asyncio.wait_for(queue.get(), timeout=1)
    queue.task_done()

    assert result.risk.state == "S2_TRUST_BUILDING"
    assert [item["source_event_id"] for item in queued.visual_inputs] == [
        "newer",
        "newest",
    ]


def test_llm_status_reports_configured_background_worker() -> None:
    settings = Settings(
        environment="test",
        fraud_llm_enabled=True,
        _env_file=None,
    )
    app = create_app(settings, fraud_llm_judge=FailingJudge())

    with TestClient(app) as client:
        response = client.get("/api/v1/fraud/llm/status")

    assert response.status_code == 200
    assert response.json()["data"] == {
        "enabled": True,
        "configured": True,
        "running": True,
        "model": "failing-llm",
        "vision_enabled": False,
        "queue_depth": 0,
        "reviews_processed": 0,
        "reviews_failed": 0,
        "last_error": None,
    }


def test_incomplete_llm_settings_do_not_break_local_backend() -> None:
    settings = Settings(
        environment="test",
        fraud_llm_enabled=True,
        _env_file=None,
    )
    app = create_app(settings)

    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        status = client.get("/api/v1/fraud/llm/status")

    assert health.status_code == 200
    assert status.status_code == 200
    assert status.json()["data"]["configured"] is False
    assert status.json()["data"]["running"] is False
    assert status.json()["data"]["last_error"] == "Fraud LLM settings are incomplete"
