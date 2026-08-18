import httpx
import pytest

from app.infrastructure.external.psychology.client import HttpPsychologySource


@pytest.mark.asyncio
async def test_client_reads_exact_latest_endpoint_without_triggering_inference() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "schema_version": "psychology_assessment_v1",
                "assessment_id": "psy-001",
                "subject_key": "elder-001",
                "status": "completed",
                "window_started_at": "2026-08-16T08:00:00+08:00",
                "window_ended_at": "2026-08-16T08:07:00+08:00",
                "estimated_phq8_score": 6.42,
                "segment_scores": [6.42],
                "clip_count": 7,
                "completed_at": "2026-08-16T08:08:00+08:00",
            },
        )

    source = HttpPsychologySource(
        base_url="https://psychology.test",
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await source.get_latest_assessment(subject_key="elder-001")
    finally:
        await source.close()

    assert result.estimated_phq8_score == 6.42
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/psychology/assessments/latest"
    assert requests[0].url.params["subject_key"] == "elder-001"
