from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from service.api import create_app
from service.result_store import LatestAssessmentStore
from service.schemas import PsychologyAssessmentSnapshot


def test_health_and_latest_are_read_only(tmp_path) -> None:
    store = LatestAssessmentStore(tmp_path)
    client = TestClient(create_app(store))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "inference_triggered_by_api": False}
    assert client.get(
        "/api/psychology/assessments/latest",
        params={"subject_key": "elder-001"},
    ).status_code == 404

    completed_at = datetime.now(timezone.utc)
    store.write(
        PsychologyAssessmentSnapshot(
            assessment_id="psy-test-001",
            subject_key="elder-001",
            status="completed",
            window_started_at=completed_at - timedelta(minutes=7),
            window_ended_at=completed_at,
            estimated_phq8_score=6.42,
            segment_scores=[6.42],
            clip_count=7,
            completed_at=completed_at,
        )
    )

    response = client.get(
        "/api/psychology/assessments/latest",
        params={"subject_key": "elder-001"},
    )
    assert response.status_code == 200
    assert response.json()["estimated_phq8_score"] == 6.42
    assert response.json()["subject_key"] == "elder-001"
    assert client.post(
        "/api/psychology/assessments/latest",
        params={"subject_key": "elder-001"},
    ).status_code == 405
