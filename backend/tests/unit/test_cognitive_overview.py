from datetime import datetime, timedelta, timezone

import pytest

from app.modules.psychology.cognitive.mapping import map_cognitive_snapshot
from app.modules.psychology.cognitive.result_store import CognitiveResultStore
from app.modules.psychology.cognitive.schemas import (
    CognitiveAssessmentSnapshot,
    CognitiveAttentionLevel,
    CognitiveDataQuality,
    CognitiveState,
)
from app.modules.psychology.cognitive.service import CognitiveOverviewService

UTC_TZ = timezone.utc  # noqa: UP017 - match the Python 3.10 Cognitive worker.


def _snapshot(
    *,
    assessment_id: str,
    status: str,
    score: float | None = None,
    speech_seconds: float = 120.0,
) -> CognitiveAssessmentSnapshot:
    now = datetime.now(UTC_TZ)
    return CognitiveAssessmentSnapshot.model_validate(
        {
            "assessment_id": assessment_id,
            "subject_key": "elder-001",
            "session_id": f"session-{assessment_id}",
            "status": status,
            "window_started_at": now,
            "window_ended_at": now if status != "processing" else None,
            "effective_speech_seconds": speech_seconds,
            "estimated_mmse_score": score,
            "audio_window_count": 11 if score is not None else 0,
            "completed_at": now if status != "processing" else None,
        }
    )


def test_completed_snapshot_maps_score_and_non_diagnostic_attention_level() -> None:
    result = map_cognitive_snapshot(
        _snapshot(assessment_id="completed", status="completed", score=23.4)
    )
    payload = result.model_dump(mode="json")

    assert result.assessment_state is CognitiveState.COMPLETED
    assert result.estimated_mmse_score == 23.4
    assert result.attention_level is CognitiveAttentionLevel.MODERATE
    assert result.data_quality is CognitiveDataQuality.USABLE
    assert result.source_modality == "voice_acoustic"
    assert "risk_level" not in payload
    assert "认知障碍或医疗诊断" in result.disclaimer


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (17.99, CognitiveAttentionLevel.HIGH),
        (18.0, CognitiveAttentionLevel.MODERATE),
        (23.99, CognitiveAttentionLevel.MODERATE),
        (24.0, CognitiveAttentionLevel.MILD),
        (26.99, CognitiveAttentionLevel.MILD),
        (27.0, CognitiveAttentionLevel.NONE),
        (30.0, CognitiveAttentionLevel.NONE),
    ],
)
def test_completed_attention_level_boundaries(
    score: float,
    expected: CognitiveAttentionLevel,
) -> None:
    result = map_cognitive_snapshot(
        _snapshot(assessment_id=f"score-{score}", status="completed", score=score)
    )

    assert result.estimated_mmse_score == score
    assert result.attention_level is expected


@pytest.mark.asyncio
async def test_processing_overview_keeps_latest_completed(tmp_path) -> None:
    store = CognitiveResultStore(tmp_path)
    store.write_snapshot(_snapshot(assessment_id="completed", status="completed", score=22.8))
    store.write_snapshot(
        _snapshot(
            assessment_id="processing",
            status="processing",
            speech_seconds=18.0,
        )
    )

    result = await CognitiveOverviewService(store).get_overview(subject_key="elder-001")

    assert result.assessment_state is CognitiveState.PROCESSING
    assert result.estimated_mmse_score is None
    assert result.attention_level is None
    assert result.latest_completed is not None
    assert result.latest_completed.estimated_mmse_score == 22.8
    assert result.latest_completed.attention_level is CognitiveAttentionLevel.MODERATE


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("insufficient_data", CognitiveState.INSUFFICIENT_DATA),
        ("failed", CognitiveState.FAILED),
    ],
)
def test_non_completed_states_remain_non_diagnostic(
    status: str,
    expected: CognitiveState,
) -> None:
    result = map_cognitive_snapshot(
        _snapshot(
            assessment_id=status,
            status=status,
            speech_seconds=30.0,
        )
    )

    assert result.assessment_state is expected
    assert result.estimated_mmse_score is None
    assert result.attention_level is None
    assert "risk_level" not in result.model_dump(mode="json")


@pytest.mark.parametrize("status", ["processing", "failed", "insufficient_data"])
def test_non_completed_state_ignores_present_score(status: str) -> None:
    result = map_cognitive_snapshot(
        _snapshot(
            assessment_id=f"invalid-{status}",
            status=status,
            score=24.0,
            speech_seconds=120.0,
        )
    )

    assert result.assessment_state.value == status
    assert result.estimated_mmse_score is None
    assert result.attention_level is None


@pytest.mark.asyncio
async def test_missing_snapshot_returns_unavailable(tmp_path) -> None:
    result = await CognitiveOverviewService(CognitiveResultStore(tmp_path)).get_overview(
        subject_key="elder-001"
    )

    assert result.assessment_state is CognitiveState.UNAVAILABLE
    assert result.updated_at is None
    assert result.attention_level is None


def test_stale_processing_maps_to_component_unavailable() -> None:
    fresh = _snapshot(assessment_id="processing", status="processing")
    stale = fresh.model_copy(
        update={
            "window_started_at": datetime.now(UTC_TZ) - timedelta(minutes=36),
        }
    )
    result = map_cognitive_snapshot(stale)
    assert result.assessment_state is CognitiveState.UNAVAILABLE
    assert "组件未就绪" in result.evidence_summary
    active = map_cognitive_snapshot(fresh)
    assert active.assessment_state is CognitiveState.PROCESSING
