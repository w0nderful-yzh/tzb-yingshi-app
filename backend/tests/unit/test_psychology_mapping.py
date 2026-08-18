from app.modules.psychology.mapping import map_psychology_snapshot
from app.modules.psychology.schemas import AssessmentState, SourceStatus
from app.modules.psychology.source_schemas import PsychologySourceSnapshot


def _completed_snapshot(score: float = 6.42) -> PsychologySourceSnapshot:
    return PsychologySourceSnapshot.model_validate(
        {
            "schema_version": "psychology_assessment_v1",
            "assessment_id": "psy-001",
            "subject_key": "elder-001",
            "status": "completed",
            "window_started_at": "2026-08-16T08:00:00+08:00",
            "window_ended_at": "2026-08-16T08:07:00+08:00",
            "estimated_phq8_score": score,
            "segment_scores": [score],
            "clip_count": 7,
            "completed_at": "2026-08-16T08:08:00+08:00",
        }
    )


def test_completed_result_maps_to_shadow_observation_with_raw_reference_scores() -> None:
    result = map_psychology_snapshot(_completed_snapshot())
    payload = result.model_dump(mode="json")

    assert result.source_status is SourceStatus.AVAILABLE
    assert result.assessment_state is AssessmentState.OBSERVATION_AVAILABLE
    assert result.operating_mode == "shadow"
    assert result.attention_level == "unknown"
    assert result.trend_state == "insufficient_history"
    # Raw research-prototype regression outputs pass through unchanged,
    # without threshold mapping or risk grading.
    assert result.estimated_phq8_score == 6.42
    assert result.segment_scores == [6.42]
    assert set(payload) == {
        "source_status",
        "operating_mode",
        "assessment_state",
        "attention_level",
        "trend_state",
        "data_quality",
        "source_modality",
        "review_status",
        "assessment_window",
        "estimated_phq8_score",
        "segment_scores",
        "evidence_summary",
        "guidance",
        "updated_at",
        "disclaimer",
    }
    assert "抑郁" not in str(payload)


def test_out_of_contract_model_score_is_not_silently_clamped_or_exposed() -> None:
    result = map_psychology_snapshot(_completed_snapshot(score=25.0))

    assert result.source_status is SourceStatus.UNAVAILABLE
    assert result.assessment_state is AssessmentState.UNAVAILABLE


def test_insufficient_data_stays_non_diagnostic() -> None:
    snapshot = PsychologySourceSnapshot.model_validate(
        {
            "schema_version": "psychology_assessment_v1",
            "assessment_id": "psy-short",
            "subject_key": "elder-001",
            "status": "insufficient_data",
            "window_started_at": "2026-08-16T08:00:00+08:00",
            "window_ended_at": "2026-08-16T08:00:23+08:00",
            "estimated_phq8_score": None,
            "segment_scores": [],
            "clip_count": 0,
            "completed_at": "2026-08-16T08:01:00+08:00",
        }
    )

    result = map_psychology_snapshot(snapshot)

    assert result.source_status is SourceStatus.INSUFFICIENT_DATA
    assert result.attention_level == "unknown"

