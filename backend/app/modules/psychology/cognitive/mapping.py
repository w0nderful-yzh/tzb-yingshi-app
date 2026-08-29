"""Map Cognitive worker snapshots to a non-diagnostic App-facing contract."""

from math import isfinite

from app.modules.psychology.cognitive.schemas import (
    CognitiveAssessmentSnapshot,
    CognitiveAssessmentWindow,
    CognitiveCompletedReference,
    CognitiveDataQuality,
    CognitiveOverview,
    CognitiveState,
)

_DISCLAIMER = "AI辅助认知状态评估仅供日常关怀参考，不构成认知障碍或医疗诊断"
_GUIDANCE = "建议结合日常沟通、生活表现和专业人员意见进行综合关注"


def map_cognitive_snapshot(
    snapshot: CognitiveAssessmentSnapshot,
    *,
    latest_completed: CognitiveAssessmentSnapshot | None = None,
) -> CognitiveOverview:
    window = CognitiveAssessmentWindow(
        started_at=snapshot.window_started_at,
        ended_at=snapshot.window_ended_at,
    )
    updated_at = snapshot.completed_at or snapshot.window_ended_at or snapshot.window_started_at
    completed_reference = _completed_reference(latest_completed)

    if snapshot.status == "processing":
        return CognitiveOverview(
            source_status=CognitiveState.PROCESSING,
            assessment_state=CognitiveState.PROCESSING,
            data_quality=CognitiveDataQuality.LIMITED,
            assessment_window=window,
            evidence_summary="正在采集语音声学资料并等待辅助分析",
            guidance=_GUIDANCE,
            updated_at=updated_at,
            disclaimer=_DISCLAIMER,
            latest_completed=completed_reference,
        )
    if snapshot.status == "insufficient_data":
        return CognitiveOverview(
            source_status=CognitiveState.INSUFFICIENT_DATA,
            assessment_state=CognitiveState.INSUFFICIENT_DATA,
            data_quality=CognitiveDataQuality.INSUFFICIENT,
            assessment_window=window,
            evidence_summary="本次有效语音不足60秒，暂无法形成认知状态辅助评估",
            guidance=_GUIDANCE,
            updated_at=updated_at,
            disclaimer=_DISCLAIMER,
            latest_completed=completed_reference,
        )
    if snapshot.status == "failed" or not _valid_completed(snapshot):
        return CognitiveOverview(
            source_status=CognitiveState.FAILED,
            assessment_state=CognitiveState.FAILED,
            data_quality=CognitiveDataQuality.INSUFFICIENT,
            assessment_window=window,
            evidence_summary="本次语音声学分析未完成，可在后续守护时重新评估",
            guidance=_GUIDANCE,
            updated_at=updated_at,
            disclaimer=_DISCLAIMER,
            latest_completed=completed_reference,
        )

    assert snapshot.estimated_mmse_score is not None
    return CognitiveOverview(
        source_status=CognitiveState.COMPLETED,
        assessment_state=CognitiveState.COMPLETED,
        data_quality=_completed_quality(snapshot),
        assessment_window=window,
        estimated_mmse_score=snapshot.estimated_mmse_score,
        evidence_summary="语音声学特征辅助分析已完成",
        guidance=_GUIDANCE,
        updated_at=updated_at,
        disclaimer=_DISCLAIMER,
    )


def unavailable_cognitive_overview() -> CognitiveOverview:
    return CognitiveOverview(
        source_status=CognitiveState.UNAVAILABLE,
        assessment_state=CognitiveState.UNAVAILABLE,
        data_quality=CognitiveDataQuality.INSUFFICIENT,
        evidence_summary="认知状态辅助评估服务暂不可用",
        guidance=_GUIDANCE,
        disclaimer=_DISCLAIMER,
    )


def _valid_completed(snapshot: CognitiveAssessmentSnapshot) -> bool:
    score = snapshot.estimated_mmse_score
    return (
        snapshot.status == "completed"
        and score is not None
        and isfinite(score)
        and 0.0 <= score <= 30.0
        and snapshot.effective_speech_seconds >= 60.0
        and snapshot.audio_window_count > 0
        and snapshot.completed_at is not None
    )


def _completed_quality(snapshot: CognitiveAssessmentSnapshot) -> CognitiveDataQuality:
    if snapshot.effective_speech_seconds >= 120.0:
        return CognitiveDataQuality.USABLE
    return CognitiveDataQuality.LIMITED


def _completed_reference(
    snapshot: CognitiveAssessmentSnapshot | None,
) -> CognitiveCompletedReference | None:
    if snapshot is None or not _valid_completed(snapshot):
        return None
    assert snapshot.estimated_mmse_score is not None
    assert snapshot.completed_at is not None
    return CognitiveCompletedReference(
        assessment_window=CognitiveAssessmentWindow(
            started_at=snapshot.window_started_at,
            ended_at=snapshot.window_ended_at,
        ),
        estimated_mmse_score=snapshot.estimated_mmse_score,
        data_quality=_completed_quality(snapshot),
        evidence_summary="上一轮语音声学特征辅助分析已完成",
        updated_at=snapshot.completed_at,
        disclaimer=_DISCLAIMER,
    )
