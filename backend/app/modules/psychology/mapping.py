"""Map raw reference scores to honest, non-diagnostic product states."""

from datetime import datetime
from math import isfinite

from app.modules.psychology.schemas import (
    AssessmentState,
    AssessmentWindow,
    CompletedAssessmentReference,
    DataQuality,
    PsychologyOverview,
    PsychologyRiskLevel,
    ReviewStatus,
    SourceStatus,
)
from app.modules.psychology.source_schemas import PsychologySourceSnapshot

_GUIDANCE = "结果仅供日常关怀参考，建议结合日常沟通和专业评估"
_DISCLAIMER = "该结果仅供日常关怀参考，不构成心理或医疗诊断"


def map_psychology_snapshot(
    snapshot: PsychologySourceSnapshot,
    *,
    latest_completed: PsychologySourceSnapshot | None = None,
) -> PsychologyOverview:
    window = AssessmentWindow(
        started_at=snapshot.window_started_at,
        ended_at=snapshot.window_ended_at,
    )
    if snapshot.status == "processing":
        return _overview(
            source_status=SourceStatus.PROCESSING,
            assessment_state=AssessmentState.COLLECTING,
            data_quality=DataQuality.LIMITED,
            review_status=ReviewStatus.NOT_AVAILABLE,
            window=window,
            summary="正在采集并分析视觉行为资料",
            updated_at=snapshot.completed_at,
            latest_completed=_completed_reference(latest_completed),
        )
    if snapshot.status == "insufficient_data":
        return _overview(
            source_status=SourceStatus.INSUFFICIENT_DATA,
            assessment_state=AssessmentState.INSUFFICIENT_DATA,
            data_quality=DataQuality.INSUFFICIENT,
            review_status=ReviewStatus.NOT_AVAILABLE,
            window=window,
            summary="本次资料不足，暂无法形成参考观察",
            updated_at=snapshot.completed_at,
        )
    if snapshot.status != "completed" or not _valid_completed_result(snapshot):
        return unavailable_overview(
            window=window,
            updated_at=snapshot.completed_at,
        )
    return _overview(
        source_status=SourceStatus.AVAILABLE,
        assessment_state=AssessmentState.OBSERVATION_AVAILABLE,
        # Phase 1 has no face-coverage contract, so never overstate technical quality.
        data_quality=DataQuality.LIMITED,
        review_status=ReviewStatus.REQUIRED,
        window=window,
        summary="近期视觉行为资料已完成参考分析",
        updated_at=snapshot.completed_at,
        score=snapshot.estimated_phq8_score,
        segment_scores=snapshot.segment_scores,
    )


def unavailable_overview(
    *,
    window: AssessmentWindow | None = None,
    updated_at: datetime | None = None,
) -> PsychologyOverview:
    return _overview(
        source_status=SourceStatus.UNAVAILABLE,
        assessment_state=AssessmentState.UNAVAILABLE,
        data_quality=DataQuality.INSUFFICIENT,
        review_status=ReviewStatus.NOT_AVAILABLE,
        window=window,
        summary="心理观察服务暂不可用",
        updated_at=updated_at,
    )


def _valid_completed_result(snapshot: PsychologySourceSnapshot) -> bool:
    score = snapshot.estimated_phq8_score
    return (
        score is not None
        and isfinite(score)
        and 0.0 <= score <= 24.0
        and bool(snapshot.segment_scores)
        and all(isfinite(value) for value in snapshot.segment_scores)
        and snapshot.clip_count >= 7
        and snapshot.completed_at is not None
    )


def _risk_level_for_score(score: float | None) -> PsychologyRiskLevel:
    if score is None:
        return PsychologyRiskLevel.UNKNOWN
    if score < 10.0:
        return PsychologyRiskLevel.NO_RISK
    if score < 15.0:
        return PsychologyRiskLevel.MILD
    if score < 20.0:
        return PsychologyRiskLevel.MODERATE
    return PsychologyRiskLevel.SEVERE


def _overview(
    *,
    source_status: SourceStatus,
    assessment_state: AssessmentState,
    data_quality: DataQuality,
    review_status: ReviewStatus,
    window: AssessmentWindow | None,
    summary: str,
    updated_at: datetime | None,
    score: float | None = None,
    segment_scores: list[float] | None = None,
    latest_completed: CompletedAssessmentReference | None = None,
) -> PsychologyOverview:
    return PsychologyOverview(
        source_status=source_status,
        assessment_state=assessment_state,
        data_quality=data_quality,
        review_status=review_status,
        assessment_window=window,
        estimated_phq8_score=score,
        risk_level=_risk_level_for_score(score),
        segment_scores=segment_scores or [],
        evidence_summary=summary,
        guidance=_GUIDANCE,
        updated_at=updated_at,
        disclaimer=_DISCLAIMER,
        latest_completed=latest_completed,
    )


def _completed_reference(
    snapshot: PsychologySourceSnapshot | None,
) -> CompletedAssessmentReference | None:
    if snapshot is None or snapshot.status != "completed" or not _valid_completed_result(snapshot):
        return None
    assert snapshot.estimated_phq8_score is not None
    assert snapshot.completed_at is not None
    return CompletedAssessmentReference(
        assessment_window=AssessmentWindow(
            started_at=snapshot.window_started_at,
            ended_at=snapshot.window_ended_at,
        ),
        data_quality=DataQuality.LIMITED,
        review_status=ReviewStatus.REQUIRED,
        estimated_phq8_score=snapshot.estimated_phq8_score,
        risk_level=_risk_level_for_score(snapshot.estimated_phq8_score),
        evidence_summary="上一轮视觉行为资料已完成参考分析",
        guidance=_GUIDANCE,
        updated_at=snapshot.completed_at,
        disclaimer=_DISCLAIMER,
    )
