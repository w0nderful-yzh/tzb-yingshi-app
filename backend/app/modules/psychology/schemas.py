"""Stable, non-diagnostic App-facing psychology contract."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class SourceStatus(StrEnum):
    AVAILABLE = "available"
    PROCESSING = "processing"
    INSUFFICIENT_DATA = "insufficient_data"
    UNAVAILABLE = "unavailable"


class AssessmentState(StrEnum):
    OBSERVATION_AVAILABLE = "observation_available"
    COLLECTING = "collecting"
    INSUFFICIENT_DATA = "insufficient_data"
    UNAVAILABLE = "unavailable"


class DataQuality(StrEnum):
    USABLE = "usable"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"


class ReviewStatus(StrEnum):
    REQUIRED = "required"
    NOT_AVAILABLE = "not_available"


class PsychologyRiskLevel(StrEnum):
    UNKNOWN = "unknown"
    NO_RISK = "no_risk"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class AssessmentWindow(BaseModel):
    started_at: datetime
    ended_at: datetime


class CompletedAssessmentReference(BaseModel):
    assessment_window: AssessmentWindow
    data_quality: DataQuality
    review_status: ReviewStatus
    estimated_phq8_score: float
    risk_level: PsychologyRiskLevel
    evidence_summary: str
    guidance: str
    updated_at: datetime
    disclaimer: str


class PsychologyOverview(BaseModel):
    source_status: SourceStatus
    operating_mode: Literal["shadow"] = "shadow"
    assessment_state: AssessmentState
    attention_level: Literal["unknown"] = "unknown"
    trend_state: Literal["insufficient_history"] = "insufficient_history"
    data_quality: DataQuality
    source_modality: Literal["camera_behavior"] = "camera_behavior"
    review_status: ReviewStatus
    assessment_window: AssessmentWindow | None = None
    # Raw research-prototype regression outputs are passed through unchanged.
    # The derived level is a non-diagnostic daily-care aid only.
    estimated_phq8_score: float | None = None
    risk_level: PsychologyRiskLevel = PsychologyRiskLevel.UNKNOWN
    segment_scores: list[float] = Field(default_factory=list)
    evidence_summary: str
    guidance: str
    updated_at: datetime | None = None
    disclaimer: str
    latest_completed: CompletedAssessmentReference | None = None
