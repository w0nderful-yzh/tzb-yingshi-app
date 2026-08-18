"""Machine-readable contracts produced by the existing offline inference chain."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AssessmentStatus = Literal["processing", "completed", "insufficient_data", "failed"]


class PsychologyAssessmentSnapshot(BaseModel):
    """Internal algorithm result. Raw model values must not be sent to Android."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal["psychology_assessment_v1"] = "psychology_assessment_v1"
    assessment_id: str = Field(min_length=1, max_length=128)
    subject_key: str = Field(min_length=1, max_length=128)
    status: AssessmentStatus
    window_started_at: datetime
    window_ended_at: datetime
    estimated_phq8_score: float | None = None
    segment_scores: list[float] = Field(default_factory=list)
    clip_count: int = Field(default=0, ge=0)
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "PsychologyAssessmentSnapshot":
        if self.window_ended_at < self.window_started_at:
            raise ValueError("window_ended_at must not precede window_started_at")
        if self.status == "completed":
            if self.estimated_phq8_score is None or not self.segment_scores:
                raise ValueError("completed assessment requires model scores")
            if self.clip_count < 7:
                raise ValueError("completed assessment requires at least seven clips")
            if self.completed_at is None:
                raise ValueError("completed assessment requires completed_at")
        return self


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    inference_triggered_by_api: Literal[False] = False

