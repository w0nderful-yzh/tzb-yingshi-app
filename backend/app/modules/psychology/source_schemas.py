"""Internal contract returned by the psychology algorithm projection service."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PsychologySourceSnapshot(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="ignore")

    schema_version: Literal["psychology_assessment_v1"]
    assessment_id: str = Field(min_length=1, max_length=128)
    subject_key: str = Field(min_length=1, max_length=128)
    status: Literal["processing", "completed", "insufficient_data", "failed"]
    window_started_at: datetime
    window_ended_at: datetime
    estimated_phq8_score: float | None = None
    segment_scores: list[float] = Field(default_factory=list)
    clip_count: int = Field(default=0, ge=0)
    completed_at: datetime | None = None
