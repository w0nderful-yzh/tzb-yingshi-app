"""Internal contracts shared by the Cognitive Collector and Worker."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

CognitiveAssessmentStatus = Literal["processing", "completed", "failed"]


class CognitiveAssessmentSnapshot(BaseModel):
    """Machine-readable worker state; not an Android-facing contract."""

    schema_version: Literal["cognitive_assessment_v1"] = "cognitive_assessment_v1"
    assessment_id: str
    subject_key: str
    session_id: str
    status: CognitiveAssessmentStatus
    window_started_at: datetime
    window_ended_at: datetime | None = None
    effective_speech_seconds: float = Field(ge=0.0)
    estimated_mmse_score: float | None = None
    audio_window_count: int = Field(default=0, ge=0)
    completed_at: datetime | None = None
    failure_code: str | None = None
    failure_message: str | None = None


class CognitiveInferenceJob(BaseModel):
    """Metadata published after enough validated speech has been collected."""

    schema_version: Literal["cognitive_inference_job_v1"] = "cognitive_inference_job_v1"
    assessment_id: str
    subject_key: str
    session_id: str
    device_id: str
    window_started_at: datetime
    window_ended_at: datetime
    effective_speech_seconds: float = Field(ge=0.0)
    sample_rate: Literal[16000] = 16000
    channels: Literal[1] = 1
    sample_width_bytes: Literal[2] = 2
    created_at: datetime
    expires_at: datetime
