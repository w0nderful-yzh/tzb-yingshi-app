"""Internal contracts shared by the Cognitive Collector and Worker."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

CognitiveAssessmentStatus = Literal[
    "processing",
    "completed",
    "failed",
    "insufficient_data",
]


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


class CognitiveState(str, Enum):  # noqa: UP042 - Python 3.10 worker compatibility.
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"
    UNAVAILABLE = "unavailable"


class CognitiveDataQuality(str, Enum):  # noqa: UP042 - Python 3.10 worker compatibility.
    USABLE = "usable"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"


class CognitiveAttentionLevel(str, Enum):  # noqa: UP042 - Python 3.10 worker compatibility.
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    HIGH = "high"


class CognitiveAssessmentWindow(BaseModel):
    started_at: datetime
    ended_at: datetime | None = None


class CognitiveCompletedReference(BaseModel):
    assessment_window: CognitiveAssessmentWindow
    estimated_mmse_score: float
    attention_level: CognitiveAttentionLevel
    data_quality: CognitiveDataQuality
    source_modality: Literal["voice_acoustic"] = "voice_acoustic"
    evidence_summary: str
    updated_at: datetime
    disclaimer: str


class CognitiveOverview(BaseModel):
    source_status: CognitiveState
    assessment_state: CognitiveState
    data_quality: CognitiveDataQuality
    source_modality: Literal["voice_acoustic"] = "voice_acoustic"
    assessment_window: CognitiveAssessmentWindow | None = None
    estimated_mmse_score: float | None = None
    attention_level: CognitiveAttentionLevel | None = None
    evidence_summary: str
    guidance: str
    updated_at: datetime | None = None
    disclaimer: str
    latest_completed: CognitiveCompletedReference | None = None
