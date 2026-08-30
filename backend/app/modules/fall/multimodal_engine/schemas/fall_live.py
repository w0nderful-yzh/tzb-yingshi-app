from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.modules.fall.multimodal_engine.schemas.risk_event import RiskLevel


class FallLiveState(str, Enum):
    DISABLED = "DISABLED"
    STARTING = "STARTING"
    LOADING_MODELS = "LOADING_MODELS"
    CONNECTING = "CONNECTING"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


class FallLiveInputState(str, Enum):
    WAITING = "WAITING"
    NO_PERSON = "NO_PERSON"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INSUFFICIENT_DENSITY = "INSUFFICIENT_DENSITY"
    INSUFFICIENT_POSE = "INSUFFICIENT_POSE"
    READY = "READY"


class PhysioHrvMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    RMSSD: float | None = Field(default=None, ge=0)
    SDNN: float | None = Field(default=None, ge=0)
    LF: float | None = Field(default=None, ge=0)
    HF: float | None = Field(default=None, ge=0)
    LF_HF: float | None = Field(default=None, ge=0)


class RppgLiveStatus(BaseModel):
    """Camera-frame physiological sidecar; never an alert input."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    available: bool = False
    assessment_ready: bool = False
    heart_rate: float | None = Field(default=None, ge=0, le=250)
    sqi: float | None = Field(default=None, ge=0, le=1)
    hrv: PhysioHrvMetrics = Field(default_factory=PhysioHrvMetrics)
    physio_level: str = "UNKNOWN"
    physio_abnormal: bool = False
    abnormal_reasons: list[str] = Field(default_factory=list)
    valid_seconds: float = Field(default=0.0, ge=0)
    frames_fed: int = Field(default=0, ge=0)
    frames_sqi_ok: int = Field(default=0, ge=0)
    quality_coverage: float = Field(default=0.0, ge=0, le=1)
    quality_reason: str = "RPPG_DISABLED"
    source_timestamp: datetime | None = None
    error_hint: str | None = None
    recent_errors: list[str] = Field(default_factory=list)
    shadow_only: bool = True
    affects_fusion: bool = False
    affects_dynamic_risk: bool = False
    affects_short_term_fall: bool = False
    affects_fall_event: bool = False


class FallLiveStatusResponse(BaseModel):
    enabled: bool
    state: FallLiveState
    device_id: str | None = None
    model_device: str | None = None
    model_version: str | None = None
    risk_score: float | None = Field(default=None, ge=0, le=1)
    risk_level: RiskLevel | None = None
    positive_votes: int | None = Field(default=None, ge=0)
    torso_inclination_deg: float | None = Field(default=None, ge=0)
    com_proxy_relative_change: float | None = Field(default=None, ge=0)
    yaw_delta_deg: float | None = None
    pose_quality: float | None = Field(default=None, ge=0, le=1)
    input_state: FallLiveInputState = FallLiveInputState.WAITING
    input_message: str = "等待摄像头输入"
    target_present: bool = False
    training_input_ready: bool = False
    frames_ready: int = Field(default=0, ge=0)
    source_window_frames: int = Field(default=0, ge=0)
    valid_pose_frames: int = Field(default=0, ge=0)
    required_source_frames: int = Field(default=45, ge=2)
    effective_sample_fps: float = Field(default=0.0, ge=0)
    mean_keypoint_confidence: float | None = Field(default=None, ge=0, le=1)
    latest_keypoint_confidence: float | None = Field(default=None, ge=0, le=1)
    max_source_gap_frames: int = Field(default=0, ge=0)
    captured_frames: int = Field(default=0, ge=0)
    processed_frames: int = Field(default=0, ge=0)
    dropped_frames: int = Field(default=0, ge=0)
    queue_dropped_frames: int = Field(default=0, ge=0)
    invalid_image_frames: int = Field(default=0, ge=0)
    no_person_frames: int = Field(default=0, ge=0)
    low_confidence_frames: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    processing_fps: float | None = Field(default=None, ge=0)
    pipeline_latency_seconds: float | None = Field(default=None, ge=0)
    source_fps: float | None = Field(default=None, ge=0)
    last_prediction_at: datetime | None = None
    last_event_id: str | None = None
    rppg: RppgLiveStatus = Field(default_factory=RppgLiveStatus)
    alignment_snapshot: "CameraAlignmentSnapshot | None" = None
    error: str | None = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CameraAlignmentSnapshot(BaseModel):
    """Compact shadow geometry from the same frame used by Camera inference."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(min_length=1, max_length=128)
    source_timestamp: datetime
    camera_person_id: int | None = Field(default=None, ge=0)
    detected: bool = False
    image_size: tuple[int, int] | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    footpoint_uv: tuple[float, float] | None = None
    footpoint_confidence: float = Field(default=0.0, ge=0, le=1)
    footpoint_source: str | None = Field(default=None, max_length=64)
    shadow_only: Literal[True] = True

    @field_validator("source_timestamp")
    @classmethod
    def source_timestamp_needs_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("camera alignment timestamp must include timezone")
        return value


FallLiveStatusResponse.model_rebuild()


class BrowserFallFrameRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=128)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    frame_base64: str = Field(min_length=32)


class BrowserFallFrameResponse(BaseModel):
    accepted: bool
    queue_depth: int = Field(ge=0)
