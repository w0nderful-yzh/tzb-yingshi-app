"""Stable App-facing contract for fall-risk summaries."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DecisionPath(StrEnum):
    CAMERA_LED_RADAR_EVIDENCE = "camera_led_radar_evidence"
    CAMERA_ONLY = "camera_only"
    RADAR_ONLY = "radar_only"
    UNAVAILABLE = "unavailable"


class RiskLevel(StrEnum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class PredictionState(StrEnum):
    STABLE = "stable"
    ELEVATED_RISK = "elevated_risk"
    SHORT_TERM_WARNING = "short_term_warning"
    FALL_SUSPECTED = "fall_suspected"
    FALL_DETECTED = "fall_detected"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


class FallEventStatus(StrEnum):
    NONE = "none"
    PREDICTED = "predicted"
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"


class SensorStatus(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class AssociationStatus(StrEnum):
    ASSOCIATED = "associated"
    NOT_ASSOCIATED = "not_associated"
    NOT_REQUIRED = "not_required"
    UNAVAILABLE = "unavailable"


class JointAssessment(StrEnum):
    CORROBORATED_HIGH = "corroborated_high"
    RADAR_SUPPORTS_CAMERA = "radar_supports_camera"
    CAMERA_LED = "camera_led"
    CAMERA_ONLY = "camera_only"
    RADAR_ONLY = "radar_only"
    MODALITY_CONFLICT = "modality_conflict"
    NOT_ASSOCIATED = "not_associated"
    MONITORING = "monitoring"
    UNAVAILABLE = "unavailable"


class CameraStreamStatus(StrEnum):
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class CameraAlgorithmStatus(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    WAITING_DATA = "waiting_data"
    STOPPED = "stopped"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class CameraMonitoringStatus(BaseModel):
    camera_stream_status: CameraStreamStatus
    camera_algorithm_status: CameraAlgorithmStatus
    detail: str
    updated_at: datetime


class RoomFallRisk(BaseModel):
    room_id: str
    room_name: str
    decision_path: DecisionPath
    risk_level: RiskLevel
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    prediction_state: PredictionState
    fall_event_status: FallEventStatus
    camera_status: SensorStatus
    radar_status: SensorStatus
    association_status: AssociationStatus
    joint_assessment: JointAssessment
    evidence_summary: str
    updated_at: datetime | None = None


class FallRiskOverview(BaseModel):
    overall_risk_level: RiskLevel
    rooms: list[RoomFallRisk]
    camera_monitoring: CameraMonitoringStatus
    generated_at: datetime
