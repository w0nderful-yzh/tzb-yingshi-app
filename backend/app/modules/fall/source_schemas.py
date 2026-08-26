"""Exact, minimal contracts for the existing fall algorithm HTTP responses.

Only fields used by the App adapter are declared. Experiment-only fields are
ignored at this boundary and can remain in upstream debug logs.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QualityLevel = Literal["GOOD", "DEGRADED", "INSUFFICIENT_DATA"]
CameraRiskState = Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
RadarTcnRiskState = Literal["NORMAL", "WATCH", "IMMINENT", "UNKNOWN"]
RadarGateState = Literal[
    "NORMAL",
    "WATCH",
    "IMMINENT",
    "SUPPRESSED_RECOVERY",
    "CONFIRMED",
]
AlignmentAssociationState = Literal[
    "MATCHED",
    "OUT_OF_SYNC",
    "CAMERA_PERSON_MISSING",
    "RADAR_TRACK_MISSING",
    "MULTIPLE_CANDIDATES",
    "TRACK_CONFLICT",
    "CALIBRATION_INVALID",
]
AssociatedEvidenceState = Literal[
    "UNKNOWN",
    "NOT_ASSOCIATED",
    "NORMAL_CORROBORATED",
    "CAMERA_ONLY_NORMAL",
    "CAMERA_ONLY_WATCH",
    "CAMERA_ONLY_HIGH",
    "CORROBORATED_WATCH",
    "CORROBORATED_HIGH",
    "RADAR_MOTION_ANOMALY",
    "MODALITY_CONFLICT",
]
CameraLedEvidenceFusionMode = Literal[
    "CAMERA_ONLY",
    "RADAR_SUPPORTED",
    "CAMERA_RADAR_CONSISTENT",
    "RADAR_CONFLICT",
    "LOW_CONFIDENCE",
]
FusionStableState = Literal["UNKNOWN", "NORMAL", "WATCH", "HIGH", "IMMINENT"]
FallLiveState = Literal[
    "DISABLED",
    "STARTING",
    "LOADING_MODELS",
    "CONNECTING",
    "RUNNING",
    "ERROR",
    "STOPPED",
]
FallLiveInputState = Literal[
    "WAITING",
    "NO_PERSON",
    "LOW_CONFIDENCE",
    "INSUFFICIENT_DENSITY",
    "INSUFFICIENT_POSE",
    "READY",
]
GuardSessionState = Literal["STOPPED", "STARTING", "ACTIVE", "DEGRADED"]
GuardCapabilityState = Literal[
    "STOPPED",
    "STARTING",
    "RUNNING",
    "DEGRADED",
    "UNAVAILABLE",
]


class AlgorithmSourceModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class CameraAlgorithmEvidence(AlgorithmSourceModel):
    camera_score: float | None = Field(default=None, ge=0.0, le=1.0)
    camera_risk_state: CameraRiskState = "UNKNOWN"
    quality_level: QualityLevel
    timestamp: datetime
    available: bool


class RadarAlgorithmEvidence(AlgorithmSourceModel):
    radar_score: float | None = Field(default=None, ge=0.0, le=1.0)
    radar_risk_state: RadarGateState | Literal["UNKNOWN"] = "UNKNOWN"
    quality_level: QualityLevel
    timestamp: datetime
    available: bool
    room: str | None = None


class AlignmentAlgorithmEvidence(AlgorithmSourceModel):
    association_state: AlignmentAssociationState = "CALIBRATION_INVALID"
    eligible_for_temporal_association: bool = False


class AssociatedRiskAugmentation(AlgorithmSourceModel):
    associated_short_term_fall_score: float | None = Field(default=None, ge=0.0, le=1.0)
    associated_risk_state: Literal["UNKNOWN", "NORMAL", "WATCH", "HIGH", "IMMINENT"] = "UNKNOWN"
    associated_evidence_state: AssociatedEvidenceState = "UNKNOWN"
    base_camera_score: float | None = Field(default=None, ge=0.0, le=1.0)
    base_camera_state: CameraRiskState = "UNKNOWN"
    radar_motion_evidence_strength: Literal["NONE", "WEAK", "STRONG", "UNKNOWN"] = "UNKNOWN"
    association_state: AlignmentAssociationState = "CALIBRATION_INVALID"
    shadow_only: bool = True
    affects_alerts: bool = False
    camera_score_unchanged: bool = True
    uses_radar_tcn_score: bool = False


class CameraLedEvidenceFusionV2Source(AlgorithmSourceModel):
    schema_version: Literal["camera_led_evidence_fusion_v2"]
    camera_led_score: float | None = Field(default=None, ge=0.0, le=1.0)
    camera_led_state: FusionStableState = "UNKNOWN"
    fusion_mode: CameraLedEvidenceFusionMode = "LOW_CONFIDENCE"
    camera_score: float | None = Field(default=None, ge=0.0, le=1.0)
    radar_score: float | None = Field(default=None, ge=0.0, le=1.0)
    radar_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    radar_eligible: bool = False
    radar_motion_evidence_strength: Literal["NONE", "WEAK", "STRONG", "UNKNOWN"] = (
        "UNKNOWN"
    )
    association_state: AlignmentAssociationState = "CALIBRATION_INVALID"
    sync_delta_ms: float | None = Field(default=None, ge=0.0)
    reason_codes: list[str] = Field(default_factory=list)
    model_version: Literal["camera-led-evidence-fusion-v2-realtime-v1"]
    realtime_active: Literal[True]
    shadow_only: Literal[False]
    affects_app_result: Literal[True]
    affects_alerts: Literal[False]


class AlgorithmFallEvent(AlgorithmSourceModel):
    fall_event_status: Literal["NO_EVENT", "SUSPECTED", "CONFIRMED", "UNKNOWN"] = "UNKNOWN"
    summary: str


class CameraLedSourceSnapshot(AlgorithmSourceModel):
    camera: CameraAlgorithmEvidence
    radar: RadarAlgorithmEvidence
    alignment: AlignmentAlgorithmEvidence
    associated_risk_augmentation: AssociatedRiskAugmentation | None = None
    camera_led_evidence_fusion_v2: CameraLedEvidenceFusionV2Source
    fall_event: AlgorithmFallEvent
    timestamp: datetime


class CameraMonitoringSourceStatus(AlgorithmSourceModel):
    enabled: bool
    state: FallLiveState
    input_state: FallLiveInputState = "WAITING"
    input_message: str = "等待摄像头输入"
    error: str | None = None
    checked_at: datetime


class GuardCapabilitySourceStatus(AlgorithmSourceModel):
    state: GuardCapabilityState
    enabled_for_session: bool
    detail: str


class GuardSessionSourceStatus(AlgorithmSourceModel):
    schema_version: Literal["multimodal_guard_session_v1"]
    session_id: str | None = None
    active: bool
    state: GuardSessionState
    camera_analysis: GuardCapabilitySourceStatus
    radar_worker: GuardCapabilitySourceStatus
    radar_participation: GuardCapabilitySourceStatus
    fusion: GuardCapabilitySourceStatus
    reason_codes: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    updated_at: datetime


class RadarTcnPredictionSource(AlgorithmSourceModel):
    schema_version: Literal["radar_tcn_live_v1"]
    timestamp: datetime
    device_id: str
    room: str
    risk_state: RadarTcnRiskState
    pre_fall_score: float = Field(ge=0.0, le=1.0)
    score_valid: bool
    event_triggered: bool = False
    data_quality: QualityLevel
    shadow_only: bool
    alert_suppressed: bool


class RadarCalibratedTcnPredictionSource(AlgorithmSourceModel):
    schema_version: Literal["radar_calibrated_tcn_live_v1"]
    timestamp: datetime
    device_id: str
    room: str
    pre_fall_score: float = Field(ge=0.0, le=1.0)
    score_valid: bool
    tcn_risk_state: RadarTcnRiskState
    gate_state: RadarGateState
    formal_alert: bool = False
    data_quality: QualityLevel
    shadow_only: bool
    alert_suppressed: bool


type RadarOnlySourceSnapshot = RadarTcnPredictionSource | RadarCalibratedTcnPredictionSource


class RadarLatestResponse(AlgorithmSourceModel):
    calibrated_tcn_prediction: RadarCalibratedTcnPredictionSource | None = None
    tcn_prediction: RadarTcnPredictionSource | None = None
    tcn_baseline: RadarTcnPredictionSource | None = None
