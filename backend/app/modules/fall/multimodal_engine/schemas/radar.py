from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Room = Literal["living_room", "bedroom", "bathroom"]
SourceMode = Literal["REAL", "REPLAY"]
LegacyModelMode = Literal["TEST_CHECKPOINT", "TRAINED_CHECKPOINT"]
ModelMode = Literal[
    "TEST_CHECKPOINT",
    "TRAINED_CHECKPOINT",
    "RESEARCH_WEAK_SUPERVISION",
]
HumanState = Literal["NO_PERSON", "NORMAL", "FALL_RISK"]
PredictionState = Literal["NORMAL", "WATCH", "IMMINENT", "UNKNOWN"]
RadarGateState = Literal[
    "NORMAL",
    "WATCH",
    "IMMINENT",
    "SUPPRESSED_RECOVERY",
    "CONFIRMED",
    "UNKNOWN",
]
DataQuality = Literal["GOOD", "DEGRADED", "INSUFFICIENT_DATA"]
FallRiskLevel = Literal["LOW", "MODERATE", "HIGH", "UNKNOWN"]


class RadarHealthPayload(BaseModel):
    """Radar FastAPI /health 的受控输入契约。"""

    model_config = ConfigDict(extra="ignore")

    status: Literal["ok", "degraded"]
    radar_connected: bool
    model_loaded: bool
    source_mode: SourceMode | None
    model_mode: ModelMode
    feature_version: str = Field(min_length=1, max_length=64)
    frame_rate_hz: float | None = Field(default=None, ge=0)
    point_count: int | None = Field(default=None, ge=0)


class RadarLatestPayload(BaseModel):
    """Radar FastAPI /api/radar/latest 的受控输入契约。"""

    model_config = ConfigDict(extra="ignore")

    room: Room
    device_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime
    source_mode: SourceMode
    human_state: HumanState
    risk_score: float = Field(ge=0, le=1)
    model_mode: ModelMode
    disclaimer: str | None = Field(default=None, max_length=500)
    event_triggered: bool = False
    research: "RadarResearchShadowPayload | None" = None

    @field_validator("device_id")
    @classmethod
    def device_id_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("device_id must not be blank")
        return normalized

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone offset")
        return value

    @model_validator(mode="after")
    def demo_checkpoint_must_be_disclosed(self) -> "RadarLatestPayload":
        if self.model_mode == "TEST_CHECKPOINT" and not self.disclaimer:
            raise ValueError("TEST_CHECKPOINT result must include a disclaimer")
        return self


class RadarTcnPredictionPayload(BaseModel):
    """Frozen TCN shadow result returned by Radar FastAPI."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["radar_tcn_live_v1"]
    timestamp: datetime
    emitted_at: datetime
    device_id: str = Field(min_length=1, max_length=64)
    room: Room
    source_mode: SourceMode
    risk_state: PredictionState
    pre_fall_score: float = Field(ge=0, le=1)
    score_valid: bool
    consecutive_high_windows: int = Field(ge=0)
    event_triggered: bool = False
    event_id: str | None = Field(default=None, max_length=128)
    unknown_reason: str | None = Field(default=None, max_length=128)
    data_quality: DataQuality
    missing_frame_ratio: float = Field(ge=0, le=1)
    longest_unresolved_gap_seconds: float = Field(ge=0)
    centroid_z: float | None = None
    vertical_velocity: float | None = None
    height_delta_0_6s: float | None = None
    feature_point_count: float | None = Field(default=None, ge=0)
    model_version: str = Field(min_length=1, max_length=128)
    model_mode: Literal["RESEARCH_WEAK_SUPERVISION"]
    architecture: Literal["causal_tcn"]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_version: str = Field(min_length=1, max_length=64)
    threshold: float = Field(gt=0, lt=1)
    threshold_policy: str = Field(min_length=1, max_length=256)
    prediction_horizon_seconds: tuple[float, float]
    positive_anchor: str = Field(min_length=1, max_length=64)
    shadow_only: Literal[True]
    alert_suppressed: Literal[True]
    disclaimer: str = Field(min_length=1, max_length=500)

    @field_validator("device_id")
    @classmethod
    def tcn_device_id_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("device_id must not be blank")
        return normalized

    @field_validator("timestamp", "emitted_at")
    @classmethod
    def tcn_timestamps_need_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("TCN timestamps must include a timezone offset")
        return value

    @field_validator("prediction_horizon_seconds")
    @classmethod
    def tcn_horizon_must_be_ordered(
        cls, value: tuple[float, float]
    ) -> tuple[float, float]:
        if not 0 < value[0] <= value[1]:
            raise ValueError("prediction horizon must be positive and ordered")
        return value


class RadarTcnLatestPayload(BaseModel):
    """TCN-only envelope; no legacy or rule score is accepted here."""

    model_config = ConfigDict(extra="forbid")

    tcn_prediction: RadarTcnPredictionPayload


class RadarPointNetPredictionPayload(BaseModel):
    """PointNet-GRU short-horizon radar evidence returned by Radar FastAPI."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["radar_pointnet_live_v1"]
    timestamp: datetime
    emitted_at: datetime
    device_id: str = Field(min_length=1, max_length=64)
    room: Room
    source_mode: SourceMode
    risk_state: PredictionState
    pre_fall_score: float = Field(ge=0, le=1)
    score_valid: bool
    consecutive_high_windows: int = Field(ge=0)
    event_triggered: bool = False
    event_id: str | None = Field(default=None, max_length=128)
    unknown_reason: str | None = Field(default=None, max_length=128)
    data_quality: DataQuality
    missing_frame_ratio: float = Field(ge=0, le=1)
    observed_frame_count: int = Field(ge=0, le=20)
    point_count: int = Field(ge=0)
    snr_available_fraction: float = Field(ge=0, le=1)
    model_version: str = Field(min_length=1, max_length=128)
    model_variant: str = Field(min_length=1, max_length=128)
    model_mode: Literal["RESEARCH_WEAK_SUPERVISION"]
    architecture: Literal["pointnet_gru"]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_version: str = Field(min_length=1, max_length=64)
    threshold: float = Field(gt=0, lt=1)
    threshold_policy: str = Field(min_length=1, max_length=256)
    prediction_horizon_seconds: tuple[float, float]
    positive_anchor: str = Field(min_length=1, max_length=64)
    shadow_only: Literal[True]
    alert_suppressed: Literal[True]
    disclaimer: str = Field(min_length=1, max_length=500)

    @field_validator("device_id")
    @classmethod
    def pointnet_device_id_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("device_id must not be blank")
        return normalized

    @field_validator("timestamp", "emitted_at")
    @classmethod
    def pointnet_timestamps_need_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("PointNet timestamps must include a timezone offset")
        return value

    @field_validator("prediction_horizon_seconds")
    @classmethod
    def pointnet_horizon_must_be_ordered(
        cls, value: tuple[float, float]
    ) -> tuple[float, float]:
        if not 0 < value[0] <= value[1]:
            raise ValueError("prediction horizon must be positive and ordered")
        return value


class RadarPointNetLatestPayload(BaseModel):
    """PointNet primary radar branch with an optional frozen TCN baseline."""

    model_config = ConfigDict(extra="forbid")

    pointnet_prediction: RadarPointNetPredictionPayload
    tcn_baseline: RadarTcnPredictionPayload | None = None


class RadarCalibratedTcnPredictionPayload(BaseModel):
    """Domain-calibrated TCN shadow result returned by Radar FastAPI."""

    model_config = ConfigDict(extra="ignore")

    schema_version: Literal["radar_calibrated_tcn_live_v1"]
    timestamp: datetime
    emitted_at: datetime
    device_id: str = Field(min_length=1, max_length=64)
    room: Room
    source_mode: SourceMode
    pre_fall_score: float = Field(ge=0, le=1)
    score_valid: bool
    tcn_risk_state: PredictionState
    gate_state: Literal["NORMAL", "WATCH", "IMMINENT", "SUPPRESSED_RECOVERY", "CONFIRMED"]
    formal_alert: bool = False
    suppressed_reason: str | None = Field(default=None, max_length=128)
    recovery_window_active: bool = False
    recovery_count: int = Field(ge=0)
    consecutive_high_windows: int = Field(ge=0)
    threshold_crossed_at: datetime | None = None
    confirmed_at: datetime | None = None
    confirmation_latency_seconds: float | None = Field(default=None, ge=0)
    unknown_reason: str | None = Field(default=None, max_length=128)
    data_quality: DataQuality
    centroid_z: float | None = None
    vertical_velocity: float | None = None
    height_delta_0_6s: float | None = None
    feature_point_count: float | None = Field(default=None, ge=0)
    model_version: str = Field(min_length=1, max_length=128)
    model_mode: Literal["RESEARCH_WEAK_SUPERVISION"]
    architecture: Literal["causal_tcn"]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_version: str = Field(min_length=1, max_length=64)
    threshold: float = Field(gt=0, lt=1)
    prediction_horizon_seconds: tuple[float, float]
    positive_anchor: str = Field(min_length=1, max_length=64)
    shadow_only: Literal[True]
    alert_suppressed: Literal[True]
    disclaimer: str = Field(min_length=1, max_length=500)

    @field_validator("device_id")
    @classmethod
    def calibrated_device_id_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("device_id must not be blank")
        return normalized

    @field_validator("timestamp", "emitted_at")
    @classmethod
    def calibrated_timestamps_need_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("calibrated TCN timestamps must include timezone")
        return value


class RadarDescentPredictionPayload(BaseModel):
    """Descent early-detection shadow result returned by Radar FastAPI."""

    model_config = ConfigDict(extra="ignore")

    schema_version: Literal["radar_descent_live_v1"]
    timestamp: datetime
    emitted_at: datetime
    device_id: str = Field(min_length=1, max_length=64)
    room: Room
    source_mode: SourceMode
    descent_score: float = Field(ge=0, le=1)
    score_valid: bool
    risk_state: PredictionState
    consecutive_high_windows: int = Field(ge=0)
    event_triggered: bool = False
    event_id: str | None = Field(default=None, max_length=128)
    unknown_reason: str | None = Field(default=None, max_length=128)
    data_quality: DataQuality
    model_version: str = Field(min_length=1, max_length=128)
    model_mode: Literal["RESEARCH_DESCENT_DETECTION_V1"]
    architecture: Literal["causal_tcn"]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_version: str = Field(min_length=1, max_length=64)
    threshold: float = Field(gt=0, lt=1)
    prediction_horizon_seconds: tuple[float, float]
    positive_anchor: str = Field(min_length=1, max_length=64)
    shadow_only: Literal[True]
    alert_suppressed: Literal[True]
    disclaimer: str = Field(min_length=1, max_length=500)

    @field_validator("device_id")
    @classmethod
    def descent_device_id_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("device_id must not be blank")
        return normalized

    @field_validator("timestamp", "emitted_at")
    @classmethod
    def descent_timestamps_need_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("descent timestamps must include timezone")
        return value


class RadarFallRiskAssessmentPayload(BaseModel):
    """Rule-based fall-risk assessment shadow result from Radar FastAPI."""

    model_config = ConfigDict(extra="ignore")

    schema_version: Literal["radar_risk_assessment_live_v1"]
    timestamp: str = Field(min_length=1, max_length=64)
    device_id: str = Field(min_length=1, max_length=64)
    room: Room
    risk_level: FallRiskLevel
    risk_score: float | None = Field(default=None, ge=0, le=1)
    sway_risk: float | None = Field(default=None, ge=0, le=1)
    mobility_risk: float | None = Field(default=None, ge=0, le=1)
    descent_risk: float | None = Field(default=None, ge=0, le=1)
    assessment_window_seconds: float = Field(ge=0)
    valid_window_count: int = Field(ge=0)
    observed_duration_seconds: float = Field(ge=0)
    unknown_reason: str | None = Field(default=None, max_length=128)
    shadow_only: Literal[True]
    alert_suppressed: Literal[True]
    disclaimer: str = Field(min_length=1, max_length=500)

    @field_validator("device_id")
    @classmethod
    def risk_assessment_device_id_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("device_id must not be blank")
        return normalized


class RadarEvidencePayload(BaseModel):
    """Independent radar evidence reserved for later multimodal fusion."""

    model_config = ConfigDict(extra="forbid")

    radar_score: float | None = Field(default=None, ge=0, le=1)
    risk_state: RadarGateState
    timestamp: datetime
    room: Room
    device_id: str = Field(min_length=1, max_length=64)
    quality: DataQuality
    model_version: str = Field(min_length=1, max_length=128)

    @field_validator("timestamp")
    @classmethod
    def evidence_timestamp_needs_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("radar evidence timestamp must include a timezone offset")
        return value

    @model_validator(mode="after")
    def unknown_evidence_has_no_score(self) -> "RadarEvidencePayload":
        if self.risk_state == "UNKNOWN" and self.radar_score is not None:
            raise ValueError("UNKNOWN radar evidence must not expose a valid score")
        if self.risk_state != "UNKNOWN" and self.radar_score is None:
            raise ValueError("known radar evidence requires radar_score")
        return self


class RadarSensorMetricsPayload(BaseModel):
    """Observed source metrics; these do not alter model inference."""

    model_config = ConfigDict(extra="forbid")

    frame_rate_hz: float | None = Field(default=None, ge=0)
    point_count: int | None = Field(default=None, ge=0)


class RadarAlignmentEvidencePayload(BaseModel):
    """Per-track geometry already emitted by the Radar service; shadow-only."""

    model_config = ConfigDict(extra="ignore")

    frame_number: int | None = Field(default=None, ge=0)
    source_timestamp: datetime
    track_id: int | None = Field(default=None, ge=0, lt=253)
    x: float | None = None
    y: float | None = None
    z: float | None = None
    vx: float | None = None
    vy: float | None = None
    vz: float | None = None
    point_count: int = Field(default=0, ge=0)
    point_cloud_spread_m: float | None = Field(default=None, ge=0)
    radar_score: float | None = Field(default=None, ge=0, le=1)
    radar_quality: float = Field(default=0.0, ge=0, le=1)
    radar_state: str = Field(default="UNKNOWN", max_length=64)
    radar_config_name: str | None = Field(default=None, max_length=128)
    target_confidence: float | None = Field(default=None, ge=0, le=1)
    shadow_only: Literal[True] = True

    @field_validator("source_timestamp")
    @classmethod
    def alignment_timestamp_needs_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("radar alignment timestamp must include timezone")
        return value


class RadarDebugPayload(BaseModel):
    """Non-authoritative branches retained strictly for engineering diagnosis."""

    model_config = ConfigDict(extra="forbid")

    descent_prediction: RadarDescentPredictionPayload | None = None
    fall_risk_assessment: RadarFallRiskAssessmentPayload | None = None
    affects_risk_state: Literal[False] = False
    affects_alerts: Literal[False] = False


class RadarResearchShadowPayload(BaseModel):
    """Non-alerting v2 research values forwarded to the dashboard."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    prediction_state: PredictionState
    pre_fall_score: float = Field(ge=0, le=1)
    fall_risk_score: float = Field(ge=0, le=1)
    fall_risk_score_5s: float = Field(ge=0, le=1)
    fall_risk_level: FallRiskLevel
    action_risk_event_triggered: bool = False
    data_quality: DataQuality
    threshold: float = Field(gt=0, lt=1)
    prediction_horizon_seconds: tuple[float, float]
    positive_anchor: str = Field(min_length=1, max_length=64)
    rule_components: dict[str, float] = Field(default_factory=dict)
    model_mode: Literal["RESEARCH_WEAK_SUPERVISION"]
    shadow_only: Literal[True]
    alert_suppressed: Literal[True]
    disclaimer: str = Field(min_length=1, max_length=500)

    @field_validator("timestamp")
    @classmethod
    def research_timestamp_needs_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("research timestamp must include a timezone offset")
        return value

    @field_validator("prediction_horizon_seconds")
    @classmethod
    def horizon_must_be_ordered(
        cls, value: tuple[float, float]
    ) -> tuple[float, float]:
        if not 0 < value[0] <= value[1]:
            raise ValueError("prediction horizon must be positive and ordered")
        return value


class RadarStatusResponse(BaseModel):
    """提供给现有Vue页面的最小雷达状态。"""

    model_config = ConfigDict(extra="forbid")

    online: bool
    room: Room | None = None
    device_id: str | None = None
    source_mode: SourceMode | None = None
    model_mode: ModelMode | None = None
    human_state: HumanState | None = None
    risk_score: float | None = Field(default=None, ge=0, le=1)
    timestamp: datetime | None = None
    disclaimer: str | None = None
    research: RadarResearchShadowPayload | None = None
    tcn_prediction: RadarTcnPredictionPayload | None = None
    pointnet_prediction: RadarPointNetPredictionPayload | None = None
    tcn_baseline: RadarTcnPredictionPayload | None = None
    calibrated_tcn_prediction: RadarCalibratedTcnPredictionPayload | None = None
    radar_debug: RadarDebugPayload | None = None
    radar_evidence: RadarEvidencePayload | None = None
    sensor_metrics: RadarSensorMetricsPayload | None = None
    alignment_evidence: list[RadarAlignmentEvidencePayload] = Field(default_factory=list)
    error: str | None = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
