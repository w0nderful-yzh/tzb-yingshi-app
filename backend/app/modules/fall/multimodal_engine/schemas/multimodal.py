from datetime import datetime, timezone
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


QualityLevel = Literal["GOOD", "DEGRADED", "INSUFFICIENT_DATA"]
MultimodalDataSource = Literal["REAL_CAMERA_RADAR", "PUBLIC_EVIDENCE_REPLAY"]
FusionRiskLevel = Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
FusionMethod = Literal[
    "fixed_weighted",
    "quality_weighted",
    "radar_quality_adaptive",
    "mlp",
    "camera_led_evidence_v2",
]
FusionStableState = Literal["UNKNOWN", "NORMAL", "WATCH", "HIGH", "IMMINENT"]
FusionOperatingState = Literal[
    "NORMAL_FUSION",
    "CAMERA_ONLY",
    "RADAR_ONLY",
    "RADAR_CONFLICT",
    "LOW_CONFIDENCE",
    "NO_EVIDENCE",
]
FusionDegradedMode = Literal[
    "NONE",
    "CAMERA_ONLY",
    "RADAR_ONLY",
    "OUT_OF_SYNC",
    "LOW_QUALITY",
    "MODALITY_CONFLICT",
    "BOTH_UNAVAILABLE",
]
DominantModality = Literal["CAMERA", "RADAR", "BALANCED", "NONE"]
TargetAssociationState = Literal[
    "MATCHED",
    "SINGLE_TARGET_ASSUMED",
    "CONFLICT",
    "UNKNOWN",
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
TemporalRelation = Literal[
    "ALIGNED",
    "RADAR_THEN_CAMERA",
    "CAMERA_THEN_RADAR",
    "NO_RISK_SEQUENCE",
    "INSUFFICIENT_EVIDENCE",
]
RadarMotionEvidenceStrength = Literal["NONE", "WEAK", "STRONG", "UNKNOWN"]
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
DynamicRiskLevel = Literal["LOW", "MODERATE", "HIGH", "UNKNOWN"]
FallEventState = Literal["NO_EVENT", "SUSPECTED", "CONFIRMED", "UNKNOWN"]
FeatureScalar: TypeAlias = float | int | bool | str | None
EvidenceFeature: TypeAlias = list[float] | dict[str, FeatureScalar]


class ExplainableRiskReason(BaseModel):
    """Human-readable evidence without implying a clinical diagnosis."""

    model_config = ConfigDict(extra="forbid")

    code: Literal[
        "POSTURE_ABNORMAL",
        "ACTIVITY_ABNORMAL",
        "MOTION_STABILITY_DECLINE",
        "RAPID_HEIGHT_CHANGE",
        "MODALITY_QUALITY_DECLINE",
    ]
    label: str = Field(min_length=1, max_length=64)
    source: Literal["CAMERA", "RADAR", "MULTIMODAL"]
    signal_value: float | None = Field(default=None, ge=0, le=1)
    detail: str = Field(min_length=1, max_length=256)
    affects_score: bool


class DynamicRiskIndex(BaseModel):
    """Continuous risk assessment, explicitly separate from short-term warning."""

    model_config = ConfigDict(extra="forbid")

    score_name: Literal["dynamic risk index"] = "dynamic risk index"
    dynamic_risk_score: float | None = Field(default=None, ge=0, le=1)
    risk_level: DynamicRiskLevel = "UNKNOWN"
    assessment_window_seconds: float = Field(default=0.0, ge=0)
    valid_window_count: int = Field(default=0, ge=0)
    observed_duration_seconds: float = Field(default=0.0, ge=0)
    source_method: Literal[
        "radar_60s_shadow_with_camera_context_v1",
        "UNAVAILABLE",
    ] = "UNAVAILABLE"
    components: dict[str, float | None] = Field(default_factory=dict)
    reasons: list[ExplainableRiskReason] = Field(default_factory=list)
    available: bool = False
    shadow_only: Literal[True] = True
    camera_context_affects_score: Literal[False] = False
    score_interpretation: Literal["SCREENING_INDEX_NOT_DIAGNOSIS"] = (
        "SCREENING_INDEX_NOT_DIAGNOSIS"
    )
    disclaimer: str = (
        "动态风险指数是持续运动学风险筛查信号，不是临床诊断或未来跌倒概率。"
    )

    @model_validator(mode="after")
    def availability_matches_dynamic_score(self) -> "DynamicRiskIndex":
        if self.available and self.dynamic_risk_score is None:
            raise ValueError("available dynamic risk index requires a score")
        if not self.available and self.dynamic_risk_score is not None:
            raise ValueError("unavailable dynamic risk index cannot expose a score")
        if not self.available and self.risk_level != "UNKNOWN":
            raise ValueError("unavailable dynamic risk index must be UNKNOWN")
        return self


class ShortTermFallWarning(BaseModel):
    """Seconds-scale warning exposed as the active realtime algorithm result."""

    model_config = ConfigDict(extra="forbid")

    score_name: Literal["short-term fall risk score"] = "short-term fall risk score"
    short_term_fall_score: float | None = Field(default=None, ge=0, le=1)
    state: FusionStableState = "UNKNOWN"
    method: FusionMethod = "fixed_weighted"
    degraded_mode: FusionDegradedMode = "BOTH_UNAVAILABLE"
    synchronized: bool = False
    reasons: list[ExplainableRiskReason] = Field(default_factory=list)
    score_interpretation: Literal["RISK_SCORE_NOT_EVENT_CONFIRMATION"] = (
        "RISK_SCORE_NOT_EVENT_CONFIRMATION"
    )


class FallEventSummary(BaseModel):
    """Observed event state; warning levels never become a confirmed fall by themselves."""

    model_config = ConfigDict(extra="forbid")

    fall_event_status: FallEventState = "UNKNOWN"
    source_event_id: str | None = Field(default=None, max_length=128)
    source: Literal["CAMERA", "RADAR", "MULTIMODAL", "NONE"] = "NONE"
    reason_codes: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1, max_length=256)
    requires_human_confirmation: bool = False


class PhysiologicalEvidence(BaseModel):
    """rPPG sidecar evidence; deliberately excluded from all fall decisions."""

    model_config = ConfigDict(extra="forbid")

    score_name: Literal["rppg physiological evidence"] = "rppg physiological evidence"
    enabled: bool = False
    available: bool = False
    assessment_ready: bool = False
    heart_rate: float | None = Field(default=None, ge=0, le=250)
    sqi: float | None = Field(default=None, ge=0, le=1)
    hrv: dict[str, float | None] = Field(default_factory=dict)
    physio_level: Literal["NORMAL", "ABNORMAL", "UNKNOWN"] = "UNKNOWN"
    abnormal_reasons: list[str] = Field(default_factory=list)
    valid_seconds: float = Field(default=0.0, ge=0)
    quality_coverage: float = Field(default=0.0, ge=0, le=1)
    quality_level: QualityLevel = "INSUFFICIENT_DATA"
    quality_reason: str = Field(default="RPPG_DISABLED", max_length=128)
    timestamp: datetime | None = None
    shadow_only: Literal[True] = True
    affects_fusion: Literal[False] = False
    affects_dynamic_risk: Literal[False] = False
    affects_short_term_fall: Literal[False] = False
    affects_fall_event: Literal[False] = False
    score_interpretation: Literal["PHYSIOLOGICAL_SIGNAL_NOT_FALL_RISK_SCORE"] = (
        "PHYSIOLOGICAL_SIGNAL_NOT_FALL_RISK_SCORE"
    )
    disclaimer: str = (
        "rPPG 是非接触式生理状态观察信号，不是跌倒概率、临床诊断或告警触发条件。"
    )

    @model_validator(mode="after")
    def unknown_until_assessment_is_ready(self) -> "PhysiologicalEvidence":
        if not self.assessment_ready and self.physio_level != "UNKNOWN":
            raise ValueError("unready rPPG evidence must remain UNKNOWN")
        if self.assessment_ready and not self.available:
            raise ValueError("rPPG assessment cannot be ready while unavailable")
        return self


class FinalDecisionContext(BaseModel):
    """Post-fusion context. It can request review but cannot rewrite fall decisions."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal["POST_FUSION_DECISION_CONTEXT"] = "POST_FUSION_DECISION_CONTEXT"
    base_short_term_state: FusionStableState = "UNKNOWN"
    base_fall_event_status: FallEventState = "UNKNOWN"
    physiological_context: PhysiologicalEvidence = Field(
        default_factory=PhysiologicalEvidence
    )
    physiological_review_level: Literal[
        "NO_ADDITIONAL_CONCERN",
        "MANUAL_REVIEW_SUGGESTED",
        "UNAVAILABLE",
    ] = "UNAVAILABLE"
    human_review_suggested: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1, max_length=256)
    affects_fusion_score: Literal[False] = False
    affects_dynamic_risk_score: Literal[False] = False
    affects_short_term_fall_score: Literal[False] = False
    affects_fall_event_status: Literal[False] = False
    can_trigger_alert: Literal[False] = False


class RiskTrendPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    dynamic_risk_score: float | None = Field(default=None, ge=0, le=1)
    short_term_fall_score: float | None = Field(default=None, ge=0, le=1)
    fall_event_status: FallEventState = "UNKNOWN"


class ContributionTrendPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    camera: float = Field(ge=0, le=1)
    radar: float = Field(ge=0, le=1)
    dominant_modality: DominantModality = "NONE"


class RuntimeUnknownRatio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera: float = Field(default=1.0, ge=0, le=1)
    radar: float = Field(default=1.0, ge=0, le=1)
    dynamic_risk: float = Field(default=1.0, ge=0, le=1)
    short_term_fall: float = Field(default=1.0, ge=0, le=1)
    fall_event: float = Field(default=1.0, ge=0, le=1)


class RuntimeAverageQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera: float = Field(default=0.0, ge=0, le=1)
    radar: float = Field(default=0.0, ge=0, le=1)
    overall: float = Field(default=0.0, ge=0, le=1)
    physiological: float | None = Field(default=None, ge=0, le=1)


class MultimodalRuntimeStatistics(BaseModel):
    """Bounded, deduplicated live observability window; not a model output."""

    model_config = ConfigDict(extra="forbid")

    sample_count: int = Field(default=0, ge=0)
    window_start: datetime | None = None
    window_end: datetime | None = None
    risk_trend: list[RiskTrendPoint] = Field(default_factory=list)
    contribution_trend: list[ContributionTrendPoint] = Field(default_factory=list)
    unknown_ratio: RuntimeUnknownRatio = Field(default_factory=RuntimeUnknownRatio)
    average_quality: RuntimeAverageQuality = Field(default_factory=RuntimeAverageQuality)
    mean_contribution_camera: float = Field(default=0.0, ge=0, le=1)
    mean_contribution_radar: float = Field(default=0.0, ge=0, le=1)
    statistics_interpretation: Literal["OBSERVABILITY_ONLY"] = "OBSERVABILITY_ONLY"


class CameraEvidence(BaseModel):
    """Stable output contract for the independent camera branch."""

    model_config = ConfigDict(extra="forbid")

    camera_score: float | None = Field(default=None, ge=0, le=1)
    camera_risk_state: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"] = "UNKNOWN"
    camera_feature: EvidenceFeature | None = None
    camera_quality: float = Field(ge=0, le=1)
    quality_level: QualityLevel
    timestamp: datetime
    source_timestamp: datetime | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_latency_ms: float | None = Field(default=None, ge=0)
    evidence_age_ms: float = Field(default=0.0, ge=0)
    timestamp_semantics: Literal["LATEST_SOURCE_FRAME_CAPTURE_TIME"] = (
        "LATEST_SOURCE_FRAME_CAPTURE_TIME"
    )
    available: bool
    device_id: str | None = Field(default=None, max_length=128)
    model_version: str | None = Field(default=None, max_length=128)
    quality_reason: str | None = Field(default=None, max_length=256)

    @field_validator("timestamp", "source_timestamp", "window_start", "window_end", "received_at")
    @classmethod
    def timestamp_needs_timezone(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("camera evidence timestamp must include timezone")
        return value

    @model_validator(mode="after")
    def availability_matches_score(self) -> "CameraEvidence":
        if self.source_timestamp is None:
            self.source_timestamp = self.timestamp
        if self.window_end is None:
            self.window_end = self.source_timestamp
        if self.window_start is None:
            self.window_start = self.window_end
        if self.window_start > self.window_end:
            raise ValueError("camera window_start must not exceed window_end")
        if self.available and self.camera_score is None:
            raise ValueError("available camera evidence requires camera_score")
        if not self.available and self.camera_score is not None:
            raise ValueError("unavailable camera evidence cannot expose camera_score")
        return self


class RadarEvidence(BaseModel):
    """Stable output contract for the frozen radar TCN branch."""

    model_config = ConfigDict(extra="forbid")

    radar_score: float | None = Field(default=None, ge=0, le=1)
    radar_risk_state: Literal[
        "NORMAL",
        "WATCH",
        "IMMINENT",
        "SUPPRESSED_RECOVERY",
        "CONFIRMED",
        "UNKNOWN",
    ] = "UNKNOWN"
    radar_feature: EvidenceFeature | None = None
    radar_quality: float = Field(ge=0, le=1)
    quality_level: QualityLevel
    timestamp: datetime
    source_timestamp: datetime | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_latency_ms: float | None = Field(default=None, ge=0)
    evidence_age_ms: float = Field(default=0.0, ge=0)
    timestamp_semantics: Literal["RADAR_WINDOW_END_FRAME_TIME"] = (
        "RADAR_WINDOW_END_FRAME_TIME"
    )
    available: bool
    room: str | None = Field(default=None, max_length=64)
    device_id: str | None = Field(default=None, max_length=128)
    model_version: str | None = Field(default=None, max_length=128)
    quality_reason: str | None = Field(default=None, max_length=256)

    @field_validator("timestamp", "source_timestamp", "window_start", "window_end", "received_at")
    @classmethod
    def timestamp_needs_timezone(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("radar evidence timestamp must include timezone")
        return value

    @model_validator(mode="after")
    def availability_matches_score(self) -> "RadarEvidence":
        if self.source_timestamp is None:
            self.source_timestamp = self.timestamp
        if self.window_end is None:
            self.window_end = self.source_timestamp
        if self.window_start is None:
            self.window_start = self.window_end
        if self.window_start > self.window_end:
            raise ValueError("radar window_start must not exceed window_end")
        if self.available and self.radar_score is None:
            raise ValueError("available radar evidence requires radar_score")
        if not self.available and self.radar_score is not None:
            raise ValueError("unavailable radar evidence cannot expose radar_score")
        return self


class RadarEligibilityDecision(BaseModel):
    """Explain whether Radar is allowed to influence decision-level Fusion."""

    model_config = ConfigDict(extra="forbid")

    assessed: bool = False
    eligible: bool = False
    target_detected: bool = False
    target_matched: bool = False
    synchronized: bool = False
    track_continuous: bool = False
    point_cloud_quality_passed: bool = False
    radar_quality: float = Field(default=0.0, ge=0, le=1)
    point_count_quality: float = Field(default=0.0, ge=0, le=1)
    track_stability: float = Field(default=0.0, ge=0, le=1)
    velocity_continuity: float = Field(default=0.0, ge=0, le=1)
    height_change_credibility: float = Field(default=0.0, ge=0, le=1)
    reason_codes: list[str] = Field(default_factory=list)


class FusionResult(BaseModel):
    """Decision-layer result. The score is not a fall probability."""

    model_config = ConfigDict(extra="forbid")

    score_name: Literal["multimodal risk score"] = "multimodal risk score"
    fusion_score: float | None = Field(default=None, ge=0, le=1)
    raw_fusion_score: float | None = Field(default=None, ge=0, le=1)
    stable_fusion_score: float | None = Field(default=None, ge=0, le=1)
    risk_level: FusionRiskLevel
    fusion_state: FusionStableState = "UNKNOWN"
    stable_fusion_state: FusionStableState = "UNKNOWN"
    fusion_risk_state: FusionStableState = "UNKNOWN"
    fusion_mode: FusionOperatingState = "NO_EVIDENCE"
    contribution_camera: float = Field(ge=0, le=1)
    contribution_radar: float = Field(ge=0, le=1)
    dominant_modality: DominantModality
    method: FusionMethod
    sync_delta_seconds: float | None = Field(default=None, ge=0)
    sync_delta_ms: float | None = Field(default=None, ge=0)
    synchronized: bool
    degraded_mode: FusionDegradedMode = "NONE"
    reason_codes: list[str] = Field(default_factory=list)
    degraded_reason: str | None = Field(default=None, max_length=256)
    radar_eligibility: RadarEligibilityDecision = Field(
        default_factory=RadarEligibilityDecision
    )

    @model_validator(mode="after")
    def result_is_internally_consistent(self) -> "FusionResult":
        if self.raw_fusion_score is None:
            self.raw_fusion_score = self.fusion_score
        if self.sync_delta_ms is None and self.sync_delta_seconds is not None:
            self.sync_delta_ms = self.sync_delta_seconds * 1000.0
        contribution_sum = self.contribution_camera + self.contribution_radar
        if self.fusion_score is None:
            if self.risk_level != "UNKNOWN" or contribution_sum != 0:
                raise ValueError("missing fusion score requires UNKNOWN and zero contribution")
        elif abs(contribution_sum - 1.0) > 1e-6:
            raise ValueError("available fusion result contributions must sum to one")
        return self


class AlignedPersonEvidence(BaseModel):
    """CameraPerson↔RadarTrack compatibility evidence."""

    model_config = ConfigDict(extra="forbid")

    association_state: AlignmentAssociationState = "CALIBRATION_INVALID"
    camera_person_id: int | None = Field(default=None, ge=0)
    radar_track_id: int | None = Field(default=None, ge=0, lt=253)
    camera_frame_id: str | None = Field(default=None, max_length=128)
    radar_frame_number: int | None = Field(default=None, ge=0)
    radar_source_timestamp: datetime | None = None
    sync_delta_ms: float | None = Field(default=None, ge=0)
    radar_position_xyz_m: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )
    radar_velocity_xyz_mps: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )
    radar_point_count: int = Field(default=0, ge=0)
    radar_point_cloud_spread_m: float | None = Field(default=None, ge=0)
    radar_target_confidence: float | None = Field(default=None, ge=0, le=1)
    radar_config_name: str | None = Field(default=None, max_length=128)
    camera_footpoint_uv: tuple[float, float] | None = None
    projected_radar_uv: tuple[float, float] | None = None
    bbox_gate_distance_px: float | None = Field(default=None, ge=0)
    association_confidence: float = Field(default=0.0, ge=0, le=1)
    calibration_version: str | None = Field(default=None, max_length=128)
    eligible_for_temporal_association: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    shadow_only: bool = True
    realtime_active: bool = False
    affects_realtime_fusion_v2: bool = False
    affects_fixed_fusion: Literal[False] = False
    affects_alerts: Literal[False] = False
    interpretation: Literal["COARSE_COMPATIBILITY_NOT_IDENTITY"] = (
        "COARSE_COMPATIBILITY_NOT_IDENTITY"
    )

    @model_validator(mode="after")
    def realtime_and_shadow_flags_are_consistent(self) -> "AlignedPersonEvidence":
        if self.realtime_active == self.shadow_only:
            raise ValueError("alignment must be either realtime-active or shadow-only")
        if self.affects_realtime_fusion_v2 != self.realtime_active:
            raise ValueError("realtime alignment must declare Fusion v2 participation")
        return self

    @field_validator("radar_source_timestamp")
    @classmethod
    def radar_source_timestamp_needs_timezone(
        cls, value: datetime | None
    ) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("aligned radar source timestamp must include timezone")
        return value


class AlignmentAwareRiskAugmentationResult(BaseModel):
    """Camera-led, non-learned Radar evidence annotation; shadow-only."""

    model_config = ConfigDict(extra="forbid")

    score_name: Literal["camera-led associated short-term fall risk score"] = (
        "camera-led associated short-term fall risk score"
    )
    associated_short_term_fall_score: float | None = Field(default=None, ge=0, le=1)
    associated_risk_state: FusionStableState = "UNKNOWN"
    associated_evidence_state: AssociatedEvidenceState = "UNKNOWN"
    base_camera_score: float | None = Field(default=None, ge=0, le=1)
    base_camera_state: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"] = "UNKNOWN"
    radar_motion_evidence_strength: RadarMotionEvidenceStrength = "UNKNOWN"
    association_state: AlignmentAssociationState = "CALIBRATION_INVALID"
    sync_delta_ms: float | None = Field(default=None, ge=0)
    radar_track_id: int | None = Field(default=None, ge=0, lt=253)
    radar_evidence_count: int = Field(default=0, ge=0)
    track_stability: float | None = Field(default=None, ge=0, le=1)
    radar_motion_features: dict[str, FeatureScalar] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    shadow_only: Literal[True] = True
    affects_fixed_fusion: Literal[False] = False
    affects_alerts: Literal[False] = False
    camera_model_unchanged: Literal[True] = True
    camera_score_unchanged: Literal[True] = True
    uses_radar_tcn_score: Literal[False] = False
    radar_can_veto_camera_when_unavailable: Literal[False] = False
    radar_can_escalate_camera_low_to_high: Literal[False] = False
    interpretation: Literal["EVIDENCE_AUGMENTATION_NOT_NEW_SCORE"] = (
        "EVIDENCE_AUGMENTATION_NOT_NEW_SCORE"
    )

    @model_validator(mode="after")
    def associated_score_is_camera_score(self) -> "AlignmentAwareRiskAugmentationResult":
        if self.associated_short_term_fall_score != self.base_camera_score:
            raise ValueError("associated score must remain identical to camera score")
        return self


class CameraLedEvidenceFusionV2Result(BaseModel):
    """Active realtime Camera-led interpretation of associated Radar evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["camera_led_evidence_fusion_v2"] = (
        "camera_led_evidence_fusion_v2"
    )
    score_name: Literal["camera-led evidence risk score"] = (
        "camera-led evidence risk score"
    )
    camera_led_score: float | None = Field(default=None, ge=0, le=1)
    camera_led_state: FusionStableState = "UNKNOWN"
    fusion_mode: CameraLedEvidenceFusionMode = "LOW_CONFIDENCE"
    camera_score: float | None = Field(default=None, ge=0, le=1)
    radar_score: float | None = Field(default=None, ge=0, le=1)
    camera_quality: float = Field(default=0.0, ge=0, le=1)
    radar_quality: float = Field(default=0.0, ge=0, le=1)
    radar_eligible: bool = False
    radar_motion_evidence_strength: RadarMotionEvidenceStrength = "UNKNOWN"
    association_state: AlignmentAssociationState = "CALIBRATION_INVALID"
    sync_delta_ms: float | None = Field(default=None, ge=0)
    reason_codes: list[str] = Field(default_factory=list)
    model_version: Literal["camera-led-evidence-fusion-v2-realtime-v1"] = (
        "camera-led-evidence-fusion-v2-realtime-v1"
    )
    realtime_active: Literal[True] = True
    shadow_only: Literal[False] = False
    affects_app_result: Literal[True] = True
    affects_fixed_fusion: Literal[False] = False
    affects_alerts: Literal[False] = False
    camera_score_unchanged: Literal[True] = True
    radar_score_affects_risk_score: Literal[False] = False
    radar_can_veto_camera_high: Literal[False] = False
    interpretation: Literal["STATE_LEVEL_EVIDENCE_NOT_WEIGHTED_AVERAGE"] = (
        "STATE_LEVEL_EVIDENCE_NOT_WEIGHTED_AVERAGE"
    )

    @model_validator(mode="after")
    def score_must_remain_camera_score(self) -> "CameraLedEvidenceFusionV2Result":
        if self.camera_led_score != self.camera_score:
            raise ValueError("Fusion v2 score must remain identical to camera score")
        return self


class TemporalAssociatedFusionResult(BaseModel):
    """Shadow-only temporal/association experiment; never a formal alert input."""

    model_config = ConfigDict(extra="forbid")

    score_name: Literal["temporal associated multimodal risk score"] = (
        "temporal associated multimodal risk score"
    )
    fusion_score: float | None = Field(default=None, ge=0, le=1)
    fusion_state: FusionStableState = "UNKNOWN"
    shadow_only: Literal[True] = True
    affects_alerts: Literal[False] = False
    window_seconds: float = Field(gt=0)
    camera_evidence_count: int = Field(ge=0)
    radar_evidence_count: int = Field(ge=0)
    continuous_camera_risk: bool = False
    continuous_radar_risk: bool = False
    target_association: TargetAssociationState = "UNKNOWN"
    alignment_state: AlignmentAssociationState = "CALIBRATION_INVALID"
    camera_target_id: str | None = Field(default=None, max_length=128)
    radar_target_id: str | None = Field(default=None, max_length=128)
    temporal_relation: TemporalRelation = "INSUFFICIENT_EVIDENCE"
    causal_consistency: bool = False
    sync_delta_ms: float | None = Field(default=None, ge=0)
    degraded_mode: FusionDegradedMode = "NONE"
    reason_codes: list[str] = Field(default_factory=list)
    radar_evidence_snapshot: dict[str, FeatureScalar] = Field(default_factory=dict)


class MultimodalQualitySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera: float = Field(ge=0, le=1)
    radar: float = Field(ge=0, le=1)
    synchronization: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)
    level: QualityLevel


class MultimodalTimingAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sync_delta_ms: float | None = Field(default=None, ge=0)
    sync_p50_ms: float | None = Field(default=None, ge=0)
    sync_p95_ms: float | None = Field(default=None, ge=0)
    sync_sample_count: int = Field(default=0, ge=0)
    tolerance_ms: float = Field(gt=0)
    timezone_policy: Literal["UTC_OFFSET_AWARE"] = "UTC_OFFSET_AWARE"


class MultimodalLatestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera: CameraEvidence
    radar: RadarEvidence
    fusion: FusionResult
    dynamic_risk: DynamicRiskIndex = Field(default_factory=DynamicRiskIndex)
    short_term_warning: ShortTermFallWarning | None = None
    fall_event: FallEventSummary = Field(
        default_factory=lambda: FallEventSummary(
            fall_event_status="UNKNOWN",
            reason_codes=["EVIDENCE_NOT_ASSESSED"],
            summary="尚未形成可判读的跌倒事件状态",
        )
    )
    physiological_evidence: PhysiologicalEvidence = Field(
        default_factory=PhysiologicalEvidence
    )
    final_decision_context: FinalDecisionContext | None = None
    runtime_statistics: MultimodalRuntimeStatistics = Field(
        default_factory=MultimodalRuntimeStatistics
    )
    temporal_associated_fusion: TemporalAssociatedFusionResult | None = None
    associated_risk_augmentation: AlignmentAwareRiskAugmentationResult | None = None
    camera_led_evidence_fusion_v2: CameraLedEvidenceFusionV2Result | None = None
    alignment: AlignedPersonEvidence = Field(default_factory=AlignedPersonEvidence)
    operating_mode: Literal["LIVE_CAMERA_RADAR", "OFFLINE_EVIDENCE_REPLAY"] = (
        "LIVE_CAMERA_RADAR"
    )
    data_source: MultimodalDataSource = "REAL_CAMERA_RADAR"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    quality: MultimodalQualitySummary
    timing: MultimodalTimingAudit | None = None

    @field_validator("timestamp")
    @classmethod
    def timestamp_needs_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("multimodal response timestamp must include timezone")
        return value

    @model_validator(mode="after")
    def expose_short_term_score_separately(self) -> "MultimodalLatestResponse":
        if self.short_term_warning is None:
            self.short_term_warning = ShortTermFallWarning(
                short_term_fall_score=(
                    self.fusion.stable_fusion_score
                    if self.fusion.stable_fusion_score is not None
                    else self.fusion.raw_fusion_score
                ),
                state=self.fusion.stable_fusion_state,
                method=self.fusion.method,
                degraded_mode=self.fusion.degraded_mode,
                synchronized=self.fusion.synchronized,
            )
        if self.final_decision_context is None:
            self.final_decision_context = FinalDecisionContext(
                base_short_term_state=self.short_term_warning.state,
                base_fall_event_status=self.fall_event.fall_event_status,
                physiological_context=self.physiological_evidence,
                reason_codes=["PHYSIOLOGICAL_CONTEXT_UNAVAILABLE"],
                summary="RPPG 最终决策上下文尚不可用，不改变现有跌倒风险结论",
            )
        return self


class CameraLedAssociatedCameraProjection(BaseModel):
    """Minimal Camera evidence exposed to the App adapter."""

    model_config = ConfigDict(extra="forbid")

    camera_score: float | None = Field(default=None, ge=0, le=1)
    camera_risk_state: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"] = "UNKNOWN"
    quality_level: QualityLevel
    timestamp: datetime
    available: bool


class CameraLedAssociatedRadarProjection(BaseModel):
    """Radar availability and quality beside the Camera-led decision."""

    model_config = ConfigDict(extra="forbid")

    radar_score: float | None = Field(default=None, ge=0, le=1)
    radar_risk_state: Literal[
        "NORMAL",
        "WATCH",
        "IMMINENT",
        "SUPPRESSED_RECOVERY",
        "CONFIRMED",
        "UNKNOWN",
    ] = "UNKNOWN"
    quality_level: QualityLevel
    timestamp: datetime
    available: bool
    room: str | None = Field(default=None, max_length=64)


class CameraLedAssociatedAlignmentProjection(BaseModel):
    """Association state without raw track, sync, or calibration diagnostics."""

    model_config = ConfigDict(extra="forbid")

    association_state: AlignmentAssociationState
    eligible_for_temporal_association: bool


class CameraLedAssociatedRiskProjection(BaseModel):
    """Existing C-path annotation, including its unchanged shadow safeguards."""

    model_config = ConfigDict(extra="forbid")

    associated_short_term_fall_score: float | None = Field(default=None, ge=0, le=1)
    associated_risk_state: FusionStableState = "UNKNOWN"
    associated_evidence_state: AssociatedEvidenceState = "UNKNOWN"
    base_camera_score: float | None = Field(default=None, ge=0, le=1)
    base_camera_state: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"] = "UNKNOWN"
    radar_motion_evidence_strength: RadarMotionEvidenceStrength = "UNKNOWN"
    association_state: AlignmentAssociationState
    shadow_only: Literal[True] = True
    affects_alerts: Literal[False] = False
    camera_score_unchanged: Literal[True] = True
    uses_radar_tcn_score: Literal[False] = False


class CameraLedAssociatedFallEventProjection(BaseModel):
    """Existing observed event summary; this endpoint never promotes an event."""

    model_config = ConfigDict(extra="forbid")

    fall_event_status: FallEventState = "UNKNOWN"
    summary: str = Field(min_length=1, max_length=256)


class CameraLedAssociatedLatestResponse(BaseModel):
    """Read-only App projection of the existing Camera-led associated C path."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["camera_led_associated_latest_v1"] = (
        "camera_led_associated_latest_v1"
    )
    camera: CameraLedAssociatedCameraProjection
    radar: CameraLedAssociatedRadarProjection
    alignment: CameraLedAssociatedAlignmentProjection
    associated_risk_augmentation: CameraLedAssociatedRiskProjection | None = None
    camera_led_evidence_fusion_v2: CameraLedEvidenceFusionV2Result | None = None
    fall_event: CameraLedAssociatedFallEventProjection
    timestamp: datetime


class OfflineReplayLatestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    subject_id: str
    recording_id: str
    cursor: int = Field(ge=0)
    next_cursor: int = Field(ge=0)
    sample_count: int = Field(ge=1)
    multimodal: MultimodalLatestResponse
