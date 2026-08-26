from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math

from app.modules.fall.multimodal_engine.schemas.fall_live import FallLiveInputState, FallLiveStatusResponse
from app.modules.fall.multimodal_engine.schemas.multimodal import (
    AlignedPersonEvidence,
    CameraEvidence,
    FinalDecisionContext,
    FusionResult,
    MultimodalLatestResponse,
    MultimodalQualitySummary,
    PhysiologicalEvidence,
    RadarEligibilityDecision,
    RadarEvidence,
)
from app.modules.fall.multimodal_engine.schemas.radar import RadarStatusResponse
from app.modules.fall.multimodal_engine.services.fusion_runtime import (
    FusionResponseCallback,
    FusionRuntimeTracker,
    FusionShadowLogger,
    FusionStateConfig,
    FusionStateMachine,
    FusionTimingTracker,
)
from app.modules.fall.multimodal_engine.services.dynamic_risk_index import DynamicRiskIndexService
from app.modules.fall.multimodal_engine.services.temporal_associated_fusion import (
    TemporalAssociatedFusion,
    TemporalAssociationConfig,
)
from app.modules.fall.multimodal_engine.services.camera_radar_alignment import CameraRadarAlignmentAdapter
from app.modules.fall.multimodal_engine.services.alignment_aware_risk_augmentation import (
    AlignmentAwareRiskAugmentation,
    AssociatedEvidenceConfig,
)
from app.modules.fall.multimodal_engine.services.camera_led_evidence_fusion_v2 import (
    CameraLedEvidenceFusionV2,
    CameraLedEvidenceFusionV2Config,
)
from app.modules.fall.multimodal_engine.services.radar_eligibility import (
    RadarEligibilityConfig,
    RadarEligibilityGate,
)


@dataclass(frozen=True)
class MlpFusionParameters:
    """Explicit, externally supplied weights for a forward-only fusion head."""

    hidden_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[float, ...]
    output_bias: float


class LightweightMlpFusionHead:
    """Small inference-only MLP; it deliberately contains no training code."""

    input_size = 5

    def __init__(self, parameters: MlpFusionParameters) -> None:
        hidden_size = len(parameters.hidden_weights)
        if hidden_size == 0:
            raise ValueError("MLP fusion head needs at least one hidden unit")
        if any(len(row) != self.input_size for row in parameters.hidden_weights):
            raise ValueError("each hidden row must accept five fusion inputs")
        if len(parameters.hidden_bias) != hidden_size:
            raise ValueError("hidden bias size does not match hidden layer")
        if len(parameters.output_weights) != hidden_size:
            raise ValueError("output weights size does not match hidden layer")
        self.parameters = parameters

    def predict(self, values: Sequence[float]) -> float:
        if len(values) != self.input_size:
            raise ValueError("MLP fusion input must contain five values")
        hidden = [
            max(
                0.0,
                sum(weight * float(value) for weight, value in zip(row, values))
                + bias,
            )
            for row, bias in zip(
                self.parameters.hidden_weights,
                self.parameters.hidden_bias,
            )
        ]
        logit = (
            sum(
                weight * value
                for weight, value in zip(self.parameters.output_weights, hidden)
            )
            + self.parameters.output_bias
        )
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))


class MultimodalFusionService:
    """Read two independent branches and add a non-invasive decision layer."""

    def __init__(
        self,
        camera_status_provider: Callable[[], FallLiveStatusResponse],
        radar_status_provider: Callable[[], RadarStatusResponse],
        *,
        camera_weight: float = 0.6,
        radar_weight: float = 0.4,
        sync_tolerance_seconds: float = 2.0,
        medium_threshold: float = 0.35,
        high_threshold: float = 0.65,
        mlp_head: LightweightMlpFusionHead | None = None,
        state_config: FusionStateConfig | None = None,
        shadow_logger: FusionShadowLogger | None = None,
        response_callback: FusionResponseCallback | None = None,
        temporal_association_config: TemporalAssociationConfig | None = None,
        alignment_adapter: CameraRadarAlignmentAdapter | None = None,
        associated_evidence_config: AssociatedEvidenceConfig | None = None,
        radar_eligibility_config: RadarEligibilityConfig | None = None,
    ) -> None:
        if camera_weight <= 0 or radar_weight <= 0:
            raise ValueError("base fusion weights must be positive")
        if sync_tolerance_seconds <= 0:
            raise ValueError("sync tolerance must be positive")
        if not 0 < medium_threshold < high_threshold < 1:
            raise ValueError("risk thresholds must be ordered within (0, 1)")
        self.camera_status_provider = camera_status_provider
        self.radar_status_provider = radar_status_provider
        self.camera_weight = camera_weight
        self.radar_weight = radar_weight
        self.sync_tolerance_seconds = sync_tolerance_seconds
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.mlp_head = mlp_head
        resolved_state_config = (
            state_config
            or FusionStateConfig(
                watch_enter=medium_threshold,
                high_enter=high_threshold,
            )
        )
        # Experimental methods must not advance or reset the formal fixed
        # baseline state. Keep a fully independent state machine per method.
        self.state_machines = {
            method: FusionStateMachine(resolved_state_config)
            for method in (
                "fixed_weighted",
                "quality_weighted",
                "radar_quality_adaptive",
                "mlp",
            )
        }
        self.state_machine = self.state_machines["fixed_weighted"]
        self.timing_tracker = FusionTimingTracker(
            tolerance_ms=sync_tolerance_seconds * 1000.0
        )
        self.shadow_logger = shadow_logger
        self.response_callback = response_callback
        self.temporal_associated_fusion = TemporalAssociatedFusion(
            temporal_association_config
        )
        self.alignment_adapter = alignment_adapter
        self.radar_eligibility_gate = RadarEligibilityGate(
            radar_eligibility_config
            or RadarEligibilityConfig(enabled=alignment_adapter is not None)
        )
        self.associated_risk_augmentation = AlignmentAwareRiskAugmentation(
            associated_evidence_config
        )
        self.camera_led_evidence_fusion_v2 = CameraLedEvidenceFusionV2(
            CameraLedEvidenceFusionV2Config(
                minimum_camera_quality=(
                    resolved_state_config.minimum_modality_quality
                )
            )
        )
        self.dynamic_risk_index = DynamicRiskIndexService()
        self.runtime_tracker = FusionRuntimeTracker()

    def get_latest(self, *, method: str = "fixed_weighted") -> MultimodalLatestResponse:
        if method not in {
            "fixed_weighted",
            "quality_weighted",
            "radar_quality_adaptive",
            "mlp",
        }:
            raise ValueError(f"unsupported fusion method: {method}")
        if method == "mlp" and self.mlp_head is None:
            raise RuntimeError("MLP fusion is not configured with reviewed weights")

        now = datetime.now(timezone.utc)
        camera_status = self.camera_status_provider()
        radar_status = self.radar_status_provider()
        camera = self.camera_evidence(camera_status, now=now)
        radar = self.radar_evidence(radar_status, now=now)
        alignment = (
            self.alignment_adapter.apply(camera_status, radar_status)
            if self.alignment_adapter is not None
            else None
        )
        radar_eligibility = self.radar_eligibility_gate.evaluate(
            camera,
            radar,
            alignment,
            sync_tolerance_seconds=self.sync_tolerance_seconds,
        )
        fusion = self.state_machines[method].apply(
            camera,
            radar,
            self.fuse(
                camera,
                radar,
                method=method,
                radar_eligibility=radar_eligibility,
            ),
        )
        temporal_associated = self.temporal_associated_fusion.apply(
            camera,
            radar,
            fusion,
            alignment=alignment,
        )
        associated_augmentation = self.associated_risk_augmentation.apply(
            camera,
            alignment or AlignedPersonEvidence(),
        )
        camera_led_v2 = self.camera_led_evidence_fusion_v2.apply(
            camera,
            radar,
            alignment or AlignedPersonEvidence(),
            radar_eligibility,
            associated_augmentation,
        )
        sync_quality = self._sync_quality(fusion.sync_delta_seconds)
        available_qualities = [
            quality
            for quality, available in (
                (camera.camera_quality, camera.available),
                (radar.radar_quality, radar.available),
            )
            if available
        ]
        if not available_qualities:
            overall = 0.0
            level = "INSUFFICIENT_DATA"
        elif len(available_qualities) == 1:
            overall = available_qualities[0] * 0.75
            level = "DEGRADED"
        else:
            overall = sum(available_qualities) / 2.0 * sync_quality
            level = "GOOD" if overall >= 0.75 else "DEGRADED"
        response = MultimodalLatestResponse(
            camera=camera,
            radar=radar,
            fusion=fusion,
            dynamic_risk=self.dynamic_risk_index.build_dynamic_risk(
                camera,
                radar,
                radar_status,
            ),
            short_term_warning=self.dynamic_risk_index.build_camera_led_short_term_warning(
                camera,
                radar,
                camera_led_v2,
            ),
            fall_event=self.dynamic_risk_index.build_fall_event(camera, radar),
            physiological_evidence=self.physiological_evidence(camera_status),
            temporal_associated_fusion=temporal_associated,
            associated_risk_augmentation=associated_augmentation,
            camera_led_evidence_fusion_v2=camera_led_v2,
            **({"alignment": alignment} if alignment is not None else {}),
            operating_mode="LIVE_CAMERA_RADAR",
            timestamp=now,
            quality=MultimodalQualitySummary(
                camera=camera.camera_quality,
                radar=radar.radar_quality,
                synchronization=sync_quality,
                overall=max(0.0, min(1.0, overall)),
                level=level,
            ),
            timing=self.timing_tracker.observe(camera, radar),
        )
        response = response.model_copy(
            update={
                "final_decision_context": self.final_decision_context(
                    response.physiological_evidence,
                    response.short_term_warning.state,
                    response.fall_event.fall_event_status,
                )
            }
        )
        response = response.model_copy(
            update={"runtime_statistics": self.runtime_tracker.observe(response)}
        )
        if self.shadow_logger is not None:
            self.shadow_logger.write(response)
        if self.response_callback is not None:
            self.response_callback(response)
        return response

    @staticmethod
    def final_decision_context(
        physiological: PhysiologicalEvidence,
        short_term_state: str,
        fall_event_status: str,
    ) -> FinalDecisionContext:
        """Append rPPG after Fusion without changing any formal decision."""

        if not physiological.assessment_ready:
            return FinalDecisionContext(
                base_short_term_state=short_term_state,
                base_fall_event_status=fall_event_status,
                physiological_context=physiological,
                physiological_review_level="UNAVAILABLE",
                human_review_suggested=False,
                reason_codes=[physiological.quality_reason],
                summary="RPPG 未通过质量或有效时长门控，不改变现有跌倒风险结论",
            )
        if physiological.physio_level == "ABNORMAL":
            return FinalDecisionContext(
                base_short_term_state=short_term_state,
                base_fall_event_status=fall_event_status,
                physiological_context=physiological,
                physiological_review_level="MANUAL_REVIEW_SUGGESTED",
                human_review_suggested=True,
                reason_codes=["PHYSIOLOGICAL_OBSERVATION_ABNORMAL"]
                + physiological.abnormal_reasons,
                summary="跌倒结论保持不变；RPPG 生理观察异常，建议结合人工复核",
            )
        return FinalDecisionContext(
            base_short_term_state=short_term_state,
            base_fall_event_status=fall_event_status,
            physiological_context=physiological,
            physiological_review_level="NO_ADDITIONAL_CONCERN",
            human_review_suggested=False,
            reason_codes=["PHYSIOLOGICAL_OBSERVATION_NORMAL"],
            summary="RPPG 当前未提供额外生理关注信号，跌倒结论保持不变",
        )

    @staticmethod
    def physiological_evidence(
        status: FallLiveStatusResponse | None,
    ) -> PhysiologicalEvidence:
        """Normalize OpenSDK rPPG output without feeding any risk decision."""

        if status is None:
            return PhysiologicalEvidence()
        source = status.rppg
        if not source.enabled:
            quality_level = "INSUFFICIENT_DATA"
        elif source.assessment_ready and source.quality_coverage >= 0.75:
            quality_level = "GOOD"
        else:
            quality_level = "DEGRADED"
        raw_level = source.physio_level.upper()
        level = raw_level if raw_level in {"NORMAL", "ABNORMAL"} else "UNKNOWN"
        if not source.assessment_ready:
            level = "UNKNOWN"
        return PhysiologicalEvidence(
            enabled=source.enabled,
            available=source.available,
            assessment_ready=source.assessment_ready,
            heart_rate=source.heart_rate,
            sqi=source.sqi,
            hrv=source.hrv.model_dump(),
            physio_level=level,
            abnormal_reasons=list(source.abnormal_reasons),
            valid_seconds=source.valid_seconds,
            quality_coverage=source.quality_coverage,
            quality_level=quality_level,
            quality_reason=source.quality_reason,
            timestamp=source.source_timestamp,
        )

    def camera_evidence(
        self,
        status: FallLiveStatusResponse,
        *,
        now: datetime | None = None,
    ) -> CameraEvidence:
        now = now or datetime.now(timezone.utc)
        timestamp = status.last_prediction_at or status.checked_at
        available = (
            status.risk_score is not None
            and status.input_state == FallLiveInputState.READY
            and status.training_input_ready
        )
        quality = self._camera_quality(status) if available else 0.0
        quality_level = self._quality_level(quality)
        feature = {
            "input_state": status.input_state.value,
            "risk_state": status.risk_level.value if status.risk_level is not None else "UNKNOWN",
            "positive_votes": status.positive_votes,
            "target_present": status.target_present,
            "source_window_frames": status.source_window_frames,
            "valid_pose_frames": status.valid_pose_frames,
            "effective_sample_fps": status.effective_sample_fps,
            "mean_keypoint_confidence": status.mean_keypoint_confidence,
            "latest_keypoint_confidence": status.latest_keypoint_confidence,
            "last_event_id": status.last_event_id,
        }
        window_end = timestamp
        if status.effective_sample_fps > 0 and status.source_window_frames > 1:
            window_start = window_end - timedelta(
                seconds=(status.source_window_frames - 1) / status.effective_sample_fps
            )
        else:
            window_start = window_end
        return CameraEvidence(
            camera_score=status.risk_score if available else None,
            camera_risk_state=(
                status.risk_level.value
                if available and status.risk_level is not None
                else "UNKNOWN"
            ),
            camera_feature=feature,
            camera_quality=quality,
            quality_level=quality_level,
            timestamp=timestamp,
            source_timestamp=timestamp,
            window_start=window_start,
            window_end=window_end,
            received_at=status.checked_at,
            processing_latency_ms=(
                status.pipeline_latency_seconds * 1000.0
                if status.pipeline_latency_seconds is not None
                else None
            ),
            evidence_age_ms=max(0.0, (now - timestamp).total_seconds() * 1000.0),
            available=available,
            device_id=status.device_id,
            model_version=status.model_version,
            quality_reason=(
                "camera pose window is ready"
                if available
                else status.input_message or "camera evidence unavailable"
            ),
        )

    def radar_evidence(
        self,
        status: RadarStatusResponse,
        *,
        now: datetime | None = None,
    ) -> RadarEvidence:
        now = now or datetime.now(timezone.utc)
        source = status.radar_evidence
        available = bool(
            status.online
            and source is not None
            and source.radar_score is not None
            and source.risk_state != "UNKNOWN"
            and source.quality != "INSUFFICIENT_DATA"
        )
        quality_map = {"GOOD": 1.0, "DEGRADED": 0.6, "INSUFFICIENT_DATA": 0.0}
        quality = quality_map.get(source.quality, 0.0) if source else 0.0
        timestamp = source.timestamp if source else status.checked_at
        emitted_at = None
        source_prediction = None
        for prediction in (
            status.tcn_prediction,
            status.tcn_baseline,
            status.calibrated_tcn_prediction,
        ):
            if prediction is not None and prediction.timestamp == timestamp:
                emitted_at = prediction.emitted_at
                source_prediction = prediction
                break
        metrics = status.sensor_metrics
        vertical_velocity = getattr(source_prediction, "vertical_velocity", None)
        height_delta = getattr(source_prediction, "height_delta_0_6s", None)
        feature_point_count = getattr(source_prediction, "feature_point_count", None)
        point_count = metrics.point_count if metrics else None
        if point_count is None and feature_point_count is not None:
            point_count = int(feature_point_count)
        motion_direction = "UNKNOWN"
        if vertical_velocity is not None:
            motion_direction = (
                "DESCENDING"
                if vertical_velocity < 0
                else "ASCENDING"
                if vertical_velocity > 0
                else "STABLE"
            )
        feature = {
            "risk_state": source.risk_state if source else "UNKNOWN",
            "frame_rate_hz": metrics.frame_rate_hz if metrics else None,
            "point_count": point_count,
            "feature_point_count": feature_point_count,
            "centroid_z": getattr(source_prediction, "centroid_z", None),
            "vertical_velocity": vertical_velocity,
            "height_delta": height_delta,
            "motion_direction": motion_direction,
            "missing_frame_ratio": getattr(
                source_prediction, "missing_frame_ratio", None
            ),
            "longest_unresolved_gap_seconds": getattr(
                source_prediction, "longest_unresolved_gap_seconds", None
            ),
            "event_triggered": getattr(source_prediction, "event_triggered", False),
            "event_id": getattr(source_prediction, "event_id", None),
            # The current API exposes no cross-modal person/track identifier.
            "target_id": None,
            "track_count": None,
        }
        return RadarEvidence(
            radar_score=source.radar_score if available and source else None,
            radar_risk_state=(
                source.risk_state if available and source else "UNKNOWN"
            ),
            radar_feature=feature,
            radar_quality=quality if available else 0.0,
            quality_level=self._quality_level(quality if available else 0.0),
            timestamp=timestamp,
            source_timestamp=timestamp,
            window_start=timestamp - timedelta(seconds=1.9),
            window_end=timestamp,
            received_at=status.checked_at,
            processing_latency_ms=(
                max(0.0, (emitted_at - timestamp).total_seconds() * 1000.0)
                if emitted_at is not None
                else None
            ),
            evidence_age_ms=max(0.0, (now - timestamp).total_seconds() * 1000.0),
            available=available,
            room=source.room if source else status.room,
            device_id=source.device_id if source else status.device_id,
            model_version=source.model_version if source else None,
            quality_reason=(
                f"radar data quality is {source.quality}"
                if source
                else status.error or "radar evidence unavailable"
            ),
        )

    def fuse(
        self,
        camera: CameraEvidence,
        radar: RadarEvidence,
        *,
        method: str = "quality_weighted",
        radar_eligibility: RadarEligibilityDecision | None = None,
    ) -> FusionResult:
        eligibility = radar_eligibility or RadarEligibilityDecision()
        if not camera.available and not radar.available:
            return FusionResult(
                fusion_score=None,
                risk_level="UNKNOWN",
                contribution_camera=0.0,
                contribution_radar=0.0,
                dominant_modality="NONE",
                method=method,
                fusion_mode="NO_EVIDENCE",
                sync_delta_seconds=None,
                synchronized=False,
                degraded_mode="BOTH_UNAVAILABLE",
                reason_codes=["CAMERA_UNAVAILABLE", "RADAR_UNAVAILABLE"],
                degraded_reason="camera and radar evidence are unavailable",
                radar_eligibility=eligibility,
            )

        sync_delta = (
            abs((camera.timestamp - radar.timestamp).total_seconds())
            if camera.available and radar.available
            else None
        )
        synchronized = sync_delta is not None and sync_delta <= self.sync_tolerance_seconds

        # The formal fixed 0.6/0.4 formula is unchanged, but Radar is not
        # allowed to reach it until the explicit eligibility contract passes.
        if (
            camera.available
            and radar.available
            and eligibility.assessed
            and not eligibility.eligible
        ):
            reasons = list(eligibility.reason_codes)
            conflict = any(
                reason in {"TRACK_CONFLICT", "MULTIPLE_CANDIDATES"}
                for reason in reasons
            )
            low_quality = "LOW_QUALITY" in reasons
            return self._single_modality_result(
                camera.camera_score,
                "CAMERA",
                method,
                sync_delta,
                "MODALITY_CONFLICT"
                if conflict
                else "LOW_QUALITY"
                if low_quality
                else "CAMERA_ONLY",
                reasons,
                "Radar failed the eligibility gate; camera-only fallback",
                fusion_mode=(
                    "RADAR_CONFLICT"
                    if conflict
                    else "LOW_CONFIDENCE"
                    if low_quality
                    else "CAMERA_ONLY"
                ),
                radar_eligibility=eligibility,
            )

        if camera.available and radar.available and not synchronized:
            if camera.timestamp >= radar.timestamp:
                return self._single_modality_result(
                    camera.camera_score,
                    "CAMERA",
                    method,
                    sync_delta,
                    "OUT_OF_SYNC",
                    ["SYNC_TOLERANCE_EXCEEDED", "RADAR_STALE"],
                    "timestamps are outside the fusion tolerance; newer camera evidence used",
                    fusion_mode="CAMERA_ONLY",
                    radar_eligibility=eligibility,
                )
            return self._single_modality_result(
                radar.radar_score,
                "RADAR",
                method,
                sync_delta,
                "OUT_OF_SYNC",
                ["SYNC_TOLERANCE_EXCEEDED", "CAMERA_STALE"],
                "timestamps are outside the fusion tolerance; newer radar evidence used",
                fusion_mode="RADAR_ONLY",
                radar_eligibility=eligibility,
            )

        if camera.available and not radar.available:
            return self._single_modality_result(
                camera.camera_score,
                "CAMERA",
                method,
                None,
                "CAMERA_ONLY",
                ["RADAR_UNAVAILABLE"],
                "radar evidence unavailable; camera-only fallback",
                fusion_mode="CAMERA_ONLY",
                radar_eligibility=eligibility,
            )
        if radar.available and not camera.available:
            return self._single_modality_result(
                radar.radar_score,
                "RADAR",
                method,
                None,
                "RADAR_ONLY",
                ["CAMERA_UNAVAILABLE"],
                "camera evidence unavailable; radar-only fallback",
                fusion_mode="RADAR_ONLY",
                radar_eligibility=eligibility,
            )

        assert camera.camera_score is not None and radar.radar_score is not None
        if method == "quality_weighted":
            camera_effective = self.camera_weight * camera.camera_quality
            radar_effective = self.radar_weight * radar.radar_quality
        elif method == "radar_quality_adaptive":
            camera_effective = self.camera_weight
            radar_effective = self.radar_weight * (
                eligibility.radar_quality
                if eligibility.assessed
                else radar.radar_quality
            )
        else:
            camera_effective = self.camera_weight
            radar_effective = self.radar_weight
        total = camera_effective + radar_effective
        if total <= 0:
            return FusionResult(
                fusion_score=None,
                risk_level="UNKNOWN",
                contribution_camera=0.0,
                contribution_radar=0.0,
                dominant_modality="NONE",
                method=method,
                sync_delta_seconds=sync_delta,
                synchronized=True,
                degraded_mode="LOW_QUALITY",
                reason_codes=["ZERO_QUALITY_ADJUSTED_WEIGHT"],
                degraded_reason="quality-adjusted weights are zero",
                fusion_mode="LOW_CONFIDENCE",
                radar_eligibility=eligibility,
            )
        contribution_camera = camera_effective / total
        contribution_radar = radar_effective / total
        if method == "mlp":
            assert self.mlp_head is not None
            score = self.mlp_head.predict(
                (
                    camera.camera_score,
                    radar.radar_score,
                    camera.camera_quality,
                    radar.radar_quality,
                    self._sync_quality(sync_delta),
                )
            )
        elif method == "radar_quality_adaptive":
            # This non-default interface implements the requested unnormalised
            # Radar reliability attenuation while preserving the 0.6/0.4 base.
            score = (
                self.camera_weight * camera.camera_score
                + self.radar_weight
                * radar.radar_score
                * (
                    eligibility.radar_quality
                    if eligibility.assessed
                    else radar.radar_quality
                )
            )
        else:
            # Both branches use the same risk direction. Subtraction would make
            # a high radar pre-fall score reduce the multimodal risk score.
            score = (
                contribution_camera * camera.camera_score
                + contribution_radar * radar.radar_score
            )
        return FusionResult(
            fusion_score=score,
            risk_level=self._risk_level(score),
            contribution_camera=contribution_camera,
            contribution_radar=contribution_radar,
            dominant_modality=self._dominant(contribution_camera, contribution_radar),
            method=method,
            fusion_mode=(
                "LOW_CONFIDENCE"
                if max(camera.camera_quality, radar.radar_quality)
                < self.state_machine.config.minimum_modality_quality
                else "NORMAL_FUSION"
            ),
            sync_delta_seconds=sync_delta,
            synchronized=True,
            degraded_mode=(
                "LOW_QUALITY"
                if min(camera.camera_quality, radar.radar_quality)
                < self.state_machine.config.minimum_modality_quality
                else "NONE"
            ),
            reason_codes=(
                ["MODALITY_QUALITY_BELOW_MINIMUM"]
                if min(camera.camera_quality, radar.radar_quality)
                < self.state_machine.config.minimum_modality_quality
                else []
            )
            + (["RADAR_ELIGIBLE"] if eligibility.eligible else []),
            radar_eligibility=eligibility,
        )

    def _single_modality_result(
        self,
        score: float | None,
        modality: str,
        method: str,
        sync_delta: float | None,
        degraded_mode: str,
        reason_codes: list[str],
        reason: str,
        *,
        fusion_mode: str,
        radar_eligibility: RadarEligibilityDecision,
    ) -> FusionResult:
        assert score is not None
        return FusionResult(
            fusion_score=score,
            risk_level=self._risk_level(score),
            contribution_camera=1.0 if modality == "CAMERA" else 0.0,
            contribution_radar=1.0 if modality == "RADAR" else 0.0,
            dominant_modality=modality,
            method=method,
            fusion_mode=fusion_mode,
            sync_delta_seconds=sync_delta,
            synchronized=False,
            degraded_mode=degraded_mode,
            reason_codes=reason_codes,
            degraded_reason=reason,
            radar_eligibility=radar_eligibility,
        )

    def _camera_quality(self, status: FallLiveStatusResponse) -> float:
        confidence = (
            status.mean_keypoint_confidence
            if status.mean_keypoint_confidence is not None
            else status.latest_keypoint_confidence
            if status.latest_keypoint_confidence is not None
            else 0.7
        )
        pose_ratio = (
            min(1.0, status.valid_pose_frames / status.source_window_frames)
            if status.source_window_frames > 0
            else 1.0
        )
        density = min(1.0, status.effective_sample_fps / 15.0)
        return max(0.0, min(1.0, 0.5 * confidence + 0.3 * pose_ratio + 0.2 * density))

    def _risk_level(self, score: float) -> str:
        if score >= self.high_threshold:
            return "HIGH"
        if score >= self.medium_threshold:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _quality_level(quality: float) -> str:
        if quality <= 0:
            return "INSUFFICIENT_DATA"
        if quality >= 0.75:
            return "GOOD"
        return "DEGRADED"

    @staticmethod
    def _dominant(camera: float, radar: float) -> str:
        if abs(camera - radar) < 0.1:
            return "BALANCED"
        return "CAMERA" if camera > radar else "RADAR"

    def _sync_quality(self, delta: float | None) -> float:
        if delta is None:
            return 0.0
        if delta > self.sync_tolerance_seconds:
            return 0.0
        # A pair at the accepted boundary is degraded, not equivalent to a
        # completely missing/misaligned pair.
        return max(0.5, 1.0 - 0.5 * delta / self.sync_tolerance_seconds)
