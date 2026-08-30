from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from app.modules.fall.multimodal_engine.schemas.fall_live import (
    FallLiveInputState,
    FallLiveStatusResponse,
)
from app.modules.fall.multimodal_engine.schemas.multimodal import (
    AlignedPersonEvidence,
    CameraEvidence,
    FinalDecisionContext,
    MultimodalLatestResponse,
    MultimodalQualitySummary,
    PhysiologicalEvidence,
    RadarEvidence,
)
from app.modules.fall.multimodal_engine.schemas.radar import RadarStatusResponse
from app.modules.fall.multimodal_engine.services.alignment_aware_risk_augmentation import (
    AlignmentAwareRiskAugmentation,
    AssociatedEvidenceConfig,
)
from app.modules.fall.multimodal_engine.services.camera_led_evidence_fusion_v2 import (
    CameraLedEvidenceFusionV2,
    CameraLedEvidenceFusionV2Config,
)
from app.modules.fall.multimodal_engine.services.camera_radar_alignment import (
    CameraRadarAlignmentAdapter,
)
from app.modules.fall.multimodal_engine.services.dynamic_risk_index import DynamicRiskIndexService
from app.modules.fall.multimodal_engine.services.fusion_runtime import (
    FusionResponseCallback,
    FusionRuntimeTracker,
    FusionShadowLogger,
    FusionTimingTracker,
)
from app.modules.fall.multimodal_engine.services.radar_eligibility import (
    RadarEligibilityConfig,
    RadarEligibilityGate,
)


class MultimodalFusionService:
    """Build the Camera-led decision with associated Radar motion evidence."""

    def __init__(
        self,
        camera_status_provider: Callable[[], FallLiveStatusResponse],
        radar_status_provider: Callable[[], RadarStatusResponse],
        *,
        sync_tolerance_seconds: float = 2.0,
        minimum_camera_quality: float = 0.25,
        shadow_logger: FusionShadowLogger | None = None,
        response_callback: FusionResponseCallback | None = None,
        alignment_adapter: CameraRadarAlignmentAdapter | None = None,
        associated_evidence_config: AssociatedEvidenceConfig | None = None,
        radar_eligibility_config: RadarEligibilityConfig | None = None,
    ) -> None:
        if sync_tolerance_seconds <= 0:
            raise ValueError("sync tolerance must be positive")
        if not 0 <= minimum_camera_quality <= 1:
            raise ValueError("minimum camera quality must be within [0, 1]")
        self.camera_status_provider = camera_status_provider
        self.radar_status_provider = radar_status_provider
        self.sync_tolerance_seconds = sync_tolerance_seconds
        self.timing_tracker = FusionTimingTracker(tolerance_ms=sync_tolerance_seconds * 1000.0)
        self.shadow_logger = shadow_logger
        self.response_callback = response_callback
        self.alignment_adapter = alignment_adapter
        self.radar_eligibility_gate = RadarEligibilityGate(
            radar_eligibility_config
            or RadarEligibilityConfig(enabled=alignment_adapter is not None)
        )
        self.associated_risk_augmentation = AlignmentAwareRiskAugmentation(
            associated_evidence_config
        )
        self.camera_led_evidence_fusion_v2 = CameraLedEvidenceFusionV2(
            CameraLedEvidenceFusionV2Config(minimum_camera_quality=minimum_camera_quality)
        )
        self.dynamic_risk_index = DynamicRiskIndexService()
        self.runtime_tracker = FusionRuntimeTracker()

    def get_latest(self) -> MultimodalLatestResponse:
        now = datetime.now(UTC)
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
        sync_delta = (
            abs((camera.timestamp - radar.timestamp).total_seconds())
            if camera.available and radar.available
            else None
        )
        sync_quality = self._sync_quality(sync_delta)
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
        now = now or datetime.now(UTC)
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
            "torso_inclination_deg": status.torso_inclination_deg,
            "com_proxy_relative_change": status.com_proxy_relative_change,
            "yaw_delta_deg": status.yaw_delta_deg,
            "pose_quality": status.pose_quality,
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
        now = now or datetime.now(UTC)
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
            "missing_frame_ratio": getattr(source_prediction, "missing_frame_ratio", None),
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
            radar_risk_state=(source.risk_state if available and source else "UNKNOWN"),
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

    @staticmethod
    def _quality_level(quality: float) -> str:
        if quality <= 0:
            return "INSUFFICIENT_DATA"
        if quality >= 0.75:
            return "GOOD"
        return "DEGRADED"

    def _sync_quality(self, delta: float | None) -> float:
        if delta is None:
            return 0.0
        if delta > self.sync_tolerance_seconds:
            return 0.0
        # A pair at the accepted boundary is degraded, not equivalent to a
        # completely missing/misaligned pair.
        return max(0.5, 1.0 - 0.5 * delta / self.sync_tolerance_seconds)
