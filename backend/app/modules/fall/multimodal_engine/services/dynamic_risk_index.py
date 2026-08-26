from __future__ import annotations

from app.modules.fall.multimodal_engine.schemas.multimodal import (
    CameraEvidence,
    CameraLedEvidenceFusionV2Result,
    DynamicRiskIndex,
    ExplainableRiskReason,
    FallEventSummary,
    FusionResult,
    RadarEvidence,
    ShortTermFallWarning,
)
from app.modules.fall.multimodal_engine.schemas.radar import RadarStatusResponse


_LABELS = {
    "POSTURE_ABNORMAL": "姿态异常",
    "ACTIVITY_ABNORMAL": "活动异常",
    "MOTION_STABILITY_DECLINE": "运动稳定性下降",
    "RAPID_HEIGHT_CHANGE": "高度快速变化",
    "MODALITY_QUALITY_DECLINE": "模态质量下降",
}


class DynamicRiskIndexService:
    """Organize existing evidence into three system layers without changing inference."""

    def build_dynamic_risk(
        self,
        camera: CameraEvidence,
        radar: RadarEvidence,
        radar_status: RadarStatusResponse | None,
    ) -> DynamicRiskIndex:
        reasons = self._context_reasons(camera, radar)
        assessment = (
            radar_status.radar_debug.fall_risk_assessment
            if radar_status is not None and radar_status.radar_debug is not None
            else None
        )
        if assessment is None or assessment.risk_score is None:
            return DynamicRiskIndex(
                reasons=reasons,
                components={
                    "camera_posture_context": camera.camera_score,
                    "radar_sway": None,
                    "radar_mobility": None,
                    "radar_descent": None,
                },
                disclaimer=(
                    "长期评估窗口尚未就绪。系统不会用短时预警分数代替动态风险指数。"
                ),
            )

        component_reasons = [
            self._component_reason(
                "MOTION_STABILITY_DECLINE",
                assessment.sway_risk,
                "雷达持续窗口观察到身体高度或质心稳定性变化",
            ),
            self._component_reason(
                "ACTIVITY_ABNORMAL",
                assessment.mobility_risk,
                "雷达持续窗口观察到活动强度偏低或运动突发",
            ),
            self._component_reason(
                "RAPID_HEIGHT_CHANGE",
                assessment.descent_risk,
                "雷达运动学分量观察到快速下降证据",
            ),
        ]
        reasons.extend(reason for reason in component_reasons if reason is not None)
        return DynamicRiskIndex(
            dynamic_risk_score=assessment.risk_score,
            risk_level=assessment.risk_level,
            assessment_window_seconds=assessment.assessment_window_seconds,
            valid_window_count=assessment.valid_window_count,
            observed_duration_seconds=assessment.observed_duration_seconds,
            source_method="radar_60s_shadow_with_camera_context_v1",
            components={
                "camera_posture_context": camera.camera_score,
                "radar_sway": assessment.sway_risk,
                "radar_mobility": assessment.mobility_risk,
                "radar_descent": assessment.descent_risk,
            },
            reasons=self._deduplicate(reasons),
            available=True,
            disclaimer=assessment.disclaimer,
        )

    def build_short_term_warning(
        self,
        camera: CameraEvidence,
        radar: RadarEvidence,
        fusion: FusionResult,
    ) -> ShortTermFallWarning:
        reasons = self._context_reasons(camera, radar)
        return ShortTermFallWarning(
            short_term_fall_score=(
                fusion.stable_fusion_score
                if fusion.stable_fusion_score is not None
                else fusion.raw_fusion_score
            ),
            state=fusion.stable_fusion_state,
            method=fusion.method,
            degraded_mode=fusion.degraded_mode,
            synchronized=fusion.synchronized,
            reasons=self._deduplicate(reasons),
        )

    def build_camera_led_short_term_warning(
        self,
        camera: CameraEvidence,
        radar: RadarEvidence,
        fusion_v2: CameraLedEvidenceFusionV2Result,
    ) -> ShortTermFallWarning:
        """Project the unchanged Fusion v2 decision into the active API layer."""

        degraded_mode = {
            "CAMERA_ONLY": "CAMERA_ONLY",
            "RADAR_SUPPORTED": "NONE",
            "CAMERA_RADAR_CONSISTENT": "NONE",
            "RADAR_CONFLICT": "MODALITY_CONFLICT",
            "LOW_CONFIDENCE": "LOW_QUALITY",
        }[fusion_v2.fusion_mode]
        return ShortTermFallWarning(
            short_term_fall_score=fusion_v2.camera_led_score,
            state=fusion_v2.camera_led_state,
            method="camera_led_evidence_v2",
            degraded_mode=degraded_mode,
            synchronized=(
                fusion_v2.association_state == "MATCHED"
                and fusion_v2.sync_delta_ms is not None
                and fusion_v2.radar_eligible
            ),
            reasons=self._deduplicate(self._context_reasons(camera, radar)),
        )

    def build_fall_event(
        self,
        camera: CameraEvidence,
        radar: RadarEvidence,
    ) -> FallEventSummary:
        camera_event_id = self._feature(camera.camera_feature, "last_event_id")
        radar_event_id = self._feature(radar.radar_feature, "event_id")
        radar_triggered = self._feature(radar.radar_feature, "event_triggered") is True
        if camera_event_id and camera.camera_risk_state == "HIGH":
            return FallEventSummary(
                fall_event_status="SUSPECTED",
                source_event_id=str(camera_event_id),
                source="CAMERA",
                reason_codes=["CAMERA_WARNING_EVENT_RECORDED"],
                summary="摄像头链路记录了风险事件，仍需人工确认是否实际跌倒",
                requires_human_confirmation=True,
            )
        if radar_triggered or radar.radar_risk_state == "CONFIRMED":
            return FallEventSummary(
                fall_event_status="SUSPECTED",
                source_event_id=str(radar_event_id) if radar_event_id else None,
                source="RADAR",
                reason_codes=["RADAR_SHADOW_GATE_TRIGGERED"],
                summary="雷达 shadow 门控达到事件条件，不能单独确认为实际跌倒",
                requires_human_confirmation=True,
            )
        if not camera.available and not radar.available:
            return FallEventSummary(
                fall_event_status="UNKNOWN",
                reason_codes=["NO_VALID_MODALITY"],
                summary="双路证据不可用，无法判断跌倒事件状态",
            )
        return FallEventSummary(
            fall_event_status="NO_EVENT",
            reason_codes=["NO_EVENT_EVIDENCE_OBSERVED"],
            summary="当前有效证据未观察到跌倒事件",
        )

    def _context_reasons(
        self,
        camera: CameraEvidence,
        radar: RadarEvidence,
    ) -> list[ExplainableRiskReason]:
        reasons: list[ExplainableRiskReason] = []
        if camera.available and camera.camera_risk_state in {"MEDIUM", "HIGH"}:
            reasons.append(
                ExplainableRiskReason(
                    code="POSTURE_ABNORMAL",
                    label=_LABELS["POSTURE_ABNORMAL"],
                    source="CAMERA",
                    signal_value=camera.camera_score,
                    detail="摄像头姿态链路输出中高风险状态，仅作为当前姿态上下文",
                    affects_score=False,
                )
            )
        radar_state_risky = radar.radar_risk_state in {"WATCH", "IMMINENT", "CONFIRMED"}
        height_delta = self._feature(radar.radar_feature, "height_delta")
        if radar.available and radar_state_risky and isinstance(height_delta, (int, float)) and height_delta < 0:
            reasons.append(
                ExplainableRiskReason(
                    code="RAPID_HEIGHT_CHANGE",
                    label=_LABELS["RAPID_HEIGHT_CHANGE"],
                    source="RADAR",
                    signal_value=radar.radar_score,
                    detail="雷达短时风险状态伴随负向高度变化",
                    affects_score=False,
                )
            )
        if camera.quality_level != "GOOD" or radar.quality_level != "GOOD":
            reasons.append(
                ExplainableRiskReason(
                    code="MODALITY_QUALITY_DECLINE",
                    label=_LABELS["MODALITY_QUALITY_DECLINE"],
                    source="MULTIMODAL",
                    signal_value=min(camera.camera_quality, radar.radar_quality),
                    detail="至少一路模态质量降级或不可用，系统按降级模式解释结果",
                    affects_score=False,
                )
            )
        return reasons

    @staticmethod
    def _component_reason(
        code: str,
        value: float | None,
        detail: str,
    ) -> ExplainableRiskReason | None:
        if value is None or value <= 0:
            return None
        return ExplainableRiskReason(
            code=code,
            label=_LABELS[code],
            source="RADAR",
            signal_value=value,
            detail=detail,
            affects_score=True,
        )

    @staticmethod
    def _feature(feature: object, key: str) -> object | None:
        return feature.get(key) if isinstance(feature, dict) else None

    @staticmethod
    def _deduplicate(
        reasons: list[ExplainableRiskReason],
    ) -> list[ExplainableRiskReason]:
        result: list[ExplainableRiskReason] = []
        seen: set[tuple[str, str, bool]] = set()
        for reason in reasons:
            key = (reason.code, reason.source, reason.affects_score)
            if key not in seen:
                result.append(reason)
                seen.add(key)
        return result
