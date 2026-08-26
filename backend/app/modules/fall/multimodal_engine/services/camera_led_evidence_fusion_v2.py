from __future__ import annotations

from dataclasses import dataclass

from app.modules.fall.multimodal_engine.schemas.multimodal import (
    AlignedPersonEvidence,
    AlignmentAwareRiskAugmentationResult,
    CameraEvidence,
    CameraLedEvidenceFusionV2Result,
    RadarEligibilityDecision,
    RadarEvidence,
)


@dataclass(frozen=True, slots=True)
class CameraLedEvidenceFusionV2Config:
    """Realtime interpretation controls; these do not change model thresholds."""

    minimum_camera_quality: float = 0.25

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_camera_quality <= 1:
            raise ValueError("minimum Camera quality must be within [0, 1]")


class CameraLedEvidenceFusionV2:
    """Interpret Radar as evidence around the unchanged Camera decision.

    The numeric result is always the Camera score. Radar can corroborate,
    expose conflict, or request WATCH when strong associated motion conflicts
    with Camera LOW. It never participates in score averaging and never lowers
    a valid Camera HIGH.
    """

    _CAMERA_STATE = {
        "LOW": "NORMAL",
        "MEDIUM": "WATCH",
        "HIGH": "HIGH",
        "UNKNOWN": "UNKNOWN",
    }

    def __init__(self, config: CameraLedEvidenceFusionV2Config | None = None) -> None:
        self.config = config or CameraLedEvidenceFusionV2Config()

    def apply(
        self,
        camera: CameraEvidence,
        radar: RadarEvidence,
        alignment: AlignedPersonEvidence,
        eligibility: RadarEligibilityDecision,
        associated: AlignmentAwareRiskAugmentationResult | None,
    ) -> CameraLedEvidenceFusionV2Result:
        reasons = [
            "REALTIME_ACTIVE",
            "CAMERA_LED_EVIDENCE_FUSION_V2",
            "RADAR_EVIDENCE_NOT_WEIGHTED_SCORE",
            *alignment.reason_codes,
            *eligibility.reason_codes,
        ]
        strength = (
            associated.radar_motion_evidence_strength
            if associated is not None
            else "UNKNOWN"
        )
        if associated is not None:
            # The associated motion assessor remains an internal shadow
            # component, while this v2 projection is now the active app
            # result. Preserve its evidence reasons without leaking the old
            # pipeline-level SHADOW_ONLY label into the active contract.
            reasons.extend(
                code for code in associated.reason_codes if code != "SHADOW_ONLY"
            )

        if not camera.available or camera.camera_score is None:
            return self._result(
                camera,
                radar,
                alignment,
                eligibility,
                strength,
                state="UNKNOWN",
                mode="LOW_CONFIDENCE",
                reasons=[*reasons, "CAMERA_EVIDENCE_UNAVAILABLE"],
            )
        if camera.camera_quality < self.config.minimum_camera_quality:
            return self._result(
                camera,
                radar,
                alignment,
                eligibility,
                strength,
                state="UNKNOWN",
                mode="LOW_CONFIDENCE",
                reasons=[*reasons, "CAMERA_QUALITY_BELOW_MINIMUM"],
            )

        camera_state = self._CAMERA_STATE[camera.camera_risk_state]
        if not radar.available or radar.radar_score is None:
            return self._result(
                camera,
                radar,
                alignment,
                eligibility,
                strength,
                state=camera_state,
                mode="CAMERA_ONLY",
                reasons=[*reasons, "RADAR_UNAVAILABLE_CAMERA_DECISION_PRESERVED"],
            )

        if alignment.association_state in {"TRACK_CONFLICT", "MULTIPLE_CANDIDATES"}:
            return self._result(
                camera,
                radar,
                alignment,
                eligibility,
                strength,
                state=camera_state,
                mode="RADAR_CONFLICT",
                reasons=[*reasons, "TARGET_ASSOCIATION_CONFLICT", "RADAR_CANNOT_VETO_CAMERA_HIGH"],
            )

        if not eligibility.eligible:
            reason_set = set(eligibility.reason_codes)
            is_conflict = bool(reason_set.intersection({"TRACK_MISMATCH", "TRACK_CONFLICT", "MULTIPLE_CANDIDATES"}))
            return self._result(
                camera,
                radar,
                alignment,
                eligibility,
                strength,
                state=camera_state,
                mode="RADAR_CONFLICT" if is_conflict else "CAMERA_ONLY",
                reasons=[
                    *reasons,
                    "RADAR_INELIGIBLE_CAMERA_DECISION_PRESERVED",
                    "RADAR_CANNOT_VETO_CAMERA_HIGH",
                ],
            )

        if associated is None:
            return self._result(
                camera,
                radar,
                alignment,
                eligibility,
                strength,
                state=camera_state,
                mode="LOW_CONFIDENCE",
                reasons=[*reasons, "ASSOCIATED_RADAR_EVIDENCE_NOT_ASSESSED"],
            )

        evidence_state = associated.associated_evidence_state
        if evidence_state == "NORMAL_CORROBORATED":
            mode = "CAMERA_RADAR_CONSISTENT"
            reasons.append("CAMERA_RADAR_NORMAL_EVIDENCE_CONSISTENT")
        elif evidence_state == "CORROBORATED_HIGH":
            mode = (
                "CAMERA_RADAR_CONSISTENT"
                if strength == "STRONG"
                else "RADAR_SUPPORTED"
            )
            reasons.append("CAMERA_RISK_SUPPORTED_BY_ASSOCIATED_RADAR_MOTION")
        elif evidence_state == "CORROBORATED_WATCH":
            mode = "RADAR_SUPPORTED"
            reasons.append("CAMERA_WATCH_SUPPORTED_BY_ASSOCIATED_RADAR_MOTION")
        elif evidence_state in {"RADAR_MOTION_ANOMALY", "MODALITY_CONFLICT"}:
            mode = "RADAR_CONFLICT"
            reasons.append("CAMERA_RADAR_STATE_CONFLICT")
        elif evidence_state in {
            "CAMERA_ONLY_NORMAL",
            "CAMERA_ONLY_WATCH",
            "CAMERA_ONLY_HIGH",
            "NOT_ASSOCIATED",
        }:
            mode = "CAMERA_ONLY"
            reasons.append("RADAR_NOT_USED_FOR_STATE_INTERPRETATION")
        else:
            mode = "LOW_CONFIDENCE"
            reasons.append("ASSOCIATED_EVIDENCE_STATE_UNKNOWN")

        # Only a strong, eligible and associated Radar motion anomaly may ask
        # for WATCH when Camera is LOW. It still cannot create HIGH.
        state = camera_state
        if evidence_state == "RADAR_MOTION_ANOMALY" and camera_state == "NORMAL":
            state = "WATCH"
            reasons.extend(
                [
                    "STRONG_ASSOCIATED_RADAR_MOTION_REQUESTS_WATCH",
                    "RADAR_CANNOT_ESCALATE_CAMERA_LOW_TO_HIGH",
                ]
            )
        if camera_state == "HIGH":
            state = "HIGH"
            reasons.append("RADAR_CANNOT_VETO_CAMERA_HIGH")

        return self._result(
            camera,
            radar,
            alignment,
            eligibility,
            strength,
            state=state,
            mode=mode,
            reasons=reasons,
        )

    @staticmethod
    def _result(
        camera: CameraEvidence,
        radar: RadarEvidence,
        alignment: AlignedPersonEvidence,
        eligibility: RadarEligibilityDecision,
        strength: str,
        *,
        state: str,
        mode: str,
        reasons: list[str],
    ) -> CameraLedEvidenceFusionV2Result:
        return CameraLedEvidenceFusionV2Result(
            camera_led_score=camera.camera_score,
            camera_led_state=state,
            fusion_mode=mode,
            camera_score=camera.camera_score,
            radar_score=radar.radar_score,
            camera_quality=camera.camera_quality,
            radar_quality=(
                eligibility.radar_quality
                if eligibility.assessed
                else radar.radar_quality
            ),
            radar_eligible=eligibility.eligible,
            radar_motion_evidence_strength=strength,
            association_state=alignment.association_state,
            sync_delta_ms=alignment.sync_delta_ms,
            reason_codes=list(dict.fromkeys(reasons)),
        )
