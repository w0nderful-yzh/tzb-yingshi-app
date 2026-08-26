from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from app.modules.fall.multimodal_engine.schemas.multimodal import (
    AlignedPersonEvidence,
    AlignmentAwareRiskAugmentationResult,
    MultimodalLatestResponse,
    MultimodalQualitySummary,
    RadarEligibilityDecision,
)
from app.modules.fall.multimodal_engine.services.camera_led_evidence_fusion_v2 import CameraLedEvidenceFusionV2
from app.modules.fall.multimodal_engine.services.fusion_event_bridge import FusionFindingFactory
from app.modules.fall.multimodal_engine.services.fusion_runtime import FusionShadowLogger
from app.modules.fall.multimodal_engine.services.multimodal_fusion import MultimodalFusionService
from app.modules.fall.multimodal_engine.tests.test_multimodal_fusion import NOW, MultimodalFusionTest, _UnusedProvider


class CameraLedEvidenceFusionV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = CameraLedEvidenceFusionV2()
        self.formal = MultimodalFusionService(_UnusedProvider(), _UnusedProvider())

    @staticmethod
    def alignment(state: str = "MATCHED") -> AlignedPersonEvidence:
        return AlignedPersonEvidence(
            association_state=state,
            camera_person_id=0,
            radar_track_id=7 if state == "MATCHED" else None,
            radar_source_timestamp=NOW,
            sync_delta_ms=10.0,
            radar_position_xyz_m=(0.1, 2.0, 1.2),
            radar_velocity_xyz_mps=(0.0, 0.0, -0.4),
            radar_point_count=30,
            association_confidence=0.8,
            eligible_for_temporal_association=state == "MATCHED",
            reason_codes=["TARGET_ASSOCIATION_MATCHED" if state == "MATCHED" else state],
        )

    @staticmethod
    def eligibility(*, eligible: bool = True, reason: str = "RADAR_ELIGIBLE") -> RadarEligibilityDecision:
        return RadarEligibilityDecision(
            assessed=True,
            eligible=eligible,
            target_detected=True,
            target_matched=eligible,
            synchronized=eligible,
            track_continuous=eligible,
            point_cloud_quality_passed=eligible,
            radar_quality=0.8 if eligible else 0.1,
            reason_codes=[reason],
        )

    @staticmethod
    def associated(camera_score: float, camera_state: str, evidence_state: str, strength: str):
        state = "HIGH" if camera_state == "HIGH" else "WATCH" if camera_state == "MEDIUM" else "NORMAL"
        if evidence_state == "RADAR_MOTION_ANOMALY":
            state = "WATCH"
        return AlignmentAwareRiskAugmentationResult(
            associated_short_term_fall_score=camera_score,
            associated_risk_state=state,
            associated_evidence_state=evidence_state,
            base_camera_score=camera_score,
            base_camera_state=camera_state,
            radar_motion_evidence_strength=strength,
            association_state="MATCHED",
            sync_delta_ms=10.0,
            radar_track_id=7,
            radar_evidence_count=3,
            track_stability=1.0,
            reason_codes=[evidence_state],
        )

    def apply(self, *, camera_score=0.8, camera_state="HIGH", evidence_state="CORROBORATED_HIGH", strength="STRONG"):
        camera = MultimodalFusionTest.camera(score=camera_score, quality=0.9)
        camera.camera_risk_state = camera_state
        radar = MultimodalFusionTest.radar(score=0.4, quality=0.9)
        return self.engine.apply(
            camera,
            radar,
            self.alignment(),
            self.eligibility(),
            self.associated(camera_score, camera_state, evidence_state, strength),
        )

    def test_strong_consistent_evidence_keeps_camera_score_and_high(self) -> None:
        result = self.apply()
        self.assertEqual(result.fusion_mode, "CAMERA_RADAR_CONSISTENT")
        self.assertEqual(result.camera_led_state, "HIGH")
        self.assertEqual(result.camera_led_score, 0.8)
        self.assertFalse(result.radar_score_affects_risk_score)
        self.assertFalse(result.affects_fixed_fusion)
        self.assertFalse(result.affects_alerts)
        self.assertTrue(result.realtime_active)
        self.assertTrue(result.affects_app_result)
        self.assertFalse(result.shadow_only)
        self.assertNotIn("SHADOW_ONLY", result.reason_codes)

    def test_weak_radar_support_is_not_score_averaging(self) -> None:
        result = self.apply(
            camera_score=0.5,
            camera_state="MEDIUM",
            evidence_state="CORROBORATED_WATCH",
            strength="WEAK",
        )
        self.assertEqual(result.fusion_mode, "RADAR_SUPPORTED")
        self.assertEqual(result.camera_led_state, "WATCH")
        self.assertEqual(result.camera_led_score, 0.5)

    def test_conflict_never_vetoes_camera_high(self) -> None:
        result = self.apply(
            evidence_state="MODALITY_CONFLICT",
            strength="NONE",
        )
        self.assertEqual(result.fusion_mode, "RADAR_CONFLICT")
        self.assertEqual(result.camera_led_state, "HIGH")
        self.assertIn("RADAR_CANNOT_VETO_CAMERA_HIGH", result.reason_codes)

    def test_strong_radar_anomaly_can_only_request_watch_from_camera_low(self) -> None:
        result = self.apply(
            camera_score=0.2,
            camera_state="LOW",
            evidence_state="RADAR_MOTION_ANOMALY",
            strength="STRONG",
        )
        self.assertEqual(result.fusion_mode, "RADAR_CONFLICT")
        self.assertEqual(result.camera_led_state, "WATCH")
        self.assertNotEqual(result.camera_led_state, "HIGH")

    def test_missing_radar_is_camera_only_and_low_camera_quality_is_unknown(self) -> None:
        camera = MultimodalFusionTest.camera(score=0.8, quality=0.9)
        camera.camera_risk_state = "HIGH"
        missing = self.engine.apply(
            camera,
            MultimodalFusionTest.radar(score=None),
            AlignedPersonEvidence(association_state="RADAR_TRACK_MISSING"),
            RadarEligibilityDecision(assessed=True, reason_codes=["RADAR_MISSING"]),
            None,
        )
        self.assertEqual(missing.fusion_mode, "CAMERA_ONLY")
        self.assertEqual(missing.camera_led_state, "HIGH")

        low_quality_camera = MultimodalFusionTest.camera(score=0.8, quality=0.1)
        low_quality_camera.camera_risk_state = "HIGH"
        low_confidence = self.engine.apply(
            low_quality_camera,
            MultimodalFusionTest.radar(),
            self.alignment(),
            self.eligibility(),
            self.associated(0.8, "HIGH", "CORROBORATED_HIGH", "STRONG"),
        )
        self.assertEqual(low_confidence.fusion_mode, "LOW_CONFIDENCE")
        self.assertEqual(low_confidence.camera_led_state, "UNKNOWN")

    def test_v2_log_is_explicit_and_alert_bridge_remains_disabled(self) -> None:
        camera = MultimodalFusionTest.camera(score=0.8, quality=0.9)
        camera.camera_risk_state = "HIGH"
        radar = MultimodalFusionTest.radar(score=0.4, quality=0.9)
        formal = self.formal.fuse(camera, radar, method="fixed_weighted")
        self.assertAlmostEqual(formal.fusion_score or 0.0, 0.64)
        v2 = self.apply()
        response = MultimodalLatestResponse(
            camera=camera,
            radar=radar,
            fusion=formal,
            camera_led_evidence_fusion_v2=v2,
            quality=MultimodalQualitySummary(
                camera=0.9,
                radar=0.9,
                synchronization=1.0,
                overall=0.9,
                level="GOOD",
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fusion_v2_shadow.jsonl"
            logger = FusionShadowLogger(path, enabled=True)
            logger.write(response)
            for handler in logger._logger.handlers:
                handler.flush()
                handler.close()
            payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

        block = payload["camera_led_evidence_fusion_v2"]
        self.assertEqual(block["camera_score"], 0.8)
        self.assertEqual(block["radar_score"], 0.4)
        self.assertEqual(block["radar_quality"], 0.8)
        self.assertEqual(block["fusion_mode"], "CAMERA_RADAR_CONSISTENT")
        self.assertIn("reason_codes", block)
        self.assertIsNone(FusionFindingFactory().create(response))


if __name__ == "__main__":
    unittest.main()
