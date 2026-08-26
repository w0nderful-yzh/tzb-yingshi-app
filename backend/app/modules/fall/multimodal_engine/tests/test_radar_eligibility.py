from datetime import timedelta
import unittest

from app.modules.fall.multimodal_engine.schemas.multimodal import (
    AlignedPersonEvidence,
    RadarEligibilityDecision,
)
from app.modules.fall.multimodal_engine.services.multimodal_fusion import MultimodalFusionService
from app.modules.fall.multimodal_engine.services.radar_eligibility import RadarEligibilityGate
from app.modules.fall.multimodal_engine.tests.test_multimodal_fusion import NOW, MultimodalFusionTest, _UnusedProvider


class RadarEligibilityGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = RadarEligibilityGate()
        self.fusion = MultimodalFusionService(_UnusedProvider(), _UnusedProvider())

    @staticmethod
    def alignment(timestamp, *, track_id: int = 7, point_count: int = 40, z: float = 1.2):
        return AlignedPersonEvidence(
            association_state="MATCHED",
            camera_person_id=0,
            radar_track_id=track_id,
            radar_source_timestamp=timestamp,
            sync_delta_ms=10.0,
            radar_position_xyz_m=(0.2, 2.0, z),
            radar_velocity_xyz_mps=(0.0, 0.0, -0.1),
            radar_point_count=point_count,
            association_confidence=0.65,
            eligible_for_temporal_association=True,
            reason_codes=["TARGET_ASSOCIATION_MATCHED"],
        )

    def evaluate(self, step: int, **alignment_kwargs):
        timestamp = NOW + timedelta(milliseconds=500 * step)
        camera = MultimodalFusionTest.camera(timestamp=timestamp)
        radar = MultimodalFusionTest.radar(timestamp=timestamp)
        decision = self.gate.evaluate(
            camera,
            radar,
            self.alignment(timestamp, **alignment_kwargs),
            sync_tolerance_seconds=2.0,
        )
        return camera, radar, decision

    def test_track_warmup_falls_back_then_matched_track_enters_fixed_baseline(self) -> None:
        camera, radar, first = self.evaluate(0, z=1.20)
        first_fusion = self.fusion.fuse(
            camera,
            radar,
            method="fixed_weighted",
            radar_eligibility=first,
        )
        self.assertFalse(first.eligible)
        self.assertEqual(first_fusion.fusion_mode, "LOW_CONFIDENCE")
        self.assertEqual(first_fusion.fusion_score, camera.camera_score)
        self.assertIn("TRACK_DISCONTINUOUS", first.reason_codes)

        camera, radar, second = self.evaluate(1, z=1.15)
        second_fusion = self.fusion.fuse(
            camera,
            radar,
            method="fixed_weighted",
            radar_eligibility=second,
        )
        self.assertTrue(second.eligible)
        self.assertEqual(second_fusion.fusion_mode, "NORMAL_FUSION")
        self.assertAlmostEqual(second_fusion.fusion_score or 0.0, 0.64)
        self.assertAlmostEqual(second_fusion.contribution_camera, 0.6)
        self.assertAlmostEqual(second_fusion.contribution_radar, 0.4)

    def test_missing_and_mismatched_radar_never_enter_weighted_formula(self) -> None:
        camera = MultimodalFusionTest.camera()
        radar = MultimodalFusionTest.radar()
        missing = RadarEligibilityDecision(
            assessed=True,
            eligible=False,
            reason_codes=["RADAR_MISSING"],
        )
        camera_only = self.fusion.fuse(
            camera,
            radar,
            method="fixed_weighted",
            radar_eligibility=missing,
        )
        self.assertEqual(camera_only.fusion_mode, "CAMERA_ONLY")
        self.assertEqual(camera_only.fusion_score, camera.camera_score)
        self.assertEqual(camera_only.contribution_radar, 0.0)

        mismatch = RadarEligibilityDecision(
            assessed=True,
            eligible=False,
            target_detected=True,
            reason_codes=["TRACK_MISMATCH", "TRACK_CONFLICT"],
        )
        conflict = self.fusion.fuse(
            camera,
            radar,
            method="fixed_weighted",
            radar_eligibility=mismatch,
        )
        self.assertEqual(conflict.fusion_mode, "RADAR_CONFLICT")
        self.assertEqual(conflict.degraded_mode, "MODALITY_CONFLICT")
        self.assertEqual(conflict.fusion_score, camera.camera_score)

    def test_low_point_count_is_explicit_low_quality(self) -> None:
        self.evaluate(0, point_count=2)
        _, _, result = self.evaluate(1, point_count=2)
        self.assertFalse(result.eligible)
        self.assertIn("LOW_QUALITY", result.reason_codes)
        self.assertIn("POINT_COUNT_BELOW_MINIMUM", result.reason_codes)

    def test_radar_quality_adaptive_is_non_default_and_uses_requested_formula(self) -> None:
        decision = RadarEligibilityDecision(
            assessed=True,
            eligible=True,
            target_detected=True,
            target_matched=True,
            synchronized=True,
            track_continuous=True,
            point_cloud_quality_passed=True,
            radar_quality=0.5,
            reason_codes=["RADAR_ELIGIBLE"],
        )
        result = self.fusion.fuse(
            MultimodalFusionTest.camera(score=0.8),
            MultimodalFusionTest.radar(score=0.4),
            method="radar_quality_adaptive",
            radar_eligibility=decision,
        )
        self.assertAlmostEqual(result.fusion_score or 0.0, 0.56)
        self.assertEqual(result.method, "radar_quality_adaptive")


if __name__ == "__main__":
    unittest.main()
