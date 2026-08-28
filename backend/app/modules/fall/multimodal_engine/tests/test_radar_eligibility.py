import unittest
from datetime import timedelta

from app.modules.fall.multimodal_engine.schemas.multimodal import (
    AlignedPersonEvidence,
)
from app.modules.fall.multimodal_engine.services.radar_eligibility import RadarEligibilityGate
from app.modules.fall.multimodal_engine.tests.test_multimodal_fusion import (
    NOW,
    MultimodalFusionTest,
)


class RadarEligibilityGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = RadarEligibilityGate()

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

    def test_track_warmup_then_matched_track_becomes_eligible(self) -> None:
        _, _, first = self.evaluate(0, z=1.20)
        self.assertFalse(first.eligible)
        self.assertIn("TRACK_DISCONTINUOUS", first.reason_codes)

        _, _, second = self.evaluate(1, z=1.15)
        self.assertTrue(second.eligible)
        self.assertIn("RADAR_ELIGIBLE", second.reason_codes)

    def test_low_point_count_is_explicit_low_quality(self) -> None:
        self.evaluate(0, point_count=2)
        _, _, result = self.evaluate(1, point_count=2)
        self.assertFalse(result.eligible)
        self.assertIn("LOW_QUALITY", result.reason_codes)
        self.assertIn("POINT_COUNT_BELOW_MINIMUM", result.reason_codes)


if __name__ == "__main__":
    unittest.main()
