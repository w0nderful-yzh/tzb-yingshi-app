from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest

from app.modules.fall.multimodal_engine.schemas.fall_live import (
    CameraAlignmentSnapshot,
    FallLiveInputState,
    FallLiveState,
    FallLiveStatusResponse,
)
from app.modules.fall.multimodal_engine.schemas.multimodal import AlignedPersonEvidence, CameraEvidence, RadarEvidence
from app.modules.fall.multimodal_engine.schemas.radar import RadarAlignmentEvidencePayload, RadarStatusResponse
from app.modules.fall.multimodal_engine.services.camera_radar_alignment import (
    CameraRadarAlignmentAdapter,
    RadarTrackEvidenceBuffer,
)
from app.modules.fall.multimodal_engine.services.alignment_aware_risk_augmentation import (
    AlignmentAwareRiskAugmentation,
)
from app.modules.fall.multimodal_engine.services.multimodal_fusion import MultimodalFusionService
from app.modules.fall.multimodal_engine.services.temporal_associated_fusion import TemporalAssociatedFusion
from app.modules.fall.multimodal_engine.tests.test_multimodal_fusion import NOW, _UnusedProvider


def _calibration(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "calibration_version": "unit_shadow_v0",
                "mapping": {
                    "matrix_2x3": [[100.0, 0.0, 500.0], [0.0, -100.0, 800.0]]
                },
                "valid_radar_region_xy_m": {
                    "x_min": -1.0,
                    "x_max": 1.0,
                    "y_min": 1.0,
                    "y_max": 4.0,
                },
                "gates": {
                    "max_sync_delta_ms": 50.0,
                    "uncertainty_radius_px": 150.0,
                },
                "training_metrics": {"leave_one_out_p95_px": 100.0},
                "runtime_enabled": False,
                "fusion_enabled": False,
            }
        ),
        encoding="utf-8",
    )


class CameraRadarAlignmentAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / "calibration.json"
        _calibration(path)
        self.adapter = CameraRadarAlignmentAdapter(path)

    def tearDown(self) -> None:
        self.directory.cleanup()

    @staticmethod
    def camera() -> FallLiveStatusResponse:
        return FallLiveStatusResponse(
            enabled=True,
            state=FallLiveState.RUNNING,
            input_state=FallLiveInputState.READY,
            target_present=True,
            training_input_ready=True,
            alignment_snapshot=CameraAlignmentSnapshot(
                frame_id="1:10",
                source_timestamp=NOW,
                camera_person_id=0,
                detected=True,
                image_size=(1280, 720),
                bbox_xyxy=(450.0, 350.0, 650.0, 720.0),
                footpoint_uv=(550.0, 700.0),
                footpoint_confidence=0.9,
                footpoint_source="ANKLE_MIDPOINT_15_16",
            ),
        )

    @staticmethod
    def radar(*, offset_ms: int = 0, second: bool = False) -> RadarStatusResponse:
        targets = [
            RadarAlignmentEvidencePayload(
                frame_number=20,
                source_timestamp=NOW + timedelta(milliseconds=offset_ms),
                track_id=7,
                x=0.5,
                y=1.0,
                z=1.2,
                vx=0.1,
                vy=-0.2,
                vz=-0.1,
                point_count=40,
                point_cloud_spread_m=0.22,
                radar_quality=0.9,
            )
        ]
        if second:
            targets.append(
                RadarAlignmentEvidencePayload(
                    frame_number=20,
                    source_timestamp=NOW + timedelta(milliseconds=offset_ms),
                    track_id=8,
                    x=0.4,
                    y=1.1,
                    z=1.1,
                    point_count=30,
                    radar_quality=0.8,
                )
            )
        return RadarStatusResponse(online=True, alignment_evidence=targets)

    def test_matches_one_track_without_affecting_fixed_fusion(self) -> None:
        alignment = self.adapter.apply(self.camera(), self.radar())
        self.assertEqual(alignment.association_state, "MATCHED")
        self.assertEqual(alignment.radar_track_id, 7)
        self.assertTrue(alignment.eligible_for_temporal_association)
        self.assertEqual(alignment.radar_point_count, 40)
        self.assertAlmostEqual(alignment.radar_point_cloud_spread_m or 0.0, 0.22)
        self.assertLessEqual(alignment.association_confidence, 0.70)
        self.assertFalse(alignment.affects_fixed_fusion)

        fusion_service = MultimodalFusionService(_UnusedProvider(), _UnusedProvider())
        camera = self._camera_evidence(NOW)
        radar = self._radar_evidence(NOW)
        before = fusion_service.fuse(camera, radar, method="fixed_weighted")
        self.adapter.apply(self.camera(), self.radar())
        after = fusion_service.fuse(camera, radar, method="fixed_weighted")
        self.assertEqual(before.model_dump(), after.model_dump())

    def test_out_of_sync_and_multiple_candidates_are_explicit(self) -> None:
        self.assertEqual(
            self.adapter.apply(self.camera(), self.radar(offset_ms=60)).association_state,
            "OUT_OF_SYNC",
        )
        self.assertEqual(
            self.adapter.apply(self.camera(), self.radar(second=True)).association_state,
            "MULTIPLE_CANDIDATES",
        )

    def test_buffer_uses_nearest_radar_frame_instead_of_latest(self) -> None:
        buffer = RadarTrackEvidenceBuffer(
            clock=lambda: NOW + timedelta(milliseconds=500),
        )
        close = self._buffered_radar(
            timestamp=NOW - timedelta(milliseconds=20),
            frame_number=20,
        )
        latest = self._buffered_radar(
            timestamp=NOW + timedelta(milliseconds=300),
            frame_number=21,
        )
        buffer.observe(close)
        buffer.observe(latest)
        adapter = CameraRadarAlignmentAdapter(
            self.adapter.calibration_path,
            radar_track_buffer=buffer,
        )

        alignment = adapter.apply(self.camera(), latest)

        self.assertEqual(alignment.association_state, "MATCHED")
        self.assertEqual(alignment.radar_frame_number, 20)
        self.assertAlmostEqual(alignment.sync_delta_ms or 0.0, 20.0)
        self.assertIn(
            "RADAR_TRACK_BUFFER_NEAREST_TIMESTAMP",
            alignment.reason_codes,
        )

    def test_buffer_keeps_existing_50ms_gate(self) -> None:
        buffer = RadarTrackEvidenceBuffer(
            clock=lambda: NOW + timedelta(milliseconds=100),
        )
        radar = self._buffered_radar(
            timestamp=NOW + timedelta(milliseconds=60),
            frame_number=20,
        )
        buffer.observe(radar)
        adapter = CameraRadarAlignmentAdapter(
            self.adapter.calibration_path,
            radar_track_buffer=buffer,
        )

        alignment = adapter.apply(self.camera(), radar)

        self.assertEqual(alignment.association_state, "OUT_OF_SYNC")
        self.assertAlmostEqual(alignment.sync_delta_ms or 0.0, 60.0)
        self.assertIn("ALIGNMENT_FRAME_SYNC_GATE_EXCEEDED", alignment.reason_codes)

    def test_buffer_rejects_stale_evidence(self) -> None:
        buffer = RadarTrackEvidenceBuffer(
            freshness_seconds=1.0,
            clock=lambda: NOW + timedelta(milliseconds=1100),
        )
        radar = self._buffered_radar(timestamp=NOW, frame_number=20)
        buffer.observe(radar)
        adapter = CameraRadarAlignmentAdapter(
            self.adapter.calibration_path,
            radar_track_buffer=buffer,
        )

        alignment = adapter.apply(self.camera(), radar)

        self.assertEqual(alignment.association_state, "OUT_OF_SYNC")
        self.assertIn(
            "RADAR_TRACK_BUFFER_EVIDENCE_STALE",
            alignment.reason_codes,
        )

    def test_radar_frame_rollback_clears_pre_reset_buffer(self) -> None:
        buffer = RadarTrackEvidenceBuffer(
            clock=lambda: NOW + timedelta(milliseconds=500),
        )
        old = self._buffered_radar(timestamp=NOW, frame_number=20)
        restarted = self._buffered_radar(
            timestamp=NOW + timedelta(milliseconds=100),
            frame_number=1,
        )
        buffer.observe(old)
        generation_before = buffer.generation
        buffer.observe(restarted)
        adapter = CameraRadarAlignmentAdapter(
            self.adapter.calibration_path,
            radar_track_buffer=buffer,
        )

        alignment = adapter.apply(self.camera(), restarted)

        self.assertGreater(buffer.generation, generation_before)
        self.assertEqual(buffer.frame_count(room="living_room", device_id="radar-1"), 1)
        self.assertEqual(alignment.radar_frame_number, 1)
        self.assertEqual(alignment.association_state, "OUT_OF_SYNC")

    def test_offline_status_clears_buffer_and_degrades_safely(self) -> None:
        buffer = RadarTrackEvidenceBuffer(clock=lambda: NOW)
        radar = self._buffered_radar(timestamp=NOW, frame_number=20)
        buffer.observe(radar)
        self.assertEqual(buffer.frame_count(room="living_room", device_id="radar-1"), 1)
        offline = RadarStatusResponse(
            online=False,
            room="living_room",
            device_id="radar-1",
            source_mode="REAL",
        )
        buffer.observe(offline)
        adapter = CameraRadarAlignmentAdapter(
            self.adapter.calibration_path,
            radar_track_buffer=buffer,
        )

        alignment = adapter.apply(self.camera(), offline)

        self.assertEqual(buffer.frame_count(room="living_room", device_id="radar-1"), 0)
        self.assertEqual(alignment.association_state, "RADAR_TRACK_MISSING")
        self.assertIn("RADAR_STREAM_OFFLINE", alignment.reason_codes)

    def test_temporal_high_requires_matched_alignment(self) -> None:
        camera_samples = [self._camera_evidence(NOW + timedelta(seconds=i)) for i in range(2)]
        radar_samples = [self._radar_evidence(NOW + timedelta(seconds=i)) for i in range(2)]
        fixed_service = MultimodalFusionService(_UnusedProvider(), _UnusedProvider())

        matched_engine = TemporalAssociatedFusion()
        unknown_engine = TemporalAssociatedFusion()
        matched_result = None
        unknown_result = None
        for camera, radar in zip(camera_samples, radar_samples):
            fixed = fixed_service.fuse(camera, radar, method="fixed_weighted")
            matched_result = matched_engine.apply(
                camera,
                radar,
                fixed,
                alignment=AlignedPersonEvidence(
                    association_state="MATCHED",
                    camera_person_id=0,
                    radar_track_id=7,
                    eligible_for_temporal_association=True,
                    reason_codes=["TARGET_ASSOCIATION_MATCHED"],
                ),
            )
            unknown_result = unknown_engine.apply(
                camera,
                radar,
                fixed,
                alignment=AlignedPersonEvidence(
                    association_state="RADAR_TRACK_MISSING",
                    camera_person_id=0,
                    reason_codes=["RADAR_TRACK_UNAVAILABLE"],
                ),
            )
        assert matched_result is not None and unknown_result is not None
        self.assertEqual(matched_result.fusion_state, "HIGH")
        self.assertEqual(unknown_result.fusion_state, "WATCH")
        self.assertEqual(unknown_result.target_association, "UNKNOWN")

    def test_spatial_conflict_without_risk_is_unknown_not_watch(self) -> None:
        camera = self._camera_evidence(NOW)
        camera.camera_risk_state = "LOW"
        radar = self._radar_evidence(NOW)
        radar.radar_risk_state = "NORMAL"
        radar.radar_feature["risk_state"] = "NORMAL"  # type: ignore[index]
        fixed_service = MultimodalFusionService(_UnusedProvider(), _UnusedProvider())
        fixed = fixed_service.fuse(camera, radar, method="fixed_weighted")
        result = TemporalAssociatedFusion().apply(
            camera,
            radar,
            fixed,
            alignment=AlignedPersonEvidence(
                association_state="TRACK_CONFLICT",
                camera_person_id=0,
                radar_track_id=7,
                reason_codes=["SPATIAL_GATE_FAILED"],
            ),
        )
        self.assertEqual(result.fusion_state, "UNKNOWN")
        self.assertEqual(result.degraded_mode, "MODALITY_CONFLICT")

    @staticmethod
    def _camera_evidence(timestamp):
        return CameraEvidence(
            camera_score=0.85,
            camera_risk_state="HIGH",
            camera_feature={"target_present": True},
            camera_quality=0.9,
            quality_level="GOOD",
            timestamp=timestamp,
            available=True,
        )

    @staticmethod
    def _buffered_radar(
        *,
        timestamp,
        frame_number: int,
    ) -> RadarStatusResponse:
        return RadarStatusResponse(
            online=True,
            room="living_room",
            device_id="radar-1",
            source_mode="REAL",
            alignment_evidence=[
                RadarAlignmentEvidencePayload(
                    frame_number=frame_number,
                    source_timestamp=timestamp,
                    track_id=7,
                    x=0.5,
                    y=1.0,
                    z=1.2,
                    vx=0.1,
                    vy=-0.2,
                    vz=-0.1,
                    point_count=40,
                    point_cloud_spread_m=0.22,
                    radar_quality=0.9,
                    radar_config_name="ISK_6m_55ms_ab.cfg",
                )
            ],
        )

    @staticmethod
    def _radar_evidence(timestamp):
        return RadarEvidence(
            radar_score=0.75,
            radar_risk_state="IMMINENT",
            radar_feature={
                "risk_state": "IMMINENT",
                "point_count": 40,
                "vertical_velocity": -0.5,
                "height_delta": -0.3,
                "motion_direction": "DESCENDING",
            },
            radar_quality=0.9,
            quality_level="GOOD",
            timestamp=timestamp,
            available=True,
        )


class AlignmentAwareRiskAugmentationTest(unittest.TestCase):
    @staticmethod
    def camera(timestamp, *, state: str = "HIGH", score: float = 0.85):
        return CameraEvidence(
            camera_score=score,
            camera_risk_state=state,
            camera_feature={"target_present": True},
            camera_quality=0.9,
            quality_level="GOOD",
            timestamp=timestamp,
            available=True,
            device_id="camera-living-room",
        )

    @staticmethod
    def alignment(timestamp, *, z: float, vz: float):
        return AlignedPersonEvidence(
            association_state="MATCHED",
            camera_person_id=0,
            radar_track_id=7,
            radar_frame_number=int(timestamp.timestamp() * 10),
            radar_source_timestamp=timestamp,
            sync_delta_ms=15.0,
            radar_position_xyz_m=(0.2, 2.0, z),
            radar_velocity_xyz_mps=(0.1, 0.0, vz),
            radar_point_count=30,
            radar_point_cloud_spread_m=0.25,
            association_confidence=0.65,
            eligible_for_temporal_association=True,
            reason_codes=["TARGET_ASSOCIATION_MATCHED"],
        )

    def test_camera_score_is_unchanged_when_descent_is_corroborated(self) -> None:
        engine = AlignmentAwareRiskAugmentation()
        engine.apply(self.camera(NOW), self.alignment(NOW, z=1.30, vz=-0.40))
        result = engine.apply(
            self.camera(NOW + timedelta(milliseconds=600)),
            self.alignment(
                NOW + timedelta(milliseconds=600),
                z=1.00,
                vz=-0.45,
            ),
        )

        self.assertEqual(result.associated_short_term_fall_score, 0.85)
        self.assertEqual(result.base_camera_score, 0.85)
        self.assertEqual(result.associated_evidence_state, "CORROBORATED_HIGH")
        self.assertEqual(result.radar_motion_evidence_strength, "STRONG")
        self.assertEqual(result.associated_risk_state, "HIGH")
        self.assertFalse(result.uses_radar_tcn_score)
        self.assertFalse(result.affects_fixed_fusion)
        self.assertFalse(result.affects_alerts)

    def test_radar_motion_cannot_raise_camera_low_to_high(self) -> None:
        engine = AlignmentAwareRiskAugmentation()
        engine.apply(
            self.camera(NOW, state="LOW", score=0.10),
            self.alignment(NOW, z=1.30, vz=-0.40),
        )
        result = engine.apply(
            self.camera(
                NOW + timedelta(milliseconds=600),
                state="LOW",
                score=0.10,
            ),
            self.alignment(
                NOW + timedelta(milliseconds=600),
                z=1.00,
                vz=-0.45,
            ),
        )

        self.assertEqual(result.associated_evidence_state, "RADAR_MOTION_ANOMALY")
        self.assertEqual(result.associated_risk_state, "WATCH")
        self.assertEqual(result.associated_short_term_fall_score, 0.10)

    def test_missing_radar_cannot_lower_camera_high(self) -> None:
        result = AlignmentAwareRiskAugmentation().apply(
            self.camera(NOW),
            AlignedPersonEvidence(
                association_state="RADAR_TRACK_MISSING",
                camera_person_id=0,
                reason_codes=["RADAR_TRACK_UNAVAILABLE"],
            ),
        )

        self.assertEqual(result.associated_evidence_state, "CAMERA_ONLY_HIGH")
        self.assertEqual(result.associated_risk_state, "HIGH")
        self.assertEqual(result.associated_short_term_fall_score, 0.85)

    def test_reliable_stable_radar_exposes_camera_high_conflict_in_shadow(self) -> None:
        engine = AlignmentAwareRiskAugmentation()
        engine.apply(self.camera(NOW), self.alignment(NOW, z=1.30, vz=0.0))
        result = engine.apply(
            self.camera(NOW + timedelta(milliseconds=600)),
            self.alignment(
                NOW + timedelta(milliseconds=600),
                z=1.30,
                vz=0.0,
            ),
        )

        self.assertEqual(result.radar_motion_evidence_strength, "NONE")
        self.assertEqual(result.associated_evidence_state, "MODALITY_CONFLICT")
        self.assertEqual(result.associated_risk_state, "WATCH")
        self.assertFalse(result.affects_alerts)


if __name__ == "__main__":
    unittest.main()
