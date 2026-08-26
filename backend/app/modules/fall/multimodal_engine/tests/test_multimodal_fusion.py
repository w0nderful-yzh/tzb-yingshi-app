from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import threading
import unittest

from fastapi.testclient import TestClient

from app.modules.fall.multimodal_engine.main import app
from app.modules.fall.multimodal_engine.schemas.multimodal import (
    CameraEvidence,
    MultimodalLatestResponse,
    MultimodalQualitySummary,
    RadarEvidence,
    TemporalAssociatedFusionResult,
)
from app.modules.fall.multimodal_engine.schemas.fall_live import FallLiveState, FallLiveStatusResponse, RppgLiveStatus
from app.modules.fall.multimodal_engine.schemas.radar import (
    RadarDebugPayload,
    RadarFallRiskAssessmentPayload,
    RadarStatusResponse,
)
from app.modules.fall.multimodal_engine.services.dynamic_risk_index import DynamicRiskIndexService
from app.modules.fall.multimodal_engine.services.multimodal_fusion import (
    LightweightMlpFusionHead,
    MlpFusionParameters,
    MultimodalFusionService,
)
from app.modules.fall.multimodal_engine.services.fusion_runtime import (
    FusionShadowLogger,
    FusionShadowSampler,
    FusionStateConfig,
    FusionStateMachine,
    FusionTimingTracker,
)
from app.modules.fall.multimodal_engine.services.fusion_event_bridge import FusionFindingFactory
from app.modules.fall.multimodal_engine.services.temporal_associated_fusion import (
    TemporalAssociatedFusion,
    TemporalAssociationConfig,
)


NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


class _UnusedProvider:
    def __call__(self):
        raise AssertionError("provider should not be called in direct fusion tests")


class MultimodalFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = MultimodalFusionService(
            _UnusedProvider(),
            _UnusedProvider(),
            camera_weight=0.6,
            radar_weight=0.4,
        )

    @staticmethod
    def camera(
        *, score: float | None = 0.8, quality: float = 1.0, timestamp: datetime = NOW
    ) -> CameraEvidence:
        return CameraEvidence(
            camera_score=score,
            camera_feature={"pose_ready": score is not None},
            camera_quality=quality if score is not None else 0.0,
            quality_level="GOOD" if score is not None else "INSUFFICIENT_DATA",
            timestamp=timestamp,
            available=score is not None,
        )

    @staticmethod
    def radar(
        *, score: float | None = 0.4, quality: float = 1.0, timestamp: datetime = NOW
    ) -> RadarEvidence:
        return RadarEvidence(
            radar_score=score,
            radar_feature={"risk_state": "WATCH"},
            radar_quality=quality if score is not None else 0.0,
            quality_level="GOOD" if score is not None else "INSUFFICIENT_DATA",
            timestamp=timestamp,
            available=score is not None,
        )

    def test_quality_weighted_fusion_uses_same_risk_direction(self) -> None:
        result = self.service.fuse(self.camera(), self.radar())

        self.assertAlmostEqual(result.fusion_score, 0.64)
        self.assertAlmostEqual(result.contribution_camera, 0.6)
        self.assertAlmostEqual(result.contribution_radar, 0.4)
        self.assertEqual(result.risk_level, "MEDIUM")
        self.assertEqual(result.score_name, "multimodal risk score")

    def test_lower_quality_reduces_modality_contribution(self) -> None:
        result = self.service.fuse(
            self.camera(score=0.8, quality=1.0),
            self.radar(score=0.4, quality=0.5),
        )

        self.assertAlmostEqual(result.contribution_camera, 0.75)
        self.assertAlmostEqual(result.contribution_radar, 0.25)
        self.assertAlmostEqual(result.fusion_score, 0.7)

    def test_fixed_baseline_ignores_quality_but_preserves_06_04(self) -> None:
        result = self.service.fuse(
            self.camera(score=0.8, quality=0.2),
            self.radar(score=0.4, quality=1.0),
            method="fixed_weighted",
        )
        self.assertAlmostEqual(result.fusion_score, 0.64)
        self.assertAlmostEqual(result.contribution_camera, 0.6)
        self.assertAlmostEqual(result.contribution_radar, 0.4)

    def test_unsynchronized_evidence_falls_back_to_newer_modality(self) -> None:
        result = self.service.fuse(
            self.camera(score=0.8, timestamp=NOW),
            self.radar(score=0.4, timestamp=NOW - timedelta(seconds=3)),
        )

        self.assertEqual(result.fusion_score, 0.8)
        self.assertEqual(result.dominant_modality, "CAMERA")
        self.assertEqual(result.contribution_camera, 1.0)
        self.assertFalse(result.synchronized)
        self.assertIn("outside the fusion tolerance", result.degraded_reason)

    def test_one_modality_fallback_and_no_evidence_are_explicit(self) -> None:
        camera_only = self.service.fuse(self.camera(), self.radar(score=None))
        self.assertEqual(camera_only.fusion_score, 0.8)
        self.assertEqual(camera_only.contribution_camera, 1.0)
        self.assertEqual(camera_only.dominant_modality, "CAMERA")

        unknown = self.service.fuse(self.camera(score=None), self.radar(score=None))
        self.assertIsNone(unknown.fusion_score)
        self.assertEqual(unknown.risk_level, "UNKNOWN")
        self.assertEqual(unknown.dominant_modality, "NONE")

    def test_mlp_is_forward_only_and_requires_explicit_weights(self) -> None:
        head = LightweightMlpFusionHead(
            MlpFusionParameters(
                hidden_weights=((1.0, 1.0, 0.0, 0.0, 0.0),),
                hidden_bias=(0.0,),
                output_weights=(1.0,),
                output_bias=0.0,
            )
        )
        score = head.predict((0.2, 0.3, 1.0, 1.0, 1.0))
        self.assertAlmostEqual(score, 0.6224593312)

        with self.assertRaisesRegex(RuntimeError, "not configured"):
            self.service.get_latest(method="mlp")

    def test_rppg_is_shadow_only_and_unknown_until_quality_gate(self) -> None:
        status = FallLiveStatusResponse(
            enabled=True,
            state=FallLiveState.RUNNING,
            rppg=RppgLiveStatus(
                enabled=True,
                available=True,
                assessment_ready=False,
                heart_rate=74.0,
                sqi=0.82,
                physio_level="NORMAL",
                valid_seconds=20.0,
                quality_coverage=0.33,
                quality_reason="RPPG_WARMING_UP",
            ),
        )
        evidence = self.service.physiological_evidence(status)
        self.assertEqual(evidence.physio_level, "UNKNOWN")
        self.assertTrue(evidence.shadow_only)
        self.assertFalse(evidence.affects_fusion)
        self.assertFalse(evidence.affects_dynamic_risk)
        self.assertFalse(evidence.affects_short_term_fall)
        self.assertFalse(evidence.affects_fall_event)

    def test_abnormal_rppg_only_suggests_review_after_fusion(self) -> None:
        physiological = self.service.physiological_evidence(
            FallLiveStatusResponse(
                enabled=True,
                state=FallLiveState.RUNNING,
                rppg=RppgLiveStatus(
                    enabled=True,
                    available=True,
                    assessment_ready=True,
                    heart_rate=102.0,
                    sqi=0.9,
                    physio_level="ABNORMAL",
                    physio_abnormal=True,
                    abnormal_reasons=["HR_HIGH"],
                    valid_seconds=70,
                    quality_coverage=1,
                    quality_reason="RPPG_ASSESSMENT_READY",
                ),
            )
        )
        context = self.service.final_decision_context(
            physiological,
            "NORMAL",
            "NO_EVENT",
        )
        self.assertTrue(context.human_review_suggested)
        self.assertFalse(context.affects_fusion_score)
        self.assertFalse(context.affects_short_term_fall_score)
        self.assertFalse(context.affects_fall_event_status)
        self.assertFalse(context.can_trigger_alert)
        self.assertEqual(context.base_short_term_state, "NORMAL")
        self.assertEqual(context.base_fall_event_status, "NO_EVENT")


class MultimodalApiTest(unittest.TestCase):
    def test_latest_endpoint_returns_contract_without_probability_language(self) -> None:
        camera = MultimodalFusionTest.camera()
        radar = MultimodalFusionTest.radar()
        service = MultimodalFusionService(
            lambda: None,  # type: ignore[arg-type,return-value]
            lambda: None,  # type: ignore[arg-type,return-value]
        )
        service.camera_evidence = lambda _status, **_kwargs: camera  # type: ignore[method-assign]
        service.radar_evidence = lambda _status, **_kwargs: radar  # type: ignore[method-assign]
        original_service = app.state.multimodal_fusion_service
        try:
            app.state.multimodal_fusion_service = service
            response = TestClient(app).get("/api/multimodal/latest")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["fusion"]["score_name"], "multimodal risk score")
            self.assertEqual(payload["fusion"]["method"], "fixed_weighted")
            self.assertTrue(payload["temporal_associated_fusion"]["shadow_only"])
            self.assertFalse(payload["temporal_associated_fusion"]["affects_alerts"])
            self.assertTrue(payload["associated_risk_augmentation"]["shadow_only"])
            self.assertFalse(
                payload["associated_risk_augmentation"]["affects_fixed_fusion"]
            )
            self.assertEqual(
                payload["associated_risk_augmentation"][
                    "associated_short_term_fall_score"
                ],
                payload["camera"]["camera_score"],
            )
            self.assertIn("stable_fusion_state", payload["fusion"])
            self.assertIn("source_timestamp", payload["camera"])
            self.assertIn("processing_latency_ms", payload["radar"])
            self.assertIn("physiological_evidence", payload)
            self.assertIn("runtime_statistics", payload)
            self.assertIn("final_decision_context", payload)
            self.assertEqual(payload["operating_mode"], "LIVE_CAMERA_RADAR")
            self.assertEqual(payload["data_source"], "REAL_CAMERA_RADAR")
            self.assertIn("camera_risk_state", payload["camera"])
            self.assertIn("radar_risk_state", payload["radar"])
            self.assertIn("fusion_risk_state", payload["fusion"])
            self.assertIn("dynamic_risk_score", payload["dynamic_risk"])
            self.assertIn("short_term_fall_score", payload["short_term_warning"])
            self.assertEqual(
                payload["short_term_warning"]["method"],
                "camera_led_evidence_v2",
            )
            self.assertEqual(
                payload["short_term_warning"]["short_term_fall_score"],
                payload["camera_led_evidence_fusion_v2"]["camera_led_score"],
            )
            self.assertTrue(
                payload["camera_led_evidence_fusion_v2"]["realtime_active"]
            )
            self.assertFalse(
                payload["camera_led_evidence_fusion_v2"]["shadow_only"]
            )
            self.assertFalse(
                payload["camera_led_evidence_fusion_v2"]["affects_alerts"]
            )
            self.assertIn("fall_event_status", payload["fall_event"])
            self.assertNotIn("probability", str(payload).lower())
            self.assertIn("camera", payload)
            self.assertIn("radar", payload)
            self.assertIn("quality", payload)
        finally:
            app.state.multimodal_fusion_service = original_service

    def test_camera_led_associated_endpoint_exposes_realtime_v2_projection(self) -> None:
        camera = MultimodalFusionTest.camera()
        radar = MultimodalFusionTest.radar()
        service = MultimodalFusionService(
            lambda: None,  # type: ignore[arg-type,return-value]
            lambda: None,  # type: ignore[arg-type,return-value]
        )
        service.camera_evidence = lambda _status, **_kwargs: camera  # type: ignore[method-assign]
        service.radar_evidence = lambda _status, **_kwargs: radar  # type: ignore[method-assign]
        original_service = app.state.multimodal_fusion_service
        try:
            app.state.multimodal_fusion_service = service
            response = TestClient(app).get(
                "/api/multimodal/camera-led-associated/latest"
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(
                payload["schema_version"], "camera_led_associated_latest_v1"
            )
            self.assertEqual(
                payload["associated_risk_augmentation"][
                    "associated_short_term_fall_score"
                ],
                payload["camera"]["camera_score"],
            )
            self.assertTrue(
                payload["associated_risk_augmentation"]["shadow_only"]
            )
            self.assertFalse(
                payload["associated_risk_augmentation"]["affects_alerts"]
            )
            self.assertTrue(
                payload["camera_led_evidence_fusion_v2"]["realtime_active"]
            )
            self.assertTrue(
                payload["camera_led_evidence_fusion_v2"]["affects_app_result"]
            )
            self.assertFalse(
                payload["camera_led_evidence_fusion_v2"]["shadow_only"]
            )
            self.assertNotIn("fusion", payload)
            self.assertNotIn("temporal_associated_fusion", payload)
            self.assertNotIn("radar_track_id", payload["alignment"])
            self.assertNotIn("sync_delta_ms", payload["alignment"])
        finally:
            app.state.multimodal_fusion_service = original_service


class DynamicRiskIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = DynamicRiskIndexService()

    def test_missing_long_window_stays_unknown_instead_of_reusing_fusion(self) -> None:
        result = self.builder.build_dynamic_risk(
            MultimodalFusionTest.camera(score=0.9),
            MultimodalFusionTest.radar(score=0.8),
            RadarStatusResponse(online=True),
        )

        self.assertFalse(result.available)
        self.assertIsNone(result.dynamic_risk_score)
        self.assertEqual(result.risk_level, "UNKNOWN")
        self.assertFalse(result.camera_context_affects_score)
        self.assertEqual(result.score_interpretation, "SCREENING_INDEX_NOT_DIAGNOSIS")

    def test_existing_radar_assessment_is_exposed_with_camera_context(self) -> None:
        status = RadarStatusResponse(
            online=True,
            radar_debug=RadarDebugPayload(
                fall_risk_assessment=RadarFallRiskAssessmentPayload(
                    schema_version="radar_risk_assessment_live_v1",
                    timestamp=NOW.isoformat(),
                    device_id="radar-1",
                    room="living_room",
                    risk_level="MODERATE",
                    risk_score=0.42,
                    sway_risk=0.6,
                    mobility_risk=0.3,
                    descent_risk=0.1,
                    assessment_window_seconds=60.0,
                    valid_window_count=18,
                    observed_duration_seconds=43.0,
                    shadow_only=True,
                    alert_suppressed=True,
                    disclaimer="shadow-only dynamic screening signal",
                )
            ),
        )
        result = self.builder.build_dynamic_risk(
            MultimodalFusionTest.camera(score=0.8),
            MultimodalFusionTest.radar(score=0.2),
            status,
        )

        self.assertTrue(result.available)
        self.assertEqual(result.dynamic_risk_score, 0.42)
        self.assertEqual(result.risk_level, "MODERATE")
        self.assertEqual(result.components["camera_posture_context"], 0.8)
        self.assertIn("运动稳定性下降", [reason.label for reason in result.reasons])

    def test_warning_is_not_reported_as_confirmed_fall_event(self) -> None:
        camera = MultimodalFusionTest.camera(score=0.9)
        radar = MultimodalFusionTest.radar(score=0.9)
        event = self.builder.build_fall_event(camera, radar)

        self.assertEqual(event.fall_event_status, "NO_EVENT")
        self.assertFalse(event.requires_human_confirmation)


class FusionStateMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.machine = FusionStateMachine(
            FusionStateConfig(
                ema_alpha=1.0,
                watch_confirmation_windows=2,
                high_confirmation_windows=3,
                normal_confirmation_windows=2,
            )
        )
        self.service = MultimodalFusionService(_UnusedProvider(), _UnusedProvider())

    def apply(self, camera_score: float | None, radar_score: float | None, step: int):
        timestamp = NOW + timedelta(seconds=step)
        camera = MultimodalFusionTest.camera(score=camera_score, timestamp=timestamp)
        radar = MultimodalFusionTest.radar(score=radar_score, timestamp=timestamp)
        raw = self.service.fuse(camera, radar, method="fixed_weighted")
        return self.machine.apply(camera, radar, raw)

    def test_confirmation_and_hysteresis_states(self) -> None:
        self.assertEqual(self.apply(0.1, 0.1, 1).stable_fusion_state, "UNKNOWN")
        self.assertEqual(self.apply(0.1, 0.1, 2).stable_fusion_state, "NORMAL")
        self.assertEqual(self.apply(0.5, 0.5, 3).stable_fusion_state, "NORMAL")
        self.assertEqual(self.apply(0.5, 0.5, 4).stable_fusion_state, "WATCH")
        self.assertEqual(self.apply(0.8, 0.8, 5).stable_fusion_state, "WATCH")
        self.assertEqual(self.apply(0.8, 0.8, 6).stable_fusion_state, "WATCH")
        self.assertEqual(self.apply(0.8, 0.8, 7).stable_fusion_state, "IMMINENT")

    def test_missing_modality_never_becomes_low(self) -> None:
        self.assertEqual(self.apply(0.1, None, 1).stable_fusion_state, "UNKNOWN")
        result = self.apply(0.1, None, 2)
        self.assertEqual(result.stable_fusion_state, "WATCH")
        self.assertEqual(result.degraded_mode, "CAMERA_ONLY")

    def test_conflicting_modalities_are_capped_at_watch(self) -> None:
        self.apply(1.0, 0.3, 1)
        result = self.apply(1.0, 0.3, 2)
        self.assertEqual(result.fusion_state, "WATCH")
        self.assertEqual(result.stable_fusion_state, "WATCH")
        self.assertEqual(result.degraded_mode, "MODALITY_CONFLICT")

    def test_duplicate_evidence_does_not_advance_confirmation(self) -> None:
        first = self.apply(0.5, 0.5, 1)
        duplicate = self.apply(0.5, 0.5, 1)
        self.assertEqual(first.stable_fusion_state, "UNKNOWN")
        self.assertEqual(duplicate.stable_fusion_state, "UNKNOWN")

    def test_experimental_method_has_independent_stable_state(self) -> None:
        fixed = self.service.state_machines["fixed_weighted"]
        quality = self.service.state_machines["quality_weighted"]
        fixed_result = None
        quality_result = None
        for step in range(1, 3):
            timestamp = NOW + timedelta(seconds=step)
            camera = MultimodalFusionTest.camera(score=0.5, timestamp=timestamp)
            radar = MultimodalFusionTest.radar(score=0.5, timestamp=timestamp)
            fixed_result = fixed.apply(
                camera,
                radar,
                self.service.fuse(camera, radar, method="fixed_weighted"),
            )
            camera_low = MultimodalFusionTest.camera(score=0.1, timestamp=timestamp)
            radar_low = MultimodalFusionTest.radar(score=0.1, timestamp=timestamp)
            quality_result = quality.apply(
                camera_low,
                radar_low,
                self.service.fuse(camera_low, radar_low, method="quality_weighted"),
            )
        assert fixed_result is not None and quality_result is not None
        self.assertEqual(fixed_result.stable_fusion_state, "WATCH")
        self.assertEqual(quality_result.stable_fusion_state, "NORMAL")

    def test_fusion_finding_requires_stable_non_degraded_high(self) -> None:
        response = None
        for step in range(1, 4):
            camera = MultimodalFusionTest.camera(score=0.8, timestamp=NOW + timedelta(seconds=step))
            radar = MultimodalFusionTest.radar(score=0.8, timestamp=NOW + timedelta(seconds=step))
            camera.device_id = "camera-1"
            radar.device_id = "radar-1"
            radar.room = "living_room"
            fusion = self.machine.apply(
                camera,
                radar,
                self.service.fuse(camera, radar, method="fixed_weighted"),
            )
            response = MultimodalLatestResponse(
                camera=camera,
                radar=radar,
                fusion=fusion,
                quality=MultimodalQualitySummary(
                    camera=1.0, radar=1.0, synchronization=1.0, overall=1.0, level="GOOD"
                ),
            )
        assert response is not None
        finding = FusionFindingFactory().create(response)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.finding.event_type, "MULTIMODAL_PRE_FALL_RISK")
        self.assertIsNone(
            FusionFindingFactory().create(
                response.model_copy(
                    update={
                        "fusion": response.fusion.model_copy(
                            update={"method": "quality_weighted"}
                        )
                    }
                )
            )
        )

    def test_shadow_temporal_high_cannot_create_formal_finding(self) -> None:
        camera = MultimodalFusionTest.camera(score=0.1)
        radar = MultimodalFusionTest.radar(score=0.1)
        fixed = self.service.state_machines["fixed_weighted"].apply(
            camera,
            radar,
            self.service.fuse(camera, radar, method="fixed_weighted"),
        )
        response = MultimodalLatestResponse(
            camera=camera,
            radar=radar,
            fusion=fixed,
            temporal_associated_fusion=TemporalAssociatedFusionResult(
                fusion_score=0.9,
                fusion_state="HIGH",
                window_seconds=2.0,
                camera_evidence_count=2,
                radar_evidence_count=2,
            ),
            quality=MultimodalQualitySummary(
                camera=1.0,
                radar=1.0,
                synchronization=1.0,
                overall=1.0,
                level="GOOD",
            ),
        )

        self.assertIsNone(FusionFindingFactory().create(response))
        self.assertIsNone(
            FusionFindingFactory().create(
                response.model_copy(update={"operating_mode": "OFFLINE_EVIDENCE_REPLAY"})
            )
        )


class FusionRuntimeAuditTest(unittest.TestCase):
    def test_shadow_sampler_runs_without_dashboard_polling(self) -> None:
        observed = threading.Event()
        sampler = FusionShadowSampler(
            observed.set,
            enabled=True,
            interval_seconds=0.01,
        )
        sampler.start()
        try:
            self.assertTrue(observed.wait(timeout=0.5))
        finally:
            sampler.stop()

    def test_timing_tracker_reports_deduplicated_p50_and_p95(self) -> None:
        tracker = FusionTimingTracker(tolerance_ms=2000.0)
        first = tracker.observe(
            MultimodalFusionTest.camera(timestamp=NOW),
            MultimodalFusionTest.radar(timestamp=NOW + timedelta(milliseconds=100)),
        )
        tracker.observe(
            MultimodalFusionTest.camera(timestamp=NOW + timedelta(seconds=1)),
            MultimodalFusionTest.radar(
                timestamp=NOW + timedelta(seconds=1, milliseconds=500)
            ),
        )
        duplicate = tracker.observe(
            MultimodalFusionTest.camera(timestamp=NOW + timedelta(seconds=1)),
            MultimodalFusionTest.radar(
                timestamp=NOW + timedelta(seconds=1, milliseconds=500)
            ),
        )

        self.assertEqual(first.sync_sample_count, 1)
        self.assertEqual(duplicate.sync_sample_count, 2)
        self.assertAlmostEqual(duplicate.sync_p50_ms, 300.0)
        self.assertAlmostEqual(duplicate.sync_p95_ms, 480.0)

    def test_shadow_logger_is_deduplicated_and_contains_audit_fields(self) -> None:
        service = MultimodalFusionService(_UnusedProvider(), _UnusedProvider())
        camera = MultimodalFusionTest.camera()
        radar = MultimodalFusionTest.radar()
        response = MultimodalLatestResponse(
            camera=camera,
            radar=radar,
            fusion=service.fuse(camera, radar, method="fixed_weighted"),
            quality=MultimodalQualitySummary(
                camera=1.0,
                radar=1.0,
                synchronization=1.0,
                overall=1.0,
                level="GOOD",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fusion_shadow.jsonl"
            logger = FusionShadowLogger(path, enabled=True)
            logger.write(response)
            logger.write(response)
            for handler in logger._logger.handlers:
                handler.flush()
                handler.close()
            payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["raw_fusion_score"], 0.64)
        for field in (
            "camera_score",
            "radar_score",
            "schema_version",
            "data_source",
            "fusion_score",
            "risk_state",
            "camera_state",
            "camera_positive_votes",
            "radar_state",
            "camera_quality",
            "radar_quality",
            "contribution_camera",
            "contribution_radar",
            "sync_delta_ms",
            "degraded_mode",
            "fusion_state",
            "fusion_method",
            "camera_processing_latency_ms",
            "radar_processing_latency_ms",
            "camera_evidence_age_ms",
            "radar_evidence_age_ms",
            "room",
            "device",
            "model_version",
            "dynamic_risk_score",
            "dynamic_risk_level",
            "dynamic_risk_available",
            "short_term_fall_score",
            "fall_event_status",
            "physiological_evidence",
            "runtime_statistics",
            "associated_risk_augmentation",
        ):
            self.assertIn(field, payloads[0])


class TemporalAssociatedFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixed_service = MultimodalFusionService(_UnusedProvider(), _UnusedProvider())
        self.temporal = TemporalAssociatedFusion(
            TemporalAssociationConfig(
                window_seconds=2.0,
                confirmation_windows=2,
                minimum_quality=0.25,
            )
        )

    @staticmethod
    def camera(timestamp: datetime, *, present: bool = True) -> CameraEvidence:
        return CameraEvidence(
            camera_score=0.85,
            camera_risk_state="HIGH",
            camera_feature={"target_present": present},
            camera_quality=0.9,
            quality_level="GOOD",
            timestamp=timestamp,
            available=True,
        )

    @staticmethod
    def radar(timestamp: datetime, *, point_count: int = 40) -> RadarEvidence:
        return RadarEvidence(
            radar_score=0.75,
            radar_risk_state="IMMINENT",
            radar_feature={
                "risk_state": "IMMINENT",
                "point_count": point_count,
                "vertical_velocity": -0.5,
                "height_delta": -0.3,
                "motion_direction": "DESCENDING",
                "missing_frame_ratio": 0.0,
                "target_id": None,
                "track_count": None,
            },
            radar_quality=0.9,
            quality_level="GOOD",
            timestamp=timestamp,
            available=True,
        )

    def apply(self, camera: CameraEvidence, radar: RadarEvidence):
        fixed = self.fixed_service.fuse(camera, radar, method="fixed_weighted")
        return self.temporal.apply(camera, radar, fixed)

    def test_unverified_single_target_assumption_cannot_enter_high(self) -> None:
        first = self.apply(
            self.camera(NOW + timedelta(milliseconds=100)),
            self.radar(NOW),
        )
        second = self.apply(
            self.camera(NOW + timedelta(seconds=1, milliseconds=100)),
            self.radar(NOW + timedelta(seconds=1)),
        )

        self.assertEqual(first.fusion_state, "WATCH")
        self.assertEqual(second.fusion_state, "WATCH")
        self.assertTrue(second.continuous_camera_risk)
        self.assertTrue(second.continuous_radar_risk)
        self.assertEqual(second.temporal_relation, "ALIGNED")
        self.assertEqual(second.target_association, "SINGLE_TARGET_ASSUMED")
        self.assertIn("NO_VERIFIED_CROSS_MODAL_TARGET_ID", second.reason_codes)

    def test_duplicate_windows_do_not_create_continuity(self) -> None:
        camera = self.camera(NOW + timedelta(milliseconds=100))
        radar = self.radar(NOW)
        self.apply(camera, radar)
        duplicate = self.apply(camera, radar)

        self.assertEqual(duplicate.fusion_state, "WATCH")
        self.assertEqual(duplicate.camera_evidence_count, 1)
        self.assertEqual(duplicate.radar_evidence_count, 1)

    def test_target_presence_conflict_caps_result_at_watch(self) -> None:
        result = None
        for offset in (0, 1):
            result = self.apply(
                self.camera(NOW + timedelta(seconds=offset), present=False),
                self.radar(NOW + timedelta(seconds=offset), point_count=50),
            )
        assert result is not None

        self.assertEqual(result.target_association, "CONFLICT")
        self.assertEqual(result.fusion_state, "WATCH")
        self.assertEqual(result.degraded_mode, "MODALITY_CONFLICT")
        self.assertFalse(result.affects_alerts)

    def test_weak_risk_is_not_held_after_latest_normal_window(self) -> None:
        weak_camera = self.camera(NOW)
        weak_camera.camera_risk_state = "MEDIUM"
        normal_radar = self.radar(NOW)
        normal_radar.radar_risk_state = "NORMAL"
        normal_radar.radar_feature["risk_state"] = "NORMAL"  # type: ignore[index]
        self.apply(weak_camera, normal_radar)

        normal_camera = self.camera(NOW + timedelta(seconds=1))
        normal_camera.camera_risk_state = "LOW"
        normal_radar = self.radar(NOW + timedelta(seconds=1))
        normal_radar.radar_risk_state = "NORMAL"
        normal_radar.radar_feature["risk_state"] = "NORMAL"  # type: ignore[index]
        result = self.apply(normal_camera, normal_radar)

        self.assertEqual(result.fusion_state, "NORMAL")
        self.assertIn("NO_CONTINUOUS_RISK_SEQUENCE", result.reason_codes)


if __name__ == "__main__":
    unittest.main()
