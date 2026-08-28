import json
import tempfile
import threading
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from app.modules.fall.multimodal_engine.main import app
from app.modules.fall.multimodal_engine.schemas.fall_live import (
    FallLiveState,
    FallLiveStatusResponse,
    RppgLiveStatus,
)
from app.modules.fall.multimodal_engine.schemas.multimodal import (
    CameraEvidence,
    RadarEvidence,
)
from app.modules.fall.multimodal_engine.schemas.radar import (
    RadarDebugPayload,
    RadarFallRiskAssessmentPayload,
    RadarStatusResponse,
)
from app.modules.fall.multimodal_engine.services.dynamic_risk_index import (
    DynamicRiskIndexService,
)
from app.modules.fall.multimodal_engine.services.fusion_runtime import (
    FusionShadowLogger,
    FusionShadowSampler,
    FusionTimingTracker,
)
from app.modules.fall.multimodal_engine.services.multimodal_fusion import (
    MultimodalFusionService,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


class _UnusedProvider:
    def __call__(self):
        raise AssertionError("provider should not be called in direct evidence tests")


class MultimodalFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = MultimodalFusionService(_UnusedProvider(), _UnusedProvider())

    @staticmethod
    def camera(
        *,
        score: float | None = 0.8,
        quality: float = 1.0,
        timestamp: datetime = NOW,
    ) -> CameraEvidence:
        state = (
            "UNKNOWN"
            if score is None
            else "HIGH"
            if score >= 0.65
            else "MEDIUM"
            if score >= 0.35
            else "LOW"
        )
        return CameraEvidence(
            camera_score=score,
            camera_risk_state=state,
            camera_feature={"pose_ready": score is not None},
            camera_quality=quality if score is not None else 0.0,
            quality_level="GOOD" if score is not None else "INSUFFICIENT_DATA",
            timestamp=timestamp,
            available=score is not None,
        )

    @staticmethod
    def radar(
        *,
        score: float | None = 0.4,
        quality: float = 1.0,
        timestamp: datetime = NOW,
    ) -> RadarEvidence:
        return RadarEvidence(
            radar_score=score,
            radar_risk_state="WATCH" if score is not None else "UNKNOWN",
            radar_feature={"risk_state": "WATCH", "point_count": 20},
            radar_quality=quality if score is not None else 0.0,
            quality_level="GOOD" if score is not None else "INSUFFICIENT_DATA",
            timestamp=timestamp,
            available=score is not None,
        )

    def test_rppg_is_observation_only_and_unknown_until_quality_gate(self) -> None:
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
        self.assertFalse(evidence.affects_short_term_fall)
        self.assertFalse(evidence.affects_fall_event)

    def test_abnormal_rppg_only_suggests_review_after_multimodal_decision(self) -> None:
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
        self.assertFalse(context.affects_fall_event_status)
        self.assertFalse(context.can_trigger_alert)


class MultimodalApiTest(unittest.TestCase):
    @staticmethod
    def _service() -> MultimodalFusionService:
        camera = MultimodalFusionTest.camera()
        radar = MultimodalFusionTest.radar()
        service = MultimodalFusionService(
            lambda: None,  # type: ignore[arg-type,return-value]
            lambda: None,  # type: ignore[arg-type,return-value]
        )
        service.camera_evidence = lambda _status, **_kwargs: camera  # type: ignore[method-assign]
        service.radar_evidence = lambda _status, **_kwargs: radar  # type: ignore[method-assign]
        return service

    def test_latest_endpoint_returns_camera_led_contract(self) -> None:
        original_service = app.state.multimodal_fusion_service
        try:
            app.state.multimodal_fusion_service = self._service()
            response = TestClient(app).get("/api/multimodal/latest")
        finally:
            app.state.multimodal_fusion_service = original_service

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertNotIn("fusion", payload)
        self.assertEqual(
            payload["short_term_warning"]["method"],
            "camera_led_evidence_v2",
        )
        self.assertEqual(
            payload["short_term_warning"]["short_term_fall_score"],
            payload["camera_led_evidence_fusion_v2"]["camera_led_score"],
        )
        self.assertTrue(payload["camera_led_evidence_fusion_v2"]["realtime_active"])
        self.assertFalse(payload["camera_led_evidence_fusion_v2"]["shadow_only"])
        self.assertNotIn("probability", str(payload).lower())

    def test_camera_led_associated_endpoint_exposes_minimal_projection(self) -> None:
        original_service = app.state.multimodal_fusion_service
        try:
            app.state.multimodal_fusion_service = self._service()
            response = TestClient(app).get("/api/multimodal/camera-led-associated/latest")
        finally:
            app.state.multimodal_fusion_service = original_service

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["schema_version"], "camera_led_associated_latest_v1")
        self.assertTrue(payload["camera_led_evidence_fusion_v2"]["affects_app_result"])
        self.assertNotIn("fusion", payload)
        self.assertNotIn("radar_track_id", payload["alignment"])
        self.assertNotIn("sync_delta_ms", payload["alignment"])


class DynamicRiskIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = DynamicRiskIndexService()

    def test_missing_long_window_stays_unknown(self) -> None:
        result = self.builder.build_dynamic_risk(
            MultimodalFusionTest.camera(score=0.9),
            MultimodalFusionTest.radar(score=0.8),
            RadarStatusResponse(online=True),
        )
        self.assertFalse(result.available)
        self.assertIsNone(result.dynamic_risk_score)
        self.assertEqual(result.risk_level, "UNKNOWN")

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


class FusionRuntimeAuditTest(unittest.TestCase):
    def test_sampler_runs_without_dashboard_polling(self) -> None:
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
            MultimodalFusionTest.radar(timestamp=NOW + timedelta(seconds=1, milliseconds=500)),
        )
        duplicate = tracker.observe(
            MultimodalFusionTest.camera(timestamp=NOW + timedelta(seconds=1)),
            MultimodalFusionTest.radar(timestamp=NOW + timedelta(seconds=1, milliseconds=500)),
        )
        self.assertEqual(first.sync_sample_count, 1)
        self.assertEqual(duplicate.sync_sample_count, 2)
        self.assertAlmostEqual(duplicate.sync_p50_ms, 300.0)
        self.assertAlmostEqual(duplicate.sync_p95_ms, 480.0)

    def test_runtime_logger_is_deduplicated_and_has_no_score_contributions(self) -> None:
        service = MultimodalApiTest._service()
        response = service.get_latest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multimodal_runtime.jsonl"
            logger = FusionShadowLogger(path, enabled=True)
            logger.write(response)
            logger.write(response)
            for handler in logger._logger.handlers:
                handler.flush()
                handler.close()
            payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(payloads), 1)
        self.assertEqual(
            payloads[0]["schema_version"],
            "camera_led_radar_evidence_runtime_v1",
        )
        self.assertIn("camera_led_evidence_fusion_v2", payloads[0])


if __name__ == "__main__":
    unittest.main()
