from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.modules.fall.multimodal_engine.api.fall_live import router as fall_live_router
from app.modules.fall.multimodal_engine.core.config import Settings
from app.modules.fall.multimodal_engine.integrations.ezviz import EzvizApiError
from app.modules.fall.multimodal_engine.schemas.fall_live import FallLiveInputState, FallLiveState, FallLiveStatusResponse
from app.modules.fall.multimodal_engine.schemas.risk_event import RiskLevel
from app.modules.fall.multimodal_engine.services.fall_live_monitor import FallLiveMonitorService


class _FakeFallLiveMonitor:
    def __init__(self) -> None:
        self.status = FallLiveStatusResponse(enabled=True, state=FallLiveState.STOPPED)

    def start(self) -> None:
        self.status = self.status.model_copy(update={"state": FallLiveState.STARTING})

    def stop(self) -> None:
        self.status = self.status.model_copy(update={"state": FallLiveState.STOPPED})

    def get_status(self) -> FallLiveStatusResponse:
        return self.status


class FallLiveControlApiTest(unittest.TestCase):
    def test_start_and_stop_are_idempotent_worker_controls(self) -> None:
        monitor = _FakeFallLiveMonitor()
        app = FastAPI()
        app.state.fall_live_monitor_service = monitor
        app.include_router(fall_live_router)

        with TestClient(app) as client:
            start = client.post("/api/fall-live/start")
            repeated_start = client.post("/api/fall-live/start")
            stop = client.post("/api/fall-live/stop")

        self.assertEqual(start.status_code, 202)
        self.assertEqual(start.json()["state"], "STARTING")
        self.assertEqual(repeated_start.status_code, 202)
        self.assertEqual(stop.status_code, 200)
        self.assertEqual(stop.json()["state"], "STOPPED")


class FallLiveMonitorStatusTest(unittest.TestCase):
    def test_encrypted_standard_stream_has_actionable_status_message(self) -> None:
        message = FallLiveMonitorService._display_error(
            EzvizApiError("加密已开启", code="60019")
        )
        self.assertIn("视频图片加密", message)
        self.assertIn("EZOPEN", message)
        self.assertIn("自动重连", message)

    def test_disabled_monitor_does_not_start_background_worker(self) -> None:
        service = FallLiveMonitorService(
            Settings(_env_file=None, fall_live_monitor_enabled=False)
        )
        service.start()
        status = service.get_status()
        self.assertEqual(status.state, FallLiveState.DISABLED)
        self.assertFalse(service.is_running)

    def test_prediction_updates_live_status_without_persisting_low_event(self) -> None:
        service = FallLiveMonitorService(
            Settings(_env_file=None, fall_live_monitor_enabled=False)
        )
        occurred_at = datetime.now(timezone.utc).isoformat()
        service._handle_message(
            "camera-01",
            {
                "type": "prediction",
                "occurred_at": occurred_at,
                "risk_score": 0.21,
                "risk_level": "LOW",
                "alert": False,
                "positive_votes": 0,
                "captured_frames": 110,
                "processed_frames": 100,
                "dropped_frames": 10,
                "queue_depth": 4,
                "processing_fps": 18.5,
                "pipeline_latency_seconds": 0.8,
            },
        )
        status = service.get_status()
        self.assertEqual(status.state, FallLiveState.RUNNING)
        self.assertEqual(status.risk_level, RiskLevel.LOW)
        self.assertEqual(status.dropped_frames, 10)
        self.assertEqual(status.processing_fps, 18.5)
        self.assertIsNone(status.last_event_id)

    def test_prediction_keeps_rppg_as_non_decision_sidecar(self) -> None:
        service = FallLiveMonitorService(
            Settings(_env_file=None, fall_live_monitor_enabled=False)
        )
        service._handle_message(
            "camera-01",
            {
                "type": "prediction",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "risk_score": 0.21,
                "risk_level": "LOW",
                "alert": False,
                "rppg": {
                    "enabled": True,
                    "available": True,
                    "assessment_ready": True,
                    "heart_rate": 72.0,
                    "sqi": 0.86,
                    "physio_level": "NORMAL",
                    "valid_seconds": 60.0,
                    "quality_coverage": 1.0,
                    "quality_reason": "RPPG_ASSESSMENT_READY",
                    "shadow_only": True,
                    "affects_fusion": False,
                },
            },
        )
        status = service.get_status()
        self.assertTrue(status.rppg.assessment_ready)
        self.assertEqual(status.rppg.heart_rate, 72.0)
        self.assertFalse(status.rppg.affects_fusion)
        self.assertEqual(status.risk_score, 0.21)

    def test_shadow_mode_keeps_high_camera_output_without_persisting_event(self) -> None:
        service = FallLiveMonitorService(
            Settings(
                _env_file=None,
                fall_live_monitor_enabled=False,
                fall_live_risk_events_enabled=False,
            )
        )
        with patch.object(service, "_persist_alert") as persist:
            service._handle_message(
                "camera-01",
                {
                    "type": "prediction",
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                    "risk_score": 0.88,
                    "risk_level": "HIGH",
                    "alert": True,
                    "positive_votes": 5,
                },
            )

        persist.assert_not_called()
        self.assertEqual(service.get_status().risk_level, RiskLevel.HIGH)

    def test_optional_alignment_log_persists_geometry_without_changing_status(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "camera_alignment.jsonl"
            service = FallLiveMonitorService(
                Settings(
                    _env_file=None,
                    fall_live_monitor_enabled=False,
                    fall_alignment_shadow_log_path=output,
                )
            )
            message = {
                "type": "input_status",
                "frame_id": "1:42",
                "source_timestamp": datetime.now(timezone.utc).isoformat(),
                "input_state": "WAITING",
                "bbox_xyxy": [1, 2, 30, 40],
                "keypoints_2d": [[10, 20]],
                "pose_3d_camera": [[0, 0, 1]],
            }
            service._handle_message("camera-01", message)

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, [message])
            self.assertEqual(service.get_status().input_state, FallLiveInputState.WAITING)

    def test_alignment_snapshot_uses_ankle_midpoint_without_changing_camera_score(self) -> None:
        service = FallLiveMonitorService(
            Settings(_env_file=None, fall_live_monitor_enabled=False)
        )
        keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
        keypoints[15] = [500.0, 690.0, 0.9]
        keypoints[16] = [540.0, 710.0, 0.8]
        service._handle_message(
            "camera-01",
            {
                "type": "prediction",
                "frame_id": "1:43",
                "source_timestamp": datetime.now(timezone.utc).isoformat(),
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "risk_score": 0.21,
                "risk_level": "LOW",
                "alert": False,
                "detected": True,
                "camera_quality": 0.85,
                "image_size": [1280, 720],
                "bbox_xyxy": [440, 300, 600, 715],
                "keypoints_2d": keypoints,
            },
        )

        status = service.get_status()
        self.assertEqual(status.risk_score, 0.21)
        self.assertIsNotNone(status.alignment_snapshot)
        self.assertEqual(status.alignment_snapshot.footpoint_uv, (520.0, 700.0))
        self.assertEqual(
            status.alignment_snapshot.footpoint_source,
            "ANKLE_MIDPOINT_15_16",
        )

    def test_browser_frame_is_accepted_into_bounded_queue(self) -> None:
        service = FallLiveMonitorService(
            Settings(
                _env_file=None,
                fall_live_monitor_enabled=True,
                fall_live_source_mode="browser_capture",
            )
        )
        depth = service.submit_browser_frame(
            device_id="camera-01",
            captured_at=datetime.now(timezone.utc),
            frame_base64="data:image/jpeg;base64," + "A" * 256,
        )
        self.assertEqual(depth, 1)
        self.assertEqual(service._browser_device_id, "camera-01")
        self.assertEqual(service._browser_received, 1)

    def test_no_person_status_clears_stale_risk_and_splits_frame_counts(self) -> None:
        service = FallLiveMonitorService(
            Settings(_env_file=None, fall_live_monitor_enabled=False)
        )
        service._handle_message(
            "camera-01",
            {
                "type": "prediction",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "risk_score": 0.44,
                "risk_level": "LOW",
                "positive_votes": 1,
            },
        )

        service._handle_message(
            "camera-01",
            {
                "type": "input_status",
                "input_state": "NO_PERSON",
                "input_message": "未检测到人体，已暂停跌倒风险推理",
                "target_present": False,
                "training_input_ready": False,
                "source_window_frames": 6,
                "valid_pose_frames": 0,
                "required_source_frames": 90,
                "effective_sample_fps": 2.0,
                "processed_frames": 20,
                "queue_dropped_frames": 3,
                "invalid_image_frames": 2,
                "no_person_frames": 15,
                "low_confidence_frames": 1,
            },
        )

        status = service.get_status()
        self.assertEqual(status.input_state, FallLiveInputState.NO_PERSON)
        self.assertFalse(status.target_present)
        self.assertFalse(status.training_input_ready)
        self.assertIsNone(status.risk_score)
        self.assertIsNone(status.risk_level)
        self.assertIsNone(status.last_prediction_at)
        self.assertEqual(status.queue_dropped_frames, 3)
        self.assertEqual(status.invalid_image_frames, 2)
        self.assertEqual(status.no_person_frames, 15)
        self.assertEqual(status.low_confidence_frames, 1)
        self.assertEqual(status.dropped_frames, 5)

    def test_browser_frame_is_rejected_in_standard_stream_mode(self) -> None:
        service = FallLiveMonitorService(
            Settings(
                _env_file=None,
                fall_live_monitor_enabled=True,
                fall_live_source_mode="ezviz_standard",
            )
        )
        with self.assertRaisesRegex(RuntimeError, "浏览器取帧模式"):
            service.submit_browser_frame(
                device_id="camera-01",
                captured_at=datetime.now(timezone.utc),
                frame_base64="A" * 256,
            )

    def test_stream_reconnect_clears_stale_prediction(self) -> None:
        service = FallLiveMonitorService(
            Settings(_env_file=None, fall_live_monitor_enabled=False)
        )
        service._handle_message(
            "camera-01",
            {
                "type": "prediction",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "risk_score": 0.72,
                "risk_level": "HIGH",
                "positive_votes": 5,
            },
        )

        service._handle_message(
            "camera-01",
            {
                "type": "stream",
                "state": "RECONNECTING",
                "message": "OpenSDK stream stalled",
            },
        )

        status = service.get_status()
        self.assertEqual(status.state, FallLiveState.CONNECTING)
        self.assertEqual(status.input_state, FallLiveInputState.WAITING)
        self.assertEqual(status.input_message, "视频流中断，正在自动重连")
        self.assertFalse(status.target_present)
        self.assertFalse(status.training_input_ready)
        self.assertIsNone(status.risk_score)
        self.assertIsNone(status.risk_level)
        self.assertIsNone(status.last_prediction_at)
        self.assertEqual(status.source_window_frames, 0)
        self.assertIn("stalled", status.error)


if __name__ == "__main__":
    unittest.main()
