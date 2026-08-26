import unittest
from datetime import datetime, timezone

from app.modules.fall.multimodal_engine.schemas.fall_live import (
    FallLiveInputState,
    FallLiveState,
    FallLiveStatusResponse,
)
from app.modules.fall.multimodal_engine.schemas.radar import RadarStatusResponse
from app.modules.fall.multimodal_engine.services.guard_session import (
    MultimodalGuardSessionService,
)


class FakeFallMonitor:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0
        self.state = FallLiveState.STOPPED

    def start(self) -> None:
        self.starts += 1
        self.state = FallLiveState.RUNNING

    def stop(self) -> None:
        self.stops += 1
        self.state = FallLiveState.STOPPED

    def get_status(self) -> FallLiveStatusResponse:
        return FallLiveStatusResponse(
            enabled=True,
            state=self.state,
            input_state=FallLiveInputState.READY,
            input_message="camera",
        )


class FakeRadarSource:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.online = available
        self.ensure_calls = 0

    def ensure_running(self) -> dict[str, object]:
        self.ensure_calls += 1
        if not self.available:
            raise RuntimeError("offline")
        self.online = True
        return {"status": "running", "already_running": True}


class FakeRadarIntegration:
    def __init__(self, *, online: bool = True) -> None:
        self.online = online
        self.session_enabled = False
        self.enables = 0
        self.disables = 0

    def enable_for_session(self, session_id: str) -> None:
        assert session_id
        self.enables += 1
        self.session_enabled = True

    def disable_for_session(self) -> None:
        self.disables += 1
        self.session_enabled = False

    def get_status(self) -> RadarStatusResponse:
        return RadarStatusResponse(
            online=self.online and self.session_enabled,
            checked_at=datetime.now(timezone.utc),
        )


class MultimodalGuardSessionTest(unittest.TestCase):
    def test_retries_are_idempotent_and_stop_only_unbinds_radar(self) -> None:
        camera = FakeFallMonitor()
        radar = FakeRadarIntegration()
        source = FakeRadarSource()
        service = MultimodalGuardSessionService(camera, radar, source)

        first = service.start("guard-test-001")
        second = service.start("guard-test-002")

        self.assertEqual(first.session_id, "guard-test-001")
        self.assertEqual(second.session_id, "guard-test-001")
        self.assertEqual(camera.starts, 1)
        self.assertEqual(source.ensure_calls, 1)
        self.assertEqual(radar.enables, 1)

        stopped = service.stop()
        stopped_again = service.stop()

        self.assertFalse(stopped.active)
        self.assertFalse(stopped_again.active)
        self.assertEqual(camera.stops, 1)
        self.assertEqual(radar.disables, 1)
        self.assertFalse(hasattr(source, "stop"))

    def test_radar_failure_degrades_to_camera_only(self) -> None:
        camera = FakeFallMonitor()
        radar = FakeRadarIntegration(online=False)
        source = FakeRadarSource(available=False)
        service = MultimodalGuardSessionService(camera, radar, source)

        status = service.start("guard-test-003")

        self.assertTrue(status.active)
        self.assertEqual(status.camera_analysis.state.value, "RUNNING")
        self.assertEqual(status.radar_participation.state.value, "DEGRADED")
        self.assertIn("RADAR_UNAVAILABLE_CAMERA_ONLY", status.reason_codes)

    def test_status_poll_recovers_radar_without_starting_a_second_session(self) -> None:
        camera = FakeFallMonitor()
        radar = FakeRadarIntegration(online=False)
        source = FakeRadarSource(available=False)
        service = MultimodalGuardSessionService(
            camera,
            radar,
            source,
            radar_retry_interval_seconds=0.0,
        )

        first = service.start("guard-test-004")
        self.assertEqual(first.state.value, "DEGRADED")

        source.available = True
        radar.online = True
        recovered = service.get_status()

        self.assertEqual(recovered.state.value, "ACTIVE")
        self.assertEqual(source.ensure_calls, 2)
        self.assertEqual(camera.starts, 1)
        self.assertEqual(radar.enables, 1)


if __name__ == "__main__":
    unittest.main()
