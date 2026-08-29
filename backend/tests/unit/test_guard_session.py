from datetime import UTC, datetime

import pytest

from app.modules.fall.source_schemas import GuardSessionSourceStatus
from app.modules.guarding.psychology_observation import PsychologyObservationController
from app.modules.guarding.schemas import LifecycleState
from app.modules.guarding.service import GuardianSessionService
from app.modules.psychology.mapping import unavailable_overview


class FakeFallControl:
    def __init__(self, *, radar_running: bool = True) -> None:
        self.radar_running = radar_running
        self.starts = 0
        self.stops = 0

    async def start_guard_session(self, session_id: str) -> GuardSessionSourceStatus:
        self.starts += 1
        return self._status(active=True, session_id=session_id)

    async def stop_guard_session(self) -> GuardSessionSourceStatus:
        self.stops += 1
        return self._status(active=False, session_id=None)

    async def get_guard_session_status(self) -> GuardSessionSourceStatus:
        return self._status(active=True, session_id="guard-existing")

    def _status(self, *, active: bool, session_id: str | None) -> GuardSessionSourceStatus:
        enabled = active
        radar_state = "RUNNING" if self.radar_running else "UNAVAILABLE"
        return GuardSessionSourceStatus.model_validate(
            {
                "schema_version": "multimodal_guard_session_v1",
                "session_id": session_id,
                "active": active,
                "state": "ACTIVE" if active and self.radar_running else "DEGRADED",
                "camera_analysis": {
                    "state": "RUNNING" if active else "STOPPED",
                    "enabled_for_session": enabled,
                    "detail": "camera",
                },
                "radar_worker": {
                    "state": radar_state,
                    "enabled_for_session": False,
                    "detail": "worker",
                },
                "radar_participation": {
                    "state": radar_state if active else "STOPPED",
                    "enabled_for_session": active and self.radar_running,
                    "detail": "participation",
                },
                "fusion": {
                    "state": "RUNNING" if active else "STOPPED",
                    "enabled_for_session": enabled,
                    "detail": "fusion",
                },
                "updated_at": datetime.now(UTC),
            }
        )


class FakePsychologyService:
    async def get_overview(self, *, subject_key: str):
        assert subject_key == "elder-001"
        return unavailable_overview()


class FakeCognitiveControl:
    enabled = True

    def __init__(self) -> None:
        self.attached: list[tuple[str, str]] = []
        self.detached: list[str] = []

    def attach(self, *, subject_key: str, session_id: str) -> bool:
        self.attached.append((subject_key, session_id))
        return True

    async def detach(self, *, subject_key: str) -> None:
        self.detached.append(subject_key)


@pytest.mark.asyncio
async def test_guard_start_and_stop_are_idempotent_and_do_not_stop_radar() -> None:
    fall = FakeFallControl()
    psychology = PsychologyObservationController(
        FakePsychologyService(),
        interval_seconds=3600,
    )
    cognitive = FakeCognitiveControl()
    service = GuardianSessionService(
        fall_control=fall,
        psychology=psychology,
        cognitive=cognitive,
        fraud_monitoring_enabled=True,
    )

    first = await service.start(subject_key="elder-001")
    second = await service.start(subject_key="elder-001")

    assert first.session_id == second.session_id
    assert fall.starts == 1
    assert second.active is True
    assert second.camera_preview_managed_by_guard is False
    assert second.radar_worker.state is LifecycleState.RUNNING
    assert cognitive.attached == [("elder-001", first.session_id)]

    stopped = await service.stop()
    stopped_again = await service.stop()

    assert fall.stops == 1
    assert stopped.active is False
    assert stopped_again.active is False
    assert cognitive.detached == ["elder-001"]
    # The upstream stop contract is a guard-session stop; Radar worker status
    # remains independent and no Radar /stop method exists on this control.
    assert not hasattr(fall, "stop_radar")


@pytest.mark.asyncio
async def test_radar_unavailable_keeps_other_guard_capabilities_active() -> None:
    fall = FakeFallControl(radar_running=False)
    psychology = PsychologyObservationController(
        FakePsychologyService(),
        interval_seconds=3600,
    )
    service = GuardianSessionService(
        fall_control=fall,
        psychology=psychology,
        cognitive=FakeCognitiveControl(),
        fraud_monitoring_enabled=True,
    )

    status = await service.start(subject_key="elder-001")

    assert status.active is True
    assert status.state is LifecycleState.DEGRADED
    assert status.camera_analysis.state is LifecycleState.RUNNING
    assert status.fraud_monitoring.state is LifecycleState.RUNNING
    assert status.psychology_observation.state is LifecycleState.RUNNING
    assert status.radar_participation.state is LifecycleState.UNAVAILABLE
    assert "RADAR_UNAVAILABLE_CAMERA_ONLY" in status.reason_codes

    await service.shutdown()
