"""Ports implemented by fall-risk algorithm adapters."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from app.modules.fall.source_schemas import (
    CameraLedSourceSnapshot,
    CameraMonitoringSourceStatus,
    GuardSessionSourceStatus,
    RadarOnlySourceSnapshot,
)


class FallRiskSourceError(RuntimeError):
    """The upstream fall-risk service could not provide a usable result."""


class FallRiskSource(Protocol):
    async def get_camera_led_risk(
        self,
        *,
        elder_id: str,
        room_id: str,
    ) -> CameraLedSourceSnapshot: ...

    async def get_radar_only_risk(
        self,
        *,
        elder_id: str,
        room_id: str,
    ) -> RadarOnlySourceSnapshot: ...


class CameraMonitoringControl(Protocol):
    async def start_camera_monitoring(self) -> CameraMonitoringSourceStatus: ...

    async def stop_camera_monitoring(self) -> CameraMonitoringSourceStatus: ...

    async def get_camera_monitoring_status(self) -> CameraMonitoringSourceStatus: ...


class GuardSessionControl(Protocol):
    async def start_guard_session(self, session_id: str) -> GuardSessionSourceStatus: ...

    async def stop_guard_session(self) -> GuardSessionSourceStatus: ...

    async def get_guard_session_status(self) -> GuardSessionSourceStatus: ...


@dataclass(frozen=True, slots=True)
class FallRiskEventWrite:
    """Formal FALL_SUSPECTED event payload persisted by the main backend."""

    source_event_id: str
    elder_user_id: str | None
    risk_level: str
    confidence: float
    summary: str
    occurred_at: datetime
    received_at: datetime
    evidence: dict[str, Any]
    model_name: str
    model_version: str
