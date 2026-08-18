"""Ports implemented by fall-risk algorithm adapters."""

from typing import Protocol

from app.modules.fall.source_schemas import CameraLedSourceSnapshot, RadarOnlySourceSnapshot


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
