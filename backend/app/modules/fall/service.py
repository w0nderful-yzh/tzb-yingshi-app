"""Room-aware fall-risk orchestration."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from app.modules.fall.mapping import (
    map_camera_led_snapshot,
    map_camera_monitoring_status,
    map_radar_only_snapshot,
    unavailable_room,
)
from app.modules.fall.ports import CameraMonitoringControl, FallRiskSource, FallRiskSourceError
from app.modules.fall.schemas import (
    CameraAlgorithmStatus,
    CameraMonitoringStatus,
    CameraStreamStatus,
    FallRiskOverview,
    RiskLevel,
    RoomFallRisk,
    SensorStatus,
)

logger = logging.getLogger(__name__)


class RoomRiskMode(StrEnum):
    CAMERA_LED = "camera_led"
    RADAR_ONLY = "radar_only"


@dataclass(frozen=True, slots=True)
class FallRiskRoomProfile:
    room_id: str
    room_name: str
    mode: RoomRiskMode


DEFAULT_ROOM_PROFILES = (
    FallRiskRoomProfile("living_room", "客厅", RoomRiskMode.CAMERA_LED),
    FallRiskRoomProfile("bathroom", "卫生间", RoomRiskMode.RADAR_ONLY),
    FallRiskRoomProfile("bedroom", "卧室", RoomRiskMode.RADAR_ONLY),
)

_RISK_RANK = {
    RiskLevel.UNKNOWN: 0,
    RiskLevel.NORMAL: 1,
    RiskLevel.LOW: 2,
    RiskLevel.MEDIUM: 3,
    RiskLevel.HIGH: 4,
    RiskLevel.CRITICAL: 5,
}


class FallRiskService:
    def __init__(
        self,
        source: FallRiskSource | None,
        *,
        camera_control: CameraMonitoringControl | None = None,
        room_profiles: tuple[FallRiskRoomProfile, ...] = DEFAULT_ROOM_PROFILES,
    ) -> None:
        self._source = source
        self._camera_control = camera_control
        self._room_profiles = room_profiles

    async def start_camera_monitoring(self) -> CameraMonitoringStatus:
        if self._camera_control is None:
            return self._camera_unavailable("摄像头跌倒预测服务未配置")
        try:
            snapshot = await self._camera_control.start_camera_monitoring()
        except (FallRiskSourceError, ValueError) as exc:
            logger.warning("camera monitoring start failed: %s", type(exc).__name__)
            return self._camera_unavailable("摄像头跌倒预测服务无法连接")
        return map_camera_monitoring_status(snapshot)

    async def stop_camera_monitoring(self) -> CameraMonitoringStatus:
        if self._camera_control is None:
            return self._camera_unavailable("摄像头跌倒预测服务未配置")
        try:
            snapshot = await self._camera_control.stop_camera_monitoring()
        except (FallRiskSourceError, ValueError) as exc:
            logger.warning("camera monitoring stop failed: %s", type(exc).__name__)
            return self._camera_unavailable("摄像头跌倒预测服务无法连接")
        return map_camera_monitoring_status(snapshot)

    async def get_camera_monitoring_status(self) -> CameraMonitoringStatus:
        if self._camera_control is None:
            return self._camera_unavailable("摄像头跌倒预测服务未配置")
        try:
            snapshot = await self._camera_control.get_camera_monitoring_status()
        except (FallRiskSourceError, ValueError) as exc:
            logger.warning("camera monitoring status failed: %s", type(exc).__name__)
            return self._camera_unavailable("摄像头跌倒预测服务无法连接")
        return map_camera_monitoring_status(snapshot)

    async def get_overview(self, *, elder_id: str) -> FallRiskOverview:
        camera_status_task = asyncio.create_task(self.get_camera_monitoring_status())
        rooms = await asyncio.gather(
            *(self._get_room(profile, elder_id=elder_id) for profile in self._room_profiles)
        )
        camera_monitoring = await camera_status_task
        overall = max(
            (room.risk_level for room in rooms),
            key=_RISK_RANK.__getitem__,
            default=RiskLevel.UNKNOWN,
        )
        return FallRiskOverview(
            overall_risk_level=overall,
            rooms=list(rooms),
            camera_monitoring=camera_monitoring,
            generated_at=datetime.now(UTC),
        )

    async def _get_room(
        self,
        profile: FallRiskRoomProfile,
        *,
        elder_id: str,
    ) -> RoomFallRisk:
        if self._source is None:
            return self._unavailable(profile)
        try:
            if profile.mode is RoomRiskMode.CAMERA_LED:
                camera_snapshot = await self._source.get_camera_led_risk(
                    elder_id=elder_id,
                    room_id=profile.room_id,
                )
                return map_camera_led_snapshot(
                    camera_snapshot,
                    room_id=profile.room_id,
                    room_name=profile.room_name,
                )
            radar_snapshot = await self._source.get_radar_only_risk(
                elder_id=elder_id,
                room_id=profile.room_id,
            )
            return map_radar_only_snapshot(radar_snapshot, room_name=profile.room_name)
        except (FallRiskSourceError, ValueError) as exc:
            logger.warning(
                "fall risk source unavailable for room=%s: %s",
                profile.room_id,
                type(exc).__name__,
            )
            return self._unavailable(profile)

    @staticmethod
    def _unavailable(profile: FallRiskRoomProfile) -> RoomFallRisk:
        return unavailable_room(
            room_id=profile.room_id,
            room_name=profile.room_name,
            camera_status=(
                SensorStatus.UNAVAILABLE
                if profile.mode is RoomRiskMode.CAMERA_LED
                else SensorStatus.NOT_APPLICABLE
            ),
            radar_status=SensorStatus.UNAVAILABLE,
        )

    @staticmethod
    def _camera_unavailable(detail: str) -> CameraMonitoringStatus:
        return CameraMonitoringStatus(
            camera_stream_status=CameraStreamStatus.UNAVAILABLE,
            camera_algorithm_status=CameraAlgorithmStatus.UNAVAILABLE,
            detail=detail,
            updated_at=datetime.now(UTC),
        )
