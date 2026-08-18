"""Room-aware fall-risk orchestration."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from app.modules.fall.mapping import (
    map_camera_led_snapshot,
    map_radar_only_snapshot,
    unavailable_room,
)
from app.modules.fall.ports import FallRiskSource, FallRiskSourceError
from app.modules.fall.schemas import FallRiskOverview, RiskLevel, RoomFallRisk, SensorStatus

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
        room_profiles: tuple[FallRiskRoomProfile, ...] = DEFAULT_ROOM_PROFILES,
    ) -> None:
        self._source = source
        self._room_profiles = room_profiles

    async def get_overview(self, *, elder_id: str) -> FallRiskOverview:
        rooms = await asyncio.gather(
            *(self._get_room(profile, elder_id=elder_id) for profile in self._room_profiles)
        )
        overall = max(
            (room.risk_level for room in rooms),
            key=_RISK_RANK.__getitem__,
            default=RiskLevel.UNKNOWN,
        )
        return FallRiskOverview(
            overall_risk_level=overall,
            rooms=list(rooms),
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
