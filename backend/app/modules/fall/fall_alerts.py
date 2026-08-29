"""Edge-triggered FALL_SUSPECTED event emission from camera-led App results.

The Android fall page polls ``/api/v1/fall-risk/overview``; every poll is an
observation opportunity. A formal RiskEvent is written only when the
camera-led risk state enters high/critical — one event per high-risk episode,
re-armed automatically once the state returns to normal/low. A short cooldown
absorbs threshold flapping so a single fall cannot spawn multiple events.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Protocol

from app.modules.fall.ports import FallRiskEventWrite
from app.modules.fall.schemas import FallRiskOverview, RoomFallRisk

logger = logging.getLogger(__name__)

_HIGH_RISK_LEVELS = {"high", "critical"}
_LEVEL_RANK = {"unknown": 0, "normal": 1, "low": 2, "medium": 3, "high": 4, "critical": 5}
_CAMERA_LED_DECISION_PATHS = {"camera_led_radar_evidence", "camera_only"}
_LEVEL_STATE_LABEL = {"high": "跌倒高风险", "critical": "疑似跌倒"}
_LEVEL_SUMMARY = {
    "high": "摄像头多模态判断为跌倒高风险，请立即关注老人状态",
    "critical": "检测到疑似跌倒，请立即确认老人状态",
}


class FallRiskEventSink(Protocol):
    async def upsert_fall_event(self, event: FallRiskEventWrite) -> None: ...


class FallAlertController:
    def __init__(self, sink: FallRiskEventSink, *, cooldown_seconds: float = 60.0) -> None:
        self._sink = sink
        self._cooldown_seconds = cooldown_seconds
        self._episode_ids: dict[str, str] = {}
        self._last_emitted_at: dict[str, datetime] = {}

    async def observe(self, *, elder_user_id: str, overview: FallRiskOverview) -> None:
        """Feed one overview poll into the episode state machine."""
        room = self._camera_led_room(overview)
        if room is None:
            return
        if room.risk_level in _HIGH_RISK_LEVELS:
            if elder_user_id in self._episode_ids:
                return
            now = datetime.now(UTC)
            last = self._last_emitted_at.get(elder_user_id)
            if last is not None and (now - last).total_seconds() < self._cooldown_seconds:
                return
            await self._emit(elder_user_id=elder_user_id, room=room, occurred_at=now)
            self._last_emitted_at[elder_user_id] = now
        else:
            # 状态回落即解除幕次；下一次进入高风险会作为新事件上报。
            self._episode_ids.pop(elder_user_id, None)

    async def _emit(
        self,
        *,
        elder_user_id: str,
        room: RoomFallRisk,
        occurred_at: datetime,
    ) -> None:
        source_event_id = f"fall-camera:{elder_user_id}:{uuid.uuid4().hex}"
        self._episode_ids[elder_user_id] = source_event_id
        evidence = {
            "state": room.risk_level.upper(),
            "state_label": _LEVEL_STATE_LABEL.get(room.risk_level, "跌倒高风险"),
            "room_id": room.room_id,
            "room_name": room.room_name,
            "risk_score": room.risk_score,
            "decision_path": room.decision_path,
            "evidence_summary": room.evidence_summary,
        }
        try:
            await self._sink.upsert_fall_event(
                FallRiskEventWrite(
                    source_event_id=source_event_id,
                    elder_user_id=elder_user_id,
                    risk_level=room.risk_level.upper(),
                    confidence=room.risk_score if room.risk_score is not None else 0.0,
                    summary=_LEVEL_SUMMARY.get(
                        room.risk_level, "检测到跌倒高风险，请立即关注老人状态"
                    ),
                    occurred_at=occurred_at,
                    received_at=occurred_at,
                    evidence=evidence,
                    model_name="camera_led_evidence_fusion_v2",
                    model_version="camera-led-evidence-fusion-v2-realtime-v1",
                )
            )
        except Exception:
            # 事件写入失败不能阻断风险状态返回；幕次随之作废以便下轮重试。
            self._episode_ids.pop(elder_user_id, None)
            logger.exception("fall alert persist failed elder=%s", elder_user_id)
            raise

    @staticmethod
    def _camera_led_room(overview: FallRiskOverview) -> RoomFallRisk | None:
        candidates = [
            room for room in overview.rooms if room.decision_path in _CAMERA_LED_DECISION_PATHS
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda room: _LEVEL_RANK.get(room.risk_level, 0))
