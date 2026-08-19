"""Read-only local fall-risk source backed by radar worker runtime snapshots.

Radar workers write their own raw runtime state (CalibratedTcnLiveResultV1)
to runtime_state/<room>_latest.json; this source reads and maps it to the
fall adapter contract (RadarCalibratedTcnPredictionSource). Missing, malformed
or stale snapshots fail closed via FallRiskSourceError.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from app.modules.fall.ports import FallRiskSourceError
from app.modules.fall.source_schemas import (
    CameraLedSourceSnapshot,
    QualityLevel,
    RadarCalibratedTcnPredictionSource,
    RadarGateState,
    RadarOnlySourceSnapshot,
    RadarTcnRiskState,
)

_RADAR_MODULE = Path(__file__).resolve().parents[3] / "modules" / "fall" / "radar_module"


class LocalRadarSource:
    """Reads the latest radar runtime state written by radar workers."""

    def __init__(
        self,
        runtime_state_root: Path | None = None,
        *,
        max_age_seconds: float = 30.0,
    ) -> None:
        self._state_root = runtime_state_root or (_RADAR_MODULE / "runtime_state")
        self._max_age_seconds = max_age_seconds

    async def get_camera_led_risk(
        self,
        *,
        elder_id: str,
        room_id: str,
    ) -> CameraLedSourceSnapshot:
        # 客厅 C 路算法侧未实现；保持 unavailable（service 捕获后转 unavailable_room）。
        raise FallRiskSourceError("camera-led fall-risk upstream is not available")

    async def get_radar_only_risk(
        self,
        *,
        elder_id: str,
        room_id: str,
    ) -> RadarOnlySourceSnapshot:
        payload = self._read_snapshot(room_id)
        try:
            return _map_calibrated_tcn_snapshot(payload)
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            raise FallRiskSourceError(
                f"radar snapshot cannot be mapped for room {room_id}"
            ) from exc

    def _read_snapshot(self, room_id: str) -> dict[str, object]:
        path = self._state_root / f"{room_id}_latest.json"
        if not path.is_file():
            raise FallRiskSourceError(f"no radar snapshot for room {room_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise FallRiskSourceError(f"radar snapshot is invalid for room {room_id}") from exc
        if not isinstance(payload, dict):
            raise FallRiskSourceError(f"radar snapshot is not an object for room {room_id}")
        if payload.get("schema_version") != "radar_calibrated_tcn_live_v1":
            raise FallRiskSourceError(f"radar snapshot schema mismatch for room {room_id}")
        self._validate_freshness(payload, room_id)
        return payload

    def _validate_freshness(self, payload: dict[str, object], room_id: str) -> None:
        raw = payload.get("timestamp")
        if not raw:
            raise FallRiskSourceError(f"radar snapshot missing timestamp for room {room_id}")
        try:
            parsed = datetime.fromisoformat(str(raw))
        except ValueError as exc:
            raise FallRiskSourceError(
                f"radar snapshot timestamp invalid for room {room_id}"
            ) from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        age = (datetime.now(UTC) - parsed).total_seconds()
        if age > self._max_age_seconds:
            raise FallRiskSourceError(f"radar snapshot is stale for room {room_id}")


def _map_calibrated_tcn_snapshot(payload: dict[str, object]) -> RadarCalibratedTcnPredictionSource:
    return RadarCalibratedTcnPredictionSource(
        schema_version="radar_calibrated_tcn_live_v1",
        timestamp=datetime.fromisoformat(str(payload["timestamp"])),
        device_id=str(payload.get("device_id", "iwr6843isk-01")),
        room=str(payload.get("room", "unknown")),
        pre_fall_score=float(str(payload["pre_fall_score"])),
        score_valid=bool(payload.get("score_valid", False)),
        tcn_risk_state=cast(RadarTcnRiskState, payload.get("tcn_risk_state", "UNKNOWN")),
        gate_state=cast(RadarGateState, payload.get("gate_state", "UNKNOWN")),
        formal_alert=bool(payload.get("formal_alert", False)),
        data_quality=cast(QualityLevel, payload.get("data_quality", "INSUFFICIENT_DATA")),
        shadow_only=True,
        alert_suppressed=True,
    )
