import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.infrastructure.external.fall_risk.local import LocalRadarSource
from app.modules.fall.ports import FallRiskSourceError
from app.modules.fall.source_schemas import RadarCalibratedTcnPredictionSource


def _payload(room: str = "bathroom", *, ts: datetime | None = None, **overrides) -> dict:
    data = {
        "schema_version": "radar_calibrated_tcn_live_v1",
        "timestamp": (ts or datetime.now(UTC)).isoformat(),
        "device_id": "iwr6843isk-01",
        "room": room,
        "pre_fall_score": 0.31,
        "score_valid": True,
        "tcn_risk_state": "WATCH",
        "gate_state": "WATCH",
        "formal_alert": False,
        "data_quality": "GOOD",
    }
    data.update(overrides)
    return data


def _write(store_root: Path, room: str, payload: dict) -> None:
    store_root.mkdir(parents=True, exist_ok=True)
    (store_root / f"{room}_latest.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.asyncio
async def test_local_source_reads_valid_snapshot(tmp_path) -> None:
    _write(tmp_path, "bathroom", _payload())
    source = LocalRadarSource(runtime_state_root=tmp_path)

    result = await source.get_radar_only_risk(elder_id="elder-001", room_id="bathroom")

    assert isinstance(result, RadarCalibratedTcnPredictionSource)
    assert result.room == "bathroom"
    assert result.pre_fall_score == 0.31
    assert result.tcn_risk_state == "WATCH"
    assert result.gate_state == "WATCH"
    assert result.shadow_only is True
    assert result.alert_suppressed is True


@pytest.mark.asyncio
async def test_local_source_missing_snapshot_raises(tmp_path) -> None:
    source = LocalRadarSource(runtime_state_root=tmp_path)

    with pytest.raises(FallRiskSourceError):
        await source.get_radar_only_risk(elder_id="elder-001", room_id="bedroom")


@pytest.mark.asyncio
async def test_local_source_malformed_snapshot_raises(tmp_path) -> None:
    (tmp_path / "bathroom_latest.json").write_text("{not-json", encoding="utf-8")
    source = LocalRadarSource(runtime_state_root=tmp_path)

    with pytest.raises(FallRiskSourceError):
        await source.get_radar_only_risk(elder_id="elder-001", room_id="bathroom")


@pytest.mark.asyncio
async def test_local_source_wrong_schema_raises(tmp_path) -> None:
    _write(tmp_path, "bathroom", _payload(schema_version="radar_risk_v1"))
    source = LocalRadarSource(runtime_state_root=tmp_path)

    with pytest.raises(FallRiskSourceError):
        await source.get_radar_only_risk(elder_id="elder-001", room_id="bathroom")


@pytest.mark.asyncio
async def test_local_source_stale_snapshot_raises(tmp_path) -> None:
    stale = datetime.now(UTC) - timedelta(seconds=120)
    _write(tmp_path, "bathroom", _payload(ts=stale))
    source = LocalRadarSource(runtime_state_root=tmp_path)

    with pytest.raises(FallRiskSourceError):
        await source.get_radar_only_risk(elder_id="elder-001", room_id="bathroom")


@pytest.mark.asyncio
async def test_local_source_camera_led_is_unavailable(tmp_path) -> None:
    source = LocalRadarSource(runtime_state_root=tmp_path)

    with pytest.raises(FallRiskSourceError):
        await source.get_camera_led_risk(elder_id="elder-001", room_id="living_room")
