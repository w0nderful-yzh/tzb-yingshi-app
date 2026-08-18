"""Tests for the pure-radar pre-fall capture tool (replay mode only).

Covers the per-repeat output structure with four time marks:
pre_start / action_start / action_end / post_end.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from radar_module.acquisition.real_prefall_capture_v1 import (
    ACTIONS,
    PILOT_ACTIONS,
    RepeatMetaV1,
    _sensor_to_world,
    _target_to_dict,
    run_capture,
)
from radar_module.acquisition.ti_reader import JsonlReplayAdapter


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _make_replay(tmp_path: Path, frames: int = 300) -> Path:
    """Write a synthetic replay jsonl."""
    path = tmp_path / "replay.jsonl"
    base = datetime(2026, 8, 17, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(frames):
            ts = (base + timedelta(milliseconds=55 * i)).isoformat()
            record = {
                "timestamp": ts,
                "device_id": "iwr6843isk-01",
                "room": "bathroom",
                "source_mode": "REPLAY",
                "points": [
                    {
                        "x": 0.1 * i,
                        "y": 0.5,
                        "z": 1.0 + 0.01 * i,
                        "velocity": 0.0,
                        "snr": 8.0,
                    }
                ],
                "targets": [
                    {
                        "track_id": 1,
                        "x": 0.2 * i,
                        "y": 1.0,
                        "z": 0.5 + 0.01 * i,
                        "velocity_x": 0.1,
                        "velocity_y": 0.2,
                        "velocity_z": 0.3,
                        "accel_x": 0.01,
                        "accel_y": 0.02,
                        "accel_z": 0.03,
                        "confidence": 0.9,
                    }
                ],
                "accepted_point_count": 1,
                "raw_point_count": 1,
                "ti_frame_number": i,
                "ti_parser_error": 0,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def test_action_catalog_and_pilot_set() -> None:
    assert len(ACTIONS) == 10
    assert PILOT_ACTIONS == [
        "standing",
        "fast_sitting",
        "forward_instability_recovery",
        "controlled_forward_fall",
    ]


def test_capture_replay_per_repeat_structure(tmp_path: Path) -> None:
    replay = _make_replay(tmp_path, frames=300)
    source = JsonlReplayAdapter(replay)
    out = tmp_path / "out"
    metas = run_capture(
        source,
        output_directory=out,
        session_id="fast_sitting_p01_test",
        action_name="fast_sitting",
        repeats=2,
        still_seconds=0.1,
        action_seconds=0.3,
        interactive_actions=False,
        max_action_seconds=5.0,
    )
    assert isinstance(metas, list) and len(metas) == 2
    assert all(isinstance(m, RepeatMetaV1) for m in metas)

    # 每个 repeat 独立目录
    repeat_dirs = [d for d in out.iterdir() if d.is_dir() and d.name.startswith("repeat_")]
    assert len(repeat_dirs) == 2

    for meta in metas:
        repeat_dir = out / f"repeat_{int(meta.repeat_id.split('_')[-1][1:]):02d}"
        assert (repeat_dir / "frames.jsonl").exists()
        assert (repeat_dir / "meta.json").exists()

        # 四个时间标记
        marks = {m["name"]: m for m in meta.marks}
        assert set(marks) == {
            "pre_start", "action_start", "action_end", "post_end",
        }
        assert marks["pre_start"]["monotonic"] <= marks["action_start"]["monotonic"]
        assert marks["action_start"]["monotonic"] <= marks["action_end"]["monotonic"]
        assert marks["action_end"]["monotonic"] <= marks["post_end"]["monotonic"]
        # 单调时钟单调递增
        assert (marks["action_start"]["monotonic"] - marks["pre_start"]["monotonic"]) >= 0.1 - 1e-3

        # 帧阶段划分
        records = _read_jsonl(repeat_dir / "frames.jsonl")
        assert records
        phases = {r["phase"] for r in records}
        assert phases <= {"still_pre", "action", "still_post"}
        assert any(r["phase"] == "action" for r in records)
        # 帧有 repeat_id 和相对单调时间
        assert all("repeat_id" in r and "monotonic_since_repeat_start" in r for r in records)
        # action 阶段帧的 monotonic_since_repeat_start 应大致 >= 0.1s
        action_frames = [r for r in records if r["phase"] == "action"]
        assert action_frames
        assert all(r["monotonic_since_repeat_start"] >= 0.05 for r in action_frames)

    # manifest
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["action_name"] == "fast_sitting"
    assert manifest["repeats"] == 2
    assert len(manifest["repeat_ids"]) == 2


def test_capture_replay_preserves_decoded_metadata(tmp_path: Path) -> None:
    replay = _make_replay(tmp_path, frames=200)
    source = JsonlReplayAdapter(replay)
    out = tmp_path / "out2"
    run_capture(
        source,
        output_directory=out,
        session_id="standing_p01_test",
        action_name="standing",
        repeats=1,
        still_seconds=0.1,
        action_seconds=0.2,
        interactive_actions=False,
    )
    repeat_dir = out / "repeat_01"
    records = _read_jsonl(repeat_dir / "frames.jsonl")
    sample = next(r for r in records if r["points_sensor"])
    assert "ti_frame_number" in sample
    assert sample["accepted_point_count"] == 1
    # 新结构：points_sensor + points_world + targets
    assert "points_sensor" in sample
    assert "points_world" in sample
    assert "targets" in sample
    # world 转换后 z 应加了 sensorHeight(默认 1m)
    world_z = sample["points_world"][0]["z"]
    sensor_z = sample["points_sensor"][0]["z"]
    assert world_z - sensor_z == pytest.approx(1.0, abs=0.05)


def test_capture_replay_preserves_targets(tmp_path: Path) -> None:
    """TI Target List TLV 应完整透传到每帧 targets 字段。"""
    replay = _make_replay(tmp_path, frames=120)
    source = JsonlReplayAdapter(replay)
    out = tmp_path / "out_targets"
    run_capture(
        source,
        output_directory=out,
        session_id="fall_p01_targets_test",
        action_name="controlled_forward_fall",
        repeats=1,
        still_seconds=0.1,
        action_seconds=0.2,
        interactive_actions=False,
    )
    records = _read_jsonl(out / "repeat_01" / "frames.jsonl")
    with_target = [r for r in records if r.get("targets")]
    assert with_target
    t = with_target[0]["targets"][0]
    assert t["track_id"] == 1
    assert "pos_x" in t and "pos_y" in t and "pos_z" in t
    assert "vel_x" in t and "vel_y" in t and "vel_z" in t
    assert "acc_x" in t and "acc_y" in t and "acc_z" in t
    assert t["confidence"] == pytest.approx(0.9)


def test_capture_requires_fixed_duration_when_noninteractive(tmp_path: Path) -> None:
    replay = _make_replay(tmp_path, frames=100)
    source = JsonlReplayAdapter(replay)
    with pytest.raises(ValueError):
        run_capture(
            source,
            output_directory=tmp_path / "bad",
            session_id="x",
            action_name="walking",
            repeats=1,
            still_seconds=0.1,
            action_seconds=None,
            interactive_actions=False,
        )


def test_unknown_action_rejected(tmp_path: Path) -> None:
    replay = _make_replay(tmp_path, frames=50)
    source = JsonlReplayAdapter(replay)
    with pytest.raises(ValueError):
        run_capture(
            source,
            output_directory=tmp_path / "bad2",
            session_id="x",
            action_name="not_an_action",
            repeats=1,
            still_seconds=0.1,
            action_seconds=0.2,
            interactive_actions=False,
        )


def test_capture_interactive_drains_stale_frames(tmp_path: Path, monkeypatch) -> None:
    """交互模式下按 Enter 前积压的帧应被丢弃，不进入 repeat。

    交互模式现在用 input() 等待（触发 + 动作结束），需要让 input() 立即
    返回；动作阶段后台线程采集，input() 返回后停止。
    """
    # 构造足够长的 replay，让积压存在；loop 模式让帧持续循环提供
    replay = _make_replay(tmp_path, frames=500)
    source = JsonlReplayAdapter(replay, loop=True)

    # 第1次 input()（触发 repeat）立即返回；第2次 input()（动作结束）延迟
    # 返回，模拟动作持续一段时间，给后台采集线程时间读帧。
    import time as _time

    input_calls = {"n": 0}

    def fake_input(*a, **k) -> str:
        input_calls["n"] += 1
        if input_calls["n"] >= 2:
            _time.sleep(0.3)
        return ""

    monkeypatch.setattr("builtins.input", fake_input)

    out = tmp_path / "interactive"
    metas = run_capture(
        source,
        output_directory=out,
        session_id="standing_p01_interactive_test",
        action_name="standing",
        repeats=1,
        still_seconds=0.1,
        action_seconds=None,
        interactive_actions=True,
        max_action_seconds=5.0,
    )
    assert len(metas) == 1
    meta = metas[0]
    marks = {m["name"]: m for m in meta.marks}
    # action_start 应晚于 pre_start
    assert marks["action_start"]["monotonic"] >= marks["pre_start"]["monotonic"]
    # action_end 应晚于 action_start
    assert marks["action_end"]["monotonic"] >= marks["action_start"]["monotonic"]
    # 帧的 monotonic_since_repeat_start 应保持阶段时长合理（无积压带来的超长 still_pre）
    records = _read_jsonl(out / "repeat_01" / "frames.jsonl")
    still_pre = [r for r in records if r["phase"] == "still_pre"]
    # 0.1s 静置 + 丢弃积压 => still_pre 帧数应远小于 500（积压全部丢弃）
    assert len(still_pre) <= 10
    # 动作阶段应采集到帧（后台线程 + input() 立即返回，至少采集到一些）
    action_frames = [r for r in records if r["phase"] == "action"]
    assert action_frames


def test_sensor_to_world_adds_height_and_tilt() -> None:
    params = {"sensor_height_m": 1.0, "elev_tilt_deg": 5.0, "azi_tilt_deg": 0.0}
    pts = [{"x": 0.0, "y": 1.0, "z": 0.2, "velocity": 0.0, "snr": 8.0}]
    world = _sensor_to_world(pts, params)
    assert world[0]["z"] > 1.0  # 1m + tilt 补偿
    assert world[0]["velocity"] == 0.0
    assert world[0]["snr"] == 8.0


def test_sensor_to_world_keeps_sensor_points() -> None:
    """world 转换不修改原始点，只返回新列表。"""
    params = {"sensor_height_m": 1.0, "elev_tilt_deg": 0.0, "azi_tilt_deg": 0.0}
    pts = [{"x": 0.0, "y": 1.0, "z": 0.2, "velocity": 0.0}]
    world = _sensor_to_world(pts, params)
    assert pts[0]["z"] == 0.2  # 原始不变
    assert world[0]["z"] == pytest.approx(1.2)


def test_target_to_dict_full_fields() -> None:
    from radar_module.contracts import RadarTarget

    t = RadarTarget(
        track_id=3, x=0.5, y=1.0, z=1.5,
        velocity_x=0.1, velocity_y=0.2, velocity_z=0.3,
        accel_x=0.01, accel_y=0.02, accel_z=0.03,
        confidence=0.85,
    )
    d = _target_to_dict(t)
    assert d["track_id"] == 3
    assert d["pos_x"] == 0.5 and d["pos_z"] == 1.5
    assert d["vel_z"] == 0.3
    assert d["acc_z"] == 0.03
    assert d["confidence"] == pytest.approx(0.85)


def test_target_to_dict_sparse_fields() -> None:
    """缺省字段不应被臆造。"""
    from radar_module.contracts import RadarTarget

    t = RadarTarget(track_id=1, x=0.0, y=0.0, z=0.0)
    d = _target_to_dict(t)
    assert d["track_id"] == 1
    assert "vel_x" not in d
    assert "acc_x" not in d
    assert "confidence" not in d
