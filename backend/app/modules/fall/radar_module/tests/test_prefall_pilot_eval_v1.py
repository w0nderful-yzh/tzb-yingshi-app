"""Tests for the repeat-level pre-fall pilot evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from radar_module.analysis.prefall_pilot_eval_v1 import (
    CANDIDATE_FEATURES,
    block_bootstrap_curve,
    load_sessions,
    rq1_direction_consistency,
    rq2_fall_vs_instability,
    rq3_recovery_vs_fall,
    rq4_recovery_trend,
    rq5_pointcount_motion_artifact,
    extract_repeat_features,
)


def _make_points(centroid: tuple[float, float, float], n: int = 10, vel: float = 0.0):
    rng = np.random.default_rng(0)
    return [
        {
            "x": float(centroid[0] + rng.normal(0, 0.05)),
            "y": float(centroid[1] + rng.normal(0, 0.05)),
            "z": float(centroid[2] + rng.normal(0, 0.05)),
            "velocity": float(vel + rng.normal(0, 0.02)),
            "snr": 8.0,
        }
        for _ in range(n)
    ]


def _frames(
    phase: str,
    base_z: float,
    count: int,
    *,
    spread_x: float = 0.05,
    point_count: int = 10,
    z_trend: float = 0.0,
) -> list[dict]:
    rng = np.random.default_rng(1)
    frames = []
    for i in range(count):
        z = base_z + z_trend * i
        pts = [
            {
                "x": float(rng.normal(0, spread_x)),
                "y": float(rng.normal(0, 0.1)),
                "z": float(z + rng.normal(0, 0.05)),
                "velocity": float(rng.normal(0, 0.02)),
                "snr": 8.0,
            }
            for _ in range(point_count)
        ]
        frames.append({
            "timestamp": (datetime(2026, 8, 17, tzinfo=timezone.utc)
                          + timedelta(milliseconds=55 * i)).isoformat(),
            "phase": phase,
            "points": pts,
        })
    return frames


def _write_session(root: Path, action: str, n_repeats: int = 5) -> None:
    """写一个动作的 n 个 repeat。构造不同动作的差异：
    - fall: action 阶段 x_range 大、z 下降、点少
    - instability: action 阶段 x_range 中等、点中等、post 回到基线
    - fast_sitting: action 阶段运动强(doppler)、点少但 x_range 小
    - standing: 全程平稳
    """
    action_dir = root / action
    action_dir.mkdir(parents=True, exist_ok=True)
    for r in range(1, n_repeats + 1):
        repeat_dir = action_dir / f"repeat_{r:02d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        # pre: 静止
        frames += _frames("still_pre", base_z=1.5, count=36,
                          spread_x=0.05, point_count=12)
        # action
        if action == "controlled_forward_fall":
            frames += _frames("action", base_z=1.5, count=72,
                              spread_x=0.6, point_count=4, z_trend=-0.02)
        elif action == "forward_instability_recovery":
            frames += _frames("action", base_z=1.5, count=72,
                              spread_x=0.35, point_count=8, z_trend=-0.005)
        elif action == "fast_sitting":
            frames += _frames("action", base_z=1.5, count=72,
                              spread_x=0.1, point_count=4, z_trend=-0.015)
        else:  # standing
            frames += _frames("action", base_z=1.5, count=72,
                              spread_x=0.05, point_count=12)
        # post
        if action == "controlled_forward_fall":
            # 跌倒后仍偏离：低位、点少
            frames += _frames("still_post", base_z=0.3, count=36,
                              spread_x=0.6, point_count=4)
        else:
            # 恢复/其他：回到基线
            frames += _frames("still_post", base_z=1.5, count=36,
                              spread_x=0.05, point_count=12)

        (repeat_dir / "frames.jsonl").write_text(
            "\n".join(json.dumps(f) for f in frames), encoding="utf-8")
        meta = {
            "repeat_id": f"{action}_r{r:02d}",
            "action_name": action,
            "pre_start": "2026-08-17T00:00:00+00:00",
            "action_start": "2026-08-17T00:00:02+00:00",
            "action_end": "2026-08-17T00:00:06+00:00",
            "post_end": "2026-08-17T00:00:08+00:00",
        }
        (repeat_dir / "meta.json").write_text(
            json.dumps(meta), encoding="utf-8")


def test_load_sessions_structure(tmp_path: Path) -> None:
    _write_session(tmp_path, "standing", n_repeats=3)
    _write_session(tmp_path, "controlled_forward_fall", n_repeats=3)
    by_action = load_sessions(tmp_path)
    assert set(by_action) == {"standing", "controlled_forward_fall"}
    assert len(by_action["standing"]) == 3
    rep = by_action["standing"][0]
    assert rep["repeat_id"].startswith("standing_")
    assert rep["per_phase_medians"]["pre"]["point_count"] == pytest.approx(12, abs=1.5)


def test_rq1_direction_consistency(tmp_path: Path) -> None:
    _write_session(tmp_path, "controlled_forward_fall", n_repeats=5)
    by_action = load_sessions(tmp_path)
    rq1 = rq1_direction_consistency(by_action)
    assert "controlled_forward_fall" in rq1
    # fall 的 x_range 应一致上升
    info = rq1["controlled_forward_fall"]["x_range"]["action_minus_pre"]
    assert info["consistent_direction"] == "positive"
    assert info["sign_consistency"] >= 0.8
    # fall 的 point_count 应一致下降
    pc = rq1["controlled_forward_fall"]["point_count"]["action_minus_pre"]
    assert pc["consistent_direction"] == "negative"


def test_rq2_fall_vs_instability(tmp_path: Path) -> None:
    _write_session(tmp_path, "controlled_forward_fall", n_repeats=5)
    _write_session(tmp_path, "forward_instability_recovery", n_repeats=5)
    by_action = load_sessions(tmp_path)
    rq2 = rq2_fall_vs_instability(by_action)
    assert "error" not in rq2
    # fall 的 x_range 增幅应大于 instability
    xr = rq2["x_range"]
    assert np.median(xr["fall_diffs"]) > np.median(xr["instability_diffs"])


def test_rq3_recovery_vs_fall(tmp_path: Path) -> None:
    _write_session(tmp_path, "controlled_forward_fall", n_repeats=5)
    _write_session(tmp_path, "forward_instability_recovery", n_repeats=5)
    by_action = load_sessions(tmp_path)
    rq3 = rq3_recovery_vs_fall(by_action)
    assert "error" not in rq3
    # fall action 阶段 point_count 应小于 instability
    assert rq3["point_count"]["fall_action_median"] < rq3["point_count"]["instability_action_median"]
    # fall x_range 应大于 instability
    assert rq3["x_range"]["fall_action_median"] > rq3["x_range"]["instability_action_median"]


def test_rq4_recovery_trend(tmp_path: Path) -> None:
    _write_session(tmp_path, "forward_instability_recovery", n_repeats=5)
    _write_session(tmp_path, "controlled_forward_fall", n_repeats=5)
    by_action = load_sessions(tmp_path)
    rq4 = rq4_recovery_trend(by_action)
    # fall: post 仍低位/偏离，point_count post-pre 应显著负
    fall_pc = rq4["controlled_forward_fall"]["point_count"]["post_minus_pre"]
    assert fall_pc["consistent_direction"] == "negative"
    # instability 恢复：x_range post-pre 应远小于 fall 的 post-pre（回到基线）
    inst_x_diffs = np.asarray(
        rq4["forward_instability_recovery"]["x_range"]["repeat_diffs"],
        dtype=np.float64,
    )
    fall_x_diffs = np.asarray(
        rq4["controlled_forward_fall"]["x_range"]["repeat_diffs"],
        dtype=np.float64,
    )
    assert np.abs(np.nanmedian(inst_x_diffs)) < np.abs(np.nanmedian(fall_x_diffs))
    # fall 的 post 仍偏离：height/point_count 应不回基线
    assert rq4["controlled_forward_fall"]["height_range"]["post_minus_pre"][
        "consistent_direction"
    ] in ("positive", "negative")


def test_rq5_pointcount_motion_artifact(tmp_path: Path) -> None:
    _write_session(tmp_path, "fast_sitting", n_repeats=5)
    _write_session(tmp_path, "controlled_forward_fall", n_repeats=5)
    by_action = load_sessions(tmp_path)
    rq5 = rq5_pointcount_motion_artifact(by_action)
    assert "error" not in rq5
    # fast_sitting 运动强(doppler_std 增), fall 也运动
    assert "point_count" in rq5
    assert "moving_fraction" in rq5


def test_block_bootstrap_curve(tmp_path: Path) -> None:
    _write_session(tmp_path, "standing", n_repeats=3)
    by_action = load_sessions(tmp_path)
    result = block_bootstrap_curve(by_action["standing"], "point_count")
    assert result["repeat_count"] == 3
    assert "boot_median_of_medians" in result
    assert len(result["boot_ci95"]) == 2


def test_candidate_features_present() -> None:
    for feat in ("x_range", "drift_xy_1p0s", "drift_xy_0p5s",
                 "drift_xy_1frame", "point_count", "height_range"):
        assert feat in CANDIDATE_FEATURES
