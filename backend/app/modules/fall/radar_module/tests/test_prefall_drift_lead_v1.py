"""Tests for drift_xy time-lead analysis."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from radar_module.analysis.prefall_drift_lead_v1 import (
    WINDOWS,
    align_drift_curves,
    apply_world_transform,
    find_max_drift_time,
    find_sustained_descent_onset,
    find_sustained_descent_onset_hr,
    timewindow_stats,
    repeat_direction_consistency,
    _compute_frame_features,
)


def _make_record(phase: str, mono: float, n_points: int, x_spread: float,
                 z: float = 1.0):
    rng = np.random.default_rng(0)
    pts = []
    for i in range(n_points):
        pts.append({
            "x": float(rng.normal(0, x_spread)),
            "y": float(rng.normal(0, 0.1)),
            "z": float(z + rng.normal(0, 0.05)),
            "velocity": float(rng.normal(0, 0.02)),
        })
    return {
        "timestamp": (datetime(2026, 8, 18, tzinfo=timezone.utc)
                      + timedelta(milliseconds=55 * int(mono * 18.18))).isoformat(),
        "phase": phase,
        "monotonic_since_repeat_start": mono,
        "points": pts,
    }


def _make_fall_session():
    """构造 fall session：pre 高点数，action 后段点云骤降。"""
    records = []
    # still_pre 2s，每帧 20 点
    for i in range(36):
        records.append(_make_record("still_pre", i * 0.055, 20, 0.3))
    # action 前半段（2-4s）点数正常，后半段（4-6s）骤降
    for i in range(110):
        t = 2.0 + i * 0.055
        n = 20 if t < 4.2 else 3  # 4.2s 后点云骤降
        records.append(_make_record("action", t, n, 0.3))
    return records


def test_onset_detected_by_point_count_drop() -> None:
    records = _make_fall_session()
    bases, dyns = _compute_frame_features(records, 1.0 / 18.18)
    onset = find_sustained_descent_onset(records, bases, period=1.0 / 18.18)
    assert onset["found"] is True
    # onset 应落在 4.2s 附近（点云骤降开始）
    assert onset["onset_time_rel_action_start_s"] >= 3.8
    assert onset["onset_time_rel_action_start_s"] <= 5.0
    assert onset["method"] == "point_count_sustained_drop"


def test_onset_not_drift_dependent() -> None:
    """onset 检测不含任何 drift_xy 水平特征。"""
    records = _make_fall_session()
    bases, dyns = _compute_frame_features(records, 1.0 / 18.18)
    onset = find_sustained_descent_onset(records, bases, period=1.0 / 18.18)
    # 检测仅用 point_count（垂直/密度），与 drift 无关
    assert "drift" not in json.dumps(onset, default=str)


def test_align_drift_curves_onset_at_zero() -> None:
    records = _make_fall_session()
    bases, dyns = _compute_frame_features(records, 1.0 / 18.18)
    onset = find_sustained_descent_onset(records, bases, period=1.0 / 18.18)
    curves = align_drift_curves(records, bases, dyns,
                                onset_rel_s=onset["onset_time_rel_action_start_s"])
    assert "drift_xy_0p5s" in curves
    times = curves["drift_xy_0p5s"]["time"]
    # 有负时间（onset 前）
    assert any(t < 0 for t in times)
    assert any(t > 0 for t in times)


def test_timewindow_stats() -> None:
    curves = {
        "drift_xy_0p5s": {
            "time": [-1.2, -0.8, -0.3, 0.1, 0.3],
            "value": [0.01, 0.02, 0.05, 0.1, 0.2],
        }
    }
    stats = timewindow_stats(curves)
    assert "[0.0,0.5)" in stats["drift_xy_0p5s"]
    assert stats["drift_xy_0p5s"]["[-1.0,-0.5)"]["n"] == 1


def test_repeat_direction_consistency() -> None:
    aligned = [
        {"drift_xy_0p5s": {"time": [-0.4, -0.1], "value": [0.1, 0.2]}},
        {"drift_xy_0p5s": {"time": [-0.4, -0.1], "value": [0.15, 0.25]}},
        {"drift_xy_0p5s": {"time": [-0.4, -0.1], "value": [0.08, 0.18]}},
    ]
    res = repeat_direction_consistency(aligned, (-0.5, -0.2))
    assert res["drift_xy_0p5s"]["direction"] == "positive"
    assert res["drift_xy_0p5s"]["consistency"] == 1.0


def _make_hr_shrink_session():
    """构造 fall session：pre 的 height_range 大，action 后段收缩。"""
    records = []
    rng = np.random.default_rng(0)
    # still_pre 2s：z 跨度大（站立，z 在 1.0-1.8）
    for i in range(36):
        mono = i * 0.055
        pts = [
            {"x": 0.0, "y": 1.0, "z": 1.0 + 0.8, "velocity": 0.0},
            {"x": 0.0, "y": 1.0, "z": 1.0, "velocity": 0.0},
            {"x": 0.0, "y": 1.0, "z": 1.4, "velocity": 0.0},
        ] * 6
        records.append({
            "timestamp": (datetime(2026, 8, 18, tzinfo=timezone.utc)
                          + timedelta(milliseconds=55 * i)).isoformat(),
            "phase": "still_pre", "monotonic_since_repeat_start": mono,
            "points": pts,
        })
    # action 前半段 hr 大，4.2s 后收缩（z 跨度小，趴地）
    for i in range(110):
        t = 2.0 + i * 0.055
        if t < 4.2:
            pts = [
                {"x": 0.0, "y": 1.0, "z": 1.0 + 0.8, "velocity": 0.0},
                {"x": 0.0, "y": 1.0, "z": 1.0, "velocity": 0.0},
                {"x": 0.0, "y": 1.0, "z": 1.4, "velocity": 0.0},
            ] * 6
        else:
            pts = [
                {"x": 0.0, "y": 1.0, "z": 1.0, "velocity": 0.0},
                {"x": 0.0, "y": 1.0, "z": 1.1, "velocity": 0.0},
            ] * 4
        records.append({
            "timestamp": (datetime(2026, 8, 18, tzinfo=timezone.utc)
                          + timedelta(milliseconds=55 * i)).isoformat(),
            "phase": "action", "monotonic_since_repeat_start": t,
            "points": pts,
        })
    return records


def test_hr_shrink_onset_detected() -> None:
    records = _make_hr_shrink_session()
    bases, dyns = _compute_frame_features(records, 1.0 / 18.18)
    onset = find_sustained_descent_onset_hr(records, bases, period=1.0 / 18.18)
    assert onset["found"] is True
    assert onset["onset_time_rel_action_start_s"] >= 3.8
    assert onset["method"] == "height_range_sustained_shrink"


def test_world_transform_adds_height() -> None:
    records = [{
        "phase": "action", "monotonic_since_repeat_start": 2.0,
        "timestamp": "2026-08-18T00:00:00+00:00",
        "points": [{"x": 0.0, "y": 1.0, "z": 0.0, "velocity": 0.0}],
    }]
    world = apply_world_transform(records)
    assert world[0]["points"][0]["z"] > 0.9  # 1m + tilt 补偿


def test_max_drift_time() -> None:
    records = [
        _make_record("action", 2.0, 10, 0.1),
        _make_record("action", 2.1, 10, 0.9),
        _make_record("action", 2.2, 10, 0.1),
    ]
    bases, dyns = _compute_frame_features(records, 1.0 / 18.18)
    peak = find_max_drift_time(records, dyns)
    assert peak["found"] is True
    assert peak["peak_time_rel_action_start_s"] == pytest.approx(2.1, abs=0.2)
