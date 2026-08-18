"""Tests for the early/middle/late stage evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from radar_module.analysis.prefall_pilot_stage_eval_v1 import (
    stage_slice_medians,
    compare_stages,
)


def _marks():
    return {
        "pre_start": {"monotonic": 0.0},
        "action_start": {"monotonic": 2.0},
        "action_end": {"monotonic": 8.0},  # 6s action
    }


def _record(phase: str, mono_rel: float, x_range: float, height_range: float, pts: int):
    return {
        "timestamp": "2026-08-18T00:00:00+00:00",
        "phase": phase,
        "monotonic_since_repeat_start": mono_rel,
        "points": [
            {"x": x_range / 2, "y": 0.0, "z": 1.0, "velocity": 0.0},
            {"x": -x_range / 2, "y": 0.0, "z": 1.0 + height_range, "velocity": 0.0},
        ] * max(1, pts // 2),
    }


def _make_action_records(per_stage_values):
    """生成 3 阶段各 N 帧的 action 记录，特征值按阶段给定。"""
    records = []
    # still_pre 2s
    for i in range(36):
        records.append(_record("still_pre", 0.05 * i, 0.05, 0.05, 4))
    for si, (xr, hr, pts) in enumerate(per_stage_values):
        start_rel = 2.0 + si * 2.0
        for i in range(36):  # 每段 2s = 36 帧
            records.append(_record(
                "action",
                start_rel + 0.05 * i,
                xr, hr, pts,
            ))
    return records


def test_stage_slice_medians_three_stages() -> None:
    records = _make_action_records([
        (0.5, 0.3, 10),   # early: x_range 0.5, height 0.3
        (1.0, 0.6, 20),   # middle
        (1.5, 0.9, 30),   # late
    ])
    marks = _marks()
    result = stage_slice_medians(records, marks, period_seconds=1.0 / 18.18)
    assert set(result) == {"early", "middle", "late"}
    # early 应反映 early 的特征
    assert result["early"]["x_range"] == pytest.approx(0.5)
    assert result["early"]["height_range"] == pytest.approx(0.3)
    assert result["middle"]["x_range"] == pytest.approx(1.0)
    assert result["late"]["x_range"] == pytest.approx(1.5)
    assert result["late"]["point_count"] == pytest.approx(30)


def test_stage_slice_medians_all_stages_populated() -> None:
    """所有阶段都应分配到帧，不能有 NaN 空洞。"""
    records = _make_action_records([(0.5, 0.3, 10)] * 3)
    marks = _marks()
    result = stage_slice_medians(records, marks, period_seconds=1.0 / 18.18)
    for stage in ("early", "middle", "late"):
        assert result[stage]["x_range"] == pytest.approx(0.5), stage
        assert result[stage]["height_range"] == pytest.approx(0.3), stage


def test_compare_stages_returns_all_stages() -> None:
    fall = [{
        "repeat_id": f"controlled_forward_fall_r{i:02d}",
        "action_name": "controlled_forward_fall",
        "stages": {
            "early": {"x_range": 1.8, "height_range": 0.8,
                      "drift_xy_1p0s": 0.4, "point_count": 30},
            "middle": {"x_range": 1.4, "height_range": 0.8,
                       "drift_xy_1p0s": 0.5, "point_count": 28},
            "late": {"x_range": 1.5, "height_range": 0.8,
                     "drift_xy_1p0s": 0.4, "point_count": 30},
        },
    } for i in range(5)]
    stand = [{
        "repeat_id": f"standing_r{i:02d}",
        "action_name": "standing",
        "stages": {
            "early": {"x_range": 0.1, "height_range": 0.1,
                      "drift_xy_1p0s": 0.3, "point_count": 2},
            "middle": {"x_range": 0.05, "height_range": 0.1,
                       "drift_xy_1p0s": 0.2, "point_count": 2},
            "late": {"x_range": 0.06, "height_range": 0.1,
                     "drift_xy_1p0s": 0.26, "point_count": 4},
        },
    } for i in range(5)]

    result = compare_stages({
        "controlled_forward_fall": fall,
        "standing": stand,
    })
    assert set(result) == {"early", "middle", "late"}
    # early height_range 应显著（fall 0.8 vs stand 0.1）
    p = result["early"]["height_range"]["standing"]["mannwhitney_p"]
    assert p < 0.1
