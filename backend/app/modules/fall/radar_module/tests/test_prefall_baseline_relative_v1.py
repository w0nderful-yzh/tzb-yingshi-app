"""Tests for baseline-relative pre-fall feature analysis."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from radar_module.analysis.prefall_baseline_relative_v1 import (
    compute_repeat_features,
    evaluate_pair,
    load_repeat_features,
    confound_removal_analysis,
    _safe_auroc,
    _safe_pr_auc,
    _cohen_d,
)


def _make_frames(phase: str, base_z: float, count: int, x_range: float, pts: int,
                 z_drift: float = 0.0, start_mono: float = 0.0):
    rng = np.random.default_rng(0)
    frames = []
    for i in range(count):
        z = base_z + z_drift * i
        mono = start_mono + 0.055 * i
        frames.append({
            "timestamp": (datetime(2026, 8, 18, tzinfo=timezone.utc)
                          + timedelta(milliseconds=55 * i)).isoformat(),
            "phase": phase,
            "monotonic_since_repeat_start": mono,
            "points": [
                {"x": x_range / 2, "y": 0.0, "z": z, "velocity": 0.0},
                {"x": -x_range / 2, "y": 0.0, "z": z + 0.5, "velocity": 0.0},
                {"x": 0.0, "y": 0.1, "z": z + 0.3, "velocity": 0.0},
            ] * max(1, pts // 3),
        })
    return frames


def _write_repeat(root: Path, action: str, rep_idx: int, *, x_range_pre: float,
                  x_range_action: float, z_drift: float = 0.0) -> None:
    action_dir = root / action
    rep_dir = action_dir / f"repeat_{rep_idx:02d}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    frames += _make_frames("still_pre", 1.5, 36, x_range_pre, 12,
                           start_mono=0.0)
    frames += _make_frames("action", 1.5, 108, x_range_action, 12,
                           z_drift=z_drift, start_mono=2.0)
    frames += _make_frames("still_post", 1.5, 36, x_range_pre, 12,
                           start_mono=8.0)
    (rep_dir / "frames.jsonl").write_text(
        "\n".join(json.dumps(f) for f in frames), encoding="utf-8")
    meta = {
        "repeat_id": f"{action}_r{rep_idx:02d}",
        "action_name": action,
        "marks": [
            {"name": "pre_start", "monotonic": 0.0},
            {"name": "action_start", "monotonic": 2.0},
            {"name": "action_end", "monotonic": 8.0},
            {"name": "post_end", "monotonic": 10.0},
        ],
    }
    (rep_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_compute_repeat_features_delta(tmp_path: Path) -> None:
    root = tmp_path / "sess"
    # fall: pre x_range 小, action x_range 大 → delta 应正
    _write_repeat(root, "controlled_forward_fall", 1,
                  x_range_pre=0.3, x_range_action=1.2)
    by_action = load_repeat_features(root)
    assert "controlled_forward_fall" in by_action
    rep = by_action["controlled_forward_fall"][0]
    assert "baseline" in rep
    assert rep["baseline"]["x_range"] > 0
    assert rep["stage_delta"]["early"]["x_range"] > 0
    assert rep["stage_relative"]["early"]["x_range"] > 0


def test_delta_removes_static_baseline(tmp_path: Path) -> None:
    """若 pre 和 action 的 x_range 相同，delta 应接近 0。"""
    root = tmp_path / "sess2"
    _write_repeat(root, "standing", 1,
                  x_range_pre=0.8, x_range_action=0.8)
    by_action = load_repeat_features(root)
    rep = by_action["standing"][0]
    assert abs(rep["stage_delta"]["early"]["x_range"]) < 0.2


def test_evaluate_pair_reports_all_stages(tmp_path: Path) -> None:
    root = tmp_path / "sess3"
    for i in range(1, 6):
        _write_repeat(root, "controlled_forward_fall", i,
                      x_range_pre=0.3, x_range_action=1.2)
        _write_repeat(root, "fast_sitting", i,
                      x_range_pre=0.3, x_range_action=0.5)
    by_action = load_repeat_features(root)
    result = evaluate_pair(by_action, "controlled_forward_fall", "fast_sitting")
    assert "error" not in result
    assert set(result["stages"]) == {"early", "middle", "late"}
    for stage in ("early", "middle", "late"):
        assert "x_range" in result["stages"][stage]
        e = result["stages"][stage]["x_range"]["delta"]
        assert e["auroc"] > 0.7


def test_confound_removal_detects_drop(tmp_path: Path) -> None:
    """构造一个绝对特征区分但 delta 无区分的 confound 场景。"""
    root = tmp_path / "sess4"
    # fall 和 neg 的 pre/action x_range 都高且相同 → absolute 不区分，delta 也不区分
    for i in range(1, 6):
        _write_repeat(root, "controlled_forward_fall", i,
                      x_range_pre=1.0, x_range_action=1.0)
        _write_repeat(root, "fast_sitting", i,
                      x_range_pre=1.0, x_range_action=1.0)
    by_action = load_repeat_features(root)
    conf = confound_removal_analysis(by_action)
    key = "controlled_forward_fall_vs_fast_sitting"
    assert key in conf
    # delta AUROC 应接近 0.5（无区分）
    e = conf[key]["x_range"]["early"]
    assert abs(e["delta_auroc"] - 0.5) < 0.4


def test_safe_auroc_and_pr() -> None:
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    assert _safe_auroc(y, s) == pytest.approx(1.0)
    assert _safe_pr_auc(y, s) > 0.95


def test_cohen_d_sign() -> None:
    a = np.array([1.0, 1.1, 1.2])
    b = np.array([0.1, 0.2, 0.3])
    assert _cohen_d(a, b) > 0
    assert _cohen_d(b, a) < 0
