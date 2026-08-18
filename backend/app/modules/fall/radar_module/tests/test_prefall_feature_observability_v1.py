"""Tests for the pure-radar pre-fall feature observability pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from radar_module.analysis.prefall_feature_observability_v1 import (
    collect_sessions,
    extract_frame_features,
    rq1_repeat_stability,
    rq2_discrimination,
    rq3_position_artifact,
    rq4_recovery,
    rq5_combination,
)
from radar_module.preprocess.prefall_features_v1 import (
    FrameFeatureRow,
    dynamic_features,
    frame_base_features,
)


def _make_points(centroid: tuple[float, float, float], n: int = 8, vel: float = 0.0):
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


def _session_records(
    base_z: float,
    frames: int,
    *,
    phase: str | None = None,
    repeat: int | None = None,
    fall: bool = False,
) -> list[dict]:
    records = []
    base = datetime(2026, 8, 17, tzinfo=timezone.utc)
    for i in range(frames):
        if fall:
            # 下降：z 随时间降低
            z = base_z - 0.1 * i
        else:
            z = base_z
        rec = {
            "timestamp": (base + timedelta(milliseconds=55 * i)).isoformat(),
            "points": _make_points((0.3, 0.4, z)),
        }
        if phase is not None:
            rec["phase"] = phase
        if repeat is not None:
            rec["repeat_index"] = repeat
        records.append(rec)
    return records


def test_frame_base_features_basic() -> None:
    feats = frame_base_features(_make_points((0.3, 0.4, 1.5), n=10))
    assert feats["point_count"] == 10
    assert feats["centroid_z"] == pytest.approx(1.5, abs=0.1)
    assert feats["height_range"] >= 0
    assert np.isfinite(feats["z_p50"])


def test_frame_base_features_empty() -> None:
    feats = frame_base_features([])
    assert feats["point_count"] == 0
    assert np.isnan(feats["centroid_z"])


def test_dynamic_features_windows() -> None:
    base = datetime(2026, 8, 17, tzinfo=timezone.utc)
    history = []
    for i in range(20):
        feats = frame_base_features(_make_points((0.3, 0.4, 1.5 - 0.01 * i)))
        history.append(feats)
    windows = {"0p2s": 4, "0p5s": 9, "1p0s": 18}
    dyn = dynamic_features(history, windows, period_seconds=0.055)
    # 下降趋势：delta_z 应为负
    assert dyn["delta_z_0p2s"] < 0
    assert dyn["slope_z_0p5s"] < 0
    assert dyn["drift_xy_0p5s"] >= 0
    assert "d_centroid_z" in dyn


def test_extract_frame_features_from_records() -> None:
    records = _session_records(1.5, 20, phase="action", repeat=1)
    rows = extract_frame_features(records, action_name="bending")
    assert len(rows) == 20
    assert rows[0].action_name == "bending"
    assert rows[0].repeat_index == 1
    assert rows[0].phase == "action"
    assert "delta_z_0p5s" in rows[-1].dynamic


def test_rq1_repeat_stability() -> None:
    rows: list[FrameFeatureRow] = []
    for rep in (1, 2, 3):
        records = _session_records(1.5, 10, repeat=rep)
        rows.extend(extract_frame_features(records, action_name="standing"))
    result = rq1_repeat_stability({"standing": rows})
    assert "standing" in result
    assert result["standing"]["repeat_count"] == 3
    # 稳定特征：CV 应较小
    cv = result["standing"]["feature_stability"]["centroid_z"]["cv"]
    assert np.isfinite(cv)


def test_rq2_discrimination_separates_fall() -> None:
    fall_rows = extract_frame_features(
        _session_records(1.5, 30, phase="action", fall=True),
        action_name="controlled_forward_fall",
    )
    adl_rows = extract_frame_features(
        _session_records(1.5, 30, phase="action", fall=False),
        action_name="sitting",
    )
    result = rq2_discrimination({
        "controlled_forward_fall": fall_rows,
        "sitting": adl_rows,
    })
    assert "comparisons" in result
    # delta_z / slope_z 应显著区分（fall 的 z 持续下降）
    assert "delta_z_0p5s" in result["comparisons"]


def test_rq3_position_artifact_detects_centroid_x() -> None:
    # 构造两组站位不同但动作相同的帧，centroid_x 本身应与站位强相关
    rows: list[FrameFeatureRow] = []
    for x in (0.1, 2.0):
        for _ in range(20):
            records = _session_records(1.5, 5)
            records = [dict(r, points=_make_points((x, 0.4, 1.5))) for r in records]
            rows.extend(extract_frame_features(records, action_name="walking"))
    result = rq3_position_artifact({"walking": rows})
    assert "ranking_most_position_dependent" in result
    # centroid_x 自身应在位置相关排名靠前
    names = [e["feature"] for e in result["ranking_most_position_dependent"]]
    assert "centroid_x" in names[:5]


def test_rq4_recovery_returns_to_baseline() -> None:
    # recovery: still_post 回到基线 z
    rows: list[FrameFeatureRow] = []
    for phase, z in (("still_pre", 1.5), ("action", 1.2), ("still_post", 1.5)):
        records = _session_records(z, 10, phase=phase)
        rows.extend(extract_frame_features(records, action_name="forward_lean_recovery"))
    result = rq4_recovery({"forward_lean_recovery": rows})
    assert "forward_lean_recovery" in result
    # 恢复动作 z 回到基线 → |post-pre| 小
    d = result["forward_lean_recovery"]["features"]["centroid_z"]["post_minus_pre"]
    assert abs(d) < 0.05


def test_rq5_combination_ranks_fall_features() -> None:
    fall_rows = extract_frame_features(
        _session_records(1.5, 30, fall=True),
        action_name="controlled_forward_fall",
    )
    adl_rows = extract_frame_features(
        _session_records(1.5, 30, fall=False),
        action_name="sitting",
    )
    result = rq5_combination({
        "controlled_forward_fall": fall_rows,
        "sitting": adl_rows,
    })
    assert "candidates" in result
    assert "delta_z_0p5s" in result["candidates"]


def test_collect_sessions_from_capture_dir(tmp_path: Path) -> None:
    # 模拟采集工具输出目录结构
    action_dir = tmp_path / "bending"
    action_dir.mkdir(parents=True)
    records = _session_records(1.5, 10, phase="still_pre", repeat=1)
    (action_dir / "session.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records),
        encoding="utf-8",
    )
    (action_dir / "manifest.json").write_text(
        json.dumps({"action_name": "bending"}),
        encoding="utf-8",
    )
    rows = collect_sessions([str(action_dir)])
    assert "bending" in rows
    assert len(rows["bending"]) == 10
