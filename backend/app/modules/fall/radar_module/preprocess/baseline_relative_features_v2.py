"""baseline-relative 雷达时序特征（v2，供状态演化 TCN 输入）。

背景
----
真机 pilot + DGUHA 审计确认：绝对 height_range / x_range 存在 posture /
站位 confound，应排除。本模块提取**相对基线（still_pre）的动态变化**
作为特征，用于因果 TCN 四状态模型。

特征（每帧）：
- horizontal drift magnitude (drift_xy_0p5s / 1p0s / 1frame)
- horizontal velocity / acceleration (d_centroid_x/y, 二阶)
- Doppler mean/std/max
- spatial spread
- relative height change（centroid_z / z_p90 相对基线 delta）
- point-count delta（相对基线）
- 0.2 / 0.5 / 1.0s delta / slope / variance
- first/second derivative
- missing/quality mask（空帧标记）

Baseline 定义：每个样本起始稳定段（前 N 帧）的中位数，或显式传入。

特征语义分组（按用户要求，避免 baseline≈0 爆炸）：
- DIFF_KEYS（doppler_mean）：差值 value - baseline，禁止除以近0 baseline
- SCALE_KEYS（doppler_std/max_abs/spatial_spread）：非负尺度，优先差值，
  baseline≈0 时用带 eps 的 log-ratio
- RAW_KEYS（drift/delta/slope/variance/point_count_delta 等动态量）：
  本身已是变化量，不二次 baseline 比值归一化

输出：per-frame 特征向量（用于 TCN 逐帧输入，TCN 自带时序建模）。

Version: radar_baseline_relative_features_v2
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np

from radar_module.preprocess.prefall_features_v1 import (
    default_window_frames,
    dynamic_features,
    frame_base_features,
)

EPS = 1e-6

# 输出特征名（baseline-relative 后）
FEATURE_NAMES = [
    "drift_xy_0p5s", "drift_xy_1p0s", "drift_xy_1frame",
    "d_centroid_x", "d_centroid_y", "d2_centroid_x", "d2_centroid_y",
    "doppler_mean", "doppler_std", "doppler_max_abs",
    "spatial_spread",
    "delta_z_0p2s", "delta_z_0p5s", "delta_z_1p0s",
    "slope_z_0p5s", "slope_z_1p0s",
    "delta_z_p90_0p5s",
    "point_count_delta",
    "var_z_0p5s", "var_doppler_0p5s",
    "moving_fraction",
]

# 特征语义分组：
# - DIFF_KEYS: 差值 (value - baseline)，禁止除以近0 baseline（centroid/height/
#   drift/velocity/accel/doppler_mean 等绝对水平量或符号量）
# - RAW_KEYS: 动态量，本身已是 delta/slope/variance，不再二次 baseline 比值
#   归一化（0.2/0.5/1.0s 动态）
# - SCALE_KEYS: 非负尺度特征，优先差值，必要时带 eps 的 log-ratio
# - COUNT_REL: point_count 相对变化，分母设合理下限
DIFF_KEYS = {
    "doppler_mean",
}
SCALE_KEYS = {
    "doppler_std", "doppler_max_abs", "spatial_spread",
}
COUNT_REL_KEYS = set()  # 用 point_count_delta（RAW）
RAW_KEYS = {
    "drift_xy_0p5s", "drift_xy_1p0s", "drift_xy_1frame",
    "d_centroid_x", "d_centroid_y", "d2_centroid_x", "d2_centroid_y",
    "delta_z_0p2s", "delta_z_0p5s", "delta_z_1p0s",
    "slope_z_0p5s", "slope_z_1p0s",
    "delta_z_p90_0p5s",
    "point_count_delta",
    "var_z_0p5s", "var_doppler_0p5s",
    "moving_fraction",
}

# 用于 baseline 差值的绝对水平量（取自 base features）
BASELINE_DIFF_FROM_BASE = {
    "centroid_z", "z_p90", "point_count",
}
# 非负尺度 log-ratio 的 eps 下限
SCALE_EPS = 1e-3
# point_count 相对变化的分母下限
COUNT_MIN_DENOM = 5.0


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _to_dict_points(points: Sequence[Any]) -> list[dict[str, Any]]:
    """把 RadarPoint / dict 统一转成 dict（frame_base_features 需 dict）。"""
    out = []
    for p in points:
        if isinstance(p, Mapping):
            out.append(dict(p))
        elif hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
            out.append({
                "x": p.x, "y": p.y, "z": p.z,
                "velocity": getattr(p, "velocity", 0.0),
                "snr": getattr(p, "snr", None),
                "track_id": getattr(p, "track_id", None),
            })
        else:
            continue
    return out


def frame_features_from_points(
    points: Sequence[Mapping[str, Any]],
    history: list[dict[str, float]],
    windows: dict[str, int],
    period: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """单帧点云 → (base, dyn)。"""
    pts = [dict(p) for p in points]
    base = frame_base_features(pts)
    history.append(base)
    dyn = dynamic_features(history, windows, period_seconds=period)
    return base, dyn


def baseline_from_frames(
    frames: Sequence[Mapping[str, Any]],
    *,
    window_size: int = 15,
) -> dict[str, float]:
    """从样本起始稳定段计算 baseline（仅对需要差值的特征）。

    只对 DIFF_KEYS + SCALE_KEYS（doppler_mean/std/max_abs/spatial_spread）
    计算 baseline。动态量（RAW_KEYS）不需要 baseline。
    """
    if len(frames) < 3:
        return {}
    frames = list(frames)[:window_size]
    need_keys = DIFF_KEYS | SCALE_KEYS
    feats: dict[str, list[float]] = {k: [] for k in need_keys}
    for frame in frames:
        pts = frame.get("points", ()) or frame.get("points_sensor", ())
        base = frame_base_features(_to_dict_points(pts))
        for name in need_keys:
            v = base.get(name, float("nan"))
            if np.isfinite(v):
                feats[name].append(float(v))
    return {name: float(np.median(vals)) if vals else float("nan")
            for name, vals in feats.items()}


def extract_sequence_features(
    frames: Sequence[Mapping[str, Any]],
    *,
    baseline: dict[str, float] | None = None,
    baseline_window: int = 15,
    sample_rate_hz: float = 20.0,
) -> tuple[np.ndarray, list[str]]:
    """从雷达帧序列提取 per-frame baseline-relative 特征。

    返回 (features[T, F], feature_names)。
    sample_rate_hz: 数据帧率（DGUHA=20Hz，IWR6843-Fall102=10Hz）。
    """
    if not frames:
        raise ValueError("empty frames")
    period = 1.0 / sample_rate_hz
    windows = default_window_frames(period)
    history: list[dict[str, float]] = []
    z_p90_history: list[float] = []
    pc_history: list[float] = []

    if baseline is None:
        baseline = baseline_from_frames(frames, window_size=baseline_window)

    feature_rows: list[np.ndarray] = []
    for frame in frames:
        pts = frame.get("points", ()) or frame.get("points_sensor", ())
        base = frame_base_features(_to_dict_points(pts))
        history.append(base)
        dyn = dynamic_features(history, windows, period_seconds=period)
        row: dict[str, float] = {}
        for name in FEATURE_NAMES:
            v = base.get(name, dyn.get(name, float("nan")))
            row[name] = float(v) if np.isfinite(v) else float("nan")
        # 自定义动态特征：delta_z_p90_0p5s、point_count_delta
        z_p90 = base.get("z_p90", float("nan"))
        pc = base.get("point_count", float("nan"))
        z_p90_history.append(z_p90)
        pc_history.append(pc)
        if len(z_p90_history) >= 10:  # 0.5s @20Hz
            row["delta_z_p90_0p5s"] = z_p90_history[-1] - z_p90_history[-10]
        else:
            row["delta_z_p90_0p5s"] = float("nan")
        if len(pc_history) >= 10:
            row["point_count_delta"] = pc_history[-1] - pc_history[-10]
        else:
            row["point_count_delta"] = float("nan")
        # 按特征语义构造
        rel_row = {}
        for name in FEATURE_NAMES:
            val = row[name]
            if name in DIFF_KEYS:
                # 差值，禁止除以近0 baseline
                bv = baseline.get(name, float("nan"))
                rel_row[name] = val - bv if np.isfinite(val) and np.isfinite(bv) else float("nan")
            elif name in SCALE_KEYS:
                # 非负尺度：优先差值；baseline≈0 时用 log-ratio
                bv = baseline.get(name, float("nan"))
                if np.isfinite(val) and np.isfinite(bv):
                    if abs(bv) > SCALE_EPS:
                        rel_row[name] = val - bv
                    else:
                        rel_row[name] = np.log((abs(val) + SCALE_EPS) / (abs(bv) + SCALE_EPS))
                else:
                    rel_row[name] = float("nan")
            else:
                # RAW_KEYS：动态量保持原值（不二次归一化）
                rel_row[name] = val
        feature_rows.append(np.asarray([rel_row[n] for n in FEATURE_NAMES],
                                       dtype=np.float64))
    return np.stack(feature_rows), FEATURE_NAMES


def default_period_hz() -> float:
    return 20.0
