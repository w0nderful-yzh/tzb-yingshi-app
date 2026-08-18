"""纯雷达 pre-fall 特征观测性：帧级基础特征 + 时间窗动态特征。

设计目标
--------
在真机 IWR6843ISK 密点云上，回答"哪些 pre-fall / instability 特征稳定可观测"。
本模块只做特征计算，不训练模型、不修改 checkpoint。可消费：
- 采集工具输出 session.jsonl（含 repeat_index / phase 标注）
- 现有真机 phase session（仅 action 段）

基础特征（单帧，点云 → 标量）：
- centroid_x/y/z: 质心坐标
- z_p10/z_p50/z_p90: z 分位数
- height_range: z 垂直跨度 = z_max - z_min
- x_range / y_range: 水平跨度
- depth_range: y 深度跨度
- range_mean/range_std/range_max: 距离统计
- spatial_spread: 点到质心距离的 std（3D 散布）
- doppler_mean/doppler_std/doppler_max_abs: 多普勒统计
- moving_fraction: |v|>=threshold 的点占比
- upper_fraction: z>=z_p50 的点占比（身体上部比例）

动态特征（含历史窗口，w ∈ {0.2, 0.5, 1.0}s）：
- drift_xy_w: 质心水平位移（|Δcentroid_xy|）
- delta_z_w / delta_height_w / delta_doppler_w: 质心高/高度跨度/多普勒变化
- slope_z_w / slope_height_w: 线性拟合斜率（正=上升，负=下降）
- var_z_w / var_height_w / var_doppler_w: 窗口内方差
- d_centroid_z: 一阶导数（垂直速度）
- d2_centroid_z: 二阶导数（垂直加速度）

Version: radar_prefall_features_v1
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Mapping, Sequence

import numpy as np


def record_points(record: Mapping[str, Any]) -> Sequence[Any]:
    """兼容新旧采集格式：旧=points，新=points_sensor。

    新采集 JSONL 同时含 points_sensor（原始 sensor-frame）和 points_world。
    分析默认用 sensor-frame（points_sensor），与旧数据（points）语义一致。
    """
    pts = record.get("points")
    if pts is None:
        pts = record.get("points_sensor", ())
    return pts


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def _safe_std(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values))


def frame_base_features(points: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """单帧点云 → 基础特征字典。空帧返回全部 NaN（point_count=0）。"""
    if not points:
        return {
            "point_count": 0.0,
            "centroid_x": float("nan"),
            "centroid_y": float("nan"),
            "centroid_z": float("nan"),
            "z_p10": float("nan"),
            "z_p50": float("nan"),
            "z_p90": float("nan"),
            "height_range": float("nan"),
            "x_range": float("nan"),
            "y_range": float("nan"),
            "depth_range": float("nan"),
            "range_mean": float("nan"),
            "range_std": float("nan"),
            "range_max": float("nan"),
            "spatial_spread": float("nan"),
            "doppler_mean": float("nan"),
            "doppler_std": float("nan"),
            "doppler_max_abs": float("nan"),
            "moving_fraction": 0.0,
            "upper_fraction": float("nan"),
        }

    coords = np.asarray(
        [[p.get("x", float("nan")), p.get("y", float("nan")), p.get("z", float("nan"))]
         for p in points],
        dtype=np.float64,
    )
    coords = coords[np.isfinite(coords).all(axis=1)]
    if coords.shape[0] == 0:
        return frame_base_features([])

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    velocities = np.asarray(
        [p.get("velocity", float("nan")) for p in points], dtype=np.float64
    )
    velocities = velocities[np.isfinite(velocities)]

    ranges = np.linalg.norm(coords, axis=1)
    centroid = coords.mean(axis=0)
    spread = np.linalg.norm(coords - centroid, axis=1)

    moving_threshold = 0.5  # m/s
    moving = np.abs(velocities) >= moving_threshold if velocities.size else np.array([])

    features: dict[str, float] = {
        "point_count": float(coords.shape[0]),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "z_p10": _quantile(z, 0.10),
        "z_p50": _quantile(z, 0.50),
        "z_p90": _quantile(z, 0.90),
        "height_range": float(z.max() - z.min()),
        "x_range": float(x.max() - x.min()),
        "y_range": float(y.max() - y.min()),
        "depth_range": float(y.max() - y.min()),
        "range_mean": float(ranges.mean()),
        "range_std": _safe_std(ranges),
        "range_max": float(ranges.max()),
        "spatial_spread": float(spread.mean()) if spread.size else float("nan"),
        "doppler_mean": float(velocities.mean()) if velocities.size else float("nan"),
        "doppler_std": _safe_std(velocities),
        "doppler_max_abs": float(np.abs(velocities).max()) if velocities.size else float("nan"),
        "moving_fraction": float(moving.mean()) if moving.size else 0.0,
        "upper_fraction": float((z >= np.median(z)).mean()) if z.size else float("nan"),
    }
    return features


DYNAMIC_BASE_KEYS = ("centroid_z", "centroid_x", "centroid_y", "height_range",
                     "doppler_mean")


def _linear_slope(values: np.ndarray) -> float:
    """对窗口内 values 做线性拟合，返回斜率（y 每帧变化）。"""
    n = values.size
    if n < 2:
        return float("nan")
    if not np.isfinite(values).any():
        return float("nan")
    x = np.arange(n, dtype=np.float64)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return float("nan")
    slope = np.polyfit(x[mask], values[mask], 1)[0]
    return float(slope)


def dynamic_features(
    base_features: Sequence[dict[str, float]],
    window_frames: Mapping[str, int],
    *,
    period_seconds: float,
) -> dict[str, float]:
    """对帧级基础特征序列计算动态特征。

    base_features: 有序帧特征列表（当前帧在末尾）。
    window_frames: {"0.2s": n_frames, ...} 窗口帧数映射。
    period_seconds: 帧间隔（默认 1/18.18 ≈ 0.055s）。
    """
    n = len(base_features)
    result: dict[str, float] = {}

    for base_key in DYNAMIC_BASE_KEYS:
        series = np.asarray(
            [f.get(base_key, float("nan")) for f in base_features], dtype=np.float64
        )
        if n >= 1:
            result[f"d_{base_key}"] = (
                (series[-1] - series[-2]) / period_seconds if n >= 2
                else float("nan")
            )
        if n >= 3:
            d1 = np.gradient(series, period_seconds)
            result[f"d2_{base_key}"] = float(
                np.gradient(d1, period_seconds)[-1]
            ) if np.isfinite(d1).sum() >= 2 else float("nan")

    # centroid 水平位移
    if n >= 2:
        cx = np.asarray([f.get("centroid_x", float("nan")) for f in base_features])
        cy = np.asarray([f.get("centroid_y", float("nan")) for f in base_features])
        last_dx = cx[-1] - cx[-2]
        last_dy = cy[-1] - cy[-2]
        result["drift_xy_1frame"] = float(math.hypot(last_dx, last_dy))
    else:
        result["drift_xy_1frame"] = float("nan")

    for label, window in window_frames.items():
        window = max(1, int(window))
        if n < 2:
            result[f"drift_xy_{label}"] = float("nan")
            result[f"delta_z_{label}"] = float("nan")
            result[f"delta_height_{label}"] = float("nan")
            result[f"delta_doppler_{label}"] = float("nan")
            result[f"slope_z_{label}"] = float("nan")
            result[f"slope_height_{label}"] = float("nan")
            result[f"var_z_{label}"] = float("nan")
            result[f"var_height_{label}"] = float("nan")
            result[f"var_doppler_{label}"] = float("nan")
            continue

        start = max(0, n - window)
        seg_z = np.asarray(
            [f.get("centroid_z", float("nan")) for f in base_features[start:n]],
            dtype=np.float64,
        )
        seg_h = np.asarray(
            [f.get("height_range", float("nan")) for f in base_features[start:n]],
            dtype=np.float64,
        )
        seg_d = np.asarray(
            [f.get("doppler_mean", float("nan")) for f in base_features[start:n]],
            dtype=np.float64,
        )
        seg_cx = np.asarray(
            [f.get("centroid_x", float("nan")) for f in base_features[start:n]],
            dtype=np.float64,
        )
        seg_cy = np.asarray(
            [f.get("centroid_y", float("nan")) for f in base_features[start:n]],
            dtype=np.float64,
        )

        result[f"drift_xy_{label}"] = float(
            math.hypot(seg_cx[-1] - seg_cx[0], seg_cy[-1] - seg_cy[0])
        )
        result[f"delta_z_{label}"] = float(seg_z[-1] - seg_z[0])
        result[f"delta_height_{label}"] = float(seg_h[-1] - seg_h[0])
        result[f"delta_doppler_{label}"] = float(seg_d[-1] - seg_d[0])
        result[f"slope_z_{label}"] = _linear_slope(seg_z)
        result[f"slope_height_{label}"] = _linear_slope(seg_h)
        result[f"var_z_{label}"] = float(np.nanvar(seg_z)) if np.isfinite(seg_z).any() else float("nan")
        result[f"var_height_{label}"] = float(np.nanvar(seg_h)) if np.isfinite(seg_h).any() else float("nan")
        result[f"var_doppler_{label}"] = float(np.nanvar(seg_d)) if np.isfinite(seg_d).any() else float("nan")

    return result


def default_window_frames(period_seconds: float) -> dict[str, int]:
    """按帧间隔把 0.2/0.5/1.0s 转成帧数。"""
    return {
        "0p2s": max(1, int(round(0.2 / period_seconds))),
        "0p5s": max(1, int(round(0.5 / period_seconds))),
        "1p0s": max(1, int(round(1.0 / period_seconds))),
    }


@dataclass(frozen=True, slots=True)
class FrameFeatureRow:
    timestamp: datetime
    action_name: str
    repeat_index: int | None
    phase: str | None
    base: dict[str, float]
    dynamic: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action_name": self.action_name,
            "repeat_index": self.repeat_index,
            "phase": self.phase,
            **self.base,
            **self.dynamic,
        }
