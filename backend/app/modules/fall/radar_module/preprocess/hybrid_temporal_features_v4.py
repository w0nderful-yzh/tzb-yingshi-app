from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    WINDOW_SIZE_V2,
)


FEATURE_VERSION_V4 = "radar_features_v4_hybrid_geometry"
FEATURE_NAMES_V4 = (
    "centroid_z_relative",
    "z_p10_centered",
    "z_p50_centered",
    "z_p90_centered",
    "height_range",
    "log_point_count",
    "mean_velocity",
    "max_abs_velocity",
    "velocity_std",
    "moving_range_centroid_relative",
    "moving_range_width",
    "centroid_z_delta_0_3s",
    "centroid_z_delta_0_6s",
    "vertical_velocity",
    "vertical_acceleration",
    "height_range_delta_0_3s",
    "observed_frame_mask",
    "point_present_mask",
    "interpolated_mask",
)
BASELINE_FRAME_COUNT_V4 = 5


def transform_v2_values_to_v4(
    values: NDArray[np.floating],
    *,
    baseline_frame_count: int = BASELINE_FRAME_COUNT_V4,
) -> NDArray[np.float32]:
    """Remove sensor-origin translations while retaining body-shape state."""

    raw = np.asarray(values, dtype=np.float32)
    squeeze = raw.ndim == 2
    if squeeze:
        raw = raw[None, ...]
    expected = (WINDOW_SIZE_V2, len(FEATURE_NAMES_V2))
    if raw.ndim != 3 or raw.shape[1:] != expected:
        raise ValueError(f"v2 values must end with shape {expected}")
    if not np.isfinite(raw).all():
        raise ValueError("v2 values must be finite")
    if not 2 <= baseline_frame_count < WINDOW_SIZE_V2:
        raise ValueError("baseline_frame_count must be in [2, window_size)")

    result = raw.copy()
    point_present = raw[..., 17] > 0.5
    interpolated = raw[..., 18] > 0.5
    baseline_mask = point_present[:, :baseline_frame_count]
    centroid_baseline = _masked_median(
        raw[:, :baseline_frame_count, 0], baseline_mask
    )
    range_baseline = _masked_median(
        raw[:, :baseline_frame_count, 9], baseline_mask
    )

    result[..., 0] = raw[..., 0] - centroid_baseline[:, None]
    result[..., 1] = raw[..., 1] - raw[..., 0]
    result[..., 2] = raw[..., 2] - raw[..., 0]
    result[..., 3] = raw[..., 3] - raw[..., 0]
    result[..., 5] = np.log1p(np.maximum(raw[..., 5], 0.0))
    result[..., 9] = raw[..., 9] - range_baseline[:, None]

    invalid = ~(point_present | interpolated)
    for index in (0, 1, 2, 3, 5, 9):
        result[..., index] = np.where(invalid, 0.0, result[..., index])
    if not np.isfinite(result).all():
        raise ValueError("v4 transformation produced non-finite values")
    converted = result.astype(np.float32, copy=False)
    return converted[0] if squeeze else converted


def _masked_median(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    baseline = np.nanmedian(np.where(mask, values, np.nan), axis=1)
    if np.isnan(baseline).any():
        raise ValueError("each window requires a valid point frame in its baseline")
    return baseline.astype(np.float32, copy=False)
