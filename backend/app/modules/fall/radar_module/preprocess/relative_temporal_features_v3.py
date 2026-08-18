from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from radar_module.contracts import RadarFrame, Room, SourceMode
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
    TemporalFeatureWindowV2,
    WINDOW_SIZE_V2,
)


FEATURE_VERSION_V3 = "radar_features_v3_relative_geometry"
BASELINE_FRAME_COUNT_V3 = 5
FEATURE_NAMES_V3 = (
    "centroid_z_relative",
    "z_p10_relative",
    "z_p50_relative",
    "z_p90_relative",
    "height_range_relative",
    "log_point_count_relative",
    "mean_velocity",
    "max_abs_velocity",
    "velocity_std",
    "moving_range_centroid_relative",
    "moving_range_width_relative",
    "centroid_z_delta_0_3s",
    "centroid_z_delta_0_6s",
    "vertical_velocity",
    "vertical_acceleration",
    "height_range_delta_0_3s",
    "observed_frame_mask",
    "point_present_mask",
    "interpolated_mask",
)
FEATURE_UNITS_V3 = (
    "m",
    "m",
    "m",
    "m",
    "m",
    "log_count",
    "m/s",
    "m/s",
    "m/s",
    "m",
    "m",
    "m",
    "m",
    "m/s",
    "m/s^2",
    "m",
    "binary",
    "binary",
    "binary",
)

_RELATIVE_FEATURE_INDICES = (0, 1, 2, 3, 4, 9, 10)
_POINT_COUNT_INDEX = 5
_OBSERVED_MASK_INDEX = 16
_POINT_PRESENT_MASK_INDEX = 17
_INTERPOLATED_MASK_INDEX = 18


@dataclass(frozen=True, slots=True)
class RelativeTemporalFeatureWindowV3:
    end_timestamp: datetime
    device_id: str
    room: Room
    source_mode: SourceMode
    version: str
    names: tuple[str, ...]
    values: NDArray[np.float32]
    observed_mask: NDArray[np.bool_]
    point_present_mask: NDArray[np.bool_]
    interpolated_mask: NDArray[np.bool_]
    data_quality: TemporalDataQuality
    missing_frame_ratio: float
    longest_unresolved_gap_seconds: float
    baseline_frame_count: int = BASELINE_FRAME_COUNT_V3

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32).copy()
        if self.version != FEATURE_VERSION_V3:
            raise ValueError(f"unsupported feature version: {self.version}")
        if self.names != FEATURE_NAMES_V3:
            raise ValueError("v3 feature names/order are incompatible")
        if values.shape != (WINDOW_SIZE_V2, len(FEATURE_NAMES_V3)):
            raise ValueError("v3 feature values have an incompatible shape")
        if not np.isfinite(values).all():
            raise ValueError("v3 feature values must be finite")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


class RadarRelativeTemporalFeatureExtractorV3:
    """Make temporal features insensitive to a constant sensor pose offset.

    The first 0.5 seconds of each two-second window define its geometric
    baseline. Absolute height, range, spread and point-density levels are
    replaced with deviations from that baseline. Doppler and already-relative
    dynamics remain unchanged. This is an experimental feature contract and is
    deliberately incompatible with v2 checkpoints.
    """

    feature_version = FEATURE_VERSION_V3
    feature_names = FEATURE_NAMES_V3
    feature_units = FEATURE_UNITS_V3
    window_size = WINDOW_SIZE_V2

    def __init__(
        self,
        *,
        baseline_frame_count: int = BASELINE_FRAME_COUNT_V3,
        base_extractor: RadarTemporalFeatureExtractorV2 | None = None,
    ) -> None:
        if not 2 <= baseline_frame_count < WINDOW_SIZE_V2:
            raise ValueError("baseline_frame_count must be in [2, window_size)")
        self.baseline_frame_count = int(baseline_frame_count)
        self.base_extractor = base_extractor or RadarTemporalFeatureExtractorV2()

    def transform(
        self,
        frames: Sequence[RadarFrame],
        *,
        end_timestamp: datetime | None = None,
        target_track_id: int | None = None,
        min_snr: float | None = None,
    ) -> RelativeTemporalFeatureWindowV3:
        v2 = self.base_extractor.transform(
            frames,
            end_timestamp=end_timestamp,
            target_track_id=target_track_id,
            min_snr=min_snr,
        )
        return self.transform_v2_window(v2)

    def transform_v2_window(
        self, window: TemporalFeatureWindowV2
    ) -> RelativeTemporalFeatureWindowV3:
        values = transform_v2_values_to_v3(
            window.values,
            baseline_frame_count=self.baseline_frame_count,
        )
        return RelativeTemporalFeatureWindowV3(
            end_timestamp=window.end_timestamp,
            device_id=window.device_id,
            room=window.room,
            source_mode=window.source_mode,
            version=FEATURE_VERSION_V3,
            names=FEATURE_NAMES_V3,
            values=values,
            observed_mask=window.observed_mask,
            point_present_mask=window.point_present_mask,
            interpolated_mask=window.interpolated_mask,
            data_quality=window.data_quality,
            missing_frame_ratio=window.missing_frame_ratio,
            longest_unresolved_gap_seconds=window.longest_unresolved_gap_seconds,
            baseline_frame_count=self.baseline_frame_count,
        )


def transform_v2_values_to_v3(
    values: NDArray[np.floating],
    *,
    baseline_frame_count: int = BASELINE_FRAME_COUNT_V3,
) -> NDArray[np.float32]:
    """Convert one or more v2 windows without changing labels or splits."""

    raw = np.asarray(values, dtype=np.float32)
    squeeze = raw.ndim == 2
    if squeeze:
        raw = raw[None, ...]
    expected = (WINDOW_SIZE_V2, len(FEATURE_NAMES_V2))
    if raw.ndim != 3 or raw.shape[1:] != expected:
        raise ValueError(f"v2 values must end with shape {expected}")
    if not 2 <= baseline_frame_count < WINDOW_SIZE_V2:
        raise ValueError("baseline_frame_count must be in [2, window_size)")
    if not np.isfinite(raw).all():
        raise ValueError("v2 values must be finite")

    result = raw.copy()
    point_present = raw[..., _POINT_PRESENT_MASK_INDEX] > 0.5
    interpolated = raw[..., _INTERPOLATED_MASK_INDEX] > 0.5
    usable = point_present | interpolated
    baseline_valid = point_present[:, :baseline_frame_count]

    for index in _RELATIVE_FEATURE_INDICES:
        baseline = _masked_median(
            raw[:, :baseline_frame_count, index], baseline_valid
        )
        result[..., index] = raw[..., index] - baseline[:, None]

    logged_count = np.log1p(np.maximum(raw[..., _POINT_COUNT_INDEX], 0.0))
    count_baseline = _masked_median(
        logged_count[:, :baseline_frame_count], baseline_valid
    )
    result[..., _POINT_COUNT_INDEX] = logged_count - count_baseline[:, None]

    invalid = ~usable
    for index in (*_RELATIVE_FEATURE_INDICES, _POINT_COUNT_INDEX):
        result[..., index] = np.where(invalid, 0.0, result[..., index])
    if not np.isfinite(result).all():
        raise ValueError("v3 transformation produced non-finite values")
    converted = result.astype(np.float32, copy=False)
    return converted[0] if squeeze else converted


def _masked_median(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, values, np.nan)
    with np.errstate(all="ignore"):
        baseline = np.nanmedian(masked, axis=1)
    if np.isnan(baseline).any():
        raise ValueError("each window requires a valid point frame in its baseline")
    return baseline.astype(np.float32, copy=False)
