from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from radar_module.contracts import RadarFrame, RadarPoint, Room, SourceMode


FEATURE_VERSION_V2 = "radar_features_v2"
TARGET_SAMPLE_RATE_HZ = 10.0
HISTORY_SECONDS = 2.0
WINDOW_SIZE_V2 = 20

FEATURE_NAMES_V2 = (
    "centroid_z",
    "z_p10",
    "z_p50",
    "z_p90",
    "height_range",
    "point_count",
    "mean_velocity",
    "max_abs_velocity",
    "velocity_std",
    "moving_range_centroid",
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

FEATURE_UNITS_V2 = (
    "m",
    "m",
    "m",
    "m",
    "m",
    "count",
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

_BASE_FEATURE_COUNT = 11


class TemporalDataQuality(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class TemporalFeatureWindowV2:
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
    target_track_id: int | None = None
    min_snr: float | None = None

    def __post_init__(self) -> None:
        if (
            self.end_timestamp.tzinfo is None
            or self.end_timestamp.utcoffset() is None
        ):
            raise ValueError("end_timestamp must include a timezone offset")
        if self.version != FEATURE_VERSION_V2:
            raise ValueError(f"unsupported feature version: {self.version}")
        if self.names != FEATURE_NAMES_V2:
            raise ValueError("v2 feature names/order are incompatible")
        values = np.asarray(self.values, dtype=np.float32).copy()
        expected_shape = (WINDOW_SIZE_V2, len(FEATURE_NAMES_V2))
        if values.shape != expected_shape:
            raise ValueError(
                f"values must have shape {expected_shape}, got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("v2 feature values must be finite")
        object.__setattr__(self, "values", _readonly(values))
        for attribute_name in (
            "observed_mask",
            "point_present_mask",
            "interpolated_mask",
        ):
            mask = np.asarray(getattr(self, attribute_name), dtype=np.bool_).copy()
            if mask.shape != (WINDOW_SIZE_V2,):
                raise ValueError(
                    f"{attribute_name} must have shape ({WINDOW_SIZE_V2},)"
                )
            object.__setattr__(self, attribute_name, _readonly(mask))
        if not 0.0 <= self.missing_frame_ratio <= 1.0:
            raise ValueError("missing_frame_ratio must be between zero and one")
        if self.longest_unresolved_gap_seconds < 0:
            raise ValueError("longest gap must not be negative")
        if self.target_track_id is not None and not (
            0 <= self.target_track_id < 255
        ):
            raise ValueError("target_track_id must be in [0, 254]")
        if self.min_snr is not None and not np.isfinite(self.min_snr):
            raise ValueError("min_snr must be finite or None")


class RadarTemporalFeatureExtractorV2:
    """Build a 2-second, 10 Hz feature window from timestamped point clouds.

    The extractor uses only already-decoded ``RadarFrame`` objects. It does not
    parse UART/TLV packets and does not infer identity. Short gaps are linearly
    interpolated while masks keep the interpolation visible to the model.
    """

    feature_version = FEATURE_VERSION_V2
    feature_names = FEATURE_NAMES_V2
    feature_units = FEATURE_UNITS_V2
    target_sample_rate_hz = TARGET_SAMPLE_RATE_HZ
    history_seconds = HISTORY_SECONDS
    window_size = WINDOW_SIZE_V2

    def __init__(
        self,
        *,
        alignment_tolerance_seconds: float = 0.06,
        max_interpolation_gap_seconds: float = 0.25,
        max_missing_frame_ratio: float = 0.20,
        moving_velocity_threshold_mps: float = 0.10,
    ) -> None:
        if alignment_tolerance_seconds <= 0:
            raise ValueError("alignment tolerance must be positive")
        if max_interpolation_gap_seconds < 0:
            raise ValueError("maximum interpolation gap must not be negative")
        if not 0 <= max_missing_frame_ratio <= 1:
            raise ValueError("maximum missing frame ratio must be in [0, 1]")
        if moving_velocity_threshold_mps < 0:
            raise ValueError("moving velocity threshold must not be negative")
        self.alignment_tolerance_seconds = alignment_tolerance_seconds
        self.max_interpolation_gap_seconds = max_interpolation_gap_seconds
        self.max_missing_frame_ratio = max_missing_frame_ratio
        self.moving_velocity_threshold_mps = moving_velocity_threshold_mps

    def transform(
        self,
        frames: Sequence[RadarFrame],
        *,
        end_timestamp: datetime | None = None,
        target_track_id: int | None = None,
        min_snr: float | None = None,
    ) -> TemporalFeatureWindowV2:
        if not frames:
            raise ValueError("frames must not be empty")
        ordered_frames = tuple(frames)
        self._validate_stream(ordered_frames)
        if end_timestamp is None:
            end_timestamp = ordered_frames[-1].timestamp
        if end_timestamp.tzinfo is None or end_timestamp.utcoffset() is None:
            raise ValueError("end_timestamp must include a timezone offset")
        if target_track_id is not None and not (0 <= target_track_id < 255):
            raise ValueError("target_track_id must be in [0, 254]")
        if min_snr is not None and not np.isfinite(min_snr):
            raise ValueError("min_snr must be finite or None")

        period_seconds = 1.0 / self.target_sample_rate_hz
        start_timestamp = end_timestamp - timedelta(
            seconds=(self.window_size - 1) * period_seconds
        )
        start_epoch = start_timestamp.timestamp()
        target_epochs = start_epoch + np.arange(
            self.window_size, dtype=np.float64
        ) * period_seconds

        aligned_frames: list[RadarFrame | None] = [None] * self.window_size
        alignment_errors = np.full(self.window_size, np.inf, dtype=np.float64)
        for frame in ordered_frames:
            relative_index = (
                frame.timestamp.timestamp() - start_epoch
            ) / period_seconds
            target_index = int(round(relative_index))
            if not 0 <= target_index < self.window_size:
                continue
            error = abs(
                frame.timestamp.timestamp() - target_epochs[target_index]
            )
            if (
                error <= self.alignment_tolerance_seconds
                and error < alignment_errors[target_index]
            ):
                aligned_frames[target_index] = frame
                alignment_errors[target_index] = error

        observed_mask = np.asarray(
            [frame is not None for frame in aligned_frames], dtype=np.bool_
        )
        selected_points = [
            self._select_points(
                frame,
                target_track_id=target_track_id,
                min_snr=min_snr,
            )
            if frame is not None
            else ()
            for frame in aligned_frames
        ]
        point_present_mask = np.asarray(
            [bool(points) for points in selected_points], dtype=np.bool_
        )
        base_values = np.full(
            (self.window_size, _BASE_FEATURE_COUNT),
            np.nan,
            dtype=np.float64,
        )
        for index, frame in enumerate(aligned_frames):
            if frame is None:
                continue
            base_values[index] = self._extract_frame_features(
                selected_points[index]
            )

        interpolated_values, interpolated_mask = self._interpolate_short_gaps(
            base_values, period_seconds
        )
        dynamics = _calculate_dynamics(interpolated_values, period_seconds)
        unresolved_mask = ~np.isfinite(interpolated_values[:, 0])
        longest_gap_seconds = _longest_run(unresolved_mask) * period_seconds
        missing_frame_ratio = float(1.0 - observed_mask.mean())

        if (
            longest_gap_seconds > self.max_interpolation_gap_seconds
            or missing_frame_ratio > self.max_missing_frame_ratio
        ):
            quality = TemporalDataQuality.INSUFFICIENT_DATA
        elif (
            not observed_mask.all()
            or not point_present_mask.all()
            or interpolated_mask.any()
        ):
            quality = TemporalDataQuality.DEGRADED
        else:
            quality = TemporalDataQuality.GOOD

        masks = np.column_stack(
            (
                observed_mask.astype(np.float64),
                point_present_mask.astype(np.float64),
                interpolated_mask.astype(np.float64),
            )
        )
        values = np.nan_to_num(
            np.column_stack((interpolated_values, dynamics, masks)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        first = ordered_frames[0]
        return TemporalFeatureWindowV2(
            end_timestamp=end_timestamp,
            device_id=first.device_id,
            room=first.room,
            source_mode=first.source_mode,
            version=self.feature_version,
            names=self.feature_names,
            values=values,
            observed_mask=observed_mask,
            point_present_mask=point_present_mask,
            interpolated_mask=interpolated_mask,
            data_quality=quality,
            missing_frame_ratio=missing_frame_ratio,
            longest_unresolved_gap_seconds=longest_gap_seconds,
            target_track_id=target_track_id,
            min_snr=min_snr,
        )

    def feature_spec(self) -> dict[str, object]:
        return {
            "feature_version": self.feature_version,
            "feature_names": list(self.feature_names),
            "feature_units": list(self.feature_units),
            "input_size": len(self.feature_names),
            "window_size": self.window_size,
            "history_seconds": self.history_seconds,
            "target_sample_rate_hz": self.target_sample_rate_hz,
            "alignment_tolerance_seconds": self.alignment_tolerance_seconds,
            "max_interpolation_gap_seconds": (
                self.max_interpolation_gap_seconds
            ),
            "max_missing_frame_ratio": self.max_missing_frame_ratio,
        }

    def _validate_stream(self, frames: tuple[RadarFrame, ...]) -> None:
        first = frames[0]
        stream_key = (first.device_id, first.room, first.source_mode)
        previous_timestamp = first.timestamp
        for frame in frames:
            if not isinstance(frame, RadarFrame):
                raise TypeError("v2 extractor only accepts RadarFrame objects")
            if (frame.device_id, frame.room, frame.source_mode) != stream_key:
                raise ValueError("all frames must belong to the same stream")
            if frame.timestamp < previous_timestamp:
                raise ValueError("frames must be ordered by timestamp")
            previous_timestamp = frame.timestamp

    def _extract_frame_features(
        self,
        points: tuple[RadarPoint, ...],
    ) -> NDArray[np.float64]:
        if not points:
            values = np.full(_BASE_FEATURE_COUNT, np.nan, dtype=np.float64)
            values[5] = 0.0
            return values
        coordinates = np.asarray(
            [(point.x, point.y, point.z) for point in points],
            dtype=np.float64,
        )
        velocities = np.asarray(
            [point.velocity for point in points], dtype=np.float64
        )
        z_values = coordinates[:, 2]
        ranges = np.linalg.norm(coordinates, axis=1)
        moving = np.abs(velocities) >= self.moving_velocity_threshold_mps
        motion_ranges = ranges[moving] if moving.any() else ranges
        return np.asarray(
            (
                float(z_values.mean()),
                float(np.quantile(z_values, 0.10)),
                float(np.quantile(z_values, 0.50)),
                float(np.quantile(z_values, 0.90)),
                float(z_values.max() - z_values.min()),
                float(len(points)),
                float(velocities.mean()),
                float(np.abs(velocities).max()),
                float(velocities.std()),
                float(motion_ranges.mean()),
                float(motion_ranges.max() - motion_ranges.min()),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _select_points(
        frame: RadarFrame,
        *,
        target_track_id: int | None,
        min_snr: float | None,
    ) -> tuple[RadarPoint, ...]:
        return tuple(
            point
            for point in frame.points
            if (
                target_track_id is None
                or point.track_id == target_track_id
            )
            and (
                min_snr is None
                or (point.snr is not None and point.snr >= min_snr)
            )
        )

    def _interpolate_short_gaps(
        self,
        values: NDArray[np.float64],
        period_seconds: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        result = values.copy()
        interpolated_mask = np.zeros(self.window_size, dtype=np.bool_)
        for column_index in range(result.shape[1]):
            column = result[:, column_index]
            valid_indices = np.flatnonzero(np.isfinite(column))
            for left, right in zip(valid_indices[:-1], valid_indices[1:]):
                missing_count = int(right - left - 1)
                if missing_count <= 0:
                    continue
                if (
                    missing_count * period_seconds
                    > self.max_interpolation_gap_seconds
                ):
                    continue
                step = (column[right] - column[left]) / (right - left)
                for index in range(left + 1, right):
                    column[index] = column[left] + step * (index - left)
                    interpolated_mask[index] = True
        return result, interpolated_mask


def _calculate_dynamics(
    base_values: NDArray[np.float64],
    period_seconds: float,
) -> NDArray[np.float64]:
    centroid_z = base_values[:, 0]
    height_range = base_values[:, 4]
    delta_03 = _lagged_delta(centroid_z, 3)
    delta_06 = _lagged_delta(centroid_z, 6)
    vertical_velocity = _lagged_delta(centroid_z, 1) / period_seconds
    vertical_acceleration = (
        _lagged_delta(vertical_velocity, 1) / period_seconds
    )
    height_delta_03 = _lagged_delta(height_range, 3)
    return np.column_stack(
        (
            delta_03,
            delta_06,
            vertical_velocity,
            vertical_acceleration,
            height_delta_03,
        )
    )


def _lagged_delta(
    values: NDArray[np.float64],
    lag: int,
) -> NDArray[np.float64]:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if lag >= len(values):
        return result
    current = values[lag:]
    previous = values[:-lag]
    valid = np.isfinite(current) & np.isfinite(previous)
    result[lag:][valid] = current[valid] - previous[valid]
    return result


def _longest_run(mask: NDArray[np.bool_]) -> int:
    longest = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _readonly(array: NDArray) -> NDArray:
    array.setflags(write=False)
    return array
