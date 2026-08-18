from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math

import numpy as np
from numpy.typing import NDArray

from radar_module.contracts import RadarFrame


POINT_FEATURE_NAMES = (
    "x_m",
    "y_m",
    "z_m",
    "radial_velocity_mps",
    "snr",
    "snr_present",
)
POINT_SEQUENCE_VERSION = "radar_point_sequence_v1"


@dataclass(frozen=True, slots=True)
class PointCloudSequence:
    """Fixed-shape point sequence used only by the research model path."""

    values: NDArray[np.float32]
    point_mask: NDArray[np.bool_]
    frame_mask: NDArray[np.bool_]
    version: str = POINT_SEQUENCE_VERSION
    feature_names: tuple[str, ...] = POINT_FEATURE_NAMES

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        point_mask = np.asarray(self.point_mask, dtype=np.bool_)
        frame_mask = np.asarray(self.frame_mask, dtype=np.bool_)
        if values.ndim != 3 or values.shape[-1] != len(POINT_FEATURE_NAMES):
            raise ValueError("values must have shape [time, point, feature]")
        if point_mask.shape != values.shape[:2]:
            raise ValueError("point_mask must match values time/point dimensions")
        if frame_mask.shape != values.shape[:1]:
            raise ValueError("frame_mask must match values time dimension")
        if not np.isfinite(values).all():
            raise ValueError("values must contain only finite values")
        if np.any(point_mask & ~frame_mask[:, None]):
            raise ValueError("a missing frame cannot contain valid points")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "point_mask", point_mask)
        object.__setattr__(self, "frame_mask", frame_mask)


class PointCloudSequenceBuilder:
    """Resample decoded frames without inventing points or behaviour scores.

    Point selection is based on a lexicographic ordering of the measured
    values, so changing the input tuple order cannot change the tensor.  SNR
    availability is explicit because several public datasets do not provide
    calibrated SNR.
    """

    def __init__(
        self,
        *,
        history_seconds: float = 2.0,
        sample_rate_hz: float = 10.0,
        max_points: int = 64,
        alignment_tolerance_seconds: float | None = None,
    ) -> None:
        if history_seconds <= 0 or sample_rate_hz <= 0 or max_points <= 0:
            raise ValueError("history_seconds, sample_rate_hz and max_points must be positive")
        raw_steps = history_seconds * sample_rate_hz
        if not math.isclose(raw_steps, round(raw_steps), abs_tol=1e-9):
            raise ValueError("history_seconds * sample_rate_hz must be an integer")
        self.history_seconds = float(history_seconds)
        self.sample_rate_hz = float(sample_rate_hz)
        self.max_points = int(max_points)
        self.time_steps = int(round(raw_steps))
        self.alignment_tolerance_seconds = (
            float(alignment_tolerance_seconds)
            if alignment_tolerance_seconds is not None
            else 0.75 / self.sample_rate_hz
        )
        if self.alignment_tolerance_seconds <= 0:
            raise ValueError("alignment_tolerance_seconds must be positive")

    def transform(
        self,
        frames: tuple[RadarFrame, ...] | list[RadarFrame],
        *,
        end_timestamp: datetime | None = None,
    ) -> PointCloudSequence:
        if not frames:
            raise ValueError("frames must not be empty")
        ordered = tuple(frames)
        if any(current.timestamp < previous.timestamp for previous, current in zip(ordered, ordered[1:])):
            raise ValueError("frame timestamps must be ordered")
        end = end_timestamp or ordered[-1].timestamp
        if end.tzinfo is None or end.utcoffset() is None:
            raise ValueError("end_timestamp must include a timezone offset")

        values = np.zeros(
            (self.time_steps, self.max_points, len(POINT_FEATURE_NAMES)),
            dtype=np.float32,
        )
        point_mask = np.zeros((self.time_steps, self.max_points), dtype=np.bool_)
        frame_mask = np.zeros(self.time_steps, dtype=np.bool_)
        timestamps = np.asarray(
            [(frame.timestamp - end).total_seconds() for frame in ordered],
            dtype=np.float64,
        )
        target_offsets = np.arange(-(self.time_steps - 1), 1, dtype=np.float64) / self.sample_rate_hz
        for target_index, offset in enumerate(target_offsets):
            source_index = int(np.argmin(np.abs(timestamps - offset)))
            if abs(float(timestamps[source_index] - offset)) > self.alignment_tolerance_seconds:
                continue
            frame_mask[target_index] = True
            frame_values = self._frame_values(ordered[source_index])
            count = min(len(frame_values), self.max_points)
            if count:
                values[target_index, :count] = frame_values[:count]
                point_mask[target_index, :count] = True
        return PointCloudSequence(values, point_mask, frame_mask)

    def _frame_values(self, frame: RadarFrame) -> NDArray[np.float32]:
        rows = np.asarray(
            [
                (
                    point.x,
                    point.y,
                    point.z,
                    point.velocity,
                    0.0 if point.snr is None else point.snr,
                    0.0 if point.snr is None else 1.0,
                )
                for point in frame.points
            ],
            dtype=np.float32,
        )
        if not len(rows):
            return np.empty((0, len(POINT_FEATURE_NAMES)), dtype=np.float32)
        if not np.isfinite(rows).all():
            raise ValueError("radar points must contain only finite values")
        order = np.lexsort(tuple(rows[:, column] for column in reversed(range(rows.shape[1]))))
        rows = rows[order]
        if len(rows) > self.max_points:
            selected = np.linspace(0, len(rows) - 1, self.max_points, dtype=np.int64)
            rows = rows[selected]
        return rows.astype(np.float32, copy=False)

