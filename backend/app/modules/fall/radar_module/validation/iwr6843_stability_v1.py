from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class FrameStabilitySampleV1:
    timestamp: datetime
    point_count: int
    ti_frame_number: int | None = None
    ti_parser_error: int | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("stability sample timestamp must include timezone")
        if self.point_count < 0:
            raise ValueError("point_count must be non-negative")


@dataclass(frozen=True, slots=True)
class Iwr6843StabilityReportV1:
    schema_version: str
    expected_frame_rate_hz: float
    expected_interval_seconds: float
    frame_count: int
    duration_seconds: float
    observed_frame_rate_hz: float
    interval_count: int
    interval_mean_seconds: float | None
    interval_median_seconds: float | None
    interval_std_seconds: float | None
    interval_p95_seconds: float | None
    interval_max_seconds: float | None
    non_increasing_timestamp_count: int
    critical_gap_threshold_seconds: float
    critical_gap_count: int
    critical_gap_ratio: float
    frame_number_coverage_ratio: float
    exact_missing_frame_count: int | None
    estimated_missing_frame_count: int
    missing_frame_rate: float
    missing_frame_method: str
    duplicate_frame_number_count: int
    out_of_order_frame_number_count: int
    parser_error_frame_count: int
    point_count_min: int | None
    point_count_mean: float | None
    point_count_median: float | None
    point_count_p95: float | None
    point_count_max: int | None
    zero_point_frame_count: int
    zero_point_frame_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_iwr6843_stability(
    samples: Sequence[FrameStabilitySampleV1],
    *,
    expected_frame_rate_hz: float,
    critical_gap_threshold_seconds: float = 0.25,
) -> Iwr6843StabilityReportV1:
    """Summarize decoded-frame delivery without changing radar parsing."""
    if not math.isfinite(expected_frame_rate_hz) or expected_frame_rate_hz <= 0:
        raise ValueError("expected_frame_rate_hz must be finite and positive")
    if (
        not math.isfinite(critical_gap_threshold_seconds)
        or critical_gap_threshold_seconds <= 0
    ):
        raise ValueError(
            "critical_gap_threshold_seconds must be finite and positive"
        )
    expected_interval = 1.0 / expected_frame_rate_hz
    ordered = list(samples)
    intervals = [
        (current.timestamp - previous.timestamp).total_seconds()
        for previous, current in zip(ordered, ordered[1:])
    ]
    positive_intervals = np.asarray(
        [interval for interval in intervals if interval > 0.0],
        dtype=np.float64,
    )
    non_increasing = sum(interval <= 0.0 for interval in intervals)
    duration = (
        (ordered[-1].timestamp - ordered[0].timestamp).total_seconds()
        if len(ordered) >= 2
        else 0.0
    )
    observed_rate = (
        (len(ordered) - 1) / duration if duration > 0.0 else 0.0
    )
    critical_gap_count = int(
        np.sum(positive_intervals > critical_gap_threshold_seconds)
    )
    estimated_missing = sum(
        max(int(round(interval / expected_interval)) - 1, 0)
        for interval in positive_intervals
        if interval > expected_interval * 1.5
    )

    frame_numbers = [sample.ti_frame_number for sample in ordered]
    numbered_count = sum(value is not None for value in frame_numbers)
    exact_missing: int | None = None
    duplicates = 0
    out_of_order = 0
    if numbered_count >= 2:
        exact_missing = 0
        numbered = [int(value) for value in frame_numbers if value is not None]
        for previous, current in zip(numbered, numbered[1:]):
            delta = current - previous
            if delta > 1:
                exact_missing += delta - 1
            elif delta == 0:
                duplicates += 1
            elif delta < 0:
                out_of_order += 1

    if exact_missing is not None:
        missing_count = exact_missing
        missing_method = "ti_frame_number"
    else:
        missing_count = estimated_missing
        missing_method = "timestamp_estimate"
    possible_frames = len(ordered) + missing_count
    missing_rate = missing_count / possible_frames if possible_frames else 0.0

    point_counts = np.asarray(
        [sample.point_count for sample in ordered], dtype=np.float64
    )
    return Iwr6843StabilityReportV1(
        schema_version="iwr6843_stability_v1",
        expected_frame_rate_hz=float(expected_frame_rate_hz),
        expected_interval_seconds=expected_interval,
        frame_count=len(ordered),
        duration_seconds=max(duration, 0.0),
        observed_frame_rate_hz=float(observed_rate),
        interval_count=len(positive_intervals),
        interval_mean_seconds=_stat(positive_intervals, "mean"),
        interval_median_seconds=_stat(positive_intervals, "median"),
        interval_std_seconds=_stat(positive_intervals, "std"),
        interval_p95_seconds=_stat(positive_intervals, "p95"),
        interval_max_seconds=_stat(positive_intervals, "max"),
        non_increasing_timestamp_count=non_increasing,
        critical_gap_threshold_seconds=critical_gap_threshold_seconds,
        critical_gap_count=critical_gap_count,
        critical_gap_ratio=(
            critical_gap_count / len(positive_intervals)
            if len(positive_intervals)
            else 0.0
        ),
        frame_number_coverage_ratio=(
            numbered_count / len(ordered) if ordered else 0.0
        ),
        exact_missing_frame_count=exact_missing,
        estimated_missing_frame_count=estimated_missing,
        missing_frame_rate=float(missing_rate),
        missing_frame_method=missing_method,
        duplicate_frame_number_count=duplicates,
        out_of_order_frame_number_count=out_of_order,
        parser_error_frame_count=sum(
            sample.ti_parser_error not in (None, 0) for sample in ordered
        ),
        point_count_min=(int(np.min(point_counts)) if len(point_counts) else None),
        point_count_mean=_stat(point_counts, "mean"),
        point_count_median=_stat(point_counts, "median"),
        point_count_p95=_stat(point_counts, "p95"),
        point_count_max=(int(np.max(point_counts)) if len(point_counts) else None),
        zero_point_frame_count=int(np.sum(point_counts == 0)),
        zero_point_frame_ratio=(
            float(np.mean(point_counts == 0)) if len(point_counts) else 0.0
        ),
    )


def _stat(values: np.ndarray, kind: str) -> float | None:
    if not len(values):
        return None
    if kind == "mean":
        return float(np.mean(values))
    if kind == "median":
        return float(np.median(values))
    if kind == "std":
        return float(np.std(values))
    if kind == "p95":
        return float(np.quantile(values, 0.95))
    if kind == "max":
        return float(np.max(values))
    raise ValueError(f"unsupported statistic: {kind}")
