from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from radar_module.contracts import RadarPoint


_X_ALIASES = ("x", "posX", "position_x")
_Y_ALIASES = ("y", "posY", "position_y")
_Z_ALIASES = ("z", "posZ", "position_z")
_VELOCITY_ALIASES = ("velocity", "doppler", "v", "radial_velocity")
_SNR_ALIASES = ("snr", "SNR")
_TRACK_ID_ALIASES = (
    "track_id",
    "target_id",
    "trackIndex",
    "target_index",
)


def map_official_points(
    raw_points: Any,
    *,
    max_distance_m: float | None = 8.0,
) -> tuple[RadarPoint, ...]:
    """把官方已解码点字段映射为RadarPoint，不执行协议或信号处理。"""

    if max_distance_m is not None and max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive or None")
    if (
        not isinstance(raw_points, Iterable)
        or isinstance(raw_points, (str, bytes, bytearray, Mapping))
    ):
        return ()

    normalized: list[RadarPoint] = []
    for raw_point in raw_points:
        values = _extract_values(raw_point)
        if values is None:
            continue
        x, y, z, velocity, snr, track_id = values
        if not all(math.isfinite(value) for value in (x, y, z, velocity)):
            continue
        if max_distance_m is not None:
            distance = math.sqrt(x * x + y * y + z * z)
            if distance > max_distance_m:
                continue
        normalized.append(
            RadarPoint(
                x=x,
                y=y,
                z=z,
                velocity=velocity,
                snr=snr,
                track_id=track_id,
            )
        )
    return tuple(normalized)


def _extract_values(
    raw_point: Any,
) -> tuple[float, float, float, float, float | None, int | None] | None:
    if isinstance(raw_point, Mapping):
        raw_values = (
            _first_value(raw_point, _X_ALIASES),
            _first_value(raw_point, _Y_ALIASES),
            _first_value(raw_point, _Z_ALIASES),
            _first_value(raw_point, _VELOCITY_ALIASES),
        )
        raw_snr = _first_value(raw_point, _SNR_ALIASES)
        raw_track_id = _first_value(raw_point, _TRACK_ID_ALIASES)
    elif (
        isinstance(raw_point, Sequence)
        or (
            hasattr(raw_point, "__len__")
            and hasattr(raw_point, "__getitem__")
        )
    ) and not isinstance(raw_point, (str, bytes, bytearray, Mapping)):
        if len(raw_point) < 4:
            return None
        raw_values = raw_point[:4]
        raw_snr = raw_point[4] if len(raw_point) >= 5 else None
        raw_track_id = raw_point[6] if len(raw_point) >= 7 else None
    else:
        return None

    if any(value is None for value in raw_values):
        return None
    try:
        x, y, z, velocity = (float(value) for value in raw_values)
    except (TypeError, ValueError):
        return None
    return (
        x,
        y,
        z,
        velocity,
        _optional_finite_float(raw_snr),
        _optional_track_id(raw_track_id),
    )


def _first_value(mapping: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in mapping:
            return mapping[alias]
    return None


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_track_id(value: Any) -> int | None:
    parsed = _optional_finite_float(value)
    if parsed is None or not parsed.is_integer():
        return None
    track_id = int(parsed)
    # TI uses 255 for a point that has not been associated with a target.
    return track_id if 0 <= track_id < 255 else None
