from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from radar_module.contracts import Room


MMFALL_RAW_POINT_WIDTH = 15
MMFALL_TARGET_ID_INDEX = 2
MMFALL_RANGE_INDEX = 9
MMFALL_AZIMUTH_INDEX = 10
MMFALL_ELEVATION_INDEX = 11
MMFALL_DOPPLER_INDEX = 12
MMFALL_SNR_INDEX = 13


@dataclass(frozen=True, slots=True)
class MmFallConversionConfig:
    """Parameters required to reproduce mmFall's raw-point visualization."""

    start_time: datetime
    device_id: str = "mmfall-ds1"
    room: Room = Room.LIVING_ROOM
    frame_rate_hz: float = 10.0
    tilt_angle_deg: float = -10.0
    radar_height_m: float = 1.8

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise ValueError("device_id must not be blank")
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be greater than zero")
        if not math.isfinite(self.tilt_angle_deg):
            raise ValueError("tilt_angle_deg must be finite")
        if not math.isfinite(self.radar_height_m):
            raise ValueError("radar_height_m must be finite")
        if (
            self.start_time.tzinfo is None
            or self.start_time.utcoffset() is None
        ):
            raise ValueError("start_time must include a timezone offset")


@dataclass(frozen=True, slots=True)
class MmFallConversionSummary:
    input_file: str
    output_file: str
    input_sha256: str
    frame_count: int
    point_count: int
    dropped_point_count: int
    frame_rate_hz: float
    device_id: str
    room: str
    tilt_angle_deg: float
    radar_height_m: float


def convert_mmfall_npy_to_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    config: MmFallConversionConfig,
    *,
    write_manifest: bool = True,
) -> MmFallConversionSummary:
    """Convert a trusted mmFall raw ``.npy`` file to RadarFrame JSONL.

    mmFall stores each point as 15 values:
    frame, point ID, target ID, centroid xyz, centroid velocity xyz,
    range, azimuth, elevation, Doppler, SNR and noise.

    This converter uses range/angles/Doppler (columns 9-12), then applies
    the same tilt and sensor-height correction as mmFall's
    ``data_visualizer.py``. The JSONL is compatible with
    ``JsonlReplayAdapter -> TiRadarReader``.

    ``allow_pickle=True`` is required by mmFall's object-array storage.
    Only convert files obtained from a trusted source.
    """

    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"mmFall input file does not exist: {source}")
    if source.suffix.lower() != ".npy":
        raise ValueError("mmFall input must be a .npy file")
    if destination.suffix.lower() != ".jsonl":
        raise ValueError("output path must end with .jsonl")

    frames = np.load(source, allow_pickle=True)
    if frames.ndim != 1:
        raise ValueError(
            f"expected one-dimensional mmFall object array, got {frames.shape}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame_step = timedelta(seconds=1.0 / config.frame_rate_hz)
    point_count = 0
    dropped_point_count = 0

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for frame_index, raw_frame in enumerate(frames):
                points, dropped = convert_mmfall_raw_frame(
                    raw_frame,
                    tilt_angle_deg=config.tilt_angle_deg,
                    radar_height_m=config.radar_height_m,
                )
                point_count += len(points)
                dropped_point_count += dropped
                timestamp = config.start_time + frame_index * frame_step
                record = {
                    "timestamp": timestamp.isoformat(timespec="milliseconds"),
                    "device_id": config.device_id,
                    "room": config.room.value,
                    "points": points,
                }
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    summary = MmFallConversionSummary(
        input_file=str(source),
        output_file=str(destination),
        input_sha256=_sha256(source),
        frame_count=int(frames.shape[0]),
        point_count=point_count,
        dropped_point_count=dropped_point_count,
        frame_rate_hz=config.frame_rate_hz,
        device_id=config.device_id,
        room=config.room.value,
        tilt_angle_deg=config.tilt_angle_deg,
        radar_height_m=config.radar_height_m,
    )
    if write_manifest:
        _write_manifest(destination, summary)
    return summary


def convert_mmfall_raw_frame(
    raw_frame: Any,
    *,
    tilt_angle_deg: float,
    radar_height_m: float,
) -> tuple[list[dict[str, float | int]], int]:
    frame = np.asarray(raw_frame)
    if frame.size == 0:
        return [], 0
    if frame.ndim != 2 or frame.shape[1] < MMFALL_RAW_POINT_WIDTH:
        raise ValueError(
            "each mmFall frame must have shape [N, >=15], "
            f"got {frame.shape}"
        )

    tilt = math.radians(tilt_angle_deg)
    cos_tilt = math.cos(tilt)
    sin_tilt = math.sin(tilt)
    points: list[dict[str, float | int]] = []
    dropped = 0

    for row in frame:
        try:
            point_range = float(row[MMFALL_RANGE_INDEX])
            azimuth = float(row[MMFALL_AZIMUTH_INDEX])
            elevation = float(row[MMFALL_ELEVATION_INDEX])
            velocity = float(row[MMFALL_DOPPLER_INDEX])
        except (TypeError, ValueError, IndexError):
            dropped += 1
            continue
        if not all(
            math.isfinite(value)
            for value in (point_range, azimuth, elevation, velocity)
        ):
            dropped += 1
            continue
        if point_range < 0:
            dropped += 1
            continue

        x = point_range * math.cos(elevation) * math.sin(azimuth)
        y = point_range * math.cos(elevation) * math.cos(azimuth)
        z = point_range * math.sin(elevation)

        # Same x-axis correction and radar height used by mmFall.
        rotated_y = cos_tilt * y - sin_tilt * z
        rotated_z = sin_tilt * y + cos_tilt * z + radar_height_m
        if not all(
            math.isfinite(value) for value in (x, rotated_y, rotated_z)
        ):
            dropped += 1
            continue
        point: dict[str, float | int] = {
            "x": x,
            "y": rotated_y,
            "z": rotated_z,
            "velocity": velocity,
        }
        try:
            snr = float(row[MMFALL_SNR_INDEX])
            target_id_value = float(row[MMFALL_TARGET_ID_INDEX])
        except (TypeError, ValueError, IndexError):
            snr = math.nan
            target_id_value = math.nan
        if math.isfinite(snr):
            point["snr"] = snr
        if (
            math.isfinite(target_id_value)
            and target_id_value.is_integer()
            and 0 <= target_id_value < 255
        ):
            point["track_id"] = int(target_id_value)
        points.append(point)
    return points, dropped


# Backward-compatible private alias for callers written before the raw-frame
# conversion was reused by the research dataset exporter.
_convert_frame = convert_mmfall_raw_frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(
    output_path: Path,
    summary: MmFallConversionSummary,
) -> None:
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            asdict(summary),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert mmFall raw NPY point clouds to RadarFrame JSONL."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device-id", default="mmfall-ds1")
    parser.add_argument(
        "--room",
        choices=[room.value for room in Room],
        default=Room.LIVING_ROOM.value,
    )
    parser.add_argument("--frame-rate-hz", type=float, default=10.0)
    parser.add_argument("--tilt-angle-deg", type=float, default=-10.0)
    parser.add_argument("--radar-height-m", type=float, default=1.8)
    parser.add_argument(
        "--start-time",
        default="2026-01-01T00:00:00+08:00",
        help="Timezone-aware ISO-8601 timestamp for replay frame zero.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write the adjacent conversion manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start_time = datetime.fromisoformat(args.start_time.replace("Z", "+00:00"))
    config = MmFallConversionConfig(
        start_time=start_time,
        device_id=args.device_id,
        room=Room(args.room),
        frame_rate_hz=args.frame_rate_hz,
        tilt_angle_deg=args.tilt_angle_deg,
        radar_height_m=args.radar_height_m,
    )
    summary = convert_mmfall_npy_to_jsonl(
        args.input,
        args.output,
        config,
        write_manifest=not args.no_manifest,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
