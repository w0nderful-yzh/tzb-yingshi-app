from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from radar_module.contracts import RadarFrame, RadarPoint, Room, SourceMode
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


RADHAR_ACTIONS = {"squats": "CROUCH", "jump": "JUMP"}


@dataclass(frozen=True, slots=True)
class RadHarExportSummary:
    source_directory: str
    output_file: str
    output_sha256: str
    source_file_count: int
    source_bytes: int
    frame_count: int
    point_count: int
    sample_count: int
    squat_count: int
    jump_count: int
    skipped_quality_count: int
    feature_version: str
    window_size: int
    input_size: int
    intensity_used_as_snr: bool
    positive_samples_available: bool
    deployment_validation_eligible: bool


def parse_radhar_text(
    input_path: str | Path,
    *,
    device_id: str | None = None,
    room: Room = Room.LIVING_ROOM,
) -> tuple[RadarFrame, ...]:
    """Parse RadHAR ROS text while preserving source frame timestamps.

    A new frame starts when ``point_id`` resets to zero.  RadHAR intensity is
    deliberately discarded because it is not documented as calibrated SNR.
    No track identity is invented.
    """

    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"RadHAR text file does not exist: {source}")
    stream_device = device_id or f"radhar-{source.stem}"
    records = _point_records(source)
    frames: list[RadarFrame] = []
    points: list[RadarPoint] = []
    frame_timestamp: datetime | None = None
    for record in records:
        point_id = int(record["point_id"])
        timestamp = datetime.fromtimestamp(
            float(record["secs"]) + float(record["nsecs"]) / 1_000_000_000.0,
            tz=timezone.utc,
        )
        if point_id == 0 and points:
            assert frame_timestamp is not None
            frames.append(
                RadarFrame(
                    timestamp=frame_timestamp,
                    device_id=stream_device,
                    room=room,
                    source_mode=SourceMode.REPLAY,
                    points=tuple(points),
                )
            )
            points = []
            frame_timestamp = None
        if frame_timestamp is None:
            frame_timestamp = timestamp
        values = tuple(float(record[name]) for name in ("x", "y", "z", "velocity"))
        if all(math.isfinite(value) for value in values):
            points.append(RadarPoint(*values, snr=None, track_id=None))
    if points:
        assert frame_timestamp is not None
        frames.append(
            RadarFrame(
                timestamp=frame_timestamp,
                device_id=stream_device,
                room=room,
                source_mode=SourceMode.REPLAY,
                points=tuple(points),
            )
        )
    if not frames:
        raise ValueError(f"RadHAR file contains no complete points: {source}")
    for previous, current in zip(frames[:-1], frames[1:]):
        if current.timestamp < previous.timestamp:
            raise ValueError("RadHAR frame timestamps are not ordered")
    return tuple(frames)


def export_radhar_hard_negative_npz(
    radhar_data_directory: str | Path,
    output_path: str | Path,
    *,
    stride_seconds: float = 0.5,
    max_windows_per_file: int = 100,
    extractor: RadarTemporalFeatureExtractorV2 | None = None,
) -> RadHarExportSummary:
    if stride_seconds <= 0 or max_windows_per_file <= 0:
        raise ValueError("stride_seconds and max_windows_per_file must be positive")
    source_root = Path(radhar_data_directory).resolve()
    destination = Path(output_path).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"RadHAR data directory does not exist: {source_root}")
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end with .npz")

    source_files = sorted(
        path
        for split_name in ("Train", "Test")
        for action in RADHAR_ACTIONS
        for path in (source_root / split_name / action).glob("*.txt")
    )
    if not source_files:
        raise ValueError("no RadHAR squat/jump source files were found")

    feature_extractor = extractor or RadarTemporalFeatureExtractorV2()
    features: list[np.ndarray] = []
    action_names: list[str] = []
    hard_negative_types: list[str] = []
    splits: list[str] = []
    relative_files: list[str] = []
    window_end_seconds: list[float] = []
    qualities: list[str] = []
    total_frames = 0
    total_points = 0
    skipped_quality = 0

    for source_path in source_files:
        relative = source_path.relative_to(source_root).as_posix()
        action = source_path.parent.name
        author_split = source_path.parent.parent.name
        frames = parse_radhar_text(source_path)
        total_frames += len(frames)
        total_points += sum(len(frame.points) for frame in frames)
        first_timestamp = frames[0].timestamp
        frame_seconds = np.asarray(
            [(frame.timestamp - first_timestamp).total_seconds() for frame in frames],
            dtype=np.float64,
        )
        minimum_end = feature_extractor.history_seconds - (
            1.0 / feature_extractor.target_sample_rate_hz
        )
        candidates = np.arange(minimum_end, frame_seconds[-1], stride_seconds)
        if len(candidates) > max_windows_per_file:
            selected = np.linspace(
                0, len(candidates) - 1, max_windows_per_file, dtype=np.int64
            )
            candidates = candidates[selected]
        tolerance = feature_extractor.alignment_tolerance_seconds
        for end_seconds in candidates:
            left = bisect.bisect_left(
                frame_seconds,
                float(end_seconds) - feature_extractor.history_seconds - tolerance,
            )
            right = bisect.bisect_right(
                frame_seconds, float(end_seconds) + tolerance
            )
            if left >= right:
                skipped_quality += 1
                continue
            window = feature_extractor.transform(
                frames[left:right],
                end_timestamp=first_timestamp + timedelta(seconds=float(end_seconds)),
            )
            if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
                skipped_quality += 1
                continue
            features.append(np.asarray(window.values, dtype=np.float32))
            action_names.append(action)
            hard_negative_types.append(RADHAR_ACTIONS[action])
            splits.append(
                "external_train_pool" if author_split == "Train" else "external_test"
            )
            relative_files.append(relative)
            window_end_seconds.append(float(end_seconds))
            qualities.append(window.data_quality.value)

    if not features:
        raise ValueError("no usable RadHAR hard-negative windows were produced")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=np.stack(features).astype(np.float32, copy=False),
                labels=np.zeros(len(features), dtype=np.int8),
                action=np.asarray(action_names),
                hard_negative_type=np.asarray(hard_negative_types),
                split=np.asarray(splits),
                source_files=np.asarray(relative_files),
                window_end_seconds=np.asarray(window_end_seconds, dtype=np.float32),
                data_quality=np.asarray(qualities),
                feature_version=np.asarray(FEATURE_VERSION_V2),
                feature_names=np.asarray(FEATURE_NAMES_V2),
                dataset_mode=np.asarray("EXTERNAL_HARD_NEGATIVE_ONLY"),
                intensity_used_as_snr=np.asarray(False),
                positive_samples_available=np.asarray(False),
                deployment_validation_eligible=np.asarray(False),
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    actions = np.asarray(action_names)
    summary = RadHarExportSummary(
        source_directory=str(source_root),
        output_file=str(destination),
        output_sha256=_sha256(destination),
        source_file_count=len(source_files),
        source_bytes=sum(path.stat().st_size for path in source_files),
        frame_count=total_frames,
        point_count=total_points,
        sample_count=len(features),
        squat_count=int(np.sum(actions == "squats")),
        jump_count=int(np.sum(actions == "jump")),
        skipped_quality_count=skipped_quality,
        feature_version=FEATURE_VERSION_V2,
        window_size=feature_extractor.window_size,
        input_size=len(FEATURE_NAMES_V2),
        intensity_used_as_snr=False,
        positive_samples_available=False,
        deployment_validation_eligible=False,
    )
    _write_manifest(destination, summary)
    return summary


def _point_records(path: Path) -> Iterator[dict[str, float]]:
    record: dict[str, float] = {}
    required = {"secs", "nsecs", "point_id", "x", "y", "z", "velocity"}
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            normalized = line.strip()
            if normalized == "---":
                if record:
                    missing = sorted(required.difference(record))
                    if missing:
                        raise ValueError(
                            f"incomplete RadHAR point before line {line_number}: {missing}"
                        )
                    yield record
                    record = {}
                continue
            if ":" not in normalized:
                continue
            name, value = (part.strip() for part in normalized.split(":", 1))
            if name not in required:
                continue
            try:
                record[name] = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid RadHAR {name} at line {line_number}"
                ) from exc
    if record:
        missing = sorted(required.difference(record))
        if missing:
            raise ValueError(f"incomplete final RadHAR point: {missing}")
        yield record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, summary: RadHarExportSummary) -> None:
    payload = asdict(summary)
    payload["known_limitations"] = [
        "RadHAR has no fall-positive samples",
        "author Train/Test filenames do not establish subject independence",
        "sensor pose and coordinate calibration differ from mmFall/local radar",
        "intensity was discarded and was not relabelled as SNR",
        "timestamps are preserved because source frame spacing is irregular",
    ]
    path.with_suffix(".manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RadHAR squat/jump v2 hard negatives.")
    parser.add_argument("--data-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--max-windows-per-file", type=int, default=100)
    args = parser.parse_args()
    summary = export_radhar_hard_negative_npz(
        args.data_directory,
        args.output,
        stride_seconds=args.stride_seconds,
        max_windows_per_file=args.max_windows_per_file,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
