from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from radar_module.contracts import RadarFrame, RadarPoint, Room, SourceMode
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


MMWAVE_OCPID_FRAME_RATE_HZ = 15.0
MMWAVE_OCPID_EXPECTED_SUBJECTS = tuple(f"Person{index}" for index in range(1, 10))
MMWAVE_OCPID_EXPECTED_CONDITIONS = (
    "Box",
    "Plant",
    "Sponge0",
    "Sponge47",
    "Sponge90",
)
MMWAVE_OCPID_SPLIT_BY_SUBJECT = {
    **{f"Person{index}": "external_train_pool" for index in range(1, 7)},
    "Person7": "external_validation",
    "Person8": "external_validation",
    "Person9": "external_test",
}
MMWAVE_OCPID_DATASET_MODE = "MMWAVE_OCPID_WALKING_HARD_NEGATIVE_V1"


@dataclass(frozen=True, slots=True)
class MmWaveOcPidExportSummary:
    source_directory: str
    source_tree_sha256: str
    output_file: str
    output_sha256: str
    point_cloud_variant: str
    source_file_count: int
    source_bytes: int
    subject_count: int
    condition_count: int
    frame_count: int
    point_count: int
    sample_count: int
    train_pool_sample_count: int
    validation_sample_count: int
    test_sample_count: int
    skipped_quality_count: int
    frame_rate_hz: float
    feature_version: str
    window_size: int
    input_size: int
    positive_samples_available: bool
    hard_negative_training_pool_eligible: bool
    deployment_validation_eligible: bool


def parse_mmwave_ocpid_text(
    input_path: str | Path,
    *,
    device_id: str | None = None,
    room: Room = Room.LIVING_ROOM,
    frame_rate_hz: float = MMWAVE_OCPID_FRAME_RATE_HZ,
) -> tuple[RadarFrame, ...]:
    """Parse one published CFAR point-cloud sequence.

    Empty lines delimit frames. Each non-empty row contains x, y, z, radial
    velocity and SNR. No identity or track id is inferred from the points.
    """

    if frame_rate_hz <= 0 or not math.isfinite(frame_rate_hz):
        raise ValueError("frame_rate_hz must be finite and positive")
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"mmWave-ocPID text file does not exist: {source}")
    stream_device = device_id or f"mmwave-ocpid-{source.stem}"
    if not stream_device.strip():
        raise ValueError("device_id must not be blank")
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    frames: list[RadarFrame] = []
    points: list[RadarPoint] = []

    def append_frame() -> None:
        nonlocal points
        if not points:
            return
        frames.append(
            RadarFrame(
                timestamp=start + timedelta(seconds=len(frames) / frame_rate_hz),
                device_id=stream_device,
                room=room,
                source_mode=SourceMode.REPLAY,
                points=tuple(points),
            )
        )
        points = []

    with source.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                append_frame()
                continue
            columns = stripped.split()
            if len(columns) != 5:
                raise ValueError(
                    f"mmWave-ocPID row {line_number} must contain five columns"
                )
            values = tuple(float(value) for value in columns)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"mmWave-ocPID row {line_number} contains non-finite values"
                )
            points.append(
                RadarPoint(
                    x=values[0],
                    y=values[1],
                    z=values[2],
                    velocity=values[3],
                    snr=values[4],
                    track_id=None,
                )
            )
    append_frame()
    if not frames:
        raise ValueError("mmWave-ocPID file contains no non-empty frames")
    return tuple(frames)


def export_mmwave_ocpid_hard_negative_npz(
    pointcloud_directory: str | Path,
    output_path: str | Path,
    *,
    stride_seconds: float = 2.0,
    max_windows_per_file: int = 100,
    extractor: RadarTemporalFeatureExtractorV2 | None = None,
) -> MmWaveOcPidExportSummary:
    """Export only the published CFAR variant as subject-isolated negatives."""

    if stride_seconds <= 0 or max_windows_per_file <= 0:
        raise ValueError("stride_seconds and max_windows_per_file must be positive")
    source_root = Path(pointcloud_directory).resolve()
    cfar_root = source_root / "CFARData"
    destination = Path(output_path).resolve()
    if not cfar_root.is_dir():
        raise FileNotFoundError("mmWave-ocPID CFARData directory does not exist")
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end with .npz")
    source_files = sorted(cfar_root.glob("Person*/*/Person*.txt"))
    metadata = [_source_metadata(path, cfar_root) for path in source_files]
    expected_cells = {
        (subject, condition)
        for subject in MMWAVE_OCPID_EXPECTED_SUBJECTS
        for condition in MMWAVE_OCPID_EXPECTED_CONDITIONS
    }
    if len(source_files) != 45 or set(metadata) != expected_cells:
        raise ValueError("mmWave-ocPID CFARData release is incomplete or unexpected")

    feature_extractor = extractor or RadarTemporalFeatureExtractorV2()
    features: list[np.ndarray] = []
    splits: list[str] = []
    subjects: list[str] = []
    conditions: list[str] = []
    relative_files: list[str] = []
    window_end_seconds: list[float] = []
    qualities: list[str] = []
    total_frames = 0
    total_points = 0
    skipped_quality = 0

    for source_path, (subject, condition) in zip(source_files, metadata):
        relative = source_path.relative_to(source_root).as_posix()
        frames = parse_mmwave_ocpid_text(
            source_path,
            device_id=f"mmwave-ocpid-{subject}-{condition}",
        )
        total_frames += len(frames)
        total_points += sum(len(frame.points) for frame in frames)
        first_timestamp = frames[0].timestamp
        frame_seconds = np.arange(len(frames), dtype=np.float64) / MMWAVE_OCPID_FRAME_RATE_HZ
        minimum_end = feature_extractor.history_seconds - (
            1.0 / feature_extractor.target_sample_rate_hz
        )
        candidates = np.arange(minimum_end, frame_seconds[-1], stride_seconds)
        if len(candidates) > max_windows_per_file:
            candidates = candidates[
                np.linspace(0, len(candidates) - 1, max_windows_per_file, dtype=np.int64)
            ]
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
            splits.append(MMWAVE_OCPID_SPLIT_BY_SUBJECT[subject])
            subjects.append(subject)
            conditions.append(condition)
            relative_files.append(relative)
            window_end_seconds.append(float(end_seconds))
            qualities.append(window.data_quality.value)
    if not features:
        raise ValueError("no usable mmWave-ocPID windows were produced")

    split_array = np.asarray(splits)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=np.stack(features).astype(np.float32, copy=False),
                labels=np.zeros(len(features), dtype=np.int8),
                action=np.asarray(["walk"] * len(features)),
                occlusion_condition=np.asarray(conditions),
                subject_id=np.asarray(subjects),
                split=split_array,
                source_files=np.asarray(relative_files),
                window_end_seconds=np.asarray(window_end_seconds, dtype=np.float32),
                data_quality=np.asarray(qualities),
                feature_version=np.asarray(FEATURE_VERSION_V2),
                feature_names=np.asarray(FEATURE_NAMES_V2),
                dataset_mode=np.asarray(MMWAVE_OCPID_DATASET_MODE),
                point_cloud_variant=np.asarray("CFARData"),
                frame_rate_hz=np.asarray(MMWAVE_OCPID_FRAME_RATE_HZ, dtype=np.float32),
                positive_samples_available=np.asarray(False),
                deployment_validation_eligible=np.asarray(False),
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    summary = MmWaveOcPidExportSummary(
        source_directory=str(source_root),
        source_tree_sha256=_tree_sha256(source_files, cfar_root),
        output_file=str(destination),
        output_sha256=_sha256(destination),
        point_cloud_variant="CFARData",
        source_file_count=len(source_files),
        source_bytes=sum(path.stat().st_size for path in source_files),
        subject_count=len(set(subjects)),
        condition_count=len(set(conditions)),
        frame_count=total_frames,
        point_count=total_points,
        sample_count=len(features),
        train_pool_sample_count=int(np.sum(split_array == "external_train_pool")),
        validation_sample_count=int(np.sum(split_array == "external_validation")),
        test_sample_count=int(np.sum(split_array == "external_test")),
        skipped_quality_count=skipped_quality,
        frame_rate_hz=MMWAVE_OCPID_FRAME_RATE_HZ,
        feature_version=FEATURE_VERSION_V2,
        window_size=feature_extractor.window_size,
        input_size=len(FEATURE_NAMES_V2),
        positive_samples_available=False,
        hard_negative_training_pool_eligible=True,
        deployment_validation_eligible=False,
    )
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _source_metadata(path: Path, cfar_root: Path) -> tuple[str, str]:
    relative = path.relative_to(cfar_root)
    if len(relative.parts) != 3:
        raise ValueError(f"unexpected mmWave-ocPID path: {relative}")
    subject, condition, file_name = relative.parts
    if file_name != f"{subject}.txt":
        raise ValueError(f"unexpected mmWave-ocPID file name: {relative}")
    return subject, condition


def _tree_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export mmWave-ocPID CFAR walking negatives.")
    parser.add_argument("--pointcloud-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--max-windows-per-file", type=int, default=100)
    args = parser.parse_args()
    result = export_mmwave_ocpid_hard_negative_npz(
        args.pointcloud_directory,
        args.output,
        stride_seconds=args.stride_seconds,
        max_windows_per_file=args.max_windows_per_file,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
