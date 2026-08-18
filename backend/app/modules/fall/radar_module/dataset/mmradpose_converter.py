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


MMRADPOSE_FRAME_RATE_HZ = 15.0
MMRADPOSE_NOMINAL_SEQUENCE_COUNT = 432
# The checksum-verified Zenodo release contains 431 target-list files for
# subjects p1-p12. Nine nominal subject/angle/action cells are absent and
# eight cells have a second trial, so 431 is the complete published archive.
MMRADPOSE_RELEASE_SEQUENCE_COUNT = 431
MMRADPOSE_EXPECTED_SUBJECTS = tuple(f"p{index}" for index in range(1, 13))
MMRADPOSE_ACTIONS = {
    0: "t_pose",
    1: "left_upper_limb_extension",
    2: "right_upper_limb_extension",
    3: "bilateral_upper_limb_extension",
    4: "bicep_curls",
    5: "front_arm_rotation",
    6: "torso_forward_bending",
    7: "left_front_lunge",
    8: "right_front_lunge",
    9: "squats",
    10: "side_lower_limb_extension",
    11: "front_lower_limb_extension",
}
MMRADPOSE_SPLIT_BY_SUBJECT = {
    **{f"p{index}": "train" for index in range(1, 9)},
    "p9": "validation",
    "p10": "validation",
    "p11": "test",
    "p12": "test",
}


@dataclass(frozen=True, slots=True)
class MmRadPoseExportSummary:
    source_directory: str
    output_file: str
    output_sha256: str
    source_complete: bool
    source_file_count: int
    source_bytes: int
    subject_count: int
    sequence_count: int
    frame_count: int
    point_count: int
    sample_count: int
    skipped_quality_count: int
    feature_version: str
    window_size: int
    input_size: int
    frame_rate_hz: float
    snr_preserved: bool
    noise_discarded: bool
    intensity_discarded: bool
    positive_samples_available: bool
    hard_negative_training_pool_eligible: bool
    deployment_validation_eligible: bool


def parse_mmradpose_targetlist(
    input_path: str | Path,
    *,
    device_id: str = "mmradpose",
    room: Room = Room.LIVING_ROOM,
    frame_rate_hz: float = MMRADPOSE_FRAME_RATE_HZ,
) -> tuple[RadarFrame, ...]:
    """Convert one published target-list tensor into decoded radar frames.

    The seven source columns are x, y, z, radial velocity, SNR, noise and
    intensity.  Only fields represented by ``RadarPoint`` are retained.  A
    row whose seven values are all zero is documented source padding and is
    not emitted as a physical point.
    """

    if frame_rate_hz <= 0 or not math.isfinite(frame_rate_hz):
        raise ValueError("frame_rate_hz must be finite and positive")
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"mmRadPose target list does not exist: {source}")
    if not device_id.strip():
        raise ValueError("device_id must not be blank")
    target_list = np.load(source, allow_pickle=False)
    if target_list.ndim != 3 or target_list.shape[1:] != (64, 7):
        raise ValueError(
            "mmRadPose target list must have shape (frames, 64, 7), "
            f"got {target_list.shape}"
        )
    if not np.issubdtype(target_list.dtype, np.number):
        raise ValueError("mmRadPose target list must contain numeric values")
    target_list = np.asarray(target_list, dtype=np.float64)
    if not np.isfinite(target_list).all():
        raise ValueError("mmRadPose target list contains non-finite values")

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    frames: list[RadarFrame] = []
    for frame_index, raw_points in enumerate(target_list):
        present = np.any(raw_points != 0.0, axis=1)
        points = tuple(
            RadarPoint(
                x=float(row[0]),
                y=float(row[1]),
                z=float(row[2]),
                velocity=float(row[3]),
                snr=float(row[4]),
                track_id=None,
            )
            for row in raw_points[present]
        )
        frames.append(
            RadarFrame(
                timestamp=start + timedelta(seconds=frame_index / frame_rate_hz),
                device_id=device_id,
                room=room,
                source_mode=SourceMode.REPLAY,
                points=points,
            )
        )
    if not frames:
        raise ValueError("mmRadPose target list contains no frames")
    return tuple(frames)


def export_mmradpose_hard_negative_npz(
    pointcloud_directory: str | Path,
    output_path: str | Path,
    *,
    allow_incomplete_source: bool = False,
    stride_seconds: float = 0.5,
    max_windows_per_sequence: int = 100,
    extractor: RadarTemporalFeatureExtractorV2 | None = None,
) -> MmRadPoseExportSummary:
    """Export negative-only v2 windows with subject-level partitions.

    Partial downloads are accepted only with explicit opt-in and are marked
    ``AUDIT_SAMPLE_ONLY``.  They must not be merged into a training pool.
    """

    if stride_seconds <= 0 or max_windows_per_sequence <= 0:
        raise ValueError(
            "stride_seconds and max_windows_per_sequence must be positive"
        )
    source_root = Path(pointcloud_directory).resolve()
    destination = Path(output_path).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"mmRadPose point-cloud directory does not exist: {source_root}"
        )
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end with .npz")

    source_files = sorted(source_root.rglob("targetlist_64.npy"))
    if not source_files:
        raise ValueError("no mmRadPose targetlist_64.npy files were found")
    metadata = [_sequence_metadata(path, source_root) for path in source_files]
    subjects = sorted({item[0] for item in metadata}, key=_subject_number)
    source_complete = (
        tuple(subjects) == MMRADPOSE_EXPECTED_SUBJECTS
        and len(source_files) == MMRADPOSE_RELEASE_SEQUENCE_COUNT
    )
    if not source_complete and not allow_incomplete_source:
        raise ValueError(
            "mmRadPose source is incomplete; pass allow_incomplete_source=True "
            "only for format-audit output"
        )

    feature_extractor = extractor or RadarTemporalFeatureExtractorV2()
    features: list[np.ndarray] = []
    actions: list[str] = []
    action_ids: list[int] = []
    splits: list[str] = []
    subject_ids: list[str] = []
    angles: list[str] = []
    relative_files: list[str] = []
    window_end_seconds: list[float] = []
    qualities: list[str] = []
    total_frames = 0
    total_points = 0
    skipped_quality = 0

    for source_path, (subject, angle, action_id, trial) in zip(
        source_files, metadata
    ):
        relative = source_path.relative_to(source_root).as_posix()
        device_id = f"mmradpose-{subject}-{angle}-a{action_id}-t{trial}"
        frames = parse_mmradpose_targetlist(source_path, device_id=device_id)
        total_frames += len(frames)
        total_points += sum(len(frame.points) for frame in frames)
        first_timestamp = frames[0].timestamp
        frame_seconds = np.asarray(
            [
                (frame.timestamp - first_timestamp).total_seconds()
                for frame in frames
            ],
            dtype=np.float64,
        )
        minimum_end = feature_extractor.history_seconds - (
            1.0 / feature_extractor.target_sample_rate_hz
        )
        candidates = np.arange(minimum_end, frame_seconds[-1], stride_seconds)
        if len(candidates) > max_windows_per_sequence:
            selected = np.linspace(
                0,
                len(candidates) - 1,
                max_windows_per_sequence,
                dtype=np.int64,
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
            actions.append(MMRADPOSE_ACTIONS[action_id])
            action_ids.append(action_id)
            splits.append(MMRADPOSE_SPLIT_BY_SUBJECT[subject])
            subject_ids.append(subject)
            angles.append(angle)
            relative_files.append(relative)
            window_end_seconds.append(float(end_seconds))
            qualities.append(window.data_quality.value)

    if not features:
        raise ValueError("no usable mmRadPose hard-negative windows were produced")
    dataset_mode = (
        "EXTERNAL_HARD_NEGATIVE_ONLY"
        if source_complete
        else "EXTERNAL_HARD_NEGATIVE_AUDIT_SAMPLE_ONLY"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=np.stack(features).astype(np.float32, copy=False),
                labels=np.zeros(len(features), dtype=np.int8),
                action=np.asarray(actions),
                action_id=np.asarray(action_ids, dtype=np.int8),
                split=np.asarray(splits),
                subject_id=np.asarray(subject_ids),
                angle=np.asarray(angles),
                source_files=np.asarray(relative_files),
                window_end_seconds=np.asarray(
                    window_end_seconds, dtype=np.float32
                ),
                data_quality=np.asarray(qualities),
                feature_version=np.asarray(FEATURE_VERSION_V2),
                feature_names=np.asarray(FEATURE_NAMES_V2),
                dataset_mode=np.asarray(dataset_mode),
                source_complete=np.asarray(source_complete),
                snr_preserved=np.asarray(True),
                noise_discarded=np.asarray(True),
                intensity_discarded=np.asarray(True),
                positive_samples_available=np.asarray(False),
                hard_negative_training_pool_eligible=np.asarray(source_complete),
                deployment_validation_eligible=np.asarray(False),
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    summary = MmRadPoseExportSummary(
        source_directory=str(source_root),
        output_file=str(destination),
        output_sha256=_sha256(destination),
        source_complete=source_complete,
        source_file_count=len(source_files),
        source_bytes=sum(path.stat().st_size for path in source_files),
        subject_count=len(subjects),
        sequence_count=len(source_files),
        frame_count=total_frames,
        point_count=total_points,
        sample_count=len(features),
        skipped_quality_count=skipped_quality,
        feature_version=FEATURE_VERSION_V2,
        window_size=feature_extractor.window_size,
        input_size=len(FEATURE_NAMES_V2),
        frame_rate_hz=MMRADPOSE_FRAME_RATE_HZ,
        snr_preserved=True,
        noise_discarded=True,
        intensity_discarded=True,
        positive_samples_available=False,
        hard_negative_training_pool_eligible=source_complete,
        deployment_validation_eligible=False,
    )
    _write_manifest(destination, summary, source_root, source_files)
    return summary


def _sequence_metadata(path: Path, source_root: Path) -> tuple[str, str, int, str]:
    relative = path.relative_to(source_root)
    if len(relative.parts) != 5 or relative.name != "targetlist_64.npy":
        raise ValueError(f"unexpected mmRadPose path layout: {relative.as_posix()}")
    subject, angle, action_text, trial, _ = relative.parts
    if subject not in MMRADPOSE_SPLIT_BY_SUBJECT:
        raise ValueError(f"unexpected mmRadPose subject: {subject}")
    if not angle.startswith("angle"):
        raise ValueError(f"unexpected mmRadPose angle: {angle}")
    try:
        action_id = int(action_text)
    except ValueError as exc:
        raise ValueError(f"invalid mmRadPose action id: {action_text}") from exc
    if action_id not in MMRADPOSE_ACTIONS:
        raise ValueError(f"unsupported mmRadPose action id: {action_id}")
    return subject, angle, action_id, trial


def _subject_number(subject: str) -> int:
    return int(subject[1:])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(
    output_path: Path,
    summary: MmRadPoseExportSummary,
    source_root: Path,
    source_files: list[Path],
) -> None:
    payload = asdict(summary)
    payload["source_files"] = [
        {
            "path": path.relative_to(source_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in source_files
    ]
    payload["column_mapping"] = [
        "x",
        "y",
        "z",
        "radial_velocity",
        "snr",
        "noise",
        "intensity",
    ]
    payload["known_limitations"] = [
        "mmRadPose contains no falls and supplies negative samples only",
        "sensor geometry and coordinate calibration differ from mmFall/local radar",
        "noise and intensity are not represented by radar_features_v2",
        "there is no persistent person identity beyond the recording subject id",
        "source-complete false means format audit only and prohibits training merge",
    ]
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export mmRadPose v2 hard-negative windows."
    )
    parser.add_argument("--pointcloud-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-incomplete-source", action="store_true")
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--max-windows-per-sequence", type=int, default=100)
    args = parser.parse_args()
    summary = export_mmradpose_hard_negative_npz(
        args.pointcloud_directory,
        args.output,
        allow_incomplete_source=args.allow_incomplete_source,
        stride_seconds=args.stride_seconds,
        max_windows_per_sequence=args.max_windows_per_sequence,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
