from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np

from radar_module.dataset.mmradpose_converter import (
    MMRADPOSE_ACTIONS,
    MMRADPOSE_RELEASE_SEQUENCE_COUNT,
    MMRADPOSE_SPLIT_BY_SUBJECT,
    _sequence_metadata,
    parse_mmradpose_targetlist,
)
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.preprocess.pointcloud_sequence import (
    POINT_FEATURE_NAMES,
    POINT_SEQUENCE_VERSION,
    PointCloudSequenceBuilder,
)


POINT_PRETRAIN_DATASET_MODE = "MMRADPOSE_ACTIVITY_PRETRAIN_V1"
POINT_PREDICTION_DATASET_MODE = "DGUHA_POINT_PREFALL_RESEARCH_V1"


@dataclass(frozen=True, slots=True)
class PointPretrainingExportSummary:
    source_directory: str
    output_file: str
    output_sha256: str
    source_complete: bool
    source_sequence_count: int
    sample_count: int
    train_count: int
    validation_count: int
    test_count: int
    subject_count: int
    class_count: int
    time_steps: int
    max_points: int
    input_size: int
    representation_pretraining_only: bool
    fall_prediction_labels_available: bool
    deployment_eligible: bool


@dataclass(frozen=True, slots=True)
class PointPredictionExportSummary:
    source_index_file: str
    output_file: str
    output_sha256: str
    sample_count: int
    positive_count: int
    negative_count: int
    train_count: int
    validation_count: int
    test_count: int
    subject_count: int
    same_recording_negative_count: int
    normal_negative_stride: int
    ambiguity_buffer_seconds: float
    minimum_lead_seconds: float
    maximum_lead_seconds: float
    kinect_used_as_model_input: bool
    deployment_eligible: bool


def export_mmradpose_point_pretraining_npz(
    pointcloud_directory: str | Path,
    output_path: str | Path,
    *,
    windows_per_sequence: int = 4,
    allow_incomplete_source: bool = False,
    builder: PointCloudSequenceBuilder | None = None,
) -> PointPretrainingExportSummary:
    if windows_per_sequence <= 0:
        raise ValueError("windows_per_sequence must be positive")
    source_root = Path(pointcloud_directory).resolve()
    destination = Path(output_path).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"mmRadPose directory does not exist: {source_root}")
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end with .npz")
    source_files = sorted(source_root.rglob("targetlist_64.npy"))
    if not source_files:
        raise ValueError("no mmRadPose target-list files were found")
    metadata = [_sequence_metadata(path, source_root) for path in source_files]
    subjects = sorted({item[0] for item in metadata})
    source_complete = (
        len(source_files) == MMRADPOSE_RELEASE_SEQUENCE_COUNT
        and set(subjects) == set(MMRADPOSE_SPLIT_BY_SUBJECT)
    )
    if not source_complete and not allow_incomplete_source:
        raise ValueError("mmRadPose source is incomplete; explicit audit opt-in is required")

    sequence_builder = builder or PointCloudSequenceBuilder()
    values: list[np.ndarray] = []
    point_masks: list[np.ndarray] = []
    frame_masks: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    source_paths: list[str] = []
    sample_subjects: list[str] = []

    for source_path, (subject, _angle, action_id, _trial) in zip(source_files, metadata):
        frames = parse_mmradpose_targetlist(source_path)
        minimum_end = sequence_builder.history_seconds - 1.0 / sequence_builder.sample_rate_hz
        duration = (frames[-1].timestamp - frames[0].timestamp).total_seconds()
        if duration < minimum_end:
            continue
        candidate_seconds = np.linspace(minimum_end, duration, windows_per_sequence)
        for end_seconds in candidate_seconds:
            sequence = sequence_builder.transform(
                frames,
                end_timestamp=frames[0].timestamp
                + (frames[-1].timestamp - frames[0].timestamp) * float(end_seconds / duration),
            )
            if int(sequence.frame_mask.sum()) < max(2, sequence_builder.time_steps // 2):
                continue
            values.append(sequence.values)
            point_masks.append(sequence.point_mask)
            frame_masks.append(sequence.frame_mask)
            labels.append(action_id)
            splits.append(MMRADPOSE_SPLIT_BY_SUBJECT[subject])
            source_paths.append(source_path.relative_to(source_root).as_posix())
            sample_subjects.append(subject)

    if not values:
        raise ValueError("no point-cloud pretraining windows were produced")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                points=np.stack(values).astype(np.float32, copy=False),
                point_mask=np.stack(point_masks).astype(np.bool_, copy=False),
                frame_mask=np.stack(frame_masks).astype(np.bool_, copy=False),
                labels=np.asarray(labels, dtype=np.int64),
                split=np.asarray(splits),
                subject_id=np.asarray(sample_subjects),
                source_files=np.asarray(source_paths),
                action_names=np.asarray(tuple(MMRADPOSE_ACTIONS[index] for index in sorted(MMRADPOSE_ACTIONS))),
                sequence_version=np.asarray(POINT_SEQUENCE_VERSION),
                feature_names=np.asarray(POINT_FEATURE_NAMES),
                dataset_mode=np.asarray(POINT_PRETRAIN_DATASET_MODE),
                source_complete=np.asarray(source_complete),
                fall_prediction_labels_available=np.asarray(False),
                deployment_eligible=np.asarray(False),
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    split_array = np.asarray(splits)
    summary = PointPretrainingExportSummary(
        source_directory=str(source_root),
        output_file=str(destination),
        output_sha256=_sha256(destination),
        source_complete=source_complete,
        source_sequence_count=len(source_files),
        sample_count=len(values),
        train_count=int(np.sum(split_array == "train")),
        validation_count=int(np.sum(split_array == "validation")),
        test_count=int(np.sum(split_array == "test")),
        subject_count=len(subjects),
        class_count=len(MMRADPOSE_ACTIONS),
        time_steps=sequence_builder.time_steps,
        max_points=sequence_builder.max_points,
        input_size=len(POINT_FEATURE_NAMES),
        representation_pretraining_only=True,
        fall_prediction_labels_available=False,
        deployment_eligible=False,
    )
    manifest = asdict(summary)
    manifest["split_policy"] = "mmRadPose subject-disjoint p1-p8/p9-p10/p11-p12"
    manifest["label_semantics"] = "recording-level human activity; no falls"
    manifest["intended_use"] = "radar point-cloud encoder representation pretraining only"
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def export_dguha_point_prediction_npz(
    index_dataset_path: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    *,
    builder: PointCloudSequenceBuilder | None = None,
    normal_negative_stride: int = 1,
    ambiguity_buffer_seconds: float = 0.2,
) -> PointPredictionExportSummary:
    """Rebuild the audited DGUHA weak labels as point-cloud windows.

    The existing v2 export remains the source of sample timestamps and subject
    splits. Kinect-derived values are copied only as label provenance; point
    tensors are rebuilt exclusively from the paired radar text files.
    """

    if normal_negative_stride <= 0 or ambiguity_buffer_seconds < 0:
        raise ValueError("negative stride must be positive and buffer non-negative")
    index_path = Path(index_dataset_path).resolve()
    source_root = Path(data_root).resolve()
    destination = Path(output_path).resolve()
    if not index_path.is_file() or not source_root.is_dir():
        raise FileNotFoundError("DGUHA index dataset or radar source root is missing")
    with np.load(index_path, allow_pickle=False) as index:
        required = {
            "labels", "split", "subject_id", "source_files", "window_end_seconds",
            "seconds_to_onset", "label_source", "dataset_mode",
            "kinect_used_as_model_input", "deployment_eligible",
        }
        missing = sorted(required.difference(index.files))
        if missing:
            raise ValueError(f"DGUHA index dataset is incomplete: {missing}")
        if str(index["dataset_mode"].item()) != "DGUHA_SKELETON_PSEUDOLABEL_RESEARCH_V2":
            raise ValueError("DGUHA index dataset mode is incompatible")
        if bool(index["kinect_used_as_model_input"].item()):
            raise ValueError("DGUHA index must be radar-only at model input")
        labels = np.asarray(index["labels"], dtype=np.int8)
        splits = np.asarray(index["split"])
        subjects = np.asarray(index["subject_id"])
        source_files = np.asarray(index["source_files"])
        window_end_seconds = np.asarray(index["window_end_seconds"], dtype=np.float32)
        seconds_to_onset = np.asarray(index["seconds_to_onset"], dtype=np.float32)
        label_source = np.asarray(index["label_source"])

    manifest_path = index_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    minimum_lead = float(manifest["minimum_lead_seconds"])
    maximum_lead = float(manifest["maximum_lead_seconds"])
    if not 0 < minimum_lead <= maximum_lead:
        raise ValueError("index prediction horizon is invalid")

    if normal_negative_stride > 1:
        keep = labels == 1
        for relative_text in np.unique(source_files[labels == 0]):
            indices = np.flatnonzero((source_files == relative_text) & (labels == 0))
            keep[indices[::normal_negative_stride]] = True
        labels = labels[keep]
        splits = splits[keep]
        subjects = subjects[keep]
        source_files = source_files[keep]
        window_end_seconds = window_end_seconds[keep]
        seconds_to_onset = seconds_to_onset[keep]
        label_source = label_source[keep]

    sequence_builder = builder or PointCloudSequenceBuilder()
    events_path = index_path.with_suffix(".events.json")
    events = json.loads(events_path.read_text(encoding="utf-8"))
    same_recording_negative_count = 0
    extra_labels: list[int] = []
    extra_splits: list[str] = []
    extra_subjects: list[str] = []
    extra_sources: list[str] = []
    extra_end_seconds: list[float] = []
    extra_seconds_to_onset: list[float] = []
    extra_label_sources: list[str] = []
    minimum_window_end = (
        sequence_builder.history_seconds - 1.0 / sequence_builder.sample_rate_hz
    )
    for event in events:
        if not bool(event["eligible_for_prediction_windows"]):
            continue
        onset_seconds = event.get("descent_onset_seconds_from_radar_start")
        if onset_seconds is None:
            continue
        # Keep same-recording negatives beyond the configured positive horizon
        # plus an explicit ambiguity buffer.  A fixed 0.8 s cutoff would
        # conflict with wider/earlier horizons such as 0.5-1.0 s.
        candidates = _same_recording_negative_endpoints(
            minimum_window_end=minimum_window_end,
            onset_seconds=float(onset_seconds),
            maximum_lead_seconds=maximum_lead,
            ambiguity_buffer_seconds=ambiguity_buffer_seconds,
        )
        for end_seconds in candidates:
            extra_labels.append(0)
            extra_splits.append(str(event["project_split"]))
            extra_subjects.append(str(event["subject_id"]))
            extra_sources.append(str(event["source_file"]))
            extra_end_seconds.append(float(end_seconds))
            extra_seconds_to_onset.append(float(onset_seconds) - float(end_seconds))
            extra_label_sources.append(
                "dguha_same_fall_recording_outside_prediction_horizon"
            )
    same_recording_negative_count = len(extra_labels)
    if extra_labels:
        labels = np.concatenate((labels, np.asarray(extra_labels, dtype=np.int8)))
        splits = np.concatenate((splits, np.asarray(extra_splits)))
        subjects = np.concatenate((subjects, np.asarray(extra_subjects)))
        source_files = np.concatenate((source_files, np.asarray(extra_sources)))
        window_end_seconds = np.concatenate(
            (window_end_seconds, np.asarray(extra_end_seconds, dtype=np.float32))
        )
        seconds_to_onset = np.concatenate(
            (seconds_to_onset, np.asarray(extra_seconds_to_onset, dtype=np.float32))
        )
        label_source = np.concatenate((label_source, np.asarray(extra_label_sources)))

    point_values: list[np.ndarray | None] = [None] * len(labels)
    point_masks: list[np.ndarray | None] = [None] * len(labels)
    frame_masks: list[np.ndarray | None] = [None] * len(labels)
    for relative_text in np.unique(source_files):
        relative = str(relative_text)
        radar_path = source_root / Path(relative)
        frames = parse_radhar_text(radar_path, device_id=f"dguha-{radar_path.stem}")
        sample_indices = np.flatnonzero(source_files == relative_text)
        for sample_index in sample_indices:
            end_timestamp = frames[0].timestamp + timedelta(
                seconds=float(window_end_seconds[sample_index])
            )
            sequence = sequence_builder.transform(frames, end_timestamp=end_timestamp)
            if int(sequence.frame_mask.sum()) < max(2, sequence_builder.time_steps // 2):
                raise ValueError(f"insufficient point frames for indexed sample {sample_index}")
            point_values[sample_index] = sequence.values
            point_masks[sample_index] = sequence.point_mask
            frame_masks[sample_index] = sequence.frame_mask
    if any(value is None for value in point_values + point_masks + frame_masks):
        raise RuntimeError("not all DGUHA point windows were rebuilt")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                points=np.stack(point_values).astype(np.float32, copy=False),
                point_mask=np.stack(point_masks).astype(np.bool_, copy=False),
                frame_mask=np.stack(frame_masks).astype(np.bool_, copy=False),
                labels=labels,
                split=splits,
                subject_id=subjects,
                source_files=source_files,
                window_end_seconds=window_end_seconds,
                seconds_to_onset=seconds_to_onset,
                label_source=label_source,
                prediction_horizon_seconds=np.asarray((minimum_lead, maximum_lead), dtype=np.float32),
                sequence_version=np.asarray(POINT_SEQUENCE_VERSION),
                feature_names=np.asarray(POINT_FEATURE_NAMES),
                dataset_mode=np.asarray(POINT_PREDICTION_DATASET_MODE),
                pretrained_encoder_required=np.asarray(True),
                kinect_used_as_model_input=np.asarray(False),
                deployment_eligible=np.asarray(False),
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    split_array = np.asarray(splits)
    summary = PointPredictionExportSummary(
        source_index_file=str(index_path),
        output_file=str(destination),
        output_sha256=_sha256(destination),
        sample_count=len(labels),
        positive_count=int(labels.sum()),
        negative_count=int(len(labels) - labels.sum()),
        train_count=int(np.sum(split_array == "train")),
        validation_count=int(np.sum(split_array == "validation")),
        test_count=int(np.sum(split_array == "test")),
        subject_count=len(np.unique(subjects)),
        same_recording_negative_count=same_recording_negative_count,
        normal_negative_stride=normal_negative_stride,
        ambiguity_buffer_seconds=ambiguity_buffer_seconds,
        minimum_lead_seconds=minimum_lead,
        maximum_lead_seconds=maximum_lead,
        kinect_used_as_model_input=False,
        deployment_eligible=False,
    )
    output_manifest = asdict(summary)
    output_manifest["label_provenance"] = manifest["positive_label_definition"]
    output_manifest["same_recording_negative_definition"] = (
        "windows from the same fall recording ending at least "
        f"{maximum_lead + ambiguity_buffer_seconds:.3f} seconds before descent onset; "
        f"positive horizon ends at {maximum_lead:.3f} seconds before onset"
    )
    output_manifest["known_limitations"] = manifest["known_limitations"]
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _same_recording_negative_endpoints(
    *,
    minimum_window_end: float,
    onset_seconds: float,
    maximum_lead_seconds: float,
    ambiguity_buffer_seconds: float,
) -> np.ndarray:
    if maximum_lead_seconds <= 0 or ambiguity_buffer_seconds < 0:
        raise ValueError("prediction horizon and ambiguity buffer are invalid")
    latest_end = onset_seconds - maximum_lead_seconds - ambiguity_buffer_seconds
    return np.arange(minimum_window_end, latest_end + 1e-9, 0.1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export mmRadPose PointNet/GRU pretraining windows.")
    parser.add_argument("--pointcloud-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--windows-per-sequence", type=int, default=4)
    args = parser.parse_args()
    summary = export_mmradpose_point_pretraining_npz(
        args.pointcloud_directory,
        args.output,
        windows_per_sequence=args.windows_per_sequence,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
