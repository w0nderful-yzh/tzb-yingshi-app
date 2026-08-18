from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from radar_module.contracts import RadarFrame, RadarPoint, Room, SourceMode
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    WINDOW_SIZE_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


DATASET_MODE = "IWR6843_FALL_SEQUENCE_AUXILIARY_V1"
DATASET_ORIGIN = "iwr6843_fall_102"
SUBJECTS = ("Areeb", "Raffay", "Towsif")
DEFAULT_SPLIT_BY_SUBJECT = {
    "Areeb": "train",
    "Raffay": "validation",
    "Towsif": "test",
}
EXPECTED_ACTION_COUNTS = {
    "back": 15,
    "front": 21,
    "side": 15,
    "bow": 15,
    "squat": 15,
    "walk": 21,
}
FALL_ACTIONS = frozenset({"back", "front", "side"})
NONFALL_ACTIONS = frozenset({"bow", "squat", "walk"})
_FILE_PATTERN = re.compile(
    r"^(?P<subject>Areeb|Raffay|Towsif)_"
    r"(?P<action>back|front|side|bow|squat|walk)_(?P<take>[1-9][0-9]*)\.csv$"
)
_REQUIRED_COLUMNS = frozenset(
    {"frame", "DetObj#", "x", "y", "z", "v", "snr", "noise"}
)


@dataclass(frozen=True, slots=True)
class Iwr6843FallExportSummary:
    source_directory: str
    output_file: str
    output_sha256: str
    source_file_count: int
    subject_count: int
    positive_count: int
    negative_count: int
    train_count: int
    validation_count: int
    test_count: int
    minimum_frame_count: int
    maximum_frame_count: int
    feature_version: str
    window_size: int
    input_size: int
    split_policy: str
    deployment_eligible: bool


def export_iwr6843_fall_sequence_npz(
    source_directory: str | Path,
    output_path: str | Path,
    *,
    split_by_subject: Mapping[str, str] | None = None,
    require_complete_release: bool = True,
    extractor: RadarTemporalFeatureExtractorV2 | None = None,
) -> Iwr6843FallExportSummary:
    """Export terminal two-second windows from the public IWR6843 CSV data.

    Labels describe the complete recording only.  A positive sample may contain
    pre-fall, fall and post-fall frames, so this artifact is deliberately marked
    as an auxiliary fall-sequence classification dataset, never a pre-fall
    prediction dataset.
    """

    source_root = _resolve_gathered_data(Path(source_directory).resolve())
    destination = Path(output_path).resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end with .npz")
    split_map = dict(split_by_subject or DEFAULT_SPLIT_BY_SUBJECT)
    _validate_split_map(split_map)
    feature_extractor = extractor or RadarTemporalFeatureExtractorV2()

    source_files = sorted(source_root.glob("*/*.csv"))
    if not source_files:
        raise FileNotFoundError(f"no IWR6843 CSV files found under {source_root}")

    samples: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    subjects: list[str] = []
    activities: list[str] = []
    relative_files: list[str] = []
    frame_counts: list[int] = []
    point_counts: list[int] = []
    qualities: list[str] = []
    action_counts: dict[str, int] = {}

    for source_file in source_files:
        metadata = _metadata_from_path(source_file)
        frames, point_count = parse_iwr6843_fall_csv(source_file)
        if len(frames) < feature_extractor.window_size:
            raise ValueError(
                f"recording is shorter than the feature window: {source_file}"
            )
        window = feature_extractor.transform(frames)
        if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
            raise ValueError(f"recording has insufficient temporal data: {source_file}")

        action = metadata["action"]
        subject = metadata["subject"]
        label = int(action in FALL_ACTIONS)
        samples.append(np.asarray(window.values, dtype=np.float32))
        labels.append(label)
        splits.append(split_map[subject])
        subjects.append(subject)
        activities.append(action)
        relative_files.append(source_file.relative_to(source_root).as_posix())
        frame_counts.append(len(frames))
        point_counts.append(point_count)
        qualities.append(window.data_quality.value)
        action_counts[action] = action_counts.get(action, 0) + 1

    if require_complete_release:
        _validate_complete_release(
            file_count=len(source_files),
            subjects=subjects,
            action_counts=action_counts,
        )

    labels_array = np.asarray(labels, dtype=np.int8)
    splits_array = np.asarray(splits)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=np.stack(samples).astype(np.float32, copy=False),
                labels=labels_array,
                split=splits_array,
                subject_id=np.asarray(subjects),
                action=np.asarray(activities),
                source_files=np.asarray(relative_files),
                frame_count=np.asarray(frame_counts, dtype=np.int16),
                point_count=np.asarray(point_counts, dtype=np.int32),
                data_quality=np.asarray(qualities),
                label_source=np.asarray(
                    ["recording_level_fall_or_nonfall"] * len(samples)
                ),
                dataset_origin=np.asarray([DATASET_ORIGIN] * len(samples)),
                feature_version=np.asarray(FEATURE_VERSION_V2),
                feature_names=np.asarray(FEATURE_NAMES_V2),
                dataset_mode=np.asarray(DATASET_MODE),
                task_type=np.asarray("fall_sequence_auxiliary"),
                source_complete=np.asarray(require_complete_release),
                deployment_eligible=np.asarray(False),
                positive_samples_available=np.asarray(True),
                positive_label_definition=np.asarray(
                    "terminal 2 s window from a recording labelled fall; onset, "
                    "impact and pre-fall horizon are not annotated"
                ),
                dataset_description=np.asarray(
                    "Public three-subject IWR6843 fall/non-fall point-cloud "
                    "recordings converted through radar_features_v2"
                ),
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    summary = Iwr6843FallExportSummary(
        source_directory=str(source_root),
        output_file=str(destination),
        output_sha256=_sha256(destination),
        source_file_count=len(source_files),
        subject_count=len(set(subjects)),
        positive_count=int(labels_array.sum()),
        negative_count=int(len(labels_array) - labels_array.sum()),
        train_count=int(np.sum(splits_array == "train")),
        validation_count=int(np.sum(splits_array == "validation")),
        test_count=int(np.sum(splits_array == "test")),
        minimum_frame_count=min(frame_counts),
        maximum_frame_count=max(frame_counts),
        feature_version=FEATURE_VERSION_V2,
        window_size=WINDOW_SIZE_V2,
        input_size=len(FEATURE_NAMES_V2),
        split_policy="subject_disjoint_train_validation_test",
        deployment_eligible=False,
    )
    _write_manifest(
        destination,
        summary,
        split_map=split_map,
        action_counts=action_counts,
    )
    return summary


def parse_iwr6843_fall_csv(path: str | Path) -> tuple[tuple[RadarFrame, ...], int]:
    source = Path(path).resolve()
    metadata = _metadata_from_path(source)
    rows_by_frame: dict[int, list[RadarPoint]] = {}
    object_ids_by_frame: dict[int, set[int]] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = frozenset(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS.difference(columns))
        if missing:
            raise ValueError(f"IWR6843 CSV columns are incomplete: {missing}")
        for row_number, row in enumerate(reader, start=2):
            try:
                frame_index = int(row["frame"])
                object_id = int(row["DetObj#"])
                values = tuple(
                    float(row[name]) for name in ("x", "y", "z", "v", "snr", "noise")
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid numeric value at {source}:{row_number}") from error
            if frame_index < 0 or object_id < 0 or not all(map(math.isfinite, values)):
                raise ValueError(f"invalid point at {source}:{row_number}")
            seen_ids = object_ids_by_frame.setdefault(frame_index, set())
            if object_id in seen_ids:
                raise ValueError(
                    f"duplicate DetObj# {object_id} in frame {frame_index}: {source}"
                )
            seen_ids.add(object_id)
            x, y, z, velocity, snr, _noise = values
            rows_by_frame.setdefault(frame_index, []).append(
                RadarPoint(x=x, y=y, z=z, velocity=velocity, snr=snr)
            )

    if not rows_by_frame:
        raise ValueError(f"IWR6843 CSV contains no points: {source}")
    frame_indices = sorted(rows_by_frame)
    if frame_indices != list(range(frame_indices[-1] + 1)):
        raise ValueError(f"IWR6843 frame numbers are not contiguous from zero: {source}")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frames = tuple(
        RadarFrame(
            timestamp=start + timedelta(seconds=frame_index / 10.0),
            device_id=f"{DATASET_ORIGIN}-{metadata['subject']}",
            room=Room.BATHROOM,
            source_mode=SourceMode.REPLAY,
            points=tuple(rows_by_frame[frame_index]),
        )
        for frame_index in frame_indices
    )
    return frames, sum(len(points) for points in rows_by_frame.values())


def _resolve_gathered_data(path: Path) -> Path:
    candidate = path / "GatheredData"
    root = candidate if candidate.is_dir() else path
    if not (root / "Fall").is_dir() or not (root / "Not").is_dir():
        raise FileNotFoundError(
            "expected GatheredData/Fall and GatheredData/Not directories"
        )
    return root


def _metadata_from_path(path: Path) -> dict[str, str | int]:
    match = _FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected IWR6843 recording name: {path.name}")
    subject = match.group("subject")
    action = match.group("action")
    expected_directory = "Fall" if action in FALL_ACTIONS else "Not"
    if path.parent.name != expected_directory:
        raise ValueError(f"recording action/directory mismatch: {path}")
    return {"subject": subject, "action": action, "take": int(match.group("take"))}


def _validate_split_map(split_map: Mapping[str, str]) -> None:
    if set(split_map) != set(SUBJECTS):
        raise ValueError("split_by_subject must contain all and only the three subjects")
    if set(split_map.values()) != {"train", "validation", "test"}:
        raise ValueError("the three subjects must map one-to-one to train/validation/test")


def _validate_complete_release(
    *, file_count: int, subjects: list[str], action_counts: Mapping[str, int]
) -> None:
    if file_count != 102:
        raise ValueError(f"expected 102 recordings, found {file_count}")
    if set(subjects) != set(SUBJECTS):
        raise ValueError("the complete release must contain all three subjects")
    if dict(action_counts) != EXPECTED_ACTION_COUNTS:
        raise ValueError(
            f"unexpected action counts: {dict(sorted(action_counts.items()))}"
        )
    subject_counts = {subject: subjects.count(subject) for subject in SUBJECTS}
    if set(subject_counts.values()) != {34}:
        raise ValueError(f"unexpected subject counts: {subject_counts}")


def _write_manifest(
    path: Path,
    summary: Iwr6843FallExportSummary,
    *,
    split_map: Mapping[str, str],
    action_counts: Mapping[str, int],
) -> None:
    payload = asdict(summary)
    payload["dataset_mode"] = DATASET_MODE
    payload["dataset_origin"] = DATASET_ORIGIN
    payload["split_by_subject"] = dict(split_map)
    payload["action_counts"] = dict(sorted(action_counts.items()))
    payload["label_scope"] = "recording_level"
    payload["license_status"] = "no explicit LICENSE file found in downloaded repository"
    payload["ignored_source_fields"] = ["noise"]
    payload["known_limitations"] = [
        "three young subjects and 102 short controlled recordings only",
        "fall onset, loss of balance and impact time are not annotated",
        "positive terminal windows can contain pre-fall, fall and post-fall frames",
        "recordings contain 23-25 frames, so only one terminal 2 s sample is exported",
        "this artifact cannot measure or train advance-prediction lead time",
        "research-only and non-deployable",
    ]
    path.with_suffix(".manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export subject-disjoint IWR6843 fall-sequence auxiliary data."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-subject", default="Areeb", choices=SUBJECTS)
    parser.add_argument("--validation-subject", default="Raffay", choices=SUBJECTS)
    parser.add_argument("--test-subject", default="Towsif", choices=SUBJECTS)
    args = parser.parse_args()
    split_map = {
        args.train_subject: "train",
        args.validation_subject: "validation",
        args.test_subject: "test",
    }
    summary = export_iwr6843_fall_sequence_npz(
        args.source,
        args.output,
        split_by_subject=split_map,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
