from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from radar_module.contracts import RadarFrame, RadarPoint, Room, SourceMode
from radar_module.dataset.mmfall_converter import convert_mmfall_raw_frame
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


FALL_PREFIXES = ("bf", "ff", "lf", "rf")
NEGATIVE_PREFIXES = ("b", "c", "j", "sf")
CATEGORY_NAMES = {
    "bf": "backward_fall",
    "ff": "forward_fall",
    "lf": "left_fall",
    "rf": "right_fall",
    "b": "bending",
    "c": "crouching",
    "j": "jump",
    "sf": "sitting_on_floor",
}

# Split whole recordings, never neighbouring windows.  The public DS2 release
# has only ten fall recordings, so validation and test necessarily cover fewer
# fall directions than training.  This limitation is emitted in the manifest.
DEFAULT_SPLIT_BY_FILE = {
    "DS2_bf_01.npy": "train",
    "DS2_bf_02.npy": "train",
    "DS2_ff_01.npy": "train",
    "DS2_ff_02.npy": "train",
    "DS2_lf_01.npy": "train",
    "DS2_rf_01.npy": "train",
    "DS2_bf_03.npy": "validation",
    "DS2_ff_03.npy": "validation",
    "DS2_lf_02.npy": "test",
    "DS2_rf_02.npy": "test",
    "DS2_b_01.npy": "train",
    "DS2_c_01.npy": "train",
    "DS2_c_02.npy": "train",
    "DS2_c_03.npy": "train",
    "DS2_j_01.npy": "train",
    "DS2_sf_01.npy": "train",
    "DS2_sf_02.npy": "train",
    "DS2_sf_03.npy": "train",
    "DS2_c_04.npy": "validation",
    "DS2_j_02.npy": "validation",
    "DS2_sf_04.npy": "validation",
    "DS2_c_05.npy": "test",
    "DS2_j_03.npy": "test",
    "DS2_sf_05.npy": "test",
}


@dataclass(frozen=True, slots=True)
class MmFallResearchExportSummary:
    source_directory: str
    output_file: str
    output_sha256: str
    sample_count: int
    positive_count: int
    negative_count: int
    train_count: int
    validation_count: int
    test_count: int
    source_file_count: int
    skipped_quality_count: int
    feature_version: str
    window_size: int
    input_size: int
    positive_start_seconds_before_anchor: float
    positive_end_seconds_before_anchor: float
    negative_stride_seconds: float
    label_status: str
    deployment_eligible: bool


def export_mmfall_ds2_research_npz(
    ds2_directory: str | Path,
    output_path: str | Path,
    *,
    allow_weak_supervision: bool = False,
    positive_start_seconds_before_anchor: float = 1.5,
    positive_end_seconds_before_anchor: float = 0.2,
    positive_stride_seconds: float = 0.1,
    negative_stride_seconds: float = 0.5,
    max_negative_windows_per_file: int = 250,
    extractor: RadarTemporalFeatureExtractorV2 | None = None,
) -> MmFallResearchExportSummary:
    """Export DS2 prediction windows with explicit weak-label provenance.

    The CSV value is an official ``fall motion happens`` frame anchor, not a
    verified impact time.  Calling code must opt in to this limitation.  The
    resulting dataset is research-only and is not accepted by the production
    checkpoint loader.
    """

    if not allow_weak_supervision:
        raise ValueError("allow_weak_supervision=True is required")
    if not (
        positive_start_seconds_before_anchor
        > positive_end_seconds_before_anchor
        >= 0.0
    ):
        raise ValueError("positive anchor offsets are invalid")
    if positive_stride_seconds <= 0 or negative_stride_seconds <= 0:
        raise ValueError("sample strides must be positive")
    if max_negative_windows_per_file <= 0:
        raise ValueError("max_negative_windows_per_file must be positive")

    source_root = Path(ds2_directory).resolve()
    destination = Path(output_path).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"DS2 directory does not exist: {source_root}")
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end with .npz")

    missing = sorted(
        file_name
        for file_name in DEFAULT_SPLIT_BY_FILE
        if not (source_root / file_name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"DS2 component files are missing: {missing}")

    feature_extractor = extractor or RadarTemporalFeatureExtractorV2()
    samples: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    source_files: list[str] = []
    source_categories: list[str] = []
    window_end_frames: list[int] = []
    anchor_frames: list[int] = []
    seconds_to_anchor: list[float] = []
    label_sources: list[str] = []
    label_confidences: list[str] = []
    qualities: list[str] = []
    skipped_quality = 0

    for file_name, split in DEFAULT_SPLIT_BY_FILE.items():
        source_path = source_root / file_name
        category = _category_from_name(file_name)
        raw_frames = np.load(source_path, allow_pickle=True)
        frames = _to_radar_frames(raw_frames, device_id=f"mmfall-{file_name[:-4]}")

        if category in FALL_PREFIXES:
            anchors = _load_anchor_frames(source_path.with_suffix(".csv"))
            candidate_pairs = _positive_window_ends(
                anchors,
                frame_count=len(frames),
                frame_rate_hz=feature_extractor.target_sample_rate_hz,
                start_before=positive_start_seconds_before_anchor,
                end_before=positive_end_seconds_before_anchor,
                stride_seconds=positive_stride_seconds,
            )
            for end_frame, anchor_frame in candidate_pairs:
                window = _extract_at_frame(frames, end_frame, feature_extractor)
                if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
                    skipped_quality += 1
                    continue
                samples.append(np.asarray(window.values, dtype=np.float32))
                labels.append(1)
                splits.append(split)
                source_files.append(file_name)
                source_categories.append(CATEGORY_NAMES[category])
                window_end_frames.append(end_frame)
                anchor_frames.append(anchor_frame)
                seconds_to_anchor.append(
                    (anchor_frame - end_frame)
                    / feature_extractor.target_sample_rate_hz
                )
                label_sources.append("mmfall_official_anchor_preceding_window")
                label_confidences.append("weak_anchor_not_verified_impact")
                qualities.append(window.data_quality.value)
        else:
            candidate_ends = _negative_window_ends(
                frame_count=len(frames),
                frame_rate_hz=feature_extractor.target_sample_rate_hz,
                history_seconds=feature_extractor.history_seconds,
                stride_seconds=negative_stride_seconds,
                maximum=max_negative_windows_per_file,
            )
            for end_frame in candidate_ends:
                window = _extract_at_frame(frames, end_frame, feature_extractor)
                if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
                    skipped_quality += 1
                    continue
                samples.append(np.asarray(window.values, dtype=np.float32))
                labels.append(0)
                splits.append(split)
                source_files.append(file_name)
                source_categories.append(CATEGORY_NAMES[category])
                window_end_frames.append(end_frame)
                anchor_frames.append(-1)
                seconds_to_anchor.append(math.nan)
                label_sources.append("mmfall_recording_activity_label")
                label_confidences.append("recording_level_negative")
                qualities.append(window.data_quality.value)

    if not samples:
        raise ValueError("no research samples were produced")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=np.stack(samples).astype(np.float32, copy=False),
                labels=np.asarray(labels, dtype=np.int8),
                split=np.asarray(splits),
                source_files=np.asarray(source_files),
                source_categories=np.asarray(source_categories),
                window_end_frames=np.asarray(window_end_frames, dtype=np.int32),
                anchor_frames=np.asarray(anchor_frames, dtype=np.int32),
                seconds_to_anchor=np.asarray(seconds_to_anchor, dtype=np.float32),
                label_source=np.asarray(label_sources),
                label_confidence=np.asarray(label_confidences),
                data_quality=np.asarray(qualities),
                feature_version=np.asarray(FEATURE_VERSION_V2),
                feature_names=np.asarray(FEATURE_NAMES_V2),
                dataset_mode=np.asarray("RESEARCH_WEAK_SUPERVISION"),
                deployment_eligible=np.asarray(False),
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    split_array = np.asarray(splits)
    summary = MmFallResearchExportSummary(
        source_directory=str(source_root),
        output_file=str(destination),
        output_sha256=_sha256(destination),
        sample_count=len(samples),
        positive_count=sum(labels),
        negative_count=len(labels) - sum(labels),
        train_count=int(np.sum(split_array == "train")),
        validation_count=int(np.sum(split_array == "validation")),
        test_count=int(np.sum(split_array == "test")),
        source_file_count=len(DEFAULT_SPLIT_BY_FILE),
        skipped_quality_count=skipped_quality,
        feature_version=FEATURE_VERSION_V2,
        window_size=feature_extractor.window_size,
        input_size=len(FEATURE_NAMES_V2),
        positive_start_seconds_before_anchor=(
            positive_start_seconds_before_anchor
        ),
        positive_end_seconds_before_anchor=positive_end_seconds_before_anchor,
        negative_stride_seconds=negative_stride_seconds,
        label_status=(
            "positive windows are relative to unverified fall-motion anchors; "
            "negative labels are recording-level activities"
        ),
        deployment_eligible=False,
    )
    _write_manifest(destination, summary)
    return summary


def _category_from_name(file_name: str) -> str:
    stem = Path(file_name).stem
    parts = stem.split("_")
    if len(parts) != 3 or parts[0] != "DS2":
        raise ValueError(f"unexpected DS2 component name: {file_name}")
    category = parts[1]
    if category not in CATEGORY_NAMES:
        raise ValueError(f"unsupported DS2 category: {category}")
    return category


def _to_radar_frames(raw_frames: np.ndarray, *, device_id: str) -> tuple[RadarFrame, ...]:
    if raw_frames.ndim != 1:
        raise ValueError(f"expected one-dimensional object array, got {raw_frames.shape}")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    converted: list[RadarFrame] = []
    for index, raw_frame in enumerate(raw_frames):
        raw_points, _ = convert_mmfall_raw_frame(
            raw_frame,
            tilt_angle_deg=-10.0,
            radar_height_m=1.8,
        )
        points = tuple(
            RadarPoint(
                x=float(point["x"]),
                y=float(point["y"]),
                z=float(point["z"]),
                velocity=float(point["velocity"]),
                snr=float(point["snr"]) if "snr" in point else None,
                track_id=int(point["track_id"]) if "track_id" in point else None,
            )
            for point in raw_points
        )
        converted.append(
            RadarFrame(
                timestamp=start + timedelta(seconds=index / 10.0),
                device_id=device_id,
                room=Room.LIVING_ROOM,
                source_mode=SourceMode.REPLAY,
                points=points,
            )
        )
    return tuple(converted)


def _load_anchor_frames(path: Path) -> tuple[int, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"fall anchor CSV is missing: {path}")
    values = np.atleast_1d(np.genfromtxt(path, delimiter=",", dtype=np.int64))
    anchors = tuple(int(value) for value in values)
    if not anchors or any(value < 0 for value in anchors):
        raise ValueError(f"invalid fall anchors in {path}")
    return anchors


def _positive_window_ends(
    anchors: tuple[int, ...],
    *,
    frame_count: int,
    frame_rate_hz: float,
    start_before: float,
    end_before: float,
    stride_seconds: float,
) -> tuple[tuple[int, int], ...]:
    offsets = np.arange(start_before, end_before - 1e-9, -stride_seconds)
    minimum_end = int(round(1.9 * frame_rate_hz))
    pairs: list[tuple[int, int]] = []
    for anchor in anchors:
        for offset in offsets:
            end_frame = anchor - int(round(float(offset) * frame_rate_hz))
            if minimum_end <= end_frame < min(anchor, frame_count):
                pairs.append((end_frame, anchor))
    return tuple(pairs)


def _negative_window_ends(
    *,
    frame_count: int,
    frame_rate_hz: float,
    history_seconds: float,
    stride_seconds: float,
    maximum: int,
) -> tuple[int, ...]:
    first = int(round((history_seconds - 0.1) * frame_rate_hz))
    step = max(1, int(round(stride_seconds * frame_rate_hz)))
    candidates = np.arange(first, frame_count, step, dtype=np.int64)
    if len(candidates) > maximum:
        indices = np.linspace(0, len(candidates) - 1, maximum, dtype=np.int64)
        candidates = candidates[indices]
    return tuple(int(value) for value in candidates)


def _extract_at_frame(
    frames: tuple[RadarFrame, ...],
    end_frame: int,
    extractor: RadarTemporalFeatureExtractorV2,
):
    start_frame = max(0, end_frame - extractor.window_size + 1)
    return extractor.transform(
        frames[start_frame : end_frame + 1],
        end_timestamp=frames[end_frame].timestamp,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, summary: MmFallResearchExportSummary) -> None:
    payload = asdict(summary)
    payload["split_policy"] = "whole_recording_fixed_split"
    payload["known_limitations"] = [
        "DS2 anchors are fall-motion anchors, not verified impact times",
        "DS2 has no participant identity metadata for subject-independent split",
        "validation contains forward/backward falls; test contains side falls",
        "same public dataset and room cannot demonstrate deployment generalization",
    ]
    path.with_suffix(".manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export weak-supervision mmFall DS2 v2 windows.")
    parser.add_argument("--ds2-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-weak-supervision", action="store_true")
    args = parser.parse_args()
    summary = export_mmfall_ds2_research_npz(
        args.ds2_directory,
        args.output,
        allow_weak_supervision=args.allow_weak_supervision,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
