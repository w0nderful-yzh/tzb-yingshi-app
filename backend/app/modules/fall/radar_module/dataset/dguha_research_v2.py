from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from radar_module.contracts import RadarFrame
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


DGUHA_DATASET_MODE = "DGUHA_SKELETON_PSEUDOLABEL_RESEARCH_V2"
DGUHA_FALL_ACTION = "5_falling_forward"
DGUHA_POSITIVE_ANCHORS = {"descent_onset", "near_floor_level_reached"}
DGUHA_ACTIONS = {
    "1_Running": "RUNNING",
    "2_Jumping": "JUMP",
    "3_Sit_down_and_stand_up": "SIT_STAND",
    "4_Both_upper_limb_extension": "UPPER_LIMB_EXTENSION",
    DGUHA_FALL_ACTION: "FORWARD_FALL",
    "6_Right_limb_extension": "RIGHT_LIMB_EXTENSION",
    "7_Left_limb_extension": "LEFT_LIMB_EXTENSION",
}

# The author Training/Test folders leak M_012 across both partitions. Keep all
# recordings from a person in exactly one project partition instead.
DGUHA_SPLIT_BY_SUBJECT = {
    **{
        subject: "train"
        for subject in (
            "F_001",
            "F_002",
            "F_003",
            "F_004",
            "F_005",
            "M_001",
            "M_002",
            "M_004",
            "M_005",
            "M_006",
            "M_007",
            "M_008",
            "M_009",
        )
    },
    "M_010": "validation",
    "M_011": "validation",
    "M_012": "validation",
    "F_006": "test",
    "M_013": "test",
    "M_014": "test",
}


@dataclass(frozen=True, slots=True)
class DguhaKinectFrame:
    timestamp: datetime
    points_mm: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("Kinect timestamp must include a timezone offset")
        points = np.asarray(self.points_mm, dtype=np.float64).copy()
        if points.shape != (25, 3) or not np.isfinite(points).all():
            raise ValueError("Kinect frame must contain 25 finite xyz points")
        points.setflags(write=False)
        object.__setattr__(self, "points_mm", points)


@dataclass(frozen=True, slots=True)
class DguhaFallEvent:
    source_file: str
    subject_id: str
    author_split: str
    project_split: str
    radar_start: datetime
    radar_end: datetime
    kinect_start: datetime
    kinect_end: datetime
    skeleton_frame_count: int
    vertical_drop_m: float
    baseline_std_m: float
    final_std_m: float
    peak_descent_speed_mps: float
    descent_onset: datetime | None
    rapid_descent_onset: datetime | None
    near_floor_level_reached: datetime | None
    eligible_for_prediction_windows: bool
    exclusion_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DguhaResearchExportSummary:
    source_directory: str
    output_file: str
    output_sha256: str
    events_file: str
    source_file_count: int
    subject_count: int
    fall_recording_count: int
    eligible_fall_recording_count: int
    excluded_fall_recording_count: int
    sample_count: int
    positive_count: int
    negative_count: int
    train_count: int
    validation_count: int
    test_count: int
    skipped_quality_count: int
    feature_version: str
    window_size: int
    input_size: int
    minimum_lead_seconds: float
    maximum_lead_seconds: float
    positive_anchor: str
    minimum_pre_descent_margin_seconds: float
    kinect_used_as_model_input: bool
    deployment_eligible: bool


def parse_dguha_kinect(input_path: str | Path) -> tuple[DguhaKinectFrame, ...]:
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"DGUHA Kinect file does not exist: {source}")
    text = source.read_text(encoding="utf-8", errors="strict")
    frames: list[DguhaKinectFrame] = []
    for chunk_index, chunk in enumerate(text.split("ply\n")[1:], start=1):
        timestamp_match = re.search(r"comment timestamp:\s*(\d+)", chunk)
        if timestamp_match is None or "end_header\n" not in chunk:
            raise ValueError(f"invalid Kinect PLY chunk {chunk_index}: {source}")
        point_lines = chunk.split("end_header\n", 1)[1].strip().splitlines()[:25]
        points: list[tuple[float, float, float]] = []
        for line_number, line in enumerate(point_lines, start=1):
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(
                    f"invalid Kinect point {line_number} in chunk {chunk_index}"
                )
            points.append(tuple(float(value) for value in fields[:3]))
        timestamp_ns = int(timestamp_match.group(1))
        frames.append(
            DguhaKinectFrame(
                timestamp=datetime.fromtimestamp(
                    timestamp_ns / 1_000_000_000.0, tz=timezone.utc
                ),
                points_mm=np.asarray(points, dtype=np.float64),
            )
        )
    if not frames:
        raise ValueError(f"DGUHA Kinect file contains no frames: {source}")
    for previous, current in zip(frames[:-1], frames[1:]):
        if current.timestamp <= previous.timestamp:
            raise ValueError("DGUHA Kinect timestamps must be strictly increasing")
    return tuple(frames)


def derive_dguha_fall_event(
    *,
    source_file: str,
    subject_id: str,
    author_split: str,
    radar_start: datetime,
    radar_end: datetime,
    kinect_frames: tuple[DguhaKinectFrame, ...],
    required_history_seconds: float = 1.9,
    minimum_lead_seconds: float = 0.1,
) -> DguhaFallEvent:
    if subject_id not in DGUHA_SPLIT_BY_SUBJECT:
        raise ValueError(f"unknown DGUHA subject: {subject_id}")
    if len(kinect_frames) < 20:
        raise ValueError("at least 20 Kinect frames are required")
    timestamps = np.asarray(
        [frame.timestamp.timestamp() for frame in kinect_frames], dtype=np.float64
    )
    relative_seconds = timestamps - timestamps[0]
    median_z_m = np.asarray(
        [np.median(frame.points_mm[:, 2]) / 1000.0 for frame in kinect_frames],
        dtype=np.float64,
    )
    baseline_mask = relative_seconds <= 0.75
    final_mask = relative_seconds >= max(relative_seconds[-1] - 3.0, 0.0)
    baseline = float(np.median(median_z_m[baseline_mask]))
    final = float(np.median(median_z_m[final_mask]))
    vertical_drop = abs(final - baseline)
    direction = 1.0 if final >= baseline else -1.0
    progress = direction * (median_z_m - baseline)
    smooth = np.convolve(
        np.pad(progress, (2, 2), mode="edge"),
        np.ones(5, dtype=np.float64) / 5.0,
        mode="valid",
    )
    descent_speed = np.gradient(smooth, timestamps)
    onset_threshold = max(0.08, 0.10 * vertical_drop)
    onset_index = _first_sustained(smooth >= onset_threshold, 3)
    rapid_index = None
    if onset_index is not None:
        local_index = _first_sustained(
            descent_speed[onset_index:] >= 0.25,
            2,
        )
        if local_index is not None:
            rapid_index = onset_index + local_index
    floor_index = None
    if onset_index is not None:
        local_index = _first_sustained(
            smooth[onset_index:] >= 0.85 * vertical_drop,
            3,
        )
        if local_index is not None:
            floor_index = onset_index + local_index

    baseline_std = float(np.std(median_z_m[baseline_mask]))
    final_std = float(np.std(median_z_m[final_mask]))
    peak_speed = float(np.max(descent_speed))
    onset = _timestamp_at(kinect_frames, onset_index)
    rapid = _timestamp_at(kinect_frames, rapid_index)
    floor = _timestamp_at(kinect_frames, floor_index)
    exclusion_reasons: list[str] = []
    if vertical_drop < 0.30:
        exclusion_reasons.append("vertical_drop_below_0.30m")
    if baseline_std > 0.10:
        exclusion_reasons.append("unstable_initial_skeleton_level")
    if final_std > 0.10:
        exclusion_reasons.append("unstable_final_skeleton_level")
    if onset is None:
        exclusion_reasons.append("descent_onset_not_found")
    if rapid is None:
        exclusion_reasons.append("rapid_descent_not_found")
    if onset is not None:
        available = (onset - radar_start).total_seconds()
        if available < required_history_seconds + minimum_lead_seconds:
            exclusion_reasons.append("insufficient_prefall_radar_history")
        if onset > radar_end:
            exclusion_reasons.append("descent_onset_outside_radar_recording")

    return DguhaFallEvent(
        source_file=source_file,
        subject_id=subject_id,
        author_split=author_split,
        project_split=DGUHA_SPLIT_BY_SUBJECT[subject_id],
        radar_start=radar_start,
        radar_end=radar_end,
        kinect_start=kinect_frames[0].timestamp,
        kinect_end=kinect_frames[-1].timestamp,
        skeleton_frame_count=len(kinect_frames),
        vertical_drop_m=vertical_drop,
        baseline_std_m=baseline_std,
        final_std_m=final_std,
        peak_descent_speed_mps=peak_speed,
        descent_onset=onset,
        rapid_descent_onset=rapid,
        near_floor_level_reached=floor,
        eligible_for_prediction_windows=not exclusion_reasons,
        exclusion_reasons=tuple(exclusion_reasons),
    )


def export_dguha_research_npz(
    data_root: str | Path,
    output_path: str | Path,
    *,
    allow_skeleton_pseudolabels: bool = False,
    minimum_lead_seconds: float = 0.1,
    maximum_lead_seconds: float = 0.6,
    positive_stride_seconds: float = 0.1,
    negative_stride_seconds: float = 1.0,
    max_negative_windows_per_recording: int = 10,
    positive_anchor: str = "descent_onset",
    minimum_pre_descent_margin_seconds: float = 0.1,
    extractor: RadarTemporalFeatureExtractorV2 | None = None,
) -> DguhaResearchExportSummary:
    if not allow_skeleton_pseudolabels:
        raise ValueError("allow_skeleton_pseudolabels=True is required")
    if not 0 < minimum_lead_seconds <= maximum_lead_seconds:
        raise ValueError("prediction lead interval is invalid")
    if positive_stride_seconds <= 0 or negative_stride_seconds <= 0:
        raise ValueError("sample strides must be positive")
    if max_negative_windows_per_recording <= 0:
        raise ValueError("max_negative_windows_per_recording must be positive")
    if positive_anchor not in DGUHA_POSITIVE_ANCHORS:
        raise ValueError(f"unsupported positive_anchor: {positive_anchor}")
    if minimum_pre_descent_margin_seconds < 0:
        raise ValueError("minimum pre-descent margin must not be negative")

    source_root = Path(data_root).resolve()
    destination = Path(output_path).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"DGUHA data root does not exist: {source_root}")
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end with .npz")
    radar_files = sorted(source_root.glob("*/*/radar/*.txt"))
    if not radar_files:
        raise ValueError("no DGUHA radar files were found")

    feature_extractor = extractor or RadarTemporalFeatureExtractorV2()
    required_history_seconds = (feature_extractor.window_size - 1) / (
        feature_extractor.target_sample_rate_hz
    )
    features: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    actions: list[str] = []
    subject_ids: list[str] = []
    source_files: list[str] = []
    window_end_seconds: list[float] = []
    seconds_to_onset: list[float] = []
    seconds_to_anchor: list[float] = []
    label_sources: list[str] = []
    label_confidences: list[str] = []
    qualities: list[str] = []
    events: list[DguhaFallEvent] = []
    skipped_quality = 0

    for radar_path in radar_files:
        relative = radar_path.relative_to(source_root)
        if len(relative.parts) != 4:
            raise ValueError(f"unexpected DGUHA path: {relative.as_posix()}")
        author_split, action, modality, file_name = relative.parts
        if modality != "radar" or action not in DGUHA_ACTIONS:
            raise ValueError(f"unexpected DGUHA path: {relative.as_posix()}")
        subject_id = _subject_from_file_name(file_name)
        project_split = DGUHA_SPLIT_BY_SUBJECT[subject_id]
        kinect_path = radar_path.parent.parent / "kinect" / file_name
        if not kinect_path.is_file():
            raise FileNotFoundError(f"paired Kinect file is missing: {kinect_path}")
        frames = parse_radhar_text(radar_path, device_id=f"dguha-{file_name[:-4]}")
        radar_start = frames[0].timestamp
        radar_end = frames[-1].timestamp
        frame_epochs = np.asarray(
            [frame.timestamp.timestamp() for frame in frames], dtype=np.float64
        )

        if action == DGUHA_FALL_ACTION:
            kinect_frames = parse_dguha_kinect(kinect_path)
            event = derive_dguha_fall_event(
                source_file=relative.as_posix(),
                subject_id=subject_id,
                author_split=author_split,
                radar_start=radar_start,
                radar_end=radar_end,
                kinect_frames=kinect_frames,
                required_history_seconds=required_history_seconds,
                minimum_lead_seconds=(
                    minimum_lead_seconds
                    if positive_anchor == "descent_onset"
                    else minimum_pre_descent_margin_seconds
                ),
            )
            events.append(event)
            if not event.eligible_for_prediction_windows:
                continue
            assert event.descent_onset is not None
            anchor_timestamp = (
                event.descent_onset
                if positive_anchor == "descent_onset"
                else event.near_floor_level_reached
            )
            if anchor_timestamp is None:
                continue
            leads = np.arange(
                maximum_lead_seconds,
                minimum_lead_seconds - 1e-9,
                -positive_stride_seconds,
            )
            for lead_seconds in leads:
                end_timestamp = anchor_timestamp - timedelta(
                    seconds=float(lead_seconds)
                )
                if (
                    end_timestamp - radar_start
                ).total_seconds() < required_history_seconds:
                    continue
                if end_timestamp > event.descent_onset - timedelta(
                    seconds=minimum_pre_descent_margin_seconds
                ):
                    continue
                window = _extract_window(
                    frames,
                    frame_epochs,
                    end_timestamp,
                    feature_extractor,
                )
                if window is None:
                    skipped_quality += 1
                    continue
                features.append(np.asarray(window.values, dtype=np.float32))
                labels.append(1)
                splits.append(project_split)
                actions.append(DGUHA_ACTIONS[action])
                subject_ids.append(subject_id)
                source_files.append(relative.as_posix())
                window_end_seconds.append(
                    (end_timestamp - radar_start).total_seconds()
                )
                seconds_to_onset.append(
                    (event.descent_onset - end_timestamp).total_seconds()
                )
                seconds_to_anchor.append(float(lead_seconds))
                label_sources.append(f"dguha_kinect_skeleton_{positive_anchor}")
                label_confidences.append("weak_skeleton_pseudolabel")
                qualities.append(window.data_quality.value)
        else:
            duration = (radar_end - radar_start).total_seconds()
            candidates = np.arange(
                required_history_seconds,
                duration,
                negative_stride_seconds,
            )
            if len(candidates) > max_negative_windows_per_recording:
                selected = np.linspace(
                    0,
                    len(candidates) - 1,
                    max_negative_windows_per_recording,
                    dtype=np.int64,
                )
                candidates = candidates[selected]
            for end_seconds in candidates:
                end_timestamp = radar_start + timedelta(seconds=float(end_seconds))
                window = _extract_window(
                    frames,
                    frame_epochs,
                    end_timestamp,
                    feature_extractor,
                )
                if window is None:
                    skipped_quality += 1
                    continue
                features.append(np.asarray(window.values, dtype=np.float32))
                labels.append(0)
                splits.append(project_split)
                actions.append(DGUHA_ACTIONS[action])
                subject_ids.append(subject_id)
                source_files.append(relative.as_posix())
                window_end_seconds.append(float(end_seconds))
                seconds_to_onset.append(math.nan)
                seconds_to_anchor.append(math.nan)
                label_sources.append("dguha_recording_activity_label")
                label_confidences.append("recording_level_negative")
                qualities.append(window.data_quality.value)

    if not features or not any(labels) or all(labels):
        raise ValueError("DGUHA export must contain both positive and negative samples")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=np.stack(features).astype(np.float32, copy=False),
                labels=np.asarray(labels, dtype=np.int8),
                split=np.asarray(splits),
                action=np.asarray(actions),
                subject_id=np.asarray(subject_ids),
                source_files=np.asarray(source_files),
                window_end_seconds=np.asarray(window_end_seconds, dtype=np.float32),
                seconds_to_onset=np.asarray(seconds_to_onset, dtype=np.float32),
                seconds_to_anchor=np.asarray(seconds_to_anchor, dtype=np.float32),
                label_source=np.asarray(label_sources),
                label_confidence=np.asarray(label_confidences),
                data_quality=np.asarray(qualities),
                feature_version=np.asarray(FEATURE_VERSION_V2),
                feature_names=np.asarray(FEATURE_NAMES_V2),
                dataset_mode=np.asarray(DGUHA_DATASET_MODE),
                positive_anchor=np.asarray(positive_anchor),
                prediction_horizon_seconds=np.asarray(
                    (minimum_lead_seconds, maximum_lead_seconds),
                    dtype=np.float32,
                ),
                minimum_pre_descent_margin_seconds=np.asarray(
                    minimum_pre_descent_margin_seconds, dtype=np.float32
                ),
                positive_label_definition=np.asarray(
                    _positive_label_definition(
                        positive_anchor,
                        minimum_lead_seconds,
                        maximum_lead_seconds,
                        minimum_pre_descent_margin_seconds,
                    )
                ),
                dataset_description=np.asarray(
                    "DGUHA radar-only features with Kinect skeleton used "
                    "only for offline weak-label timing"
                ),
                kinect_used_as_model_input=np.asarray(False),
                deployment_eligible=np.asarray(False),
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    events_path = destination.with_suffix(".events.json")
    events_path.write_text(
        json.dumps(
            [_event_payload(event) for event in events],
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    labels_array = np.asarray(labels, dtype=np.int8)
    splits_array = np.asarray(splits)
    summary = DguhaResearchExportSummary(
        source_directory=str(source_root),
        output_file=str(destination),
        output_sha256=_sha256(destination),
        events_file=str(events_path),
        source_file_count=len(radar_files),
        subject_count=len(set(subject_ids)),
        fall_recording_count=len(events),
        eligible_fall_recording_count=sum(
            event.eligible_for_prediction_windows for event in events
        ),
        excluded_fall_recording_count=sum(
            not event.eligible_for_prediction_windows for event in events
        ),
        sample_count=len(features),
        positive_count=int(labels_array.sum()),
        negative_count=int(len(labels_array) - labels_array.sum()),
        train_count=int(np.sum(splits_array == "train")),
        validation_count=int(np.sum(splits_array == "validation")),
        test_count=int(np.sum(splits_array == "test")),
        skipped_quality_count=skipped_quality,
        feature_version=FEATURE_VERSION_V2,
        window_size=feature_extractor.window_size,
        input_size=len(FEATURE_NAMES_V2),
        minimum_lead_seconds=minimum_lead_seconds,
        maximum_lead_seconds=maximum_lead_seconds,
        positive_anchor=positive_anchor,
        minimum_pre_descent_margin_seconds=minimum_pre_descent_margin_seconds,
        kinect_used_as_model_input=False,
        deployment_eligible=False,
    )
    _write_manifest(destination, summary)
    return summary


def _extract_window(
    frames: tuple[RadarFrame, ...],
    frame_epochs: NDArray[np.float64],
    end_timestamp: datetime,
    extractor: RadarTemporalFeatureExtractorV2,
):
    tolerance = extractor.alignment_tolerance_seconds
    left_epoch = (
        end_timestamp - timedelta(seconds=extractor.history_seconds + tolerance)
    ).timestamp()
    right_epoch = (end_timestamp + timedelta(seconds=tolerance)).timestamp()
    left = bisect.bisect_left(frame_epochs, left_epoch)
    right = bisect.bisect_right(frame_epochs, right_epoch)
    if left >= right:
        return None
    window = extractor.transform(frames[left:right], end_timestamp=end_timestamp)
    if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
        return None
    return window


def _subject_from_file_name(file_name: str) -> str:
    match = re.fullmatch(r"([FM]_\d{3})_A\d+_\d+\.txt", file_name)
    if match is None:
        raise ValueError(f"unexpected DGUHA file name: {file_name}")
    subject_id = match.group(1)
    if subject_id not in DGUHA_SPLIT_BY_SUBJECT:
        raise ValueError(f"unknown DGUHA subject: {subject_id}")
    return subject_id


def _first_sustained(mask: NDArray[np.bool_], count: int) -> int | None:
    if count <= 0:
        raise ValueError("sustained count must be positive")
    for index in range(0, len(mask) - count + 1):
        if bool(np.all(mask[index : index + count])):
            return index
    return None


def _timestamp_at(
    frames: tuple[DguhaKinectFrame, ...], index: int | None
) -> datetime | None:
    return None if index is None else frames[index].timestamp


def _event_payload(event: DguhaFallEvent) -> dict[str, object]:
    payload = asdict(event)
    for name in (
        "radar_start",
        "radar_end",
        "kinect_start",
        "kinect_end",
        "descent_onset",
        "rapid_descent_onset",
        "near_floor_level_reached",
    ):
        value = getattr(event, name)
        payload[name] = None if value is None else value.isoformat()
    payload["exclusion_reasons"] = list(event.exclusion_reasons)
    if event.descent_onset is not None:
        payload["descent_onset_seconds_from_radar_start"] = (
            event.descent_onset - event.radar_start
        ).total_seconds()
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, summary: DguhaResearchExportSummary) -> None:
    payload = asdict(summary)
    payload["dataset_mode"] = DGUHA_DATASET_MODE
    payload["positive_label_definition"] = _positive_label_definition(
        summary.positive_anchor,
        summary.minimum_lead_seconds,
        summary.maximum_lead_seconds,
        summary.minimum_pre_descent_margin_seconds,
    )
    payload["known_limitations"] = [
        "Kinect is used only offline for pseudo-label timestamps, never as model input",
        "only forward falls from young healthy subjects are present",
        "the source has no clinically verified loss-of-balance or impact time",
        "recordings with insufficient 2-second pre-fall radar history are excluded",
        "author Training/Test folders leak M_012; project splits override them by subject",
        "research-only output is not deployment validation",
    ]
    path.with_suffix(".manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _positive_label_definition(
    positive_anchor: str,
    minimum_lead_seconds: float,
    maximum_lead_seconds: float,
    minimum_pre_descent_margin_seconds: float,
) -> str:
    anchor_label = (
        "whole-body descent onset"
        if positive_anchor == "descent_onset"
        else "near-floor level reached (a proxy, not measured impact)"
    )
    return (
        f"Radar-only windows ending {minimum_lead_seconds:.1f}-"
        f"{maximum_lead_seconds:.1f} s before skeleton-derived {anchor_label}; "
        f"every positive window must still end at least "
        f"{minimum_pre_descent_margin_seconds:.1f} s before descent onset. "
        "This is a weak pseudo-label, not a clinically verified loss-of-balance "
        "or impact annotation."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DGUHA radar-only v2 windows with skeleton pseudo-labels."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-skeleton-pseudolabels", action="store_true")
    parser.add_argument("--minimum-lead-seconds", type=float, default=0.1)
    parser.add_argument("--maximum-lead-seconds", type=float, default=0.6)
    parser.add_argument("--positive-stride-seconds", type=float, default=0.1)
    parser.add_argument("--negative-stride-seconds", type=float, default=1.0)
    parser.add_argument("--max-negative-windows-per-recording", type=int, default=10)
    parser.add_argument(
        "--positive-anchor",
        choices=sorted(DGUHA_POSITIVE_ANCHORS),
        default="descent_onset",
    )
    parser.add_argument("--minimum-pre-descent-margin-seconds", type=float, default=0.1)
    args = parser.parse_args()
    summary = export_dguha_research_npz(
        args.data_root,
        args.output,
        allow_skeleton_pseudolabels=args.allow_skeleton_pseudolabels,
        minimum_lead_seconds=args.minimum_lead_seconds,
        maximum_lead_seconds=args.maximum_lead_seconds,
        positive_stride_seconds=args.positive_stride_seconds,
        negative_stride_seconds=args.negative_stride_seconds,
        max_negative_windows_per_recording=args.max_negative_windows_per_recording,
        positive_anchor=args.positive_anchor,
        minimum_pre_descent_margin_seconds=(
            args.minimum_pre_descent_margin_seconds
        ),
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
