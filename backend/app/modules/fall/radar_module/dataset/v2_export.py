from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from radar_module.contracts import RadarFrame, Room, SourceMode
from radar_module.dataset.annotations import (
    PredictionWindowLabel,
    decide_prediction_window,
    load_annotation_document,
)
from radar_module.preprocess.pointcloud_processing import map_official_points
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


@dataclass(frozen=True, slots=True)
class V2DatasetExportSummary:
    replay_file: str
    annotations_file: str
    output_file: str
    source_sha256: str
    output_sha256: str
    session_id: str
    frame_count: int
    sample_count: int
    positive_count: int
    negative_count: int
    skipped_excluded_count: int
    skipped_detection_count: int
    skipped_quality_count: int
    feature_version: str
    window_size: int
    input_size: int
    stride_seconds: float


def export_v2_training_npz(
    replay_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path,
    *,
    stride_seconds: float = 0.1,
    allow_degraded_quality: bool = True,
    extractor: RadarTemporalFeatureExtractorV2 | None = None,
) -> V2DatasetExportSummary:
    """Export room-level v2 windows after label and checksum validation."""

    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    source = Path(replay_path).resolve()
    annotation_file = Path(annotations_path).resolve()
    destination = Path(output_path).resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end with .npz")
    if not source.is_file():
        raise FileNotFoundError(f"replay file does not exist: {source}")

    annotations = load_annotation_document(annotation_file)
    source_sha256 = _sha256(source)
    if source_sha256 != annotations.session.source_sha256:
        raise ValueError(
            "replay SHA-256 does not match annotation session metadata"
        )
    frames = _load_replay_frames(source, default_room=annotations.session.room)
    if frames[0].room is not annotations.session.room:
        raise ValueError("replay room does not match annotation room")

    feature_extractor = extractor or RadarTemporalFeatureExtractorV2()
    first_timestamp = frames[0].timestamp
    frame_seconds = [
        (frame.timestamp - first_timestamp).total_seconds() for frame in frames
    ]
    actual_duration = frame_seconds[-1]
    usable_duration = min(actual_duration, annotations.session.duration_seconds)
    minimum_end_seconds = (
        feature_extractor.window_size - 1
    ) / feature_extractor.target_sample_rate_hz
    if usable_duration < minimum_end_seconds:
        raise ValueError("session is shorter than one v2 history window")

    event_subjects = {
        event.event_id: event.subject_group_id
        for event in annotations.fall_events
    }
    unique_subjects = sorted(set(event_subjects.values()))
    default_subject = unique_subjects[0] if len(unique_subjects) == 1 else ""

    features: list[np.ndarray] = []
    labels: list[int] = []
    window_ends: list[float] = []
    event_ids: list[str] = []
    subject_group_ids: list[str] = []
    seconds_to_impact: list[float] = []
    qualities: list[str] = []
    skipped_excluded = 0
    skipped_detection = 0
    skipped_quality = 0

    end_seconds = minimum_end_seconds
    tolerance = feature_extractor.alignment_tolerance_seconds
    while end_seconds <= usable_duration + 1e-9:
        decision = decide_prediction_window(annotations, end_seconds)
        if decision.label is PredictionWindowLabel.EXCLUDED:
            skipped_excluded += 1
            end_seconds += stride_seconds
            continue
        if decision.label is PredictionWindowLabel.DETECTION_ONLY:
            skipped_detection += 1
            end_seconds += stride_seconds
            continue

        slice_start = end_seconds - feature_extractor.history_seconds - tolerance
        slice_end = end_seconds + tolerance
        left = bisect.bisect_left(frame_seconds, slice_start)
        right = bisect.bisect_right(frame_seconds, slice_end)
        if left >= right:
            skipped_quality += 1
            end_seconds += stride_seconds
            continue
        window = feature_extractor.transform(
            frames[left:right],
            end_timestamp=first_timestamp + timedelta(seconds=end_seconds),
        )
        if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA or (
            not allow_degraded_quality
            and window.data_quality is TemporalDataQuality.DEGRADED
        ):
            skipped_quality += 1
            end_seconds += stride_seconds
            continue

        event_id = decision.event_id or ""
        features.append(np.asarray(window.values, dtype=np.float32))
        labels.append(
            1 if decision.label is PredictionWindowLabel.PRE_FALL else 0
        )
        window_ends.append(end_seconds)
        event_ids.append(event_id)
        subject_group_ids.append(event_subjects.get(event_id, default_subject))
        seconds_to_impact.append(
            float(decision.seconds_to_impact)
            if decision.seconds_to_impact is not None
            else np.nan
        )
        qualities.append(window.data_quality.value)
        end_seconds += stride_seconds

    if not features:
        raise ValueError("no trainable v2 samples were produced")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=np.stack(features).astype(np.float32, copy=False),
                labels=np.asarray(labels, dtype=np.int8),
                window_end_seconds=np.asarray(window_ends, dtype=np.float32),
                session_ids=np.asarray(
                    [annotations.session.session_id] * len(features)
                ),
                event_ids=np.asarray(event_ids),
                subject_group_ids=np.asarray(subject_group_ids),
                seconds_to_impact=np.asarray(
                    seconds_to_impact, dtype=np.float32
                ),
                data_quality=np.asarray(qualities),
                feature_version=np.asarray(FEATURE_VERSION_V2),
                feature_names=np.asarray(FEATURE_NAMES_V2),
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    summary = V2DatasetExportSummary(
        replay_file=str(source),
        annotations_file=str(annotation_file),
        output_file=str(destination),
        source_sha256=source_sha256,
        output_sha256=_sha256(destination),
        session_id=annotations.session.session_id,
        frame_count=len(frames),
        sample_count=len(features),
        positive_count=sum(labels),
        negative_count=len(labels) - sum(labels),
        skipped_excluded_count=skipped_excluded,
        skipped_detection_count=skipped_detection,
        skipped_quality_count=skipped_quality,
        feature_version=FEATURE_VERSION_V2,
        window_size=feature_extractor.window_size,
        input_size=len(FEATURE_NAMES_V2),
        stride_seconds=stride_seconds,
    )
    _write_manifest(destination, summary)
    return summary


def _load_replay_frames(
    path: Path,
    *,
    default_room: Room,
) -> tuple[RadarFrame, ...]:
    frames: list[RadarFrame] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            normalized = line.strip()
            if not normalized:
                continue
            try:
                payload = json.loads(normalized)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL line {line_number} must be an object")
            timestamp = _parse_timestamp(payload.get("timestamp"), line_number)
            device_id = str(payload.get("device_id") or "").strip()
            if not device_id:
                raise ValueError(
                    f"JSONL line {line_number} has no device_id"
                )
            points = map_official_points(
                payload.get("points", ()), max_distance_m=8.0
            )
            frames.append(
                RadarFrame(
                    timestamp=timestamp,
                    device_id=device_id,
                    room=Room(payload.get("room") or default_room.value),
                    source_mode=SourceMode.REPLAY,
                    points=points,
                )
            )
    if not frames:
        raise ValueError("replay file contains no frames")
    for previous, current in zip(frames[:-1], frames[1:]):
        if current.timestamp < previous.timestamp:
            raise ValueError("replay frames must be ordered by timestamp")
    return tuple(frames)


def _parse_timestamp(value: Any, line_number: int) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"JSONL line {line_number} timestamp must be ISO-8601")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"JSONL line {line_number} timestamp needs timezone")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(
    output_path: Path,
    summary: V2DatasetExportSummary,
) -> None:
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export labeled radar JSONL sessions to v2 training NPZ."
    )
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride-seconds", type=float, default=0.1)
    parser.add_argument(
        "--good-quality-only",
        action="store_true",
        help="Exclude DEGRADED windows in addition to INSUFFICIENT_DATA.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = export_v2_training_npz(
        args.replay,
        args.annotations,
        args.output,
        stride_seconds=args.stride_seconds,
        allow_degraded_quality=not args.good_quality_only,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
