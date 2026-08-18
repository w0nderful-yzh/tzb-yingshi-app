"""Build a "descent early detection" dataset from DGUHA.

Motivation
----------
The existing pre-fall positive windows (ending 0.1-0.6 s before descent onset)
contain no descent within the window: the body is still standing. That is why
both binary classifiers and time-to-impact regression fail to learn temporal
evolution -- the signal is simply not in the window.

This module constructs a *descent early detection* dataset:
- Positive windows are sampled *during the descent*, from
  ``descent_onset + lead_min`` to ``near_floor_level_reached - margin``.
  A window ending at time t is labeled positive if the body is actively
  descending and there is at least ``min_lead_before_floor`` seconds left
  before the body reaches the floor. This makes the model learn "the body is
  falling right now", which is detectable from z_p90 / z_p50 / height_range
  dynamics.
- Negative windows come from normal actions (sit-stand, limb extension,
  running, jumping, etc.) and from the fall recording before onset.

Contract
--------
- Subject-isolated splits from DGUHA project_split.
- ``deployment_eligible=false``.
- The positive windows carry ``seconds_to_floor`` (time from window end to
  near_floor_level_reached), so downstream can evaluate early-warning lead.

Version: radar_descent_early_detection_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from radar_module.dataset.dguha_research_v2 import (
    DGUHA_ACTIONS,
    DGUHA_SPLIT_BY_SUBJECT,
    _subject_from_file_name,
    derive_dguha_fall_event,
    parse_dguha_kinect,
)
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


DATASET_MODE = "DGUHA_DESCENT_EARLY_DETECTION_V1"
DESCENT_DETECTION_FALL_ACTION = "5_falling_forward"


def build_descent_early_detection_npz(
    data_root: str | Path,
    output_path: str | Path,
    events_path: str | Path,
    *,
    descent_lead_min_seconds: float = 0.3,
    floor_margin_seconds: float = 0.3,
    descent_stride_seconds: float = 0.2,
    normal_stride_seconds: float = 0.5,
    max_normal_windows_per_recording: int = 30,
    extractor: RadarTemporalFeatureExtractorV2 | None = None,
) -> dict[str, Any]:
    source_root = Path(data_root).resolve()
    destination = Path(output_path).resolve()
    events_file = Path(events_path).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"DGUHA data root does not exist: {source_root}")
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end with .npz")
    if descent_lead_min_seconds < 0 or floor_margin_seconds < 0:
        raise ValueError("leads must be non-negative")
    if descent_stride_seconds <= 0 or normal_stride_seconds <= 0:
        raise ValueError("strides must be positive")

    feature_extractor = extractor or RadarTemporalFeatureExtractorV2()
    required_history_seconds = (feature_extractor.window_size - 1) / (
        feature_extractor.target_sample_rate_hz
    )

    events = json.loads(events_file.read_text(encoding="utf-8"))
    event_by_source = {ev["source_file"]: ev for ev in events}

    features: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    actions: list[str] = []
    subject_ids: list[str] = []
    source_files: list[str] = []
    window_end_seconds: list[float] = []
    seconds_to_floor: list[float] = []
    label_sources: list[str] = []
    qualities: list[str] = []
    skipped_quality = 0
    fall_recordings = 0

    radar_files = sorted(source_root.glob("*/*/radar/*.txt"))
    for radar_path in radar_files:
        relative = radar_path.relative_to(source_root)
        if len(relative.parts) != 4:
            continue
        author_split, action, modality, file_name = relative.parts
        if modality != "radar" or action not in DGUHA_ACTIONS:
            continue
        subject_id = _subject_from_file_name(file_name)
        project_split = DGUHA_SPLIT_BY_SUBJECT[subject_id]
        frames = parse_radhar_text(radar_path, device_id=f"dguha-{file_name[:-4]}")
        radar_start = frames[0].timestamp
        radar_start_epoch = radar_start.timestamp()

        if action == DESCENT_DETECTION_FALL_ACTION:
            # Positive windows during the descent.
            event = event_by_source.get(relative.as_posix())
            if event is None or not event.get("eligible_for_prediction_windows"):
                continue
            fall_recordings += 1
            onset = datetime.fromisoformat(event["descent_onset"])
            near_floor = datetime.fromisoformat(event["near_floor_level_reached"])
            onset_epoch = onset.timestamp()
            near_floor_epoch = near_floor.timestamp()

            t = onset_epoch + descent_lead_min_seconds
            while t <= near_floor_epoch - floor_margin_seconds:
                end_timestamp = datetime.fromtimestamp(t, tz=timezone.utc)
                window_frames = [
                    f
                    for f in frames
                    if f.timestamp <= end_timestamp
                    and f.timestamp
                    >= end_timestamp - timedelta(seconds=2.0)
                ]
                window = _extract_window(
                    window_frames,
                    end_timestamp,
                    feature_extractor,
                )
                if window is None:
                    skipped_quality += 1
                    t += descent_stride_seconds
                    continue
                features.append(np.asarray(window.values, dtype=np.float32))
                labels.append(1)
                splits.append(project_split)
                actions.append("falling_descent")
                subject_ids.append(subject_id)
                source_files.append(relative.as_posix())
                window_end_seconds.append(t - radar_start_epoch)
                seconds_to_floor.append(
                    max(0.0, near_floor_epoch - t)
                )
                label_sources.append("dguha_kinect_descent_interval")
                qualities.append(window.data_quality.value)
                t += descent_stride_seconds
        else:
            # Negative windows from normal actions.
            duration = (frames[-1].timestamp - frames[0].timestamp).total_seconds()
            candidates = np.arange(
                required_history_seconds,
                duration,
                normal_stride_seconds,
            )
            if len(candidates) > max_normal_windows_per_recording:
                candidates = np.linspace(
                    0,
                    len(candidates) - 1,
                    max_normal_windows_per_recording,
                    dtype=np.int64,
                )
                candidates = np.arange(
                    required_history_seconds, duration, normal_stride_seconds
                )[candidates]
            for end_seconds in candidates:
                end_timestamp = radar_start + timedelta(seconds=float(end_seconds))
                window_frames = [
                    f
                    for f in frames
                    if f.timestamp <= end_timestamp
                    and f.timestamp
                    >= end_timestamp - timedelta(seconds=2.0)
                ]
                window = _extract_window(
                    window_frames,
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
                seconds_to_floor.append(np.nan)
                label_sources.append("dguha_recording_activity_label")
                qualities.append(window.data_quality.value)

    if not features:
        raise ValueError("no windows produced")

    features_arr = np.stack(features).astype(np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64)
    splits_arr = np.asarray(splits)
    summary = {
        "dataset_mode": DATASET_MODE,
        "window_count": int(len(features_arr)),
        "positive_count": int(labels_arr.sum()),
        "negative_count": int((labels_arr == 0).sum()),
        "fall_recording_count": fall_recordings,
        "skipped_quality": skipped_quality,
        "feature_version": FEATURE_VERSION_V2,
        "feature_names": list(FEATURE_NAMES_V2),
        "positive_label_definition": (
            f"window end in [descent_onset+{descent_lead_min_seconds}s, "
            f"near_floor-{floor_margin_seconds}s]"
        ),
        "seconds_to_floor_defined": "time from window end to near_floor",
        "split_counts": {
            name: int((splits_arr == name).sum()) for name in np.unique(splits_arr)
        },
        "deployment_eligible": False,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        features=features_arr,
        labels=labels_arr,
        split=splits_arr,
        action=actions,
        subject_id=subject_ids,
        source_files=source_files,
        window_end_seconds=window_end_seconds,
        seconds_to_floor=seconds_to_floor,
        label_source=label_sources,
        data_quality=qualities,
        feature_version=FEATURE_VERSION_V2,
        feature_names=list(FEATURE_NAMES_V2),
        dataset_mode=DATASET_MODE,
        positive_label_definition=summary["positive_label_definition"],
        deployment_eligible=False,
    )
    manifest = destination.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _extract_window(frames, end_timestamp, extractor):
    if not frames:
        return None
    try:
        window = extractor.transform(tuple(frames), end_timestamp=end_timestamp)
    except ValueError:
        return None
    if window.data_quality is not TemporalDataQuality.GOOD:
        return None
    return window


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build descent early detection dataset from DGUHA."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--descent-lead-min-seconds", type=float, default=0.3)
    parser.add_argument("--floor-margin-seconds", type=float, default=0.3)
    parser.add_argument("--descent-stride-seconds", type=float, default=0.2)
    parser.add_argument("--normal-stride-seconds", type=float, default=0.5)
    parser.add_argument("--max-normal-windows-per-recording", type=int, default=30)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = build_descent_early_detection_npz(
        data_root=args.data_root,
        events_path=args.events,
        output_path=args.output,
        descent_lead_min_seconds=args.descent_lead_min_seconds,
        floor_margin_seconds=args.floor_margin_seconds,
        descent_stride_seconds=args.descent_stride_seconds,
        normal_stride_seconds=args.normal_stride_seconds,
        max_normal_windows_per_recording=args.max_normal_windows_per_recording,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
