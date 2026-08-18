"""Build old vs new pre-fall label datasets for comparison.

This constructs two datasets from DGUHA radar windows, both using radar
features only (Kinect is used offline to define event boundaries, never as
model input):

- OLD label: window end in [descent_onset - 1.0, descent_onset - 0.5] s
  (the existing 0.5-1.0 s before onset positive definition).
- NEW label: window end in [sustained_descent_onset - 0.5,
  sustained_descent_onset - 0.2] s, where sustained_descent_onset is the
  Kinect head-height sustained descent start (re-located with absolute-time
  alignment).

Negative windows are sampled from the same recordings before the precursor
window and from normal actions. Splits are strict subject-isolated.

Important: Kinect is used ONLY to re-locate sustained_descent_onset for the
NEW label. The radar feature windows themselves never contain Kinect data.

Version: radar_dguha_label_comparison_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radar_module.dataset.dguha_research_v2 import (
    DGUHA_SPLIT_BY_SUBJECT,
    _subject_from_file_name,
    parse_dguha_kinect,
)
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)

HEAD_JOINTS = (0, 1, 2, 3, 4)


def locate_sustained_descent_onset(kpath: Path):
    """Locate head-height sustained descent onset (absolute epoch)."""
    frames = parse_dguha_kinect(kpath)
    valid = [f for f in frames if f.points_mm.any()]
    if len(valid) < 30:
        return None
    t = np.asarray([f.timestamp.timestamp() for f in valid])
    head = np.asarray([np.max(f.points_mm[:, 2]) / 1000.0 for f in valid])
    base = np.median(head[:15])
    v = np.gradient(head, t)
    for i in range(3, len(v)):
        if v[i] < -0.15 and head[i] < base - 0.03:
            return t[i]
    return None


def build_label_dataset(
    data_root: str | Path,
    events_path: str | Path,
    output_prefix: str | Path,
    *,
    negative_stride_seconds: float = 1.0,
    max_negative_per_recording: int = 10,
):
    """Build old and new label npz files."""
    source_root = Path(data_root).resolve()
    events = json.loads(Path(events_path).read_text())
    event_by_source = {e["source_file"]: e for e in events}
    eligible = [e for e in events if e.get("eligible_for_prediction_windows")]

    extractor = RadarTemporalFeatureExtractorV2()
    window_size = extractor.window_size
    target_rate = extractor.target_sample_rate_hz
    required_history = (window_size - 1) / target_rate

    radar_files = sorted(source_root.glob("*/*/radar/*.txt"))
    print(f"radar 文件数: {len(radar_files)}")

    # We'll build per-recording window lists for old and new labels.
    old_records: list[dict] = []  # {features, label, split, subject, source}
    new_records: list[dict] = []

    for radar_path in radar_files:
        relative = radar_path.relative_to(source_root)
        parts = relative.parts
        if len(parts) != 4:
            continue
        author_split, action, modality, fname = parts
        if modality != "radar" or action not in {
            "1_Running", "2_Jumping", "3_Sit_down_and_stand_up",
            "4_Both_upper_limb_extension", "5_falling_forward",
            "6_Right_limb_extension", "7_Left_limb_extension",
        }:
            continue
        subject_id = _subject_from_file_name(fname)
        project_split = DGUHA_SPLIT_BY_SUBJECT.get(subject_id, "train")
        frames = parse_radhar_text(radar_path, device_id=f"dguha-{fname[:-4]}")
        radar_start_abs = frames[0].timestamp.timestamp()

        if action == "5_falling_forward":
            ev = event_by_source.get(relative.as_posix())
            if ev is None or not ev.get("eligible_for_prediction_windows"):
                continue
            # OLD anchor: descent_onset (radar relative seconds)
            old_onset = ev["descent_onset_seconds_from_radar_start"]
            if old_onset is None:
                continue
            old_onset_abs = radar_start_abs + old_onset
            # NEW anchor: sustained descent from Kinect head height
            kpath = source_root / relative.as_posix().replace("/radar/", "/kinect/")
            new_onset_abs = locate_sustained_descent_onset(kpath)
            if new_onset_abs is None:
                continue

            # Positive windows
            # OLD: end in [onset-1.0, onset-0.5]
            for lead in np.arange(0.5, 1.0 + 1e-9, 0.1):
                end_abs = old_onset_abs - lead
                end_rel = end_abs - radar_start_abs
                w = _extract_window_abs(frames, end_abs, extractor)
                if w is not None:
                    old_records.append(
                        {"features": w, "label": 1, "split": project_split,
                         "subject": subject_id, "source": relative.as_posix(),
                         "seconds_to_old_onset": lead,
                         "seconds_to_new_onset": (new_onset_abs - end_abs)}
                    )
            # NEW: end in [new_onset-0.5, new_onset-0.2]
            for lead in np.arange(0.2, 0.5 + 1e-9, 0.1):
                end_abs = new_onset_abs - lead
                end_rel = end_abs - radar_start_abs
                w = _extract_window_abs(frames, end_abs, extractor)
                if w is not None:
                    new_records.append(
                        {"features": w, "label": 1, "split": project_split,
                         "subject": subject_id, "source": relative.as_posix(),
                         "seconds_to_old_onset": (old_onset_abs - end_abs),
                         "seconds_to_new_onset": lead}
                    )
            # Negatives: windows before precursor region
            neg_region_start = min(old_onset_abs, new_onset_abs) - 3.0
            neg_region_end = min(old_onset_abs, new_onset_abs) - 0.6
            for end_rel in np.arange(
                required_history,
                (neg_region_end - radar_start_abs),
                negative_stride_seconds,
            ):
                end_abs = radar_start_abs + end_rel
                if end_abs > neg_region_end:
                    break
                w = _extract_window_abs(frames, end_abs, extractor)
                if w is not None:
                    old_records.append(
                        {"features": w, "label": 0, "split": project_split,
                         "subject": subject_id, "source": relative.as_posix(),
                         "seconds_to_old_onset": None, "seconds_to_new_onset": None}
                    )
                    new_records.append(
                        {"features": w, "label": 0, "split": project_split,
                         "subject": subject_id, "source": relative.as_posix(),
                         "seconds_to_old_onset": None, "seconds_to_new_onset": None}
                    )
        else:
            # Normal actions -> negatives only
            duration = (frames[-1].timestamp - frames[0].timestamp).total_seconds()
            for end_rel in np.arange(required_history, duration, negative_stride_seconds):
                if len(old_records) and len(new_records):
                    pass
                end_abs = radar_start_abs + end_rel
                w = _extract_window_abs(frames, end_abs, extractor)
                if w is not None:
                    old_records.append(
                        {"features": w, "label": 0, "split": project_split,
                         "subject": subject_id, "source": relative.as_posix(),
                         "seconds_to_old_onset": None, "seconds_to_new_onset": None}
                    )
                    new_records.append(
                        {"features": w, "label": 0, "split": project_split,
                         "subject": subject_id, "source": relative.as_posix(),
                         "seconds_to_old_onset": None, "seconds_to_new_onset": None}
                    )

    # Write npz
    _write_npz(Path(output_prefix) / "dguha_old_label_v1.npz", old_records, "old")
    _write_npz(Path(output_prefix) / "dguha_new_label_v1.npz", new_records, "new")
    print(f"OLD: {len(old_records)} windows")
    print(f"NEW: {len(new_records)} windows")


def _extract_window_abs(frames, end_abs, extractor):
    from datetime import datetime, timedelta, timezone

    end_ts = datetime.fromtimestamp(end_abs, tz=timezone.utc)
    wf = [f for f in frames if f.timestamp <= end_ts and f.timestamp >= end_ts - timedelta(seconds=2)]
    if not wf:
        return None
    try:
        w = extractor.transform(tuple(wf), end_timestamp=end_ts)
    except ValueError:
        return None
    if w.data_quality is not TemporalDataQuality.GOOD:
        return None
    return np.asarray(w.values, dtype=np.float32)


def _write_npz(path, records, label_name):
    if not records:
        raise ValueError(f"no records for {label_name}")
    feats = np.stack([r["features"] for r in records])
    labels = np.asarray([r["label"] for r in records], dtype=np.int64)
    splits = np.asarray([r["split"] for r in records])
    subjects = np.asarray([r["subject"] for r in records])
    sources = np.asarray([r["source"] for r in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=feats, labels=labels, split=splits,
        subject_id=subjects, source_files=sources,
        feature_version=FEATURE_VERSION_V2,
        feature_names=list(FEATURE_NAMES_V2),
        dataset_mode=f"DGUHA_LABEL_COMPARISON_{label_name.upper()}_V1",
        deployment_eligible=False,
    )
    print(f"  wrote {path}: {len(feats)} windows, pos={int(labels.sum())}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/external/dguha/raw")
    parser.add_argument("--events", default="data/processed/dguha_prefall_0p5_1p0_dense_v3.events.json")
    parser.add_argument("--output-prefix", default="data/processed/experiments_v11")
    args = parser.parse_args()
    build_label_dataset(args.data_root, args.events, args.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
