from __future__ import annotations

import argparse
import bisect
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from radar_module.dataset.dguha_research_v2 import (
    DGUHA_FALL_ACTION,
    DGUHA_SPLIT_BY_SUBJECT,
)
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.model.dguha_event_evaluation_v2 import (
    _build_prefall_model,
    _confirmed_run_end_indices,
    _load_checkpoint,
    _subject_id,
)
from radar_module.model.point_prediction_v1 import POINT_PREDICTION_MODEL_VERSION
from radar_module.preprocess.pointcloud_sequence import PointCloudSequenceBuilder
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


ANALYSIS_VERSION = "temporal_dependency_diagnostic_v1"
RANDOM_SEEDS = (20260809, 20260810, 20260811, 20260812, 20260813)
MASK_SECONDS = (0.3, 0.5, 1.0)
DYNAMIC_FEATURE_NAMES = (
    "centroid_z_delta_0_3s",
    "centroid_z_delta_0_6s",
    "vertical_velocity",
    "vertical_acceleration",
    "height_range_delta_0_3s",
)


def run_diagnostic(
    *,
    data_root: str | Path,
    events_file: str | Path,
    checkpoints: dict[str, str | Path],
    output_directory: str | Path,
    split: str = "validation",
    step_seconds: float = 0.1,
    confirmation_windows: int = 3,
    minimum_lead_seconds: float = 0.5,
    maximum_lead_seconds: float = 1.0,
    early_negative_minimum_lead_seconds: float = 1.2,
) -> dict[str, Any]:
    if split != "validation":
        raise ValueError("this diagnostic is locked to the validation split")
    if confirmation_windows < 2 or not math.isclose(step_seconds, 0.1):
        raise ValueError("diagnostic requires 0.1 s steps and at least two confirmations")
    root = Path(data_root).resolve()
    event_path = Path(events_file).resolve()
    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    events = json.loads(event_path.read_text(encoding="utf-8"))
    event_by_source = {str(event["source_file"]): event for event in events}

    run_rows: list[dict[str, Any]] = []
    checkpoint_contracts: dict[str, dict[str, Any]] = {}
    for model_name, checkpoint_value in checkpoints.items():
        checkpoint_path = Path(checkpoint_value).resolve()
        checkpoint = _load_checkpoint(checkpoint_path)
        model = _build_prefall_model(checkpoint)
        model.eval()
        point_input = checkpoint["model_version"] == POINT_PREDICTION_MODEL_VERSION
        conditions = _conditions(20)
        accumulators = {
            condition["run_id"]: _empty_accumulator(condition)
            for condition in conditions
        }
        extractor = RadarTemporalFeatureExtractorV2()
        point_builder = PointCloudSequenceBuilder()
        required_history = 1.9
        mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
        threshold = float(checkpoint["decision_threshold"])

        radar_files = sorted(root.glob("*/*/radar/*.txt"))
        processed = 0
        for radar_path in radar_files:
            subject = _subject_id(radar_path.name)
            if DGUHA_SPLIT_BY_SUBJECT[subject] != split:
                continue
            relative = radar_path.relative_to(root).as_posix()
            action = radar_path.parent.parent.name
            event = event_by_source.get(relative)
            if action == DGUHA_FALL_ACTION and (
                event is None
                or not bool(event["eligible_for_prediction_windows"])
                or event.get("descent_onset") is None
            ):
                continue
            frames = parse_radhar_text(
                radar_path, device_id=f"temporal-diagnostic-{radar_path.stem}"
            )
            radar_start = frames[0].timestamp
            duration = (frames[-1].timestamp - radar_start).total_seconds()
            if action == DGUHA_FALL_ACTION:
                onset = datetime.fromisoformat(str(event["descent_onset"]))
                maximum_end = (onset - radar_start).total_seconds() - 0.1
            else:
                onset = None
                maximum_end = duration
            endpoints = np.arange(
                required_history,
                maximum_end + step_seconds * 0.25,
                step_seconds,
                dtype=np.float64,
            )
            if point_input:
                times, raw_inputs = _extract_point_inputs(
                    frames, endpoints, radar_start, point_builder
                )
            else:
                times, raw_inputs = _extract_tcn_inputs(
                    frames, endpoints, radar_start, extractor
                )
            if not len(times):
                continue
            for condition in conditions:
                if point_input:
                    scores = _score_point_condition(
                        model, raw_inputs, mean, std, condition
                    )
                else:
                    scores = _score_tcn_condition(
                        model, raw_inputs, mean, std, condition
                    )
                _accumulate_recording(
                    accumulators[condition["run_id"]],
                    relative=relative,
                    action=action,
                    times=times,
                    scores=scores,
                    threshold=threshold,
                    duration=duration,
                    onset=onset,
                    radar_start=radar_start,
                    confirmation_windows=confirmation_windows,
                    step_seconds=step_seconds,
                    minimum_lead_seconds=minimum_lead_seconds,
                    maximum_lead_seconds=maximum_lead_seconds,
                    early_negative_minimum_lead_seconds=(
                        early_negative_minimum_lead_seconds
                    ),
                )
            processed += 1
            if processed % 25 == 0:
                print(f"{model_name}: processed {processed} validation recordings")

        for condition in conditions:
            run_rows.append(
                _finalize_accumulator(
                    model_name=model_name,
                    checkpoint=checkpoint,
                    condition=condition,
                    accumulator=accumulators[condition["run_id"]],
                )
            )
        checkpoint_contracts[model_name] = {
            "checkpoint_file": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "model_version": str(checkpoint["model_version"]),
            "threshold": threshold,
            "prediction_horizon_seconds": list(
                checkpoint["prediction_horizon_seconds"]
            ),
            "input_family": "point_cloud" if point_input else "temporal_features_v2",
        }

    runs = pd.DataFrame(run_rows)
    summary = _summarize_runs(runs)
    runs.to_csv(destination / "temporal_dependency_runs.csv", index=False)
    summary.to_csv(destination / "temporal_dependency_summary.csv", index=False)
    verdict = _diagnostic_verdict(summary)
    report: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "split": split,
        "test_split_inspected": False,
        "model_training_performed": False,
        "model_parameters_modified": False,
        "checkpoint_contracts": checkpoint_contracts,
        "evaluation_contract": {
            "step_seconds": step_seconds,
            "confirmation_windows": confirmation_windows,
            "prediction_lead_interval_seconds": [
                minimum_lead_seconds,
                maximum_lead_seconds,
            ],
            "same_recording_early_negative_definition": (
                f"fall-recording windows ending at least "
                f"{early_negative_minimum_lead_seconds:.1f} s before descent_onset"
            ),
            "false_alarm_rate_definition": (
                "confirmed high-score run starts per total duration of all validation "
                "normal recordings"
            ),
            "random_shuffle_seeds": list(RANDOM_SEEDS),
            "threshold_policy": "locked checkpoint thresholds; no recalibration",
        },
        "transformation_contract": {
            "reverse": "reverse all 20 input time steps",
            "random_shuffle": (
                "one deterministic 20-step permutation per seed, shared by every "
                "recording and both model families"
            ),
            "repeat_last_frame": (
                "repeat the last observed point frame; for TCN repeat the last base-feature "
                "row and zero its engineered delta/velocity/acceleration fields"
            ),
            "mask_suffix": (
                "set the final 0.3/0.5/1.0 s to missing/zero while retaining the original "
                "window endpoint and frozen model"
            ),
        },
        "summary": summary.to_dict(orient="records"),
        "verdict": verdict,
        "deployment_eligible": False,
    }
    report = _json_safe(report)
    (destination / "temporal_dependency_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _conditions(time_steps: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [
        {"condition": "original", "run_id": "original", "seed": None, "permutation": None},
        {
            "condition": "reverse",
            "run_id": "reverse",
            "seed": None,
            "permutation": np.arange(time_steps - 1, -1, -1),
        },
    ]
    for seed in RANDOM_SEEDS:
        result.append(
            {
                "condition": "random_shuffle",
                "run_id": f"random_shuffle_seed_{seed}",
                "seed": seed,
                "permutation": np.random.default_rng(seed).permutation(time_steps),
            }
        )
    result.append(
        {
            "condition": "repeat_last_frame",
            "run_id": "repeat_last_frame",
            "seed": None,
            "permutation": None,
        }
    )
    for seconds in MASK_SECONDS:
        result.append(
            {
                "condition": f"mask_last_{seconds:.1f}s",
                "run_id": f"mask_last_{seconds:.1f}s",
                "seed": None,
                "permutation": None,
                "mask_steps": int(round(seconds / 0.1)),
            }
        )
    return result


def _extract_tcn_inputs(
    frames,
    endpoints: np.ndarray,
    radar_start: datetime,
    extractor: RadarTemporalFeatureExtractorV2,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame_epochs = np.asarray([frame.timestamp.timestamp() for frame in frames])
    values: list[np.ndarray] = []
    times: list[float] = []
    for end_seconds in endpoints:
        end = radar_start + timedelta(seconds=float(end_seconds))
        left = bisect.bisect_left(
            frame_epochs,
            (end - timedelta(seconds=extractor.history_seconds + 0.06)).timestamp(),
        )
        right = bisect.bisect_right(
            frame_epochs, (end + timedelta(seconds=0.06)).timestamp()
        )
        if left >= right:
            continue
        window = extractor.transform(frames[left:right], end_timestamp=end)
        if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
            continue
        values.append(np.asarray(window.values, dtype=np.float32))
        times.append(float(end_seconds))
    return np.asarray(times), {"features": np.stack(values) if values else np.empty((0, 20, 19), np.float32)}


def _extract_point_inputs(
    frames,
    endpoints: np.ndarray,
    radar_start: datetime,
    builder: PointCloudSequenceBuilder,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame_epochs = np.asarray([frame.timestamp.timestamp() for frame in frames])
    values: list[np.ndarray] = []
    point_masks: list[np.ndarray] = []
    frame_masks: list[np.ndarray] = []
    times: list[float] = []
    for end_seconds in endpoints:
        end = radar_start + timedelta(seconds=float(end_seconds))
        left = bisect.bisect_left(
            frame_epochs,
            (end - timedelta(seconds=builder.history_seconds + 0.08)).timestamp(),
        )
        right = bisect.bisect_right(
            frame_epochs, (end + timedelta(seconds=0.08)).timestamp()
        )
        if left >= right:
            continue
        sequence = builder.transform(frames[left:right], end_timestamp=end)
        if int(sequence.frame_mask.sum()) < max(2, builder.time_steps // 2):
            continue
        values.append(sequence.values)
        point_masks.append(sequence.point_mask)
        frame_masks.append(sequence.frame_mask)
        times.append(float(end_seconds))
    return np.asarray(times), {
        "points": np.stack(values) if values else np.empty((0, 20, 64, 6), np.float32),
        "point_mask": np.stack(point_masks) if point_masks else np.empty((0, 20, 64), bool),
        "frame_mask": np.stack(frame_masks) if frame_masks else np.empty((0, 20), bool),
    }


def _score_tcn_condition(
    model: torch.nn.Module,
    raw_inputs: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    condition: dict[str, Any],
) -> np.ndarray:
    features = _transform_tcn(raw_inputs["features"], condition)
    normalized = ((features - mean[None, None]) / std[None, None]).astype(np.float32)
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(normalized))).numpy().astype(float)


def _score_point_condition(
    model: torch.nn.Module,
    raw_inputs: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    condition: dict[str, Any],
) -> np.ndarray:
    points, point_mask, frame_mask = _transform_point(raw_inputs, condition)
    normalized = (points - mean[None, None, None, :]) / std[None, None, None, :]
    normalized[..., 4] = np.where(points[..., 5] > 0.5, normalized[..., 4], 0.0)
    normalized[..., 5] = points[..., 5]
    normalized *= point_mask[..., None]
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(points), 256):
            logits = model(
                torch.from_numpy(normalized[start : start + 256].astype(np.float32)),
                torch.from_numpy(point_mask[start : start + 256]),
                torch.from_numpy(frame_mask[start : start + 256]),
            ).squeeze(-1)
            outputs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(outputs).astype(float)


def _transform_tcn(features: np.ndarray, condition: dict[str, Any]) -> np.ndarray:
    name = str(condition["condition"])
    if name == "original":
        return features
    if name in {"reverse", "random_shuffle"}:
        return features[:, np.asarray(condition["permutation"], dtype=int)].copy()
    if name == "repeat_last_frame":
        result = np.repeat(features[:, -1:, :], features.shape[1], axis=1).copy()
        for feature_name in DYNAMIC_FEATURE_NAMES:
            result[..., FEATURE_NAMES_V2.index(feature_name)] = 0.0
        result[..., FEATURE_NAMES_V2.index("interpolated_mask")] = 0.0
        return result
    if name.startswith("mask_last_"):
        result = features.copy()
        result[:, -int(condition["mask_steps"]) :, :] = 0.0
        return result
    raise ValueError(f"unsupported condition: {name}")


def _transform_point(
    raw_inputs: dict[str, np.ndarray], condition: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = raw_inputs["points"]
    point_mask = raw_inputs["point_mask"]
    frame_mask = raw_inputs["frame_mask"]
    name = str(condition["condition"])
    if name == "original":
        return points, point_mask, frame_mask
    if name in {"reverse", "random_shuffle"}:
        order = np.asarray(condition["permutation"], dtype=int)
        return points[:, order].copy(), point_mask[:, order].copy(), frame_mask[:, order].copy()
    if name == "repeat_last_frame":
        time_index = np.arange(frame_mask.shape[1])[None, :]
        last_valid = np.where(frame_mask, time_index, -1).max(axis=1)
        rows = np.arange(len(points))
        selected_points = points[rows, last_valid]
        selected_masks = point_mask[rows, last_valid]
        return (
            np.repeat(selected_points[:, None], points.shape[1], axis=1).copy(),
            np.repeat(selected_masks[:, None], points.shape[1], axis=1).copy(),
            np.ones_like(frame_mask),
        )
    if name.startswith("mask_last_"):
        result_points = points.copy()
        result_point_mask = point_mask.copy()
        result_frame_mask = frame_mask.copy()
        steps = int(condition["mask_steps"])
        result_points[:, -steps:] = 0.0
        result_point_mask[:, -steps:] = False
        result_frame_mask[:, -steps:] = False
        return result_points, result_point_mask, result_frame_mask
    raise ValueError(f"unsupported condition: {name}")


def _empty_accumulator(condition: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition": condition,
        "fall_event_count": 0,
        "detected_event_count": 0,
        "lead_values": [],
        "normal_recording_count": 0,
        "normal_duration_seconds": 0.0,
        "normal_confirmed_run_count": 0,
        "early_negative_window_count": 0,
        "early_negative_high_window_count": 0,
        "early_negative_recording_count": 0,
        "early_negative_recordings_with_high": 0,
    }


def _accumulate_recording(
    accumulator: dict[str, Any],
    *,
    relative: str,
    action: str,
    times: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    duration: float,
    onset: datetime | None,
    radar_start: datetime,
    confirmation_windows: int,
    step_seconds: float,
    minimum_lead_seconds: float,
    maximum_lead_seconds: float,
    early_negative_minimum_lead_seconds: float,
) -> None:
    high = scores >= threshold
    if action != DGUHA_FALL_ACTION:
        accumulator["normal_recording_count"] += 1
        accumulator["normal_duration_seconds"] += duration
        accumulator["normal_confirmed_run_count"] += len(
            _confirmed_run_end_indices(high, times, confirmation_windows, step_seconds)
        )
        return
    assert onset is not None
    accumulator["fall_event_count"] += 1
    onset_seconds = (onset - radar_start).total_seconds()
    leads = onset_seconds - times
    corridor = (
        (leads >= minimum_lead_seconds - 1e-6)
        & (leads <= maximum_lead_seconds + 1e-6)
    )
    corridor_indices = np.flatnonzero(corridor)
    run_ends = _confirmed_run_end_indices(
        high[corridor], times[corridor], confirmation_windows, step_seconds
    )
    if run_ends:
        accumulator["detected_event_count"] += 1
        accumulator["lead_values"].append(
            max(float(leads[corridor_indices[index]]) for index in run_ends)
        )
    early = leads >= early_negative_minimum_lead_seconds - 1e-6
    early_count = int(early.sum())
    if early_count:
        early_high = int(high[early].sum())
        accumulator["early_negative_window_count"] += early_count
        accumulator["early_negative_high_window_count"] += early_high
        accumulator["early_negative_recording_count"] += 1
        accumulator["early_negative_recordings_with_high"] += int(early_high > 0)


def _finalize_accumulator(
    *,
    model_name: str,
    checkpoint: dict[str, Any],
    condition: dict[str, Any],
    accumulator: dict[str, Any],
) -> dict[str, Any]:
    events = int(accumulator["fall_event_count"])
    duration = float(accumulator["normal_duration_seconds"])
    early_count = int(accumulator["early_negative_window_count"])
    leads = np.asarray(accumulator["lead_values"], dtype=float)
    return {
        "model": model_name,
        "condition": condition["condition"],
        "run_id": condition["run_id"],
        "seed": condition.get("seed"),
        "threshold": float(checkpoint["decision_threshold"]),
        "fall_event_count": events,
        "detected_event_count": int(accumulator["detected_event_count"]),
        "event_recall": int(accumulator["detected_event_count"]) / events if events else 0.0,
        "median_lead_seconds": float(np.median(leads)) if len(leads) else math.nan,
        "minimum_lead_seconds": float(np.min(leads)) if len(leads) else math.nan,
        "maximum_lead_seconds": float(np.max(leads)) if len(leads) else math.nan,
        "normal_recording_count": int(accumulator["normal_recording_count"]),
        "normal_duration_seconds": duration,
        "normal_confirmed_run_count": int(accumulator["normal_confirmed_run_count"]),
        "false_alarms_per_hour": (
            int(accumulator["normal_confirmed_run_count"]) * 3600.0 / duration
            if duration
            else 0.0
        ),
        "same_recording_early_negative_window_count": early_count,
        "same_recording_early_negative_high_window_count": int(
            accumulator["early_negative_high_window_count"]
        ),
        "same_recording_early_negative_false_positive_rate": (
            int(accumulator["early_negative_high_window_count"]) / early_count
            if early_count
            else math.nan
        ),
        "same_recording_early_negative_recording_count": int(
            accumulator["early_negative_recording_count"]
        ),
        "same_recording_early_recordings_with_any_false_positive": int(
            accumulator["early_negative_recordings_with_high"]
        ),
    }


def _summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    metrics = (
        "event_recall",
        "median_lead_seconds",
        "false_alarms_per_hour",
        "same_recording_early_negative_false_positive_rate",
    )
    for (model, condition), group in runs.groupby(["model", "condition"], sort=False):
        row: dict[str, Any] = {
            "model": model,
            "condition": condition,
            "run_count": len(group),
            "fall_event_count": int(group["fall_event_count"].iloc[0]),
            "normal_recording_count": int(group["normal_recording_count"].iloc[0]),
        }
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[metric] = float(np.mean(values)) if len(values) else math.nan
            row[f"{metric}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(np.min(values)) if len(values) else math.nan
            row[f"{metric}_max"] = float(np.max(values)) if len(values) else math.nan
        detected = group["detected_event_count"].to_numpy(dtype=float)
        row["detected_event_count_mean"] = float(np.mean(detected))
        row["detected_event_count_min"] = int(np.min(detected))
        row["detected_event_count_max"] = int(np.max(detected))
        output.append(row)
    return pd.DataFrame(output)


def _diagnostic_verdict(summary: pd.DataFrame) -> dict[str, Any]:
    verdict: dict[str, Any] = {}
    for model, group in summary.groupby("model"):
        indexed = group.set_index("condition")
        original = indexed.loc["original"]
        reverse = indexed.loc["reverse"]
        shuffled = indexed.loc["random_shuffle"]
        repeated = indexed.loc["repeat_last_frame"]
        verdict[str(model)] = {
            "reverse_event_recall_change": float(
                reverse["event_recall"] - original["event_recall"]
            ),
            "shuffle_event_recall_change": float(
                shuffled["event_recall"] - original["event_recall"]
            ),
            "repeat_last_event_recall_retention": float(
                repeated["event_recall"] / original["event_recall"]
                if original["event_recall"] > 0
                else 0.0
            ),
            "reverse_false_alarm_rate_change": float(
                reverse["false_alarms_per_hour"] - original["false_alarms_per_hour"]
            ),
            "interpretation_gate": (
                "Strong temporal dependence requires reverse/shuffle to materially reduce "
                "event recall and repeat-last/masked suffixes not to preserve the original "
                "recall-error profile."
            ),
        }
    return verdict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen TCN/PointNet temporal-dependency diagnostic")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--tcn-checkpoint", type=Path)
    parser.add_argument("--point-checkpoint", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoints = {}
    if args.tcn_checkpoint is not None:
        checkpoints["TCN_0p5_1p0"] = args.tcn_checkpoint
    if args.point_checkpoint is not None:
        checkpoints["PointNet_GRU_0p5_1p0"] = args.point_checkpoint
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    report = run_diagnostic(
        data_root=args.data_root,
        events_file=args.events,
        checkpoints=checkpoints,
        output_directory=args.output_directory,
    )
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
