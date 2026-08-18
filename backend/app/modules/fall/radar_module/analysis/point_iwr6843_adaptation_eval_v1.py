from __future__ import annotations

import argparse
import bisect
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_module.contracts import RadarFrame, RadarPoint, Room, SourceMode
from radar_module.model.dguha_event_evaluation_v2 import evaluate_dguha_events
from radar_module.model.point_iwr6843_adaptation_v1 import (
    FEATURE_NAMES,
    PREDICTION_VERSION,
)
from radar_module.model.pointnet_formal_prediction_v1 import FORMAL_MODEL_VERSION
from radar_module.model.point_temporal import PointTemporalEncoder, PointTemporalPredictionHead
from radar_module.preprocess.pointcloud_sequence import PointCloudSequenceBuilder


def evaluate_first_stage(
    checkpoint_directory: str | Path,
    dguha_data_root: str | Path,
    events_path: str | Path,
    output_directory: str | Path,
    *,
    replay_paths: tuple[str | Path, ...] = (),
    minimum_detected_events: int = 8,
) -> dict[str, Any]:
    checkpoints = Path(checkpoint_directory).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for checkpoint_path in sorted(checkpoints.glob("B[12]_seed*.pt")):
        variant = checkpoint_path.stem.split("_")[0]
        seed = int(checkpoint_path.stem.split("seed")[-1])
        calibration_path = output / f"{checkpoint_path.stem}_calibration_sweep.json"
        calibration = evaluate_dguha_events(
            dguha_data_root, events_path, checkpoint_path, calibration_path,
            split="validation", confirmation_windows=3,
            minimum_lead_seconds=0.5, maximum_lead_seconds=1.0,
            minimum_pre_descent_margin_seconds=0.1,
            early_negative_minimum_lead_seconds=1.2,
            decision_threshold_override=0.5,
        )
        selection = _select_event_threshold(calibration["threshold_sweep"], minimum_detected_events)
        locked_path = output / f"{checkpoint_path.stem}_locked_event_report.json"
        locked = evaluate_dguha_events(
            dguha_data_root, events_path, checkpoint_path, locked_path,
            split="validation", confirmation_windows=3,
            minimum_lead_seconds=0.5, maximum_lead_seconds=1.0,
            minimum_pre_descent_margin_seconds=0.1,
            early_negative_minimum_lead_seconds=1.2,
            decision_threshold_override=float(selection["threshold"]),
        )
        train_report = json.loads(checkpoint_path.with_suffix(".report.json").read_text(encoding="utf-8"))
        replay_results: list[dict[str, Any]] = []
        for replay_value in replay_paths:
            replay_path = Path(replay_value).resolve()
            replay_results.append(
                score_jsonl_replay(checkpoint_path, replay_path, float(selection["threshold"]))
            )
        result = {
            "variant": variant, "seed": seed,
            "checkpoint": str(checkpoint_path),
            "threshold_selection": selection,
            "dguha_validation": {
                "eligible_events": locked["eligible_fall_recording_count"],
                "detected_events": locked["prediction_corridor_detected_event_count"],
                "event_recall": locked["prediction_corridor_event_recall"],
                "median_lead_seconds": locked["corridor_confirmation_lead_seconds"]["median"],
                "normal_false_alarms_per_hour": locked["normal_confirmed_runs_per_hour"],
                "same_recording_early_false_positive_rate": locked["same_recording_early_negative_false_positive_rate"],
                "window_auroc": train_report["metrics"]["validation"]["auroc"],
                "normal_score_distribution": locked["normal_score_distribution"],
                "prediction_corridor_score_distribution": locked["prediction_corridor_score_distribution"],
            },
            "iwr6843_replay_audit_only": replay_results,
            "locked_report": str(locked_path),
        }
        results.append(result)
        (output / "evaluation_progress.json").write_text(
            json.dumps({"runs": results}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if not results:
        raise FileNotFoundError(f"no B1/B2 checkpoints found below {checkpoints}")
    aggregate = _aggregate(results)
    payload = {
        "experiment": "pointnet_iwr6843_adaptation_phase1",
        "threshold_policy": {
            "split": "DGUHA validation only",
            "minimum_detected_events": minimum_detected_events,
            "confirmation_windows": 3,
            "selection": "among candidates meeting recall floor, minimize normal FA/h then early FPR; prefer higher threshold on exact ties",
            "iwr_replay_used_for_selection": False,
        },
        "B0_frozen_reference": _load_b0_reference(),
        "runs": results,
        "aggregate": aggregate,
        "realtime_chain_modified": False,
    }
    (output / "phase1_evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return payload


def score_jsonl_replay(checkpoint_path: Path, replay_path: Path, threshold: float) -> dict[str, Any]:
    if not replay_path.is_file():
        return {"path": str(replay_path), "exists": False}
    checkpoint = _load_checkpoint(checkpoint_path)
    model = _build_model(checkpoint)
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    frames = _read_jsonl_frames(replay_path)
    builder = PointCloudSequenceBuilder()
    frame_epochs = np.asarray([frame.timestamp.timestamp() for frame in frames], dtype=np.float64)
    start = frames[0].timestamp
    duration = (frames[-1].timestamp - start).total_seconds()
    endpoints = np.arange(1.9, duration + 0.025, 0.1, dtype=np.float64)
    point_values: list[np.ndarray] = []
    point_masks: list[np.ndarray] = []
    frame_masks: list[np.ndarray] = []
    point_counts: list[int] = []
    times: list[float] = []
    skipped = 0
    for end_seconds in endpoints:
        end = start + timedelta(seconds=float(end_seconds))
        left = bisect.bisect_left(frame_epochs, (end - timedelta(seconds=2.08)).timestamp())
        right = bisect.bisect_right(frame_epochs, (end + timedelta(seconds=0.08)).timestamp())
        if left >= right:
            skipped += 1
            continue
        sequence = builder.transform(frames[left:right], end_timestamp=end)
        if int(sequence.frame_mask.sum()) < 10:
            skipped += 1
            continue
        point_values.append(sequence.values[..., :5])
        point_masks.append(sequence.point_mask)
        frame_masks.append(sequence.frame_mask)
        point_counts.append(int(sequence.point_mask.sum()))
        times.append(float(end_seconds))
    if not point_values:
        return {"path": str(replay_path), "exists": True, "valid_window_count": 0, "skipped_window_count": skipped}
    raw = np.stack(point_values).astype(np.float32, copy=False)
    point_mask = np.stack(point_masks)
    frame_mask = np.stack(frame_masks)
    modes: dict[str, Any] = {}
    for snr_mode in ("measured_snr", "zero_snr"):
        values = raw.copy()
        normalized = (values - mean[None, None, None, :]) / std[None, None, None, :]
        if snr_mode == "zero_snr":
            normalized[..., 4] = 0.0
        normalized[~point_mask] = 0.0
        scores = _score_batches(model, normalized, point_mask, frame_mask)
        high = scores >= threshold
        run_count = len(_confirmed_runs(high, np.asarray(times), 3, 0.1))
        modes[snr_mode] = {
            "score_distribution": _describe(scores),
            "above_threshold_window_count": int(high.sum()),
            "above_threshold_window_fraction": float(high.mean()),
            "confirmed_run_count": run_count,
        }
    intervals = np.diff([frame.timestamp.timestamp() for frame in frames])
    return {
        "path": str(replay_path), "exists": True,
        "frame_count": len(frames), "duration_seconds": duration,
        "frame_rate_hz": float((len(frames) - 1) / duration) if duration > 0 else 0.0,
        "median_frame_interval_seconds": float(np.median(intervals)) if len(intervals) else None,
        "valid_window_count": len(raw), "skipped_window_count": skipped,
        "points_per_window": _describe(point_counts),
        "threshold": threshold,
        "snr_contract_warning": (
            "Fall-102 was converted from 0.1 dB side-info counts to dB using TI official parser semantics; zero_snr is an ablation"
            if checkpoint.get("model_version") == FORMAL_MODEL_VERSION
            else "Fall-102 and live TLV SNR units are not proven compatible; both modes are audit outputs"
        ),
        "modes": modes,
    }


def _select_event_threshold(sweep: list[dict[str, Any]], minimum_detected_events: int) -> dict[str, Any]:
    eligible = [row for row in sweep if int(row["prediction_corridor_detected_event_count"]) >= minimum_detected_events]
    floor_met = bool(eligible)
    candidates = eligible or sweep
    if floor_met:
        selected = min(candidates, key=lambda row: (
            float(row["normal_confirmed_active_seconds_per_hour"]),
            float(row["same_recording_early_negative_false_positive_rate"]),
            int(row["normal_recordings_with_confirmed_run"]),
            float(row["normal_above_threshold_window_fraction"]),
            float(row["normal_confirmed_runs_per_hour"]),
            -float(row["threshold"]),
        ))
    else:
        selected = min(candidates, key=lambda row: (
            -int(row["prediction_corridor_detected_event_count"]),
            float(row["normal_confirmed_active_seconds_per_hour"]),
            float(row["same_recording_early_negative_false_positive_rate"]),
            int(row["normal_recordings_with_confirmed_run"]),
            float(row["normal_above_threshold_window_fraction"]),
            float(row["normal_confirmed_runs_per_hour"]),
            -float(row["threshold"]),
        ))
    return {
        "threshold": float(selected["threshold"]), "recall_floor_met": floor_met,
        "detected_events_at_selection": int(selected["prediction_corridor_detected_event_count"]),
        "normal_false_alarms_per_hour_at_selection": float(selected["normal_confirmed_runs_per_hour"]),
        "normal_confirmed_active_seconds_per_hour_at_selection": float(
            selected["normal_confirmed_active_seconds_per_hour"]
        ),
        "normal_confirmed_active_window_fraction_at_selection": float(
            selected["normal_confirmed_active_window_fraction"]
        ),
        "normal_above_threshold_window_fraction_at_selection": float(
            selected["normal_above_threshold_window_fraction"]
        ),
        "normal_recordings_with_confirmed_run_at_selection": int(
            selected["normal_recordings_with_confirmed_run"]
        ),
        "early_false_positive_rate_at_selection": float(selected["same_recording_early_negative_false_positive_rate"]),
    }


def _read_jsonl_frames(path: Path) -> tuple[RadarFrame, ...]:
    frames: list[RadarFrame] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            points = tuple(RadarPoint(
                x=float(point["x"]), y=float(point["y"]), z=float(point["z"]),
                velocity=float(point.get("velocity", 0.0)),
                snr=float(point["snr"]) if point.get("snr") is not None else None,
            ) for point in payload.get("points", ()))
            frames.append(RadarFrame(
                timestamp=datetime.fromisoformat(payload["timestamp"]),
                device_id=str(payload.get("device_id", "iwr6843-replay")),
                room=Room.BATHROOM, source_mode=SourceMode.REPLAY, points=points,
            ))
    if not frames:
        raise ValueError(f"empty JSONL replay: {path}")
    return tuple(frames)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_version") not in {PREDICTION_VERSION, FORMAL_MODEL_VERSION}:
        raise ValueError("checkpoint version mismatch")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("checkpoint feature order mismatch")
    return checkpoint


def _build_model(checkpoint: dict[str, Any]) -> PointTemporalPredictionHead:
    encoder = PointTemporalEncoder(
        input_size=len(FEATURE_NAMES), frame_hidden_size=int(checkpoint["frame_hidden_size"]),
        temporal_hidden_size=int(checkpoint["temporal_hidden_size"]),
    )
    model = PointTemporalPredictionHead(encoder, horizon_count=1)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _score_batches(model: torch.nn.Module, values: np.ndarray, point_mask: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), 256):
            logits = model(
                torch.from_numpy(values[start : start + 256]),
                torch.from_numpy(point_mask[start : start + 256]),
                torch.from_numpy(frame_mask[start : start + 256]),
            ).squeeze(-1)
            scores.append(torch.sigmoid(logits).numpy())
    return np.concatenate(scores).astype(np.float64)


def _confirmed_runs(high: np.ndarray, times: np.ndarray, count: int, step: float) -> list[int]:
    result: list[int] = []
    length = 0
    latched = False
    previous: float | None = None
    for index, (value, timestamp) in enumerate(zip(high, times)):
        contiguous = previous is not None and timestamp - previous <= step * 1.5 + 1e-9
        if not value or (previous is not None and not contiguous):
            length, latched = 0, False
        if value:
            length += 1
            if length >= count and not latched:
                result.append(index)
                latched = True
        previous = float(timestamp)
    return result


def _describe(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(len(array)), "min": float(array.min()), "median": float(np.median(array)),
        "p90": float(np.quantile(array, .9)), "p95": float(np.quantile(array, .95)),
        "p99": float(np.quantile(array, .99)), "max": float(array.max()),
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for variant in ("B1", "B2"):
        selected = [item for item in results if item["variant"] == variant]
        metrics = selected[0]["dguha_validation"].keys()
        output[variant] = {
            metric: _describe([item["dguha_validation"][metric] for item in selected])
            for metric in metrics
            if isinstance(selected[0]["dguha_validation"][metric], (int, float))
        }
        output[variant]["threshold"] = _describe([item["threshold_selection"]["threshold"] for item in selected])
    return output


def _load_b0_reference() -> dict[str, Any]:
    path = Path("reports/competition_b0_v1/metrics.json").resolve()
    if not path.is_file():
        return {"available": False, "touched": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"available": True, "source": str(path), "touched": False, **payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate first-stage PointNet IWR6843 adaptation.")
    parser.add_argument("--checkpoint-directory", required=True, type=Path)
    parser.add_argument("--dguha-data-root", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--replay", action="append", default=[], type=Path)
    args = parser.parse_args()
    result = evaluate_first_stage(
        args.checkpoint_directory, args.dguha_data_root, args.events, args.output_directory,
        replay_paths=tuple(args.replay),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
