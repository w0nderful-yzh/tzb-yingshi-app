from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_module.analysis.point_iwr6843_adaptation_eval_v1 import (
    _select_event_threshold,
    score_jsonl_replay,
)
from radar_module.dataset.point_iwr6843_adaptation_v1 import DGUHA_MODE, IWR_MODE
from radar_module.model.dguha_event_evaluation_v2 import evaluate_dguha_events
from radar_module.model.point_iwr6843_adaptation_v1 import _load_dataset
from radar_module.model.pointnet_formal_prediction_v1 import (
    FORMAL_MODEL_VERSION,
    IWR_HARD_ACTIONS,
    SIT_STAND_TOKEN,
    _model_from_prediction,
    _scores,
)


def evaluate_formal_models(
    checkpoint_directory: str | Path,
    dguha_dataset: str | Path,
    iwr_dataset: str | Path,
    dguha_data_root: str | Path,
    events_path: str | Path,
    output_directory: str | Path,
    *,
    replay_paths: tuple[str | Path, ...] = (),
    minimum_detected_events: int = 8,
) -> dict[str, Any]:
    checkpoint_root = Path(checkpoint_directory).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dguha_path = Path(dguha_dataset).resolve()
    iwr_path = Path(iwr_dataset).resolve()
    dguha = _load_dataset(dguha_path, DGUHA_MODE)
    iwr = _load_dataset(iwr_path, IWR_MODE)
    results: list[dict[str, Any]] = []
    for checkpoint_path in sorted(checkpoint_root.glob("P[23]_*_seed*.pt")):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_version") != FORMAL_MODEL_VERSION:
            continue
        variant = str(checkpoint["variant"])
        seed = int(checkpoint["seed"])
        calibration_path = output / f"{checkpoint_path.stem}_calibration_sweep.json"
        calibration = evaluate_dguha_events(
            dguha_data_root, events_path, checkpoint_path, calibration_path,
            split="validation", confirmation_windows=3,
            minimum_lead_seconds=0.5, maximum_lead_seconds=1.0,
            early_negative_minimum_lead_seconds=1.2,
            decision_threshold_override=0.5,
        )
        selection = _select_event_threshold(calibration["threshold_sweep"], minimum_detected_events)
        threshold = float(selection["threshold"])
        locked_path = output / f"{checkpoint_path.stem}_locked_event_report.json"
        locked = evaluate_dguha_events(
            dguha_data_root, events_path, checkpoint_path, locked_path,
            split="validation", confirmation_windows=3,
            minimum_lead_seconds=0.5, maximum_lead_seconds=1.0,
            early_negative_minimum_lead_seconds=1.2,
            decision_threshold_override=threshold,
        )
        subgroup = _subgroup_metrics(checkpoint, dguha, iwr, threshold)
        replay = [score_jsonl_replay(checkpoint_path, Path(value).resolve(), threshold) for value in replay_paths]
        results.append({
            "variant": variant, "seed": seed,
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
            "threshold_selection": selection,
            "dguha_validation": {
                "eligible_events": locked["eligible_fall_recording_count"],
                "detected_events": locked["prediction_corridor_detected_event_count"],
                "event_recall": locked["prediction_corridor_event_recall"],
                "median_lead_seconds": locked["corridor_confirmation_lead_seconds"]["median"],
                "normal_false_alarms_per_hour": locked["normal_confirmed_runs_per_hour"],
                "normal_confirmed_active_seconds_per_hour": locked["normal_confirmed_active_seconds_per_hour"],
                "normal_confirmed_active_window_fraction": locked["normal_confirmed_active_window_fraction"],
                "normal_above_threshold_window_fraction": locked["normal_above_threshold_window_fraction"],
                "normal_recordings_with_confirmed_run": locked["normal_recordings_with_confirmed_run"],
                "same_recording_early_false_positive_rate": locked["same_recording_early_negative_false_positive_rate"],
                "normal_score_distribution": locked["normal_score_distribution"],
                "prediction_corridor_score_distribution": locked["prediction_corridor_score_distribution"],
            },
            "hard_negative_validation": subgroup,
            "iwr6843_replay_audit_only": replay,
            "locked_report": str(locked_path),
        })
        (output / "evaluation_progress.json").write_text(
            json.dumps({"runs": results}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if not results:
        raise FileNotFoundError("no formal P2/P3 checkpoints found")
    selected = _select_final(results)
    final_path = checkpoint_root / "pointnet_gru_radar_branch_v1.pt"
    _write_locked_checkpoint(Path(selected["checkpoint"]), final_path, float(selected["threshold_selection"]["threshold"]), selected)
    payload = {
        "experiment": "pointnet_gru_formal_prediction_v1",
        "selection_priority": [
            "at least 8/14 DGUHA validation events",
            "lower IWR bow/squat and DGUHA sit/stand false positives",
            "lower same-recording early false positives",
            "lower confirmed-risk active seconds/hour and high-score window fraction",
            "lower continuous normal false-alarm run count/hour",
        ],
        "runs": results,
        "selected": {**selected, "locked_checkpoint": str(final_path), "locked_checkpoint_sha256": _sha256(final_path)},
        "tcn_b0_modified": False, "realtime_chain_modified": False,
    }
    (output / "formal_evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return payload


def _subgroup_metrics(
    checkpoint: dict[str, Any], dguha: dict[str, np.ndarray], iwr: dict[str, np.ndarray], threshold: float,
) -> dict[str, Any]:
    model = _model_from_prediction(checkpoint)
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    dguha_validation = dguha["split"] == "validation"
    dguha_indices = np.flatnonzero(dguha_validation)
    dguha_scores = _scores(model, dguha, dguha_indices, mean, std, torch.device("cpu"), 256)
    labels = dguha["labels"][dguha_indices]
    sources = dguha["source_files"][dguha_indices]
    label_source = dguha["label_source"][dguha_indices]
    sit = np.asarray([SIT_STAND_TOKEN in str(path) for path in sources]) & (labels == 0)
    early = label_source == "dguha_same_fall_recording_outside_prediction_horizon"
    iwr_validation = iwr["split"] == "validation"
    iwr_indices = np.flatnonzero(iwr_validation)
    iwr_scores = _scores(model, iwr, iwr_indices, mean, std, torch.device("cpu"), 256)
    actions = iwr["action"][iwr_indices]
    output: dict[str, Any] = {
        "dguha_sit_stand": _negative_metrics(dguha_scores[sit], threshold),
        "dguha_same_recording_early": _negative_metrics(dguha_scores[early], threshold),
    }
    for action in ("bow", "squat", "walk"):
        output[f"iwr_{action}"] = _negative_metrics(iwr_scores[actions == action], threshold)
    return output


def _negative_metrics(scores: np.ndarray, threshold: float) -> dict[str, Any]:
    return {
        "count": int(len(scores)), "above_threshold_count": int(np.sum(scores >= threshold)),
        "false_positive_rate": float(np.mean(scores >= threshold)) if len(scores) else 0.0,
        "score_median": float(np.median(scores)) if len(scores) else None,
        "score_p95": float(np.quantile(scores, .95)) if len(scores) else None,
        "score_max": float(scores.max()) if len(scores) else None,
    }


def _select_final(results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in results if item["dguha_validation"]["detected_events"] >= 8]
    candidates = eligible or results
    return min(candidates, key=lambda item: (
        max(
            item["hard_negative_validation"]["iwr_bow"]["false_positive_rate"],
            item["hard_negative_validation"]["iwr_squat"]["false_positive_rate"],
            item["hard_negative_validation"]["dguha_sit_stand"]["false_positive_rate"],
        ),
        item["hard_negative_validation"]["dguha_same_recording_early"]["false_positive_rate"],
        item["dguha_validation"]["normal_confirmed_active_seconds_per_hour"],
        item["dguha_validation"]["normal_above_threshold_window_fraction"],
        item["dguha_validation"]["normal_recordings_with_confirmed_run"],
        item["dguha_validation"]["normal_false_alarms_per_hour"],
        -item["dguha_validation"]["detected_events"],
    ))


def _write_locked_checkpoint(source: Path, destination: Path, threshold: float, selection: dict[str, Any]) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    checkpoint["decision_threshold"] = threshold
    checkpoint["decision_threshold_policy"] = "locked DGUHA continuous-event validation; minimum 8/14 events then low false alarms"
    checkpoint["selected_for_radar_branch"] = True
    checkpoint["selection_summary"] = {
        "event_recall": selection["dguha_validation"]["event_recall"],
        "median_lead_seconds": selection["dguha_validation"]["median_lead_seconds"],
        "normal_false_alarms_per_hour": selection["dguha_validation"]["normal_false_alarms_per_hour"],
        "normal_confirmed_active_seconds_per_hour": selection["dguha_validation"]["normal_confirmed_active_seconds_per_hour"],
        "normal_confirmed_active_window_fraction": selection["dguha_validation"]["normal_confirmed_active_window_fraction"],
        "same_recording_early_false_positive_rate": selection["dguha_validation"]["same_recording_early_false_positive_rate"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate formal PointNet-GRU radar branch candidates.")
    parser.add_argument("--checkpoint-directory", required=True, type=Path)
    parser.add_argument("--dguha-dataset", required=True, type=Path)
    parser.add_argument("--iwr-dataset", required=True, type=Path)
    parser.add_argument("--dguha-data-root", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--replay", action="append", default=[], type=Path)
    args = parser.parse_args()
    result = evaluate_formal_models(
        args.checkpoint_directory, args.dguha_dataset, args.iwr_dataset,
        args.dguha_data_root, args.events, args.output_directory,
        replay_paths=tuple(args.replay),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
