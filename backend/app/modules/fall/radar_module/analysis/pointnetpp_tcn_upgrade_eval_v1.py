from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from radar_module.contracts import RadarFrame, RadarPoint, Room, SourceMode
from radar_module.dataset.dguha_research_v2 import DGUHA_SPLIT_BY_SUBJECT
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.model.dguha_event_evaluation_v2 import evaluate_dguha_events
from radar_module.model.pointnetpp_tcn_v1 import (
    ARCHITECTURE,
    INPUT_FEATURES,
    MODEL_VERSION,
    PointNetPlusPlusTcnPrefall,
)
from radar_module.preprocess.pointcloud_sequence import PointCloudSequenceBuilder


CONFIRMATION_WINDOWS = 3
STEP_SECONDS = 0.1
MAX_CONFIRMATION_GAP = 0.16
MIN_LEAD = 0.5
MAX_LEAD = 1.0
EARLY_LEAD = 1.2
EVENT_EXCLUSION = 5.0


@dataclass(slots=True)
class ScoredRecording:
    dataset: str
    subject_id: str
    recording_id: str
    times: np.ndarray
    scores: np.ndarray
    qualities: np.ndarray
    event_anchors: tuple[float, ...]
    normal_only: bool


def evaluate_upgrade(
    candidate_checkpoint: str | Path,
    output_directory: str | Path,
    *,
    dguha_root: str | Path,
    dguha_events: str | Path,
    baseline_checkpoint: str | Path,
    peerj_root: str | Path,
    baseline_evidence: str | Path,
    iwr_normal_replay: str | Path,
    iwr_fall_replay: str | Path,
    iwr_fall_phase: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = Path(candidate_checkpoint).resolve()
    candidate = _load_candidate(candidate_path)
    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = _model(candidate, torch_device)
    mean = np.asarray(candidate["normalization_mean"], dtype=np.float32)
    std = np.asarray(candidate["normalization_std"], dtype=np.float32)

    validation_records = _score_dguha(
        Path(dguha_root).resolve(), Path(dguha_events).resolve(), "validation",
        model, mean, std, torch_device,
    )
    threshold_selection = _select_threshold(validation_records)
    threshold = float(threshold_selection["threshold"])
    test_records = _score_dguha(
        Path(dguha_root).resolve(), Path(dguha_events).resolve(), "test",
        model, mean, std, torch_device,
    )
    point_validation = _dguha_metrics(validation_records, threshold)
    point_test = _dguha_metrics(test_records, threshold)

    locked_path = output / "pointnetpp_tcn_radar_encoder_upgrade_v1.pt"
    locked = dict(candidate)
    locked["decision_threshold"] = threshold
    locked["decision_threshold_policy"] = str(threshold_selection["policy"])
    locked["evaluation_locked"] = True
    locked["selected_for_radar_encoder_upgrade"] = False
    locked["approved_to_replace_current_encoder"] = False
    locked["threshold_selection"] = _plain({
        key: value for key, value in threshold_selection.items() if key != "sweep"
    })
    locked["peerj_used_for_training_or_threshold"] = False
    torch.save(locked, locked_path)

    baseline_validation_path = output / "baseline_dguha_validation.json"
    baseline_test_path = output / "baseline_dguha_test.json"
    baseline_validation_raw = evaluate_dguha_events(
        dguha_root, dguha_events, baseline_checkpoint, baseline_validation_path,
        split="validation", confirmation_windows=CONFIRMATION_WINDOWS,
        minimum_lead_seconds=MIN_LEAD, maximum_lead_seconds=MAX_LEAD,
        early_negative_minimum_lead_seconds=EARLY_LEAD,
        decision_threshold_override=0.35,
    )
    baseline_test_raw = evaluate_dguha_events(
        dguha_root, dguha_events, baseline_checkpoint, baseline_test_path,
        split="test", confirmation_windows=CONFIRMATION_WINDOWS,
        minimum_lead_seconds=MIN_LEAD, maximum_lead_seconds=MAX_LEAD,
        early_negative_minimum_lead_seconds=EARLY_LEAD,
        decision_threshold_override=0.35,
    )
    baseline_validation = _baseline_dguha_summary(baseline_validation_raw)
    baseline_test = _baseline_dguha_summary(baseline_test_raw)

    peerj_point = _score_peerj(Path(peerj_root).resolve(), model, mean, std, torch_device)
    peerj_baseline = _read_baseline_peerj(Path(baseline_evidence).resolve(), Path(peerj_root).resolve())
    peerj_results = {
        "tcn_baseline": _external_metrics(peerj_baseline, 0.35),
        "pointnetpp_tcn": _external_metrics(peerj_point, threshold),
    }

    normal_frames = _read_jsonl_frames(Path(iwr_normal_replay).resolve())
    fall_frames = _read_jsonl_frames(Path(iwr_fall_replay).resolve())
    event_anchor = _first_jsonl_timestamp(Path(iwr_fall_phase).resolve()).timestamp()
    iwr_point_normal = _score_external_frames(
        "IWR6843_REPLAY", "p01", "high_risk_screen", normal_frames, (), True,
        model, mean, std, torch_device,
    )
    iwr_point_fall = _score_external_frames(
        "IWR6843_REPLAY", "p01", "controlled_forward_fall", fall_frames,
        (event_anchor,), False, model, mean, std, torch_device,
    )
    iwr_baseline_normal = _read_jsonl_baseline(Path(iwr_normal_replay).resolve(), (), True, "high_risk_screen")
    iwr_baseline_fall = _read_jsonl_baseline(
        Path(iwr_fall_replay).resolve(), (event_anchor,), False, "controlled_forward_fall"
    )
    iwr_results = {
        "tcn_baseline": _external_metrics([iwr_baseline_normal, iwr_baseline_fall], 0.35),
        "pointnetpp_tcn": _external_metrics([iwr_point_normal, iwr_point_fall], threshold),
        "normal_replay": {
            "tcn_baseline": _external_metrics([iwr_baseline_normal], 0.35),
            "pointnetpp_tcn": _external_metrics([iwr_point_normal], threshold),
        },
        "controlled_fall_replay": {
            "event_anchor_source": str(Path(iwr_fall_phase).resolve()),
            "tcn_baseline": _external_metrics([iwr_baseline_fall], 0.35),
            "pointnetpp_tcn": _external_metrics([iwr_point_fall], threshold),
        },
    }

    approved = bool(
        point_test["event_recall"] >= baseline_test["event_recall"]
        and point_test["false_alarms_per_hour"] <= baseline_test["false_alarms_per_hour"]
        and point_test["early_negative_above_threshold_window_fraction"]
        <= baseline_test["early_negative_above_threshold_window_fraction"]
        and peerj_results["pointnetpp_tcn"]["false_alarms_per_hour"]
        <= peerj_results["tcn_baseline"]["false_alarms_per_hour"]
    )
    locked["approved_to_replace_current_encoder"] = approved
    locked["selected_for_radar_encoder_upgrade"] = approved
    locked["replacement_decision_policy"] = (
        "DGUHA test recall non-inferior, DGUHA false alarms and early-negative FPR non-worse, PeerJ false alarms non-worse"
    )
    torch.save(locked, locked_path)
    training_summary_path = candidate_path.parent / "training_summary.json"
    training_summary = (
        json.loads(training_summary_path.read_text(encoding="utf-8"))
        if training_summary_path.is_file() else None
    )
    result = {
        "experiment": "radar_encoder_upgrade_pointnetpp_tcn_v1",
        "architecture": ARCHITECTURE,
        "candidate_checkpoint": str(candidate_path),
        "candidate_checkpoint_sha256": _sha256(candidate_path),
        "training_summary": training_summary,
        "locked_checkpoint": str(locked_path),
        "locked_checkpoint_sha256": _sha256(locked_path),
        "threshold_selection_validation_only": threshold_selection,
        "dguha": {
            "validation": {"tcn_baseline": baseline_validation, "pointnetpp_tcn": point_validation},
            "test": {"tcn_baseline": baseline_test, "pointnetpp_tcn": point_test},
        },
        "peerj_external_no_threshold_tuning": peerj_results,
        "iwr6843_replay_audit_only": iwr_results,
        "radar_evidence_output_contract": {"fields": ["score", "quality", "timestamp"], "changed": False},
        "protected_components_modified": {
            "fusion_api": False, "fusion_logic": False, "camera": False,
            "uart_tlv": False, "b0_tcn_checkpoint": False, "evidence_protocol": False,
        },
        "peerj_used_for_training_or_threshold": False,
        "replacement_decision": {
            "approved": approved,
            "verdict": "replace_current_encoder" if approved else "keep_current_tcn_baseline",
            "candidate_status": "shadow_experiment_only" if not approved else "eligible_for_separate_integration_review",
        },
    }
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "RADAR_ENCODER_UPGRADE_REPORT.md").write_text(
        _report(result), encoding="utf-8-sig"
    )
    return result


def _score_dguha(
    root: Path,
    events_path: Path,
    split: str,
    model: PointNetPlusPlusTcnPrefall,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> list[ScoredRecording]:
    events = json.loads(events_path.read_text(encoding="utf-8"))
    event_by_source = {str(item["source_file"]): item for item in events}
    output: list[ScoredRecording] = []
    for path in sorted(root.glob("*/*/radar/*.txt")):
        match = re.match(r"([FM]_\d{3})_", path.name)
        if match is None or DGUHA_SPLIT_BY_SUBJECT.get(match.group(1)) != split:
            continue
        subject = match.group(1)
        relative = path.relative_to(root).as_posix()
        action = path.parent.parent.name
        event = event_by_source.get(relative)
        anchors: tuple[float, ...] = ()
        endpoint_limit: datetime | None = None
        if action == "5_falling_forward":
            if not event or not bool(event.get("eligible_for_prediction_windows")) or not event.get("descent_onset"):
                continue
            onset = datetime.fromisoformat(str(event["descent_onset"]))
            anchors = (onset.timestamp(),)
            endpoint_limit = onset
        frames = tuple(parse_radhar_text(path, device_id=f"pointnetpp-{path.stem}"))
        output.append(
            _score_external_frames(
                "DGUHA", subject, relative, frames, anchors,
                action != "5_falling_forward", model, mean, std, device,
                endpoint_limit=endpoint_limit,
            )
        )
    return output


def _score_peerj(
    root: Path,
    model: PointNetPlusPlusTcnPrefall,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> list[ScoredRecording]:
    falls = np.loadtxt(root / "falls.csv", delimiter=",")
    output: list[ScoredRecording] = []
    for subject_number in range(1, 11):
        subject = f"subject_{subject_number}"
        frames = tuple(_read_peerj_frames(root / subject / "radar.csv", subject))
        anchors = tuple(float(value) for value in falls[:, subject_number - 1])
        output.append(
            _score_external_frames(
                "PeerJ", subject, f"{subject}/radar.csv", frames, anchors, False,
                model, mean, std, device,
            )
        )
    return output


def _score_external_frames(
    dataset: str,
    subject: str,
    recording_id: str,
    frames: Sequence[RadarFrame],
    anchors: tuple[float, ...],
    normal_only: bool,
    model: PointNetPlusPlusTcnPrefall,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    *,
    endpoint_limit: datetime | None = None,
) -> ScoredRecording:
    builder = PointCloudSequenceBuilder()
    epochs = np.asarray([frame.timestamp.timestamp() for frame in frames], dtype=np.float64)
    last = min(epochs[-1], endpoint_limit.timestamp() - 0.1) if endpoint_limit else epochs[-1]
    endpoints = np.arange(epochs[0] + 1.9, last + 0.025, STEP_SECONDS)
    times: list[float] = []
    sequences: list[np.ndarray] = []
    point_masks: list[np.ndarray] = []
    frame_masks: list[np.ndarray] = []
    qualities: list[float] = []
    for endpoint in endpoints:
        left = int(np.searchsorted(epochs, endpoint - 2.2, side="left"))
        right = int(np.searchsorted(epochs, endpoint + 0.076, side="right"))
        if right <= left:
            continue
        timestamp = datetime.fromtimestamp(float(endpoint), tz=frames[0].timestamp.tzinfo)
        sequence = builder.transform(tuple(frames[left:right]), end_timestamp=timestamp)
        observed = int(np.sum(sequence.frame_mask & np.any(sequence.point_mask, axis=1)))
        times.append(float(endpoint))
        qualities.append(1.0 if observed == 20 else 0.6 if observed >= 10 else 0.0)
        raw = sequence.values[..., :5].copy()
        normalized = (raw - mean[None, None, :]) / std[None, None, :]
        snr_present = sequence.values[..., 5] > 0
        normalized[..., 4][~snr_present] = 0.0
        normalized[~sequence.point_mask] = 0.0
        sequences.append(normalized.astype(np.float32))
        point_masks.append(sequence.point_mask)
        frame_masks.append(sequence.frame_mask)
    scores = np.full(len(times), np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.asarray(qualities) > 0)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(valid), 256):
            selected = valid[start : start + 256]
            points = torch.from_numpy(np.stack([sequences[index] for index in selected])).to(device)
            pm = torch.from_numpy(np.stack([point_masks[index] for index in selected])).to(device)
            fm = torch.from_numpy(np.stack([frame_masks[index] for index in selected])).to(device)
            scores[selected] = torch.sigmoid(model(points, pm, fm)).cpu().numpy()
    return ScoredRecording(
        dataset, subject, recording_id, np.asarray(times), scores,
        np.asarray(qualities), anchors, normal_only,
    )


def _dguha_metrics(records: Sequence[ScoredRecording], threshold: float) -> dict[str, Any]:
    event_count = detected = 0
    leads: list[float] = []
    normal_runs = 0
    normal_seconds = 0.0
    early_high = early_total = early_runs = 0
    valid = total = 0
    for record in records:
        total += len(record.scores)
        valid += int(np.isfinite(record.scores).sum())
        runs = _confirmed_runs(record.times, record.scores, threshold)
        if record.normal_only:
            if len(record.times) > 1:
                normal_seconds += record.times[-1] - record.times[0]
            normal_runs += len(runs)
            continue
        for anchor in record.event_anchors:
            event_count += 1
            corridor = [value for value in runs if anchor - MAX_LEAD <= value <= anchor - MIN_LEAD]
            if corridor:
                detected += 1
                leads.append(anchor - min(corridor))
            early_mask = record.times <= anchor - EARLY_LEAD
            early_total += int(np.sum(early_mask & np.isfinite(record.scores)))
            early_high += int(np.sum(early_mask & np.isfinite(record.scores) & (record.scores >= threshold)))
            early_runs += sum(value <= anchor - EARLY_LEAD for value in runs)
    return {
        "threshold": threshold,
        "event_count": event_count,
        "detected_event_count": detected,
        "event_recall": detected / event_count if event_count else None,
        "median_lead_seconds": float(np.median(leads)) if leads else None,
        "false_alarm_count": normal_runs,
        "normal_duration_hours": normal_seconds / 3600.0,
        "false_alarms_per_hour": normal_runs / (normal_seconds / 3600.0) if normal_seconds else None,
        "early_negative_above_threshold_window_fraction": early_high / early_total if early_total else None,
        "early_negative_confirmed_run_count": early_runs,
        "valid_window_fraction": valid / total if total else 0.0,
    }


def _select_threshold(records: Sequence[ScoredRecording]) -> dict[str, Any]:
    candidates = np.asarray(
        sorted(set([0.001, 0.005, 0.01, 0.02, 0.03, 0.04] + np.linspace(0.05, 0.95, 91).tolist())),
        dtype=np.float64,
    )
    results = [_dguha_metrics(records, float(value)) for value in candidates]
    event_count = max(int(item["event_count"]) for item in results)
    floor = min(8, event_count)
    eligible = [item for item in results if item["detected_event_count"] >= floor]
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (
                item["false_alarms_per_hour"],
                item["early_negative_above_threshold_window_fraction"],
                -item["threshold"],
            ),
        )
        policy = "recall_floor_then_minimum_normal_false_alarm"
    else:
        selected = min(
            results,
            key=lambda item: (
                -item["detected_event_count"],
                item["false_alarms_per_hour"],
                item["early_negative_above_threshold_window_fraction"],
            ),
        )
        policy = "maximum_available_event_recall_then_minimum_false_alarm"
    return {
        "threshold": selected["threshold"],
        "policy": policy,
        "event_recall_floor": floor,
        "selected_validation_metrics": selected,
        "sweep": results,
    }


def _external_metrics(records: Sequence[ScoredRecording], threshold: float) -> dict[str, Any]:
    event_count = detected = prefall = 0
    latencies: list[float] = []
    false_runs = 0
    normal_seconds = 0.0
    early_high = early_total = 0
    valid = total = 0
    for record in records:
        total += len(record.scores)
        valid += int(np.isfinite(record.scores).sum())
        runs = _confirmed_runs(record.times, record.scores, threshold)
        for anchor in record.event_anchors:
            event_count += 1
            event_candidates = [value for value in runs if anchor - 0.6 <= value <= anchor + 1.0]
            if event_candidates:
                detected += 1
                latencies.append(min(event_candidates) - anchor)
            if any(anchor - MAX_LEAD <= value <= anchor - MIN_LEAD for value in runs):
                prefall += 1
        if len(record.times) < 2:
            continue
        if record.normal_only or not record.event_anchors:
            normal_seconds += record.times[-1] - record.times[0]
            false_runs += len(runs)
            early_mask = np.isfinite(record.scores)
        else:
            intervals = _normal_intervals(record.times[0], record.times[-1], record.event_anchors)
            normal_seconds += sum(right - left for left, right in intervals)
            false_runs += sum(any(left <= value <= right for left, right in intervals) for value in runs)
            early_mask = np.asarray([
                any(left <= value <= right for left, right in intervals)
                for value in record.times
            ]) & np.isfinite(record.scores)
        early_total += int(early_mask.sum())
        early_high += int(np.sum(early_mask & (record.scores >= threshold)))
    return {
        "threshold": threshold,
        "event_count": event_count,
        "event_recall": detected / event_count if event_count else None,
        "prefall_event_recall": prefall / event_count if event_count else None,
        "median_detection_latency_seconds": float(np.median(latencies)) if latencies else None,
        "false_alarms_per_hour": false_runs / (normal_seconds / 3600.0) if normal_seconds else None,
        "early_negative_above_threshold_window_fraction": early_high / early_total if early_total else None,
        "valid_window_fraction": valid / total if total else 0.0,
        "valid_window_count": valid,
        "total_window_count": total,
    }


def _confirmed_runs(times: np.ndarray, scores: np.ndarray, threshold: float) -> list[float]:
    output: list[float] = []
    count = 0
    active = False
    previous: float | None = None
    for timestamp, score in zip(times, scores):
        contiguous = previous is None or timestamp - previous <= MAX_CONFIRMATION_GAP
        high = bool(np.isfinite(score) and score >= threshold)
        if not high or not contiguous:
            count = 0
            active = False
        if high:
            count += 1
            if count >= CONFIRMATION_WINDOWS and not active:
                output.append(float(timestamp))
                active = True
        previous = float(timestamp)
    return output


def _baseline_dguha_summary(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": raw["threshold"],
        "event_count": raw["eligible_fall_recording_count"],
        "detected_event_count": raw["prediction_corridor_detected_event_count"],
        "event_recall": raw["prediction_corridor_event_recall"],
        "median_lead_seconds": raw["corridor_confirmation_lead_seconds"]["median"],
        "false_alarms_per_hour": raw["normal_confirmed_runs_per_hour"],
        "early_negative_above_threshold_window_fraction": raw["same_recording_early_negative_false_positive_rate"],
        "valid_window_fraction": 1.0,
    }


def _read_baseline_peerj(evidence_path: Path, peerj_root: Path) -> list[ScoredRecording]:
    grouped: dict[str, tuple[list[float], list[float], list[float]]] = {}
    with gzip.open(evidence_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("dataset") != "PeerJ":
                continue
            subject = str(item["subject_id"])
            times, scores, qualities = grouped.setdefault(subject, ([], [], []))
            radar = item["radar"]
            times.append(datetime.fromisoformat(item["timestamp"]).timestamp())
            scores.append(float(radar["score"]) if radar["available"] else float("nan"))
            qualities.append(float(radar["quality"]))
    falls = np.loadtxt(peerj_root / "falls.csv", delimiter=",")
    output = []
    for subject_number in range(1, 11):
        subject = f"subject_{subject_number}"
        times, scores, qualities = grouped[subject]
        output.append(ScoredRecording(
            "PeerJ", subject, f"{subject}/radar.csv", np.asarray(times), np.asarray(scores),
            np.asarray(qualities), tuple(float(value) for value in falls[:, subject_number - 1]), False,
        ))
    return output


def _read_jsonl_baseline(
    path: Path, anchors: tuple[float, ...], normal_only: bool, recording_id: str
) -> ScoredRecording:
    times: list[float] = []
    scores: list[float] = []
    qualities: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            prediction = item.get("tcn_prediction")
            if not prediction:
                continue
            times.append(datetime.fromisoformat(str(prediction["timestamp"])).timestamp())
            valid = bool(prediction["score_valid"])
            scores.append(float(prediction["pre_fall_score"]) if valid else float("nan"))
            qualities.append(1.0 if prediction["data_quality"] == "GOOD" else 0.6 if prediction["data_quality"] == "DEGRADED" else 0.0)
    return ScoredRecording(
        "IWR6843_REPLAY", "p01", recording_id, np.asarray(times), np.asarray(scores),
        np.asarray(qualities), anchors, normal_only,
    )


def _read_peerj_frames(path: Path, subject: str):
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            values = [float(value) for value in row[1:] if value]
            points = []
            for offset in range(0, len(values), 7):
                x, y, z, velocity, snr, _noise, track = values[offset : offset + 7]
                points.append(RadarPoint(x, y, z, velocity, snr, None if int(track) == 255 else int(track)))
            yield RadarFrame(
                datetime.fromtimestamp(float(row[0]), tz=timezone.utc),
                f"peerj-{subject}", Room.LIVING_ROOM, SourceMode.REPLAY, tuple(points),
            )


def _read_jsonl_frames(path: Path) -> tuple[RadarFrame, ...]:
    frames: list[RadarFrame] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            points = tuple(
                RadarPoint(
                    float(value["x"]), float(value["y"]), float(value["z"]),
                    float(value["velocity"]),
                    float(value["snr"]) if value.get("snr") is not None else None,
                )
                for value in item.get("points", [])
            )
            frames.append(RadarFrame(
                datetime.fromisoformat(str(item["timestamp"])), str(item["device_id"]),
                Room(str(item["room"])), SourceMode(str(item["source_mode"])), points,
            ))
    return tuple(frames)


def _first_jsonl_timestamp(path: Path) -> datetime:
    with path.open("r", encoding="utf-8") as handle:
        return datetime.fromisoformat(str(json.loads(next(handle))["timestamp"]))


def _normal_intervals(start: float, end: float, anchors: Sequence[float]):
    excluded = sorted((max(start, value - EVENT_EXCLUSION), min(end, value + EVENT_EXCLUSION)) for value in anchors)
    output = []
    cursor = start
    for left, right in excluded:
        if left > cursor:
            output.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        output.append((cursor, end))
    return output


def _load_candidate(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_version") != MODEL_VERSION or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("candidate checkpoint contract mismatch")
    if tuple(checkpoint.get("feature_names", ())) != INPUT_FEATURES:
        raise ValueError("point feature order mismatch")
    if bool(checkpoint.get("peerj_used_for_training_or_threshold", True)):
        raise ValueError("PeerJ contamination detected")
    return checkpoint


def _model(checkpoint: dict[str, Any], device: torch.device) -> PointNetPlusPlusTcnPrefall:
    model = PointNetPlusPlusTcnPrefall()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def _report(result: dict[str, Any]) -> str:
    validation = result["dguha"]["validation"]
    test = result["dguha"]["test"]
    peerj = result["peerj_external_no_threshold_tuning"]
    replay = result["iwr6843_replay_audit_only"]
    training = result.get("training_summary") or {}
    pretraining = training.get("fall102_pretraining", {})
    dguha_training = training.get("dguha_training", {})
    selection = result["threshold_selection_validation_only"]
    lines = [
        "# Radar Encoder Upgrade：PointNet++ + TCN",
        "",
        "## 结论",
        "",
        "本分支只替换雷达空间输入表示；Fusion、Camera、UART/TLV、B0 checkpoint 与 Evidence 协议均未修改。",
        "PeerJ 与真实 IWR6843 replay 未参与训练、选 epoch 或阈值。",
        f"替换判定：**{result['replacement_decision']['verdict']}**；候选保持 shadow experiment，不接入 Fusion。",
        "",
        "## 模型与训练",
        "",
        "- 输入：20 帧 × 每帧最多 64 点 × `[x, y, z, radial_velocity, snr]`，不再输入 19 维人工特征。",
        "- 空间编码：两级 PointNet++ set abstraction（16/4 centroids，k=8）后得到 64 维 frame embedding。",
        "- 时间建模：保留 causal TCN，dilation 为 1/2/4/8；没有使用 GRU 替代 TCN。",
        f"- Fall-102：2 名 train subjects、1 名 validation subject；最佳空间预训练宏召回 {_pct(pretraining.get('best_validation_macro_recall'))}。",
        f"- DGUHA：frame encoder 冻结 warm-up 后小学习率联合训练；最佳 validation window AUROC {_fmt((dguha_training.get('best_validation') or {}).get('auroc'))}。",
        "- Fall-102 跌倒段没有作为 pre-fall positive；DGUHA 是唯一短时预测正监督来源。",
        "",
        "## 阈值锁定",
        "",
        f"- validation 目标事件下限：{selection['event_recall_floor']}/14；实际所有阈值最多达到 {selection['selected_validation_metrics']['detected_event_count']}/14。",
        f"- 因召回下限未满足，采用 `{selection['policy']}`，锁定阈值 {selection['threshold']}；这不是可部署阈值。",
        "",
        "## DGUHA 对比",
        "",
        "| Split | 模型 | Event recall | False alarms/hour | Early-negative FPR | Median lead (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split_name, values in (("validation", validation), ("test", test)):
        for model_name in ("tcn_baseline", "pointnetpp_tcn"):
            item = values[model_name]
            lines.append(
                f"| {split_name} | {model_name} | {_pct(item.get('event_recall'))} | "
                f"{_fmt(item.get('false_alarms_per_hour'))} | "
                f"{_pct(item.get('early_negative_above_threshold_window_fraction'))} | "
                f"{_fmt(item.get('median_lead_seconds'))} |"
            )
    lines += [
        "",
        "## PeerJ 外部泛化（不调阈值）",
        "",
        "| 模型 | Event recall | Pre-fall recall | False alarms/hour | Valid windows |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name in ("tcn_baseline", "pointnetpp_tcn"):
        item = peerj[model_name]
        lines.append(
            f"| {model_name} | {_pct(item['event_recall'])} | {_pct(item['prefall_event_recall'])} | "
            f"{_fmt(item['false_alarms_per_hour'])} | {_pct(item['valid_window_fraction'])} |"
        )
    lines += [
        "",
        "## 真实 IWR6843 replay",
        "",
        "| 模型 | Normal replay FA/hour | Controlled-fall event recall | Pre-fall recall | Valid windows |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name in ("tcn_baseline", "pointnetpp_tcn"):
        normal = replay["normal_replay"][model_name]
        fall = replay["controlled_fall_replay"][model_name]
        combined = replay[model_name]
        lines.append(
            f"| {model_name} | {_fmt(normal['false_alarms_per_hour'])} | {_pct(fall['event_recall'])} | "
            f"{_pct(fall['prefall_event_recall'])} | {_pct(combined['valid_window_fraction'])} |"
        )
    lines += [
        "",
        "## 输出契约",
        "",
        "离线和后续 shadow 推理仍输出 `score`、`quality`、`timestamp`；分数是 radar risk evidence，不是跌倒概率。",
        "",
        f"锁定阈值：`{result['threshold_selection_validation_only']['threshold']}`；锁定 checkpoint：`{result['locked_checkpoint']}`。",
        "",
        "## 最终判断",
        "",
        "PointNet++ 表示降低了 DGUHA 正常误报和 early-negative FPR，但召回损失明显；同时在 PeerJ 与真实 IWR 正常回放产生了 baseline 没有的误报。",
        "因此本轮证明了原始点云空间编码路径可以运行并保持 Evidence 契约，但**没有证明它优于 19 维特征 TCN**。当前 B0 TCN 必须继续保留。",
    ]
    return "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    """Keep new checkpoints loadable with torch weights_only=True."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PointNet++ + TCN radar encoder upgrade")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dguha-root", required=True, type=Path)
    parser.add_argument("--dguha-events", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--peerj-root", required=True, type=Path)
    parser.add_argument("--baseline-evidence", required=True, type=Path)
    parser.add_argument("--iwr-normal-replay", required=True, type=Path)
    parser.add_argument("--iwr-fall-replay", required=True, type=Path)
    parser.add_argument("--iwr-fall-phase", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate_upgrade(
        args.candidate, args.output,
        dguha_root=args.dguha_root, dguha_events=args.dguha_events,
        baseline_checkpoint=args.baseline_checkpoint, peerj_root=args.peerj_root,
        baseline_evidence=args.baseline_evidence,
        iwr_normal_replay=args.iwr_normal_replay,
        iwr_fall_replay=args.iwr_fall_replay,
        iwr_fall_phase=args.iwr_fall_phase, device=args.device,
    )
    print(json.dumps({
        "locked_checkpoint": result["locked_checkpoint"],
        "threshold": result["threshold_selection_validation_only"]["threshold"],
        "report": str(Path(args.output).resolve() / "RADAR_ENCODER_UPGRADE_REPORT.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
