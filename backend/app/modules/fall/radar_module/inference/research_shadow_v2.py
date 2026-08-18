from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_module.contracts import Room
from radar_module.dataset.v2_export import _load_replay_frames
from radar_module.inference.prefall_rule_v2 import PreFallRulePredictorV2
from radar_module.model.radar_lstm import RadarLSTM
from radar_module.model.research_training_v2 import (
    RESEARCH_MODEL_MODE,
    RESEARCH_MODEL_VERSION,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
    WINDOW_SIZE_V2,
)


SHADOW_DISCLAIMER = (
    "弱监督研究模型的离线影子输出，不触发告警，不代表真实跌倒预测能力"
)


@dataclass(frozen=True, slots=True)
class ShadowReplaySummary:
    replay_file: str
    replay_sha256: str
    checkpoint_file: str
    checkpoint_sha256: str
    output_file: str
    report_file: str
    recording_semantics: str
    duration_seconds: float
    evaluated_window_count: int
    good_window_count: int
    degraded_window_count: int
    insufficient_window_count: int
    threshold: float
    prediction_score_max: float
    prediction_score_p95: float
    above_threshold_window_count: int
    confirmed_prediction_run_count: int
    confirmed_prediction_runs_per_hour: float
    maximum_fall_risk_score: float
    shadow_only: bool
    deployment_eligible: bool


def run_research_shadow_replay(
    replay_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    recording_semantics: str = "unknown",
    confirmation_windows: int = 3,
    device: str | torch.device = "cpu",
) -> ShadowReplaySummary:
    """Run research inference offline without touching the alert path."""

    if confirmation_windows < 2:
        raise ValueError("confirmation_windows must be at least two")
    replay_file = Path(replay_path).resolve()
    checkpoint_file = Path(checkpoint_path).resolve()
    destination = Path(output_path).resolve()
    if not replay_file.is_file() or not checkpoint_file.is_file():
        raise FileNotFoundError("replay and research checkpoint must exist")
    if destination.suffix.lower() != ".jsonl":
        raise ValueError("output_path must end with .jsonl")

    checkpoint = _safe_torch_load(checkpoint_file, device)
    _validate_checkpoint(checkpoint)
    extractor = RadarTemporalFeatureExtractorV2()
    rule_predictor = PreFallRulePredictorV2()
    frames = _load_replay_frames(replay_file, default_room=Room.LIVING_ROOM)
    first_timestamp = frames[0].timestamp
    frame_seconds = [
        (frame.timestamp - first_timestamp).total_seconds() for frame in frames
    ]
    minimum_end = (extractor.window_size - 1) / extractor.target_sample_rate_hz
    windows = []
    for end_index, end_seconds in enumerate(frame_seconds):
        if end_seconds + 1e-9 < minimum_end:
            continue
        left = bisect.bisect_left(
            frame_seconds,
            end_seconds - extractor.history_seconds - extractor.alignment_tolerance_seconds,
        )
        window = extractor.transform(
            frames[left : end_index + 1], end_timestamp=frames[end_index].timestamp
        )
        windows.append(window)
    if not windows:
        raise ValueError("replay is shorter than one v2 feature window")

    feature_tensor = np.stack([window.values for window in windows]).astype(np.float32)
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    normalized = ((feature_tensor - mean[None, None, :]) / std[None, None, :]).astype(
        np.float32
    )
    torch_device = torch.device(device)
    model = RadarLSTM(
        input_size=len(FEATURE_NAMES_V2), hidden_size=int(checkpoint["hidden_size"])
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(torch_device)
    model.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(
            model(torch.from_numpy(normalized).to(torch_device))
        ).cpu().numpy().astype(np.float64)

    threshold = float(checkpoint["decision_threshold"])
    high_run = 0
    run_latched = False
    confirmed_runs = 0
    risk_history: deque[tuple[float, float]] = deque()
    records: list[dict[str, object]] = []
    risk_scores: list[float] = []
    quality_counts = {quality.value: 0 for quality in TemporalDataQuality}
    above_threshold = 0

    for window, raw_model_score in zip(windows, scores):
        quality_counts[window.data_quality.value] += 1
        rule = rule_predictor.predict(window)
        if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
            model_score = 0.0
            prediction_state = "UNKNOWN"
            high_run = 0
            run_latched = False
            risk_history.clear()
            fall_risk_score = 0.0
        else:
            model_score = float(raw_model_score)
            is_high = model_score >= threshold
            above_threshold += int(is_high)
            high_run = high_run + 1 if is_high else 0
            confirmed = high_run >= confirmation_windows
            if confirmed and not run_latched:
                confirmed_runs += 1
                run_latched = True
            elif not is_high:
                run_latched = False
            prediction_state = (
                "IMMINENT" if confirmed else "WATCH" if is_high else "NORMAL"
            )
            fall_risk_score = max(model_score, rule.fall_risk_score)
            elapsed = (window.end_timestamp - first_timestamp).total_seconds()
            risk_history.append((elapsed, fall_risk_score))
            while risk_history and elapsed - risk_history[0][0] > 5.0:
                risk_history.popleft()
        fall_risk_score_5s = max((value for _, value in risk_history), default=0.0)
        risk_scores.append(fall_risk_score)
        records.append(
            {
                "timestamp": window.end_timestamp.isoformat(),
                "room": window.room.value,
                "device_id": window.device_id,
                "prediction_state": prediction_state,
                "pre_fall_score": model_score,
                "rule_pre_fall_score": rule.pre_fall_score,
                "fall_risk_score": fall_risk_score,
                "fall_risk_score_5s": fall_risk_score_5s,
                "fall_risk_level": _risk_level(fall_risk_score_5s),
                "data_quality": window.data_quality.value,
                "model_mode": RESEARCH_MODEL_MODE,
                "shadow_only": True,
                "alert_suppressed": True,
                "disclaimer": SHADOW_DISCLAIMER,
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    duration = frame_seconds[-1]
    report_path = destination.with_suffix(".report.json")
    summary = ShadowReplaySummary(
        replay_file=str(replay_file),
        replay_sha256=_sha256(replay_file),
        checkpoint_file=str(checkpoint_file),
        checkpoint_sha256=_sha256(checkpoint_file),
        output_file=str(destination),
        report_file=str(report_path),
        recording_semantics=recording_semantics,
        duration_seconds=duration,
        evaluated_window_count=len(windows),
        good_window_count=quality_counts[TemporalDataQuality.GOOD.value],
        degraded_window_count=quality_counts[TemporalDataQuality.DEGRADED.value],
        insufficient_window_count=quality_counts[
            TemporalDataQuality.INSUFFICIENT_DATA.value
        ],
        threshold=threshold,
        prediction_score_max=float(np.max(scores)),
        prediction_score_p95=float(np.quantile(scores, 0.95)),
        above_threshold_window_count=above_threshold,
        confirmed_prediction_run_count=confirmed_runs,
        confirmed_prediction_runs_per_hour=(
            confirmed_runs * 3600.0 / duration if duration > 0 else 0.0
        ),
        maximum_fall_risk_score=float(max(risk_scores)),
        shadow_only=True,
        deployment_eligible=False,
    )
    report = asdict(summary)
    report["interpretation_limit"] = (
        "Recording-level semantics cannot locate fall events or measure advance time. "
        "Only a fully normal recording may use confirmed runs as provisional false alarms."
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _safe_torch_load(path: Path, device: str | torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("research checkpoint root must be a mapping")
    return payload


def _validate_checkpoint(payload: dict[str, Any]) -> None:
    if payload.get("model_version") != RESEARCH_MODEL_VERSION:
        raise ValueError("unsupported research checkpoint")
    if payload.get("model_mode") != RESEARCH_MODEL_MODE:
        raise ValueError("checkpoint is not weak-supervision research")
    if bool(payload.get("deployment_eligible", True)):
        raise ValueError("shadow runner requires a non-deployable checkpoint")
    if payload.get("feature_version") != FEATURE_VERSION_V2:
        raise ValueError("checkpoint feature version is incompatible")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES_V2:
        raise ValueError("checkpoint feature names/order are incompatible")
    if int(payload.get("window_size", -1)) != WINDOW_SIZE_V2:
        raise ValueError("checkpoint window size is incompatible")


def _risk_level(score: float) -> str:
    if score >= 0.60:
        return "HIGH"
    if score >= 0.30:
        return "MODERATE"
    return "LOW"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-alerting research shadow replay.")
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recording-semantics", default="unknown")
    args = parser.parse_args()
    summary = run_research_shadow_replay(
        args.replay,
        args.checkpoint,
        args.output,
        recording_semantics=args.recording_semantics,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
