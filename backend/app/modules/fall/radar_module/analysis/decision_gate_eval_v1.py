"""Evaluate the decision-level gate over real-sensor replays.

Pipes per-session feature windows through the frozen TCN with a candidate
domain-calibrated normalization, then feeds the score stream through
DecisionGateV1. Reports state counts, formal-alert events, and how the
recovery gate suppresses controlled-lowering episodes.

This is a research/evaluation script. It never modifies the frozen checkpoint,
the model threshold, the feature extractor, or the live inference chain.

Version: radar_decision_gate_eval_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_module.contracts import Room
from radar_module.dataset.v2_export import _load_replay_frames
from radar_module.inference.decision_gate_v1 import DecisionGateV1, GateState
from radar_module.model.temporal_models_v3 import (
    EXPERIMENT_MODEL_VERSION,
    TemporalBinaryModel,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


EVAL_VERSION = "radar_decision_gate_eval_v1"
HISTORY_SECONDS = 2.0


def _load_checkpoint(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("model_version") != EXPERIMENT_MODEL_VERSION:
        raise ValueError("unsupported model_version")
    if ckpt.get("model_architecture") != "causal_tcn":
        raise ValueError("expected causal_tcn checkpoint")
    if ckpt.get("feature_version") != FEATURE_VERSION_V2:
        raise ValueError("expected v2 features")
    if tuple(ckpt.get("feature_names", ())) != FEATURE_NAMES_V2:
        raise ValueError("feature names/order mismatch")
    return ckpt


def _infer_scores(
    windows: np.ndarray,
    ckpt: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    torch_device = torch.device(device)
    model = TemporalBinaryModel(
        architecture="causal_tcn",
        input_size=len(FEATURE_NAMES_V2),
        hidden_size=int(ckpt["hidden_size"]),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(torch_device)
    model.eval()
    normalized = ((windows - mean[None, None, :]) / std[None, None, :]).astype(
        np.float32
    )
    with torch.inference_mode():
        logits = model(torch.from_numpy(normalized).to(torch_device))
        scores = torch.sigmoid(logits).squeeze(-1)
    return scores.detach().cpu().numpy()


def _compute_centroid_z(frame: Any) -> float | None:
    if not frame.points:
        return None
    return float(np.mean([p.z for p in frame.points]))


def evaluate_gate(
    *,
    sessions: Mapping[str, str | Path],
    checkpoint_path: str | Path,
    calibration_report_path: str | Path,
    calibration_candidate: str,
    output_dir: str | Path,
    threshold: float = 0.35,
    confirmation_windows: int = 3,
    recovery_windows: int = 2,
    recovery_window_seconds: float = 1.5,
    persist_confirm_seconds: float = 0.0,
    stride_seconds: float = 0.2,
    max_windows_per_session: int = 800,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_file = Path(checkpoint_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    ckpt = _load_checkpoint(checkpoint_file)
    report = json.loads(Path(calibration_report_path).read_text(encoding="utf-8"))
    cand = report["domain_calibrated_normalization"][calibration_candidate]
    mean = np.asarray(cand["mean"], dtype=np.float32)
    std = np.asarray(cand["std"], dtype=np.float32)

    extractor = RadarTemporalFeatureExtractorV2()
    result: dict[str, Any] = {
        "eval_version": EVAL_VERSION,
        "checkpoint_file": str(checkpoint_file),
        "calibration_candidate": calibration_candidate,
        "threshold": threshold,
        "confirmation_windows": confirmation_windows,
        "recovery_windows": recovery_windows,
        "recovery_window_seconds": recovery_window_seconds,
        "persist_confirm_seconds": persist_confirm_seconds,
        "sessions": {},
        "pooled": {},
    }

    all_states: list[str] = []
    all_alerts: list[int] = []
    for name in sessions:
        session_file = Path(sessions[name]).resolve()
        frames = _load_replay_frames(session_file, default_room=Room.BATHROOM)
        gate = DecisionGateV1(
            threshold=threshold,
            confirmation_windows=confirmation_windows,
            recovery_windows=recovery_windows,
            recovery_window_seconds=recovery_window_seconds,
            persist_confirm_seconds=persist_confirm_seconds,
        )
        # Build windows at a regular stride aligned to frame timestamps.
        windows: list[np.ndarray] = []
        timestamps: list[Any] = []
        centroids: list[float | None] = []
        point_counts: list[int] = []
        start = frames[0].timestamp
        end = frames[-1].timestamp
        current = start
        stride = timedelta(seconds=stride_seconds)
        history = timedelta(seconds=HISTORY_SECONDS)
        while current <= end and len(windows) < max_windows_per_session:
            lo = current - history
            window_frames = [f for f in frames if lo <= f.timestamp <= current]
            if window_frames:
                try:
                    window = extractor.transform(
                        tuple(window_frames), end_timestamp=current
                    )
                except ValueError:
                    current += stride
                    continue
                if window.data_quality is TemporalDataQuality.GOOD:
                    windows.append(np.asarray(window.values, dtype=np.float32))
                    timestamps.append(current)
                    # centroid of the last frame's points (heuristic)
                    last_frame = window_frames[-1]
                    centroids.append(_compute_centroid_z(last_frame))
                    point_counts.append(len(last_frame.points))
            current += stride
        if not windows:
            result["sessions"][name] = {
                "error": "no GOOD-quality windows",
                "frame_count": len(frames),
            }
            continue

        stacked = np.stack(windows)
        scores = _infer_scores(stacked, ckpt, mean, std, device=device)

        decisions = []
        for i, ts in enumerate(timestamps):
            d = gate.consume(
                timestamp=ts,
                score=float(scores[i]),
                centroid_z=centroids[i],
                point_count=point_counts[i],
            )
            decisions.append(d.to_dict())
            all_states.append(d.state.value)
            all_alerts.append(int(d.formal_alert))

        state_counts: dict[str, int] = {}
        for d in decisions:
            state_counts[d["state"]] = state_counts.get(d["state"], 0) + 1
        episode_count = sum(1 for d in decisions if d["formal_alert"])
        suppressed_count = state_counts.get(
            GateState.SUPPRESSED_RECOVERY.value, 0
        )

        result["sessions"][name] = {
            "frame_count": len(frames),
            "window_count": len(windows),
            "score_min": float(np.min(scores)),
            "score_median": float(np.median(scores)),
            "score_p95": float(np.percentile(scores, 95)),
            "score_max": float(np.max(scores)),
            "state_counts": state_counts,
            "formal_alert_episodes": episode_count,
            "suppressed_recovery_windows": suppressed_count,
            "last_decision": decisions[-1] if decisions else None,
        }

    pooled_counts: dict[str, int] = {}
    for s in all_states:
        pooled_counts[s] = pooled_counts.get(s, 0) + 1
    result["pooled"] = {
        "state_counts": pooled_counts,
        "formal_alert_window_count": sum(all_alerts),
        "total_window_count": len(all_states),
    }

    (destination / "decision_gate_eval_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate DecisionGateV1 over real-sensor replays."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--calibration-candidate", default="real_gaussian")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--session",
        action="append",
        metavar="NAME=PATH",
        required=True,
    )
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--confirmation-windows", type=int, default=3)
    parser.add_argument("--recovery-windows", type=int, default=2)
    parser.add_argument("--recovery-window-seconds", type=float, default=1.5)
    parser.add_argument("--persist-confirm-seconds", type=float, default=0.0)
    parser.add_argument("--max-windows-per-session", type=int, default=800)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    sessions: dict[str, str] = {}
    for spec in args.session:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"invalid --session spec: {spec!r}")
        sessions[name.strip()] = path.strip()
    result = evaluate_gate(
        sessions=sessions,
        checkpoint_path=args.checkpoint,
        calibration_report_path=args.calibration_report,
        calibration_candidate=args.calibration_candidate,
        output_dir=args.output_dir,
        threshold=args.threshold,
        confirmation_windows=args.confirmation_windows,
        recovery_windows=args.recovery_windows,
        recovery_window_seconds=args.recovery_window_seconds,
        persist_confirm_seconds=args.persist_confirm_seconds,
        max_windows_per_session=args.max_windows_per_session,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
