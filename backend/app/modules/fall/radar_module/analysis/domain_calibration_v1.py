from __future__ import annotations

"""Domain calibration analysis for the frozen TCN checkpoint.

The live IWR6843 scores collapse to ~1e-11 because the checkpoint was trained
on DGUHA (IWR1443, low-mounted, forward-only falls) while the real sensor is
IWR6843ISK mounted high. This script:

1. Replays real-sensor sessions with the frozen checkpoint (baseline scores).
2. Computes the real-sensor feature distribution and compares it against the
   checkpoint's stored normalization_mean/std (feature-level domain gap).
3. Recomputes candidate domain-calibrated normalizations from real-sensor
   windows, then replays the same sessions to show how scores shift.

Important contract: this script NEVER modifies the frozen checkpoint, the TCN
architecture, the feature extractor, the threshold, or the live inference
chain. It only *reports* what a domain-calibrated normalization would look
like, as a diagnostic artifact. A human decides whether to use it.

Version: radar_domain_calibration_v1
"""

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radar_module.contracts import Room
from radar_module.dataset.v2_export import _load_replay_frames
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


ANALYSIS_VERSION = "radar_domain_calibration_v1"
FROZEN_CHECKPOINT_SHA256 = (
    "0792a712b57ae89875b2d57e6ba7a20763618a2718e961cf8c48acebe34970ef"
)
FEATURE_NAMES_LIST = list(FEATURE_NAMES_V2)
HISTORY_SECONDS = 2.0
WINDOW_SIZE = 20


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint root must be a mapping")
    if ckpt.get("model_version") != EXPERIMENT_MODEL_VERSION:
        raise ValueError("unsupported model_version")
    if ckpt.get("model_architecture") != "causal_tcn":
        raise ValueError("expected causal_tcn checkpoint")
    if ckpt.get("feature_version") != FEATURE_VERSION_V2:
        raise ValueError("expected v2 features")
    if tuple(ckpt.get("feature_names", ())) != FEATURE_NAMES_V2:
        raise ValueError("feature names/order mismatch")
    return ckpt


def _sliding_windows(
    frames: Sequence[object],
    *,
    extractor: RadarTemporalFeatureExtractorV2,
    max_windows: int = 2000,
    stride_seconds: float = 0.2,
) -> np.ndarray:
    """Slide a 2 s window over frames, returning GOOD-quality feature windows.

    Returns an array of shape (N, WINDOW_SIZE_V2, 19).
    """
    values: list[np.ndarray] = []
    start = frames[0].timestamp
    end = frames[-1].timestamp
    current = start
    stride = timedelta(seconds=stride_seconds)
    history = timedelta(seconds=HISTORY_SECONDS)
    count = 0
    while current <= end and count < max_windows:
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
                values.append(np.asarray(window.values, dtype=np.float32))
                count += 1
        current += stride
    if not values:
        raise ValueError("no GOOD-quality windows produced from this replay")
    return np.stack(values)


def _infer_scores(
    windows: np.ndarray,
    ckpt: dict[str, Any],
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Run frozen TCN on windows, optionally with custom normalization."""
    if mean is None:
        mean = np.asarray(ckpt["normalization_mean"], dtype=np.float32)
    if std is None:
        std = np.asarray(ckpt["normalization_std"], dtype=np.float32)
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


def _score_summary(scores: np.ndarray) -> dict[str, float | None]:
    if scores.size == 0:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
            "mean": None,
            "above_threshold_frac": None,
        }
    return {
        "count": int(scores.size),
        "min": float(np.min(scores)),
        "median": float(np.median(scores)),
        "p95": float(np.percentile(scores, 95)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "above_threshold_frac": float(np.mean(scores >= 0.35)),
    }


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------

def analyze_domain_calibration(
    *,
    sessions: Mapping[str, str | Path],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    stride_seconds: float = 0.2,
    max_windows_per_session: int = 2000,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_file = Path(checkpoint_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    ckpt = _load_checkpoint(checkpoint_file)
    actual_sha = _sha256(checkpoint_file)
    if actual_sha != FROZEN_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA256 mismatch: expected {FROZEN_CHECKPOINT_SHA256}, "
            f"got {actual_sha}"
        )

    stored_mean = np.asarray(ckpt["normalization_mean"], dtype=np.float64)
    stored_std = np.asarray(ckpt["normalization_std"], dtype=np.float64)
    stored_std = np.where(stored_std < 1e-9, 1e-9, stored_std)
    extractor = RadarTemporalFeatureExtractorV2()

    # ---- pass 1: collect real-sensor windows + baseline scores ----
    all_windows: list[np.ndarray] = []
    session_names = list(sessions.keys())
    session_reports: dict[str, dict[str, Any]] = {}
    for name in session_names:
        session_file = Path(sessions[name]).resolve()
        frames = _load_replay_frames(session_file, default_room=Room.BATHROOM)
        windows = _sliding_windows(
            frames,
            extractor=extractor,
            max_windows=max_windows_per_session,
            stride_seconds=stride_seconds,
        )
        all_windows.append(windows)
        baseline = _infer_scores(windows, ckpt, device=device)
        session_reports[name] = {
            "session_file": str(session_file),
            "frame_count": len(frames),
            "window_count": int(windows.shape[0]),
            "baseline_scores": _score_summary(baseline),
        }

    stacked = np.concatenate(all_windows, axis=0)  # (N, 20, 19)
    n_total = stacked.shape[0]

    # ---- feature-level domain gap (flatten over frames) ----
    flat = stacked.reshape(-1, len(FEATURE_NAMES_V2))  # (N*20, 19)
    real_mean = np.mean(flat, axis=0)
    real_std = np.std(flat, axis=0)
    real_median = np.median(flat, axis=0)
    q75 = np.percentile(flat, 75, axis=0)
    q25 = np.percentile(flat, 25, axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr < 1e-9, 1e-9, iqr)
    # Protect against zero-variance features (e.g. mask columns that are
    # constant in the real-sensor domain).
    real_std = np.where(real_std < 1e-9, 1e-9, real_std)

    z_gap = (real_mean - stored_mean) / stored_std
    feature_gap: list[dict[str, Any]] = []
    for i, name in enumerate(FEATURE_NAMES_LIST):
        feature_gap.append(
            {
                "feature": name,
                "stored_mean": float(stored_mean[i]),
                "stored_std": float(stored_std[i]),
                "real_mean": float(real_mean[i]),
                "real_std": float(real_std[i]),
                "real_median": float(real_median[i]),
                "z_gap_of_mean": float(z_gap[i]),
                "abs_z_gap": float(abs(z_gap[i])),
            }
        )
    feature_gap.sort(key=lambda item: item["abs_z_gap"], reverse=True)

    # ---- candidate calibrated normalizations ----
    candidates: dict[str, dict[str, Any]] = {
        "real_gaussian": {
            "mean": real_mean.tolist(),
            "std": real_std.tolist(),
            "description": "real-sensor Gaussian mean/std from replay windows",
        },
        "real_robust": {
            "mean": real_median.tolist(),
            "std": iqr.tolist(),
            "description": "real-sensor robust median/IQR from replay windows",
        },
    }

    # ---- pass 2: per-session scores under each candidate normalization ----
    calibrated_reports: dict[str, dict[str, Any]] = {}
    for cand_name, cand in candidates.items():
        cand_mean = np.asarray(cand["mean"], dtype=np.float32)
        cand_std = np.asarray(cand["std"], dtype=np.float32)
        pooled_scores = _infer_scores(
            stacked,
            ckpt,
            mean=cand_mean,
            std=cand_std,
            device=device,
        )
        per_session: dict[str, dict[str, Any]] = {}
        offset = 0
        for i, name in enumerate(session_names):
            n = all_windows[i].shape[0]
            per_session[name] = _score_summary(
                pooled_scores[offset : offset + n]
            )
            offset += n
        calibrated_reports[cand_name] = {
            "pooled": _score_summary(pooled_scores),
            "per_session": per_session,
        }

    result = {
        "analysis_version": ANALYSIS_VERSION,
        "checkpoint_file": str(checkpoint_file),
        "checkpoint_sha256": actual_sha,
        "threshold": float(ckpt["decision_threshold"]),
        "session_count": len(sessions),
        "total_window_count": n_total,
        "feature_gap_ranking": feature_gap,
        "stored_normalization": {
            "mean": stored_mean.tolist(),
            "std": stored_std.tolist(),
        },
        "domain_calibrated_normalization": {
            name: {
                "mean": cand["mean"],
                "std": cand["std"],
                "description": cand["description"],
            }
            for name, cand in candidates.items()
        },
        "session_reports_baseline": session_reports,
        "calibrated_scores": calibrated_reports,
        "note": (
            "Diagnostic only. This script does not modify the frozen checkpoint, "
            "threshold, feature extractor, or live inference chain. Any calibrated "
            "normalization must be validated on held-out subjects before use."
        ),
    }
    (destination / "domain_calibration_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze real-sensor domain gap and candidate calibrated normalization."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--session",
        action="append",
        metavar="NAME=PATH",
        required=True,
        help="repeatable session spec, e.g. walk=path/to/session.jsonl",
    )
    parser.add_argument("--stride-seconds", type=float, default=0.2)
    parser.add_argument("--max-windows-per-session", type=int, default=2000)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    sessions: dict[str, str] = {}
    for spec in args.session:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"invalid --session spec: {spec!r}")
        sessions[name.strip()] = path.strip()
    result = analyze_domain_calibration(
        sessions=sessions,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        stride_seconds=args.stride_seconds,
        max_windows_per_session=args.max_windows_per_session,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
