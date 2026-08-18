from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_module.model.radar_lstm import RadarLSTM
from radar_module.model.research_training_v2 import (
    RESEARCH_MODEL_MODE,
    RESEARCH_MODEL_VERSION,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    WINDOW_SIZE_V2,
)


@dataclass(frozen=True, slots=True)
class NegativeGroupMetrics:
    sample_count: int
    false_positive_count: int
    false_positive_rate: float
    score_median: float
    score_p90: float
    score_p95: float
    score_p99: float


@dataclass(frozen=True, slots=True)
class ExternalNegativeEvaluationSummary:
    checkpoint_file: str
    checkpoint_sha256: str
    dataset_file: str
    dataset_sha256: str
    report_file: str
    decision_threshold: float
    dataset_mode: str
    source_complete: bool
    overall: NegativeGroupMetrics
    by_action: dict[str, NegativeGroupMetrics]
    by_split: dict[str, NegativeGroupMetrics]
    deployment_validation_eligible: bool


def evaluate_external_hard_negatives(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    report_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
    allow_audit_sample: bool = False,
) -> ExternalNegativeEvaluationSummary:
    checkpoint_file = Path(checkpoint_path).resolve()
    dataset_file = Path(dataset_path).resolve()
    destination = Path(report_path).resolve()
    if not checkpoint_file.is_file() or not dataset_file.is_file():
        raise FileNotFoundError("checkpoint and external dataset must exist")

    payload = _safe_torch_load(checkpoint_file, device)
    _validate_checkpoint(payload)
    with np.load(dataset_file, allow_pickle=False) as dataset:
        dataset_mode = str(dataset["dataset_mode"].item())
        accepted_modes = {"EXTERNAL_HARD_NEGATIVE_ONLY"}
        if allow_audit_sample:
            accepted_modes.add("EXTERNAL_HARD_NEGATIVE_AUDIT_SAMPLE_ONLY")
        if dataset_mode not in accepted_modes:
            raise ValueError("dataset is not an external hard-negative export")
        source_complete = (
            bool(dataset["source_complete"].item())
            if "source_complete" in dataset.files
            else True
        )
        if not source_complete and not allow_audit_sample:
            raise ValueError("incomplete external dataset requires audit opt-in")
        if bool(dataset["positive_samples_available"].item()):
            raise ValueError("external hard-negative dataset unexpectedly has positives")
        if str(dataset["feature_version"].item()) != FEATURE_VERSION_V2:
            raise ValueError("external dataset feature version is incompatible")
        if tuple(str(value) for value in dataset["feature_names"]) != FEATURE_NAMES_V2:
            raise ValueError("external dataset feature names/order are incompatible")
        features = np.asarray(dataset["features"], dtype=np.float32)
        actions = np.asarray(dataset["action"])
        splits = np.asarray(dataset["split"])

    mean = np.asarray(payload["normalization_mean"], dtype=np.float32)
    std = np.asarray(payload["normalization_std"], dtype=np.float32)
    normalized = ((features - mean[None, None, :]) / std[None, None, :]).astype(
        np.float32
    )
    torch_device = torch.device(device)
    model = RadarLSTM(
        input_size=len(FEATURE_NAMES_V2), hidden_size=int(payload["hidden_size"])
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(torch_device)
    model.eval()
    score_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            tensor = torch.from_numpy(normalized[start : start + batch_size]).to(
                torch_device
            )
            score_batches.append(torch.sigmoid(model(tensor)).cpu().numpy())
    scores = np.concatenate(score_batches).astype(np.float64)
    threshold = float(payload["decision_threshold"])
    by_action = {
        str(action): _negative_metrics(scores[actions == action], threshold)
        for action in sorted(set(str(value) for value in actions))
    }
    by_split = {
        str(split): _negative_metrics(scores[splits == split], threshold)
        for split in sorted(set(str(value) for value in splits))
    }
    summary = ExternalNegativeEvaluationSummary(
        checkpoint_file=str(checkpoint_file),
        checkpoint_sha256=_sha256(checkpoint_file),
        dataset_file=str(dataset_file),
        dataset_sha256=_sha256(dataset_file),
        report_file=str(destination),
        decision_threshold=threshold,
        dataset_mode=dataset_mode,
        source_complete=source_complete,
        overall=_negative_metrics(scores, threshold),
        by_action=by_action,
        by_split=by_split,
        deployment_validation_eligible=False,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = asdict(summary)
    report["interpretation"] = (
        "Incomplete cross-sensor format-audit sample only; these scores cannot "
        "be used for model selection, training claims, sensitivity, or deployment."
        if not source_complete
        else "Cross-sensor hard-negative stress test only. The dataset has no "
        "falls, so this report estimates false positives but cannot measure "
        "sensitivity."
    )
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _negative_metrics(scores: np.ndarray, threshold: float) -> NegativeGroupMetrics:
    return NegativeGroupMetrics(
        sample_count=len(scores),
        false_positive_count=int(np.sum(scores >= threshold)),
        false_positive_rate=float(np.mean(scores >= threshold)),
        score_median=float(np.quantile(scores, 0.50)),
        score_p90=float(np.quantile(scores, 0.90)),
        score_p95=float(np.quantile(scores, 0.95)),
        score_p99=float(np.quantile(scores, 0.99)),
    )


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
        raise ValueError("unsupported research checkpoint model_version")
    if payload.get("model_mode") != RESEARCH_MODEL_MODE:
        raise ValueError("checkpoint is not marked weak-supervision research")
    if bool(payload.get("deployment_eligible", True)):
        raise ValueError("research checkpoint must be non-deployable")
    if payload.get("feature_version") != FEATURE_VERSION_V2:
        raise ValueError("research checkpoint feature_version is incompatible")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES_V2:
        raise ValueError("research checkpoint feature names/order are incompatible")
    if int(payload.get("window_size", -1)) != WINDOW_SIZE_V2:
        raise ValueError("research checkpoint window size is incompatible")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate research model on external negatives.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--allow-audit-sample", action="store_true")
    args = parser.parse_args()
    summary = evaluate_external_hard_negatives(
        args.checkpoint,
        args.dataset,
        args.report,
        allow_audit_sample=args.allow_audit_sample,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
