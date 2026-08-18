from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from radar_module.model.research_training_v2 import (
    RESEARCH_MODEL_MODE,
    _auroc,
    _binary_metrics,
    _scores_and_loss,
    _select_threshold,
    _set_seed,
    _sha256,
)
from radar_module.model.temporal_models_v3 import (
    EXPERIMENT_MODEL_VERSION,
    TemporalBinaryModel,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    WINDOW_SIZE_V2,
)
from radar_module.preprocess.relative_temporal_features_v3 import (
    FEATURE_NAMES_V3,
    FEATURE_VERSION_V3,
)
from radar_module.preprocess.hybrid_temporal_features_v4 import (
    FEATURE_NAMES_V4,
    FEATURE_VERSION_V4,
)


_SUPPORTED_FEATURE_CONTRACTS = {
    FEATURE_VERSION_V2: FEATURE_NAMES_V2,
    FEATURE_VERSION_V3: FEATURE_NAMES_V3,
    FEATURE_VERSION_V4: FEATURE_NAMES_V4,
}


def slice_dguha_horizon_dataset(
    source_path: str | Path,
    output_path: str | Path,
    *,
    minimum_lead_seconds: float,
    maximum_lead_seconds: float,
) -> dict[str, object]:
    """Create a narrower horizon artifact from an already exported superset."""

    if not 0 < minimum_lead_seconds <= maximum_lead_seconds:
        raise ValueError("prediction horizon is invalid")
    source = Path(source_path).resolve()
    destination = Path(output_path).resolve()
    with np.load(source, allow_pickle=False) as dataset:
        labels = np.asarray(dataset["labels"], dtype=np.int8)
        seconds = np.asarray(dataset["seconds_to_anchor"], dtype=np.float32)
        source_horizon = np.asarray(dataset["prediction_horizon_seconds"], dtype=np.float32)
        if (
            minimum_lead_seconds < float(source_horizon[0]) - 1e-6
            or maximum_lead_seconds > float(source_horizon[1]) + 1e-6
        ):
            raise ValueError("requested horizon is outside the source superset")
        keep = (labels == 0) | (
            (seconds >= minimum_lead_seconds - 1e-6)
            & (seconds <= maximum_lead_seconds + 1e-6)
        )
        arrays: dict[str, np.ndarray] = {}
        for name in dataset.files:
            value = np.asarray(dataset[name])
            arrays[name] = value[keep] if value.shape[:1] == (len(labels),) else value
    arrays["prediction_horizon_seconds"] = np.asarray(
        (minimum_lead_seconds, maximum_lead_seconds), dtype=np.float32
    )
    arrays["positive_label_definition"] = np.asarray(
        f"window ends {minimum_lead_seconds:.1f}-{maximum_lead_seconds:.1f} s "
        "before skeleton-derived descent onset"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    kept_labels = np.asarray(arrays["labels"], dtype=np.int8)
    splits = np.asarray(arrays["split"])
    report = {
        "output_file": str(destination),
        "output_sha256": _sha256(destination),
        "source_file": str(source),
        "source_sha256": _sha256(source),
        "source_prediction_horizon_seconds": [float(value) for value in source_horizon],
        "prediction_horizon_seconds": [minimum_lead_seconds, maximum_lead_seconds],
        "sample_count": len(kept_labels),
        "positive_count": int(kept_labels.sum()),
        "negative_count": int(len(kept_labels) - kept_labels.sum()),
        "split_counts": {
            name: int(np.sum(splits == name))
            for name in ("train", "validation", "test")
        },
        "derivation": "filtered positive windows from superset; retained identical negatives",
        "deployment_eligible": False,
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def train_prefall_architecture_experiment(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    architecture: str,
    epochs: int = 20,
    hidden_size: int = 32,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    positive_weight_cap: float | None = 32.0,
    hard_negative_checkpoint: str | Path | None = None,
    hard_negative_quantile: float = 0.9,
    hard_negative_weight: float = 4.0,
    external_negative_dataset: str | Path | None = None,
    external_negative_weight: float = 1.0,
    source_specific_training_normalization: bool = False,
    specificity_priority_minimum_sensitivity: float | None = None,
    evaluate_test_split: bool = True,
    seed: int = 20260809,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    if architecture not in {"lstm", "causal_tcn"}:
        raise ValueError("architecture must be lstm or causal_tcn")
    if epochs <= 0 or hidden_size <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("training parameters must be positive")
    if not 0.0 < hard_negative_quantile < 1.0:
        raise ValueError("hard-negative quantile must be between 0 and 1")
    if hard_negative_weight < 1.0:
        raise ValueError("hard-negative weight must be at least 1")
    if external_negative_weight <= 0.0:
        raise ValueError("external-negative weight must be positive")
    if specificity_priority_minimum_sensitivity is not None and not (
        0.0 < specificity_priority_minimum_sensitivity <= 1.0
    ):
        raise ValueError("specificity-priority minimum sensitivity must be in (0, 1]")
    source = Path(dataset_path).resolve()
    destination = Path(checkpoint_path).resolve()
    arrays = _load_dataset(source)
    features = arrays["features"]
    labels = arrays["labels"]
    splits = arrays["split"]
    feature_version = str(arrays["feature_version"].item())
    feature_names = tuple(str(value) for value in arrays["feature_names"])
    if features.ndim != 3 or features.shape[2] != len(feature_names):
        raise ValueError("dataset feature tensor does not match its feature contract")
    window_size = int(features.shape[1])
    masks = {name: splits == name for name in ("train", "validation", "test")}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("dataset must contain train, validation and test")

    normalization_reference = arrays.get(
        "normalization_reference", np.ones(len(features), dtype=bool)
    )
    normalization_mask = masks["train"] & normalization_reference
    if not normalization_mask.any():
        raise ValueError("training split has no normalization-reference samples")
    mean = features[normalization_mask].mean(axis=(0, 1), dtype=np.float64).astype(
        np.float32
    )
    std = features[normalization_mask].std(axis=(0, 1), dtype=np.float64).astype(
        np.float32
    )
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    normalized = ((features - mean[None, None]) / std[None, None]).astype(
        np.float32
    )
    source_normalization_metadata: dict[str, dict[str, object]] = {}
    if source_specific_training_normalization:
        if "dataset_origin" not in arrays:
            raise ValueError(
                "source-specific training normalization requires dataset_origin"
            )
        origins = arrays["dataset_origin"]
        external_training = masks["train"] & ~normalization_reference
        for origin in np.unique(origins[external_training]):
            origin_mask = external_training & (origins == origin)
            origin_features = features[origin_mask]
            origin_mean = origin_features.mean(
                axis=(0, 1), dtype=np.float64
            ).astype(np.float32)
            origin_std = origin_features.std(
                axis=(0, 1), dtype=np.float64
            ).astype(np.float32)
            origin_std = np.where(origin_std < 1e-6, 1.0, origin_std).astype(
                np.float32
            )
            normalized[origin_mask] = (
                (origin_features - origin_mean[None, None])
                / origin_std[None, None]
            ).astype(np.float32)
            source_normalization_metadata[str(origin)] = {
                "sample_count": int(origin_mask.sum()),
                "mean": origin_mean.tolist(),
                "std": origin_std.tolist(),
            }
    train_labels = labels[masks["train"]]
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    if not positives or not negatives:
        raise ValueError("training split must contain both labels")
    positive_weight = negatives / positives
    if positive_weight_cap is not None:
        positive_weight = min(positive_weight, positive_weight_cap)

    _set_seed(seed)
    torch_device = torch.device(device)
    model = TemporalBinaryModel(
        architecture=architecture,
        input_size=len(feature_names),
        hidden_size=hidden_size,
    ).to(torch_device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], device=torch_device)
    )
    training_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], device=torch_device),
        reduction="none",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    all_sample_weights = arrays.get(
        "sample_weight", np.ones(len(features), dtype=np.float32)
    )
    if all_sample_weights.shape != (len(features),):
        raise ValueError("sample_weight must match the dataset length")
    if not np.isfinite(all_sample_weights).all() or np.any(all_sample_weights <= 0):
        raise ValueError("sample_weight values must be finite and positive")
    training_weights = all_sample_weights[masks["train"]].astype(
        np.float32, copy=True
    )
    hard_negative_metadata: dict[str, object] | None = None
    if hard_negative_checkpoint is not None:
        teacher_path = Path(hard_negative_checkpoint).resolve()
        teacher_scores = _hard_negative_teacher_scores(
            teacher_path,
            features[masks["train"]],
            expected_feature_version=feature_version,
            expected_feature_names=feature_names,
            expected_horizon=tuple(
                float(value) for value in arrays["prediction_horizon_seconds"]
            ),
            device=torch_device,
        )
        negative_positions = np.flatnonzero(train_labels == 0)
        cutoff = float(
            np.quantile(teacher_scores[negative_positions], hard_negative_quantile)
        )
        hard_positions = negative_positions[
            teacher_scores[negative_positions] >= cutoff
        ]
        training_weights[hard_positions] *= hard_negative_weight
        hard_negative_metadata = {
            "teacher_checkpoint": str(teacher_path),
            "teacher_checkpoint_sha256": _sha256(teacher_path),
            "selection_split": "train_negative_only",
            "selection_quantile": hard_negative_quantile,
            "score_cutoff": cutoff,
            "selected_count": int(len(hard_positions)),
            "available_negative_count": int(len(negative_positions)),
            "sample_weight": hard_negative_weight,
        }
    training_features = normalized[masks["train"]]
    training_labels = train_labels.astype(np.float32)
    external_negative_metadata: dict[str, object] | None = None
    if external_negative_dataset is not None:
        external_path = Path(external_negative_dataset).resolve()
        external = _load_external_negative_dataset(
            external_path,
            expected_feature_version=feature_version,
            expected_feature_names=feature_names,
        )
        external_train_mask = external["split"] == "external_train_pool"
        if not external_train_mask.any():
            raise ValueError("external-negative dataset has no training-pool samples")
        external_train = external["features"][external_train_mask]
        external_mean = external_train.mean(axis=(0, 1), dtype=np.float64).astype(
            np.float32
        )
        external_std = external_train.std(axis=(0, 1), dtype=np.float64).astype(
            np.float32
        )
        external_std = np.where(external_std < 1e-6, 1.0, external_std).astype(
            np.float32
        )
        external_normalized = (
            (external_train - external_mean[None, None]) / external_std[None, None]
        ).astype(np.float32)
        training_features = np.concatenate(
            (training_features, external_normalized), axis=0
        )
        training_labels = np.concatenate(
            (training_labels, np.zeros(len(external_normalized), dtype=np.float32))
        )
        training_weights = np.concatenate(
            (
                training_weights,
                np.full(
                    len(external_normalized),
                    external_negative_weight,
                    dtype=np.float32,
                ),
            )
        )
        external_negative_metadata = {
            "dataset_file": str(external_path),
            "dataset_sha256": _sha256(external_path),
            "dataset_mode": str(external["dataset_mode"].item()),
            "selection_split": "external_train_pool",
            "selected_count": int(len(external_normalized)),
            "heldout_validation_count": int(
                np.sum(external["split"] == "external_validation")
            ),
            "heldout_test_count": int(
                np.sum(external["split"] == "external_test")
            ),
            "source_specific_normalization": True,
            "normalization_mean": external_mean.tolist(),
            "normalization_std": external_std.tolist(),
            "sample_weight": external_negative_weight,
        }
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(training_features),
            torch.from_numpy(training_labels),
            torch.from_numpy(training_weights),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_features, batch_labels, batch_weights in loader:
            optimizer.zero_grad(set_to_none=True)
            losses = training_criterion(
                model(batch_features.to(torch_device)),
                batch_labels.to(torch_device),
            )
            device_weights = batch_weights.to(torch_device)
            loss = (losses * device_weights).sum() / device_weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        validation_scores, validation_loss = _scores_and_loss(
            model,
            normalized[masks["validation"]],
            labels[masks["validation"]],
            criterion,
            torch_device,
            batch_size,
        )
        weighted_validation_loss = _weighted_binary_loss_from_scores(
            labels[masks["validation"]],
            validation_scores,
            all_sample_weights[masks["validation"]],
            positive_weight=positive_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "validation_loss": validation_loss,
                "weighted_validation_loss": weighted_validation_loss,
                "validation_auroc": _auroc(
                    labels[masks["validation"]], validation_scores
                ),
            }
        )
        if weighted_validation_loss < best_loss:
            best_loss = weighted_validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    model.eval()

    score_by_split: dict[str, np.ndarray] = {}
    loss_by_split: dict[str, float] = {}
    evaluation_splits = (
        ("train", "validation", "test")
        if evaluate_test_split
        else ("train", "validation")
    )
    for name in evaluation_splits:
        mask = masks[name]
        score_by_split[name], loss_by_split[name] = _scores_and_loss(
            model,
            normalized[mask],
            labels[mask],
            criterion,
            torch_device,
            batch_size,
        )
    if specificity_priority_minimum_sensitivity is None:
        threshold = _select_threshold(
            labels[masks["validation"]], score_by_split["validation"]
        )
        threshold_policy = "validation_balanced_accuracy"
    else:
        threshold = _select_specificity_priority_threshold(
            labels[masks["validation"]],
            score_by_split["validation"],
            minimum_sensitivity=specificity_priority_minimum_sensitivity,
        )
        threshold_policy = (
            "validation_maximum_specificity_with_window_sensitivity_at_least_"
            f"{specificity_priority_minimum_sensitivity:.3f}"
        )
    metrics = {
        name: asdict(
            _binary_metrics(labels[mask], score_by_split[name], loss_by_split[name], threshold)
        )
        for name, mask in masks.items()
        if name in evaluation_splits
    }
    horizon = tuple(float(value) for value in arrays["prediction_horizon_seconds"])
    checkpoint = {
        "model_version": EXPERIMENT_MODEL_VERSION,
        "model_mode": RESEARCH_MODEL_MODE,
        "model_architecture": architecture,
        "task_type": "prefall_prediction",
        "deployment_eligible": False,
        "shadow_only": True,
        "feature_version": feature_version,
        "feature_names": feature_names,
        "window_size": window_size,
        "input_size": len(feature_names),
        "hidden_size": hidden_size,
        "state_dict": best_state,
        "normalization_mean": torch.from_numpy(mean.copy()),
        "normalization_std": torch.from_numpy(std.copy()),
        "decision_threshold": threshold,
        "decision_threshold_policy": threshold_policy,
        "prediction_horizon_seconds": horizon,
        "positive_anchor": str(arrays["positive_anchor"].item()),
        "positive_weight": positive_weight,
        "hard_negative_mining": hard_negative_metadata,
        "external_negative_training": external_negative_metadata,
        "dataset_sample_weighting": {
            "enabled": "sample_weight" in arrays,
            "minimum": float(all_sample_weights.min()),
            "maximum": float(all_sample_weights.max()),
            "mean": float(all_sample_weights.mean()),
            "weighted_validation_model_selection": True,
        },
        "normalization_reference_count": int(normalization_mask.sum()),
        "source_specific_training_normalization": {
            "enabled": source_specific_training_normalization,
            "sources": source_normalization_metadata,
            "inference_normalization_unchanged": True,
        },
        "test_split_evaluated": evaluate_test_split,
        "dataset_sha256": _sha256(source),
        "seed": seed,
        "warning": "Architecture/horizon research experiment only; not accepted by live inference.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    report = {
        "dataset_file": str(source),
        "dataset_sha256": _sha256(source),
        "checkpoint_file": str(destination),
        "checkpoint_sha256": _sha256(destination),
        "architecture": architecture,
        "prediction_horizon_seconds": horizon,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "hidden_size": hidden_size,
        "positive_weight": positive_weight,
        "hard_negative_mining": hard_negative_metadata,
        "external_negative_training": external_negative_metadata,
        "test_split_evaluated": evaluate_test_split,
        "decision_threshold": threshold,
        "decision_threshold_policy": threshold_policy,
        "dataset_sample_weighting": checkpoint["dataset_sample_weighting"],
        "normalization_reference_count": int(normalization_mask.sum()),
        "source_specific_training_normalization": checkpoint[
            "source_specific_training_normalization"
        ],
        "train": metrics["train"],
        "validation": metrics["validation"],
        "test": metrics.get("test"),
        "history": history,
        "deployment_eligible": False,
    }
    destination.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _hard_negative_teacher_scores(
    checkpoint_path: Path,
    raw_features: np.ndarray,
    *,
    expected_feature_version: str,
    expected_feature_names: tuple[str, ...],
    expected_horizon: tuple[float, float],
    device: torch.device,
) -> np.ndarray:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"hard-negative checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_version") != EXPERIMENT_MODEL_VERSION:
        raise ValueError("hard-negative teacher must be a temporal experiment checkpoint")
    if checkpoint.get("feature_version") != expected_feature_version:
        raise ValueError("hard-negative teacher feature version is incompatible")
    if tuple(checkpoint.get("feature_names", ())) != expected_feature_names:
        raise ValueError("hard-negative teacher feature order is incompatible")
    teacher_horizon = tuple(
        float(value) for value in checkpoint.get("prediction_horizon_seconds", ())
    )
    if len(teacher_horizon) != 2 or not np.allclose(
        teacher_horizon, expected_horizon, atol=1e-6
    ):
        raise ValueError("hard-negative teacher prediction horizon is incompatible")
    model = TemporalBinaryModel(
        architecture=str(checkpoint["model_architecture"]),
        input_size=len(expected_feature_names),
        hidden_size=int(checkpoint["hidden_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    normalized = ((raw_features - mean[None, None]) / std[None, None]).astype(
        np.float32
    )
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(normalized), 512):
            logits = model(torch.from_numpy(normalized[start : start + 512]).to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores).astype(np.float64, copy=False)


def _weighted_binary_loss_from_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    sample_weights: np.ndarray,
    *,
    positive_weight: float,
) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if labels.shape != scores.shape or labels.shape != weights.shape:
        raise ValueError("weighted validation arrays must have matching shapes")
    if not len(labels) or np.any(weights <= 0.0) or not np.isfinite(weights).all():
        raise ValueError("weighted validation requires finite positive weights")
    losses = -(
        positive_weight * labels * np.log(scores)
        + (1.0 - labels) * np.log(1.0 - scores)
    )
    return float(np.sum(losses * weights) / np.sum(weights))


def _select_specificity_priority_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    minimum_sensitivity: float,
) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or not len(labels):
        raise ValueError("threshold arrays must be non-empty and aligned")
    positive_scores = np.sort(scores[labels == 1])[::-1]
    if not len(positive_scores):
        raise ValueError("threshold selection requires validation positives")
    required_positive_count = int(
        np.ceil(minimum_sensitivity * len(positive_scores) - 1e-12)
    )
    required_positive_count = min(max(required_positive_count, 1), len(positive_scores))
    return float(positive_scores[required_positive_count - 1])


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {path}")
    with np.load(path, allow_pickle=False) as dataset:
        required = {
            "features",
            "labels",
            "split",
            "feature_version",
            "feature_names",
            "dataset_mode",
            "positive_anchor",
            "prediction_horizon_seconds",
            "deployment_eligible",
        }
        missing = sorted(required.difference(dataset.files))
        if missing:
            raise ValueError(f"dataset is incomplete: {missing}")
        feature_version = str(dataset["feature_version"].item())
        expected_names = _SUPPORTED_FEATURE_CONTRACTS.get(feature_version)
        if expected_names is None:
            raise ValueError("feature version is incompatible")
        if tuple(str(value) for value in dataset["feature_names"]) != expected_names:
            raise ValueError("feature names/order are incompatible")
        if bool(dataset["deployment_eligible"].item()):
            raise ValueError("experiment dataset must be non-deployable")
        result = {name: np.asarray(dataset[name]) for name in required}
        for name in ("sample_weight", "normalization_reference", "dataset_origin"):
            if name in dataset.files:
                result[name] = np.asarray(dataset[name])
    result["features"] = np.asarray(result["features"], dtype=np.float32)
    result["labels"] = np.asarray(result["labels"], dtype=np.int64)
    if "sample_weight" in result:
        result["sample_weight"] = np.asarray(result["sample_weight"], dtype=np.float32)
    if "normalization_reference" in result:
        result["normalization_reference"] = np.asarray(
            result["normalization_reference"], dtype=bool
        )
    if "dataset_origin" in result:
        result["dataset_origin"] = np.asarray(result["dataset_origin"])
    return result


def _load_external_negative_dataset(
    path: Path,
    *,
    expected_feature_version: str = FEATURE_VERSION_V2,
    expected_feature_names: tuple[str, ...] = FEATURE_NAMES_V2,
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"external-negative dataset does not exist: {path}")
    with np.load(path, allow_pickle=False) as dataset:
        required = {
            "features",
            "labels",
            "split",
            "feature_version",
            "feature_names",
            "dataset_mode",
            "positive_samples_available",
            "deployment_validation_eligible",
        }
        missing = sorted(required.difference(dataset.files))
        if missing:
            raise ValueError(f"external-negative dataset is incomplete: {missing}")
        if str(dataset["feature_version"].item()) != expected_feature_version:
            raise ValueError("external-negative feature version is incompatible")
        if tuple(str(value) for value in dataset["feature_names"]) != expected_feature_names:
            raise ValueError("external-negative feature order is incompatible")
        labels = np.asarray(dataset["labels"], dtype=np.int8)
        if np.any(labels != 0) or bool(dataset["positive_samples_available"].item()):
            raise ValueError("external-negative dataset must contain only negative labels")
        if bool(dataset["deployment_validation_eligible"].item()):
            raise ValueError("external-negative dataset cannot be deployment validation")
        return {name: np.asarray(dataset[name]) for name in required}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM/causal-TCN pre-fall experiment.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--architecture", required=True, choices=("lstm", "causal_tcn"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--positive-weight-cap", type=float, default=32.0)
    parser.add_argument("--hard-negative-checkpoint", type=Path)
    parser.add_argument("--hard-negative-quantile", type=float, default=0.9)
    parser.add_argument("--hard-negative-weight", type=float, default=4.0)
    parser.add_argument("--external-negative-dataset", type=Path)
    parser.add_argument("--external-negative-weight", type=float, default=1.0)
    parser.add_argument("--source-specific-training-normalization", action="store_true")
    parser.add_argument("--specificity-priority-minimum-sensitivity", type=float)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = train_prefall_architecture_experiment(
        args.dataset,
        args.checkpoint,
        architecture=args.architecture,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        positive_weight_cap=args.positive_weight_cap,
        hard_negative_checkpoint=args.hard_negative_checkpoint,
        hard_negative_quantile=args.hard_negative_quantile,
        hard_negative_weight=args.hard_negative_weight,
        external_negative_dataset=args.external_negative_dataset,
        external_negative_weight=args.external_negative_weight,
        source_specific_training_normalization=(
            args.source_specific_training_normalization
        ),
        specificity_priority_minimum_sensitivity=(
            args.specificity_priority_minimum_sensitivity
        ),
        evaluate_test_split=not args.skip_test_evaluation,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
