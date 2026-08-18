from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from radar_module.model.prefall_experiment_v3 import _load_dataset
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
    MULTIHORIZON_MODEL_VERSION,
    MultiHorizonTemporalModel,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    WINDOW_SIZE_V2,
)


def train_multihorizon_experiment(
    early_dataset_path: str | Path,
    imminent_dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    architecture: str = "causal_tcn",
    epochs: int = 20,
    hidden_size: int = 24,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    positive_weight_cap: float = 32.0,
    imminent_loss_weight: float = 0.5,
    evaluate_test_split: bool = False,
    seed: int = 20260809,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Train an early-warning head with a nearer-horizon auxiliary objective."""

    if architecture not in {"lstm", "causal_tcn"}:
        raise ValueError("architecture must be lstm or causal_tcn")
    if min(epochs, hidden_size, batch_size) <= 0 or learning_rate <= 0:
        raise ValueError("training parameters must be positive")
    if positive_weight_cap <= 0 or imminent_loss_weight <= 0:
        raise ValueError("loss weights must be positive")

    early_path = Path(early_dataset_path).resolve()
    imminent_path = Path(imminent_dataset_path).resolve()
    destination = Path(checkpoint_path).resolve()
    early = _load_dataset(early_path)
    imminent = _load_dataset(imminent_path)
    early_horizon = _horizon(early)
    imminent_horizon = _horizon(imminent)
    if early_horizon[0] < imminent_horizon[0] or early_horizon[1] <= imminent_horizon[1]:
        raise ValueError("early dataset must end farther from the fall than imminent dataset")

    early_splits = np.asarray(early["split"])
    imminent_splits = np.asarray(imminent["split"])
    early_masks = {name: early_splits == name for name in ("train", "validation", "test")}
    imminent_masks = {
        name: imminent_splits == name for name in ("train", "validation", "test")
    }
    if any(not mask.any() for mask in (*early_masks.values(), *imminent_masks.values())):
        raise ValueError("both datasets must contain train, validation and test")

    early_features = np.asarray(early["features"], dtype=np.float32)
    imminent_features = np.asarray(imminent["features"], dtype=np.float32)
    early_labels = np.asarray(early["labels"], dtype=np.int64)
    imminent_labels = np.asarray(imminent["labels"], dtype=np.int64)
    mean = early_features[early_masks["train"]].mean(
        axis=(0, 1), dtype=np.float64
    ).astype(np.float32)
    std = early_features[early_masks["train"]].std(
        axis=(0, 1), dtype=np.float64
    ).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    early_normalized = ((early_features - mean[None, None]) / std[None, None]).astype(np.float32)
    imminent_normalized = ((imminent_features - mean[None, None]) / std[None, None]).astype(np.float32)

    early_positive_weight = _positive_weight(
        early_labels[early_masks["train"]], positive_weight_cap
    )
    imminent_positive_weight = _positive_weight(
        imminent_labels[imminent_masks["train"]], positive_weight_cap
    )
    torch_device = torch.device(device)
    _set_seed(seed)
    model = MultiHorizonTemporalModel(
        architecture=architecture,
        input_size=len(FEATURE_NAMES_V2),
        hidden_size=hidden_size,
    ).to(torch_device)
    early_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([early_positive_weight], device=torch_device)
    )
    imminent_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([imminent_positive_weight], device=torch_device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_loader = _loader(
        early_normalized[early_masks["train"]],
        early_labels[early_masks["train"]],
        batch_size,
        seed,
    )
    imminent_loader = _loader(
        imminent_normalized[imminent_masks["train"]],
        imminent_labels[imminent_masks["train"]],
        batch_size,
        seed + 1,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        early_batches = cycle(early_loader)
        imminent_batches = cycle(imminent_loader)
        for _ in range(max(len(early_loader), len(imminent_loader))):
            early_batch, early_target = next(early_batches)
            imminent_batch, imminent_target = next(imminent_batches)
            early_batch = early_batch.to(torch_device)
            imminent_batch = imminent_batch.to(torch_device)
            early_target = early_target.to(torch_device)
            imminent_target = imminent_target.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            early_logits, _ = model.forward_all(early_batch)
            _, imminent_logits = model.forward_all(imminent_batch)
            loss = early_criterion(early_logits, early_target)
            loss = loss + imminent_loss_weight * imminent_criterion(
                imminent_logits, imminent_target
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        early_scores, early_validation_loss = _scores_and_loss(
            model,
            early_normalized[early_masks["validation"]],
            early_labels[early_masks["validation"]],
            early_criterion,
            torch_device,
            batch_size,
        )
        imminent_scores, imminent_validation_loss = _head_scores_and_loss(
            model,
            imminent_normalized[imminent_masks["validation"]],
            imminent_labels[imminent_masks["validation"]],
            imminent_criterion,
            torch_device,
            batch_size,
        )
        joint_validation_loss = (
            early_validation_loss + imminent_loss_weight * imminent_validation_loss
        )
        history.append(
            {
                "epoch": epoch,
                "early_validation_loss": early_validation_loss,
                "early_validation_auroc": _auroc(
                    early_labels[early_masks["validation"]], early_scores
                ),
                "imminent_validation_loss": imminent_validation_loss,
                "imminent_validation_auroc": _auroc(
                    imminent_labels[imminent_masks["validation"]], imminent_scores
                ),
                "joint_validation_loss": joint_validation_loss,
            }
        )
        if early_validation_loss < best_loss:
            best_loss = early_validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    model.eval()
    evaluation_splits = (
        ("train", "validation", "test")
        if evaluate_test_split
        else ("train", "validation")
    )
    early_scores: dict[str, np.ndarray] = {}
    early_losses: dict[str, float] = {}
    imminent_scores: dict[str, np.ndarray] = {}
    imminent_losses: dict[str, float] = {}
    for name in evaluation_splits:
        early_scores[name], early_losses[name] = _scores_and_loss(
            model,
            early_normalized[early_masks[name]],
            early_labels[early_masks[name]],
            early_criterion,
            torch_device,
            batch_size,
        )
        imminent_scores[name], imminent_losses[name] = _head_scores_and_loss(
            model,
            imminent_normalized[imminent_masks[name]],
            imminent_labels[imminent_masks[name]],
            imminent_criterion,
            torch_device,
            batch_size,
        )
    threshold = _select_threshold(
        early_labels[early_masks["validation"]], early_scores["validation"]
    )
    imminent_threshold = _select_threshold(
        imminent_labels[imminent_masks["validation"]],
        imminent_scores["validation"],
    )
    early_metrics = {
        name: asdict(
            _binary_metrics(
                early_labels[early_masks[name]],
                early_scores[name],
                early_losses[name],
                threshold,
            )
        )
        for name in evaluation_splits
    }
    imminent_metrics = {
        name: asdict(
            _binary_metrics(
                imminent_labels[imminent_masks[name]],
                imminent_scores[name],
                imminent_losses[name],
                imminent_threshold,
            )
        )
        for name in evaluation_splits
    }
    checkpoint = {
        "model_version": MULTIHORIZON_MODEL_VERSION,
        "model_mode": RESEARCH_MODEL_MODE,
        "model_architecture": architecture,
        "task_type": "multihorizon_prefall_prediction",
        "deployment_eligible": False,
        "shadow_only": True,
        "feature_version": FEATURE_VERSION_V2,
        "feature_names": FEATURE_NAMES_V2,
        "window_size": WINDOW_SIZE_V2,
        "input_size": len(FEATURE_NAMES_V2),
        "hidden_size": hidden_size,
        "state_dict": best_state,
        "normalization_mean": torch.from_numpy(mean.copy()),
        "normalization_std": torch.from_numpy(std.copy()),
        "decision_threshold": threshold,
        "imminent_decision_threshold": imminent_threshold,
        "prediction_horizon_seconds": early_horizon,
        "auxiliary_prediction_horizon_seconds": imminent_horizon,
        "positive_anchor": str(early["positive_anchor"].item()),
        "positive_weight": early_positive_weight,
        "auxiliary_positive_weight": imminent_positive_weight,
        "auxiliary_loss_weight": imminent_loss_weight,
        "test_split_evaluated": evaluate_test_split,
        "dataset_sha256": _sha256(early_path),
        "auxiliary_dataset_sha256": _sha256(imminent_path),
        "seed": seed,
        "warning": "Multi-horizon research experiment only; not accepted by live inference.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    report: dict[str, object] = {
        "early_dataset_file": str(early_path),
        "early_dataset_sha256": _sha256(early_path),
        "imminent_dataset_file": str(imminent_path),
        "imminent_dataset_sha256": _sha256(imminent_path),
        "checkpoint_file": str(destination),
        "checkpoint_sha256": _sha256(destination),
        "architecture": architecture,
        "prediction_horizon_seconds": early_horizon,
        "auxiliary_prediction_horizon_seconds": imminent_horizon,
        "auxiliary_loss_weight": imminent_loss_weight,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "hidden_size": hidden_size,
        "decision_threshold": threshold,
        "imminent_decision_threshold": imminent_threshold,
        "test_split_evaluated": evaluate_test_split,
        "early": {**early_metrics, "test": early_metrics.get("test")},
        "imminent": {**imminent_metrics, "test": imminent_metrics.get("test")},
        "history": history,
        "deployment_eligible": False,
    }
    destination.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _horizon(dataset: dict[str, np.ndarray]) -> tuple[float, float]:
    values = tuple(float(value) for value in dataset["prediction_horizon_seconds"])
    if len(values) != 2 or not 0 < values[0] <= values[1]:
        raise ValueError("dataset prediction horizon is invalid")
    return values


def _positive_weight(labels: np.ndarray, cap: float) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if not positives or not negatives:
        raise ValueError("training split must contain both labels")
    return min(negatives / positives, cap)


def _loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(labels.astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _head_scores_and_loss(
    model: MultiHorizonTemporalModel,
    features: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    scores: list[np.ndarray] = []
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            target = torch.from_numpy(
                labels[start : start + batch_size].astype(np.float32)
            ).to(device)
            _, logits = model.forward_all(batch)
            total_loss += float(criterion(logits, target).item()) * len(batch)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores), total_loss / len(features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train shared multi-horizon pre-fall model.")
    parser.add_argument("--early-dataset", required=True, type=Path)
    parser.add_argument("--imminent-dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--architecture", default="causal_tcn", choices=("lstm", "causal_tcn"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--positive-weight-cap", type=float, default=32.0)
    parser.add_argument("--imminent-loss-weight", type=float, default=0.5)
    parser.add_argument("--evaluate-test-split", action="store_true")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = train_multihorizon_experiment(
        args.early_dataset,
        args.imminent_dataset,
        args.checkpoint,
        architecture=args.architecture,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        positive_weight_cap=args.positive_weight_cap,
        imminent_loss_weight=args.imminent_loss_weight,
        evaluate_test_split=args.evaluate_test_split,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
