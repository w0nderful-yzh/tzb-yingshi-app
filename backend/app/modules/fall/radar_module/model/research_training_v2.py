from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from radar_module.model.radar_lstm import RadarLSTM
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    WINDOW_SIZE_V2,
)


RESEARCH_MODEL_VERSION = "radar_lstm_research_weak_v2"
RESEARCH_MODEL_MODE = "RESEARCH_WEAK_SUPERVISION"


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    sample_count: int
    positive_count: int
    negative_count: int
    loss: float
    accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    auroc: float


@dataclass(frozen=True, slots=True)
class ResearchTrainingSummary:
    dataset_file: str
    dataset_sha256: str
    checkpoint_file: str
    checkpoint_sha256: str
    report_file: str
    seed: int
    epochs_requested: int
    epochs_completed: int
    best_epoch: int
    hidden_size: int
    batch_size: int
    learning_rate: float
    positive_weight: float
    positive_weight_cap: float | None
    decision_threshold: float
    source_label_group_balancing: bool
    train: BinaryMetrics
    validation: BinaryMetrics
    test: BinaryMetrics
    model_mode: str
    deployment_eligible: bool


def train_research_lstm_v2(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 15,
    hidden_size: int = 32,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 20260807,
    device: str | torch.device = "cpu",
    balance_source_label_groups: bool = False,
    positive_weight_cap: float | None = None,
) -> ResearchTrainingSummary:
    """Train a non-deployable prediction head from a compatible research export."""

    if epochs <= 0 or hidden_size <= 0 or batch_size <= 0:
        raise ValueError("epochs, hidden_size and batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if positive_weight_cap is not None and positive_weight_cap <= 0:
        raise ValueError("positive_weight_cap must be positive when provided")

    source = Path(dataset_path).resolve()
    destination = Path(checkpoint_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"research dataset does not exist: {source}")
    if destination.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("checkpoint_path must end with .pt or .pth")

    arrays = _load_dataset(source)
    features = arrays["features"]
    labels = arrays["labels"]
    splits = arrays["split"]
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("dataset must contain train, validation and test splits")

    mean = features[train_mask].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = features[train_mask].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    normalized = ((features - mean[None, None, :]) / std[None, None, :]).astype(
        np.float32
    )

    _set_seed(seed)
    torch_device = torch.device(device)
    model = RadarLSTM(input_size=len(FEATURE_NAMES_V2), hidden_size=hidden_size)
    model.to(torch_device)
    train_labels = labels[train_mask]
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("training split must contain both classes")
    effective_positive_weight = negatives / positives
    if positive_weight_cap is not None:
        effective_positive_weight = min(
            effective_positive_weight, positive_weight_cap
        )
    positive_weight = torch.tensor(
        [effective_positive_weight], dtype=torch.float32, device=torch_device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    train_loss_function = nn.BCEWithLogitsLoss(
        pos_weight=None if balance_source_label_groups else positive_weight,
        reduction="none",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_sample_weights = np.ones(len(train_labels), dtype=np.float32)
    if balance_source_label_groups:
        if "dataset_origin" not in arrays:
            raise ValueError(
                "source-label balancing requires dataset_origin metadata"
            )
        train_origins = arrays["dataset_origin"][train_mask]
        groups = np.asarray(
            [f"{origin}\x1f{label}" for origin, label in zip(train_origins, train_labels)]
        )
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        for group, count in zip(unique_groups, group_counts):
            train_sample_weights[groups == group] = len(groups) / (
                len(unique_groups) * int(count)
            )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized[train_mask]),
            torch.from_numpy(labels[train_mask].astype(np.float32)),
            torch.from_numpy(train_sample_weights),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    best_epoch = 0
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        cumulative_loss = 0.0
        seen = 0
        for batch_features, batch_labels, batch_weights in train_loader:
            batch_features = batch_features.to(torch_device)
            batch_labels = batch_labels.to(torch_device)
            batch_weights = batch_weights.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = (
                train_loss_function(logits, batch_labels) * batch_weights
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            cumulative_loss += float(loss.item()) * len(batch_labels)
            seen += len(batch_labels)

        validation_scores, validation_loss = _scores_and_loss(
            model,
            normalized[validation_mask],
            labels[validation_mask],
            criterion,
            torch_device,
            batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "train_weighted_loss": cumulative_loss / max(seen, 1),
                "validation_weighted_loss": validation_loss,
                "validation_auroc": _auroc(
                    labels[validation_mask], validation_scores
                ),
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
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
    for split_name, mask in (
        ("train", train_mask),
        ("validation", validation_mask),
        ("test", test_mask),
    ):
        score_by_split[split_name], loss_by_split[split_name] = _scores_and_loss(
            model,
            normalized[mask],
            labels[mask],
            criterion,
            torch_device,
            batch_size,
        )

    threshold = _select_threshold(
        labels[validation_mask], score_by_split["validation"]
    )
    metrics = {
        name: _binary_metrics(
            labels[mask], score_by_split[name], loss_by_split[name], threshold
        )
        for name, mask in (
            ("train", train_mask),
            ("validation", validation_mask),
            ("test", test_mask),
        )
    }

    source_dataset_mode = str(arrays["dataset_mode"].item())
    default_positive_definition = (
        "0.1-0.6 s before skeleton-derived whole-body descent onset; "
        "Kinect skeleton is used for offline timing only"
        if source_dataset_mode == "DGUHA_SKELETON_PSEUDOLABEL_RESEARCH_V2"
        else (
            "0.2-1.5 s before mmFall official fall-motion frame anchor; "
            "anchor is not verified impact"
        )
    )
    default_dataset_description = (
        "DGUHA radar-only features with skeleton-derived offline pseudo-labels"
        if source_dataset_mode == "DGUHA_SKELETON_PSEUDOLABEL_RESEARCH_V2"
        else "mmFall DS2 weak-supervision research export"
    )
    positive_label_definition = str(
        arrays.get(
            "positive_label_definition",
            np.asarray(default_positive_definition),
        ).item()
    )
    dataset_description = str(
        arrays.get(
            "dataset_description",
            np.asarray(default_dataset_description),
        ).item()
    )
    prediction_horizon = tuple(
        float(value)
        for value in np.asarray(
            arrays.get("prediction_horizon_seconds", np.asarray((0.1, 0.6)))
        ).reshape(-1)
    )
    if len(prediction_horizon) != 2 or not (
        0 < prediction_horizon[0] <= prediction_horizon[1]
    ):
        raise ValueError("dataset prediction horizon metadata is invalid")
    positive_anchor = str(
        arrays.get("positive_anchor", np.asarray("descent_onset")).item()
    )
    checkpoint: dict[str, Any] = {
        "model_version": RESEARCH_MODEL_VERSION,
        "model_mode": RESEARCH_MODEL_MODE,
        "deployment_eligible": False,
        "feature_version": FEATURE_VERSION_V2,
        "feature_names": tuple(FEATURE_NAMES_V2),
        "window_size": WINDOW_SIZE_V2,
        "input_size": len(FEATURE_NAMES_V2),
        "hidden_size": hidden_size,
        "state_dict": best_state,
        "normalization_mean": torch.from_numpy(mean.copy()),
        "normalization_std": torch.from_numpy(std.copy()),
        "decision_threshold": threshold,
        "seed": seed,
        "dataset_sha256": _sha256(source),
        "label_quality": "weak_supervision",
        "positive_label_definition": positive_label_definition,
        "dataset_description": dataset_description,
        "prediction_horizon_seconds": prediction_horizon,
        "positive_anchor": positive_anchor,
        "source_label_group_balancing": balance_source_label_groups,
        "positive_weight": effective_positive_weight,
        "positive_weight_cap": positive_weight_cap,
        "risk_output": (
            "not trained; runtime fall_risk_score remains rule motion risk "
            "upper-bounded by prediction evidence"
        ),
        "warning": (
            "Research artifact only. It has no subject-independent or "
            "deployment validation and must not be loaded as a trained "
            "production checkpoint."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    report_path = destination.with_suffix(".report.json")

    summary = ResearchTrainingSummary(
        dataset_file=str(source),
        dataset_sha256=_sha256(source),
        checkpoint_file=str(destination),
        checkpoint_sha256=_sha256(destination),
        report_file=str(report_path),
        seed=seed,
        epochs_requested=epochs,
        epochs_completed=len(history),
        best_epoch=best_epoch,
        hidden_size=hidden_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        positive_weight=effective_positive_weight,
        positive_weight_cap=positive_weight_cap,
        decision_threshold=threshold,
        source_label_group_balancing=balance_source_label_groups,
        train=metrics["train"],
        validation=metrics["validation"],
        test=metrics["test"],
        model_mode=RESEARCH_MODEL_MODE,
        deployment_eligible=False,
    )
    report = asdict(summary)
    report["training_history"] = history
    report["dataset_description"] = dataset_description
    report["prediction_horizon_seconds"] = prediction_horizon
    report["positive_anchor"] = positive_anchor
    if "dataset_origin" in arrays:
        origins = arrays["dataset_origin"]
        report["metrics_by_origin"] = {
            split_name: {
                str(origin): _group_metrics(
                    labels[mask][origins[mask] == origin],
                    score_by_split[split_name][origins[mask] == origin],
                    threshold,
                )
                for origin in np.unique(origins[mask])
            }
            for split_name, mask in (
                ("train", train_mask),
                ("validation", validation_mask),
                ("test", test_mask),
            )
        }
    report["evaluation_warning"] = (
        "All positive labels are weak or pseudo-labeled. Metrics do not "
        "estimate clinical or real-world advance-prediction performance."
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as dataset:
        required = {
            "features",
            "labels",
            "split",
            "feature_version",
            "feature_names",
            "dataset_mode",
            "deployment_eligible",
        }
        missing = sorted(required.difference(dataset.files))
        if missing:
            raise ValueError(f"research dataset is incomplete: {missing}")
        if str(dataset["feature_version"].item()) != FEATURE_VERSION_V2:
            raise ValueError("research dataset feature_version is incompatible")
        if tuple(str(value) for value in dataset["feature_names"]) != FEATURE_NAMES_V2:
            raise ValueError("research dataset feature names/order are incompatible")
        dataset_mode = str(dataset["dataset_mode"].item())
        if dataset_mode not in {
            RESEARCH_MODEL_MODE,
            "DGUHA_SKELETON_PSEUDOLABEL_RESEARCH_V2",
        }:
            raise ValueError("dataset is not an explicit weak-supervision export")
        if bool(dataset["deployment_eligible"].item()):
            raise ValueError("research dataset must be marked non-deployable")
        features = np.asarray(dataset["features"], dtype=np.float32)
        labels = np.asarray(dataset["labels"], dtype=np.int64)
        splits = np.asarray(dataset["split"])
        optional = {
            name: np.asarray(dataset[name])
            for name in (
                "positive_label_definition",
                "dataset_description",
                "dataset_origin",
                "prediction_horizon_seconds",
                "positive_anchor",
            )
            if name in dataset.files
        }
    if features.ndim != 3 or features.shape[1:] != (
        WINDOW_SIZE_V2,
        len(FEATURE_NAMES_V2),
    ):
        raise ValueError("research feature tensor has incompatible shape")
    if labels.shape != (len(features),) or splits.shape != (len(features),):
        raise ValueError("research labels/splits have incompatible shape")
    if not np.isfinite(features).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("research features or labels are invalid")
    return {
        "features": features,
        "labels": labels,
        "split": splits,
        "dataset_mode": np.asarray(dataset_mode),
        **optional,
    }


def _scores_and_loss(
    model: RadarLSTM,
    features: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    scores: list[np.ndarray] = []
    total_loss = 0.0
    seen = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            batch_features = torch.from_numpy(features[start:end]).to(device)
            batch_labels = torch.from_numpy(
                labels[start:end].astype(np.float32)
            ).to(device)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            total_loss += float(loss.item()) * (end - start)
            seen += end - start
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores).astype(np.float64), total_loss / max(seen, 1)


def _select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_threshold = 0.5
    best_balanced = -1.0
    for threshold in candidates:
        metrics = _binary_metrics(labels, scores, 0.0, float(threshold))
        if metrics.balanced_accuracy > best_balanced:
            best_balanced = metrics.balanced_accuracy
            best_threshold = float(threshold)
    return best_threshold


def _binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    loss: float,
    threshold: float,
) -> BinaryMetrics:
    predicted = scores >= threshold
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.sum(predicted & positive))
    true_negative = int(np.sum(~predicted & negative))
    sensitivity = true_positive / max(int(positive.sum()), 1)
    specificity = true_negative / max(int(negative.sum()), 1)
    return BinaryMetrics(
        sample_count=len(labels),
        positive_count=int(positive.sum()),
        negative_count=int(negative.sum()),
        loss=float(loss),
        accuracy=float(np.mean(predicted == positive)),
        balanced_accuracy=float((sensitivity + specificity) / 2.0),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        auroc=_auroc(labels, scores),
    )


def _group_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    metrics = asdict(_binary_metrics(labels, scores, 0.0, threshold))
    metrics.pop("loss")
    return metrics


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    if not len(positive_scores) or not len(negative_scores):
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            average = float(np.mean(ranks[order[start:end]]))
            ranks[order[start:end]] = average
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum
        - len(positive_scores) * (len(positive_scores) + 1) / 2.0
    ) / (len(positive_scores) * len(negative_scores))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a research-only v2 LSTM prediction head.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--balance-source-label-groups", action="store_true")
    parser.add_argument(
        "--positive-weight-cap",
        type=float,
        default=None,
        help=(
            "Optional upper bound for the negative/positive BCE class weight. "
            "Useful for dense-negative research exports where the raw ratio "
            "can over-amplify ambiguous positive pseudo-labels."
        ),
    )
    args = parser.parse_args()
    summary = train_research_lstm_v2(
        args.dataset,
        args.checkpoint,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        balance_source_label_groups=args.balance_source_label_groups,
        positive_weight_cap=args.positive_weight_cap,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
