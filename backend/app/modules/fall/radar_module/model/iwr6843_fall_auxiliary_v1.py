from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from radar_module.dataset.iwr6843_fall_v1 import (
    DATASET_MODE,
    FALL_ACTIONS,
    NONFALL_ACTIONS,
    SUBJECTS,
    export_iwr6843_fall_sequence_npz,
)
from radar_module.model.radar_lstm import RadarLSTM
from radar_module.model.research_training_v2 import (
    _auroc,
    _binary_metrics,
    _scores_and_loss,
    _select_threshold,
    _set_seed,
    _sha256,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    WINDOW_SIZE_V2,
)


MODEL_VERSION = "radar_lstm_iwr6843_fall_sequence_auxiliary_v1"
MODEL_MODE = "IWR6843_FALL_SEQUENCE_AUXILIARY"


@dataclass(frozen=True, slots=True)
class FoldTrainingResult:
    test_subject: str
    validation_subject: str
    train_subject: str
    dataset_file: str
    dataset_sha256: str
    checkpoint_file: str
    checkpoint_sha256: str
    best_epoch: int
    decision_threshold: float
    train: dict[str, float | int]
    validation: dict[str, float | int]
    test: dict[str, float | int]
    test_by_action: dict[str, dict[str, float | int]]


@dataclass(frozen=True, slots=True)
class LosoTrainingSummary:
    source_directory: str
    output_directory: str
    report_file: str
    folds: tuple[FoldTrainingResult, ...]
    aggregate_test: dict[str, float | int]
    aggregate_test_by_action: dict[str, dict[str, float | int]]
    epochs: int
    hidden_size: int
    batch_size: int
    learning_rate: float
    seed: int
    task_type: str
    lead_time_evaluation: str
    deployment_eligible: bool


def train_iwr6843_fall_loso(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    epochs: int = 100,
    hidden_size: int = 16,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 20260808,
    device: str | torch.device = "cpu",
    require_complete_release: bool = True,
) -> LosoTrainingSummary:
    """Train three subject-disjoint auxiliary folds.

    Each fold uses one subject for training, one for threshold/epoch selection,
    and the third for testing.  This is the only leakage-safe arrangement
    possible with the three available subjects.
    """

    if epochs <= 0 or hidden_size <= 0 or batch_size <= 0:
        raise ValueError("epochs, hidden_size and batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    source = Path(source_directory).resolve()
    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)

    fold_results: list[FoldTrainingResult] = []
    aggregate_labels: list[np.ndarray] = []
    aggregate_scores: list[np.ndarray] = []
    aggregate_predictions: list[np.ndarray] = []
    aggregate_actions: list[np.ndarray] = []
    fold_aurocs: list[float] = []

    for fold_index, test_subject in enumerate(SUBJECTS):
        validation_subject = SUBJECTS[(fold_index + 1) % len(SUBJECTS)]
        train_subject = SUBJECTS[(fold_index + 2) % len(SUBJECTS)]
        split_map = {
            train_subject: "train",
            validation_subject: "validation",
            test_subject: "test",
        }
        fold_stem = f"iwr6843_fall_aux_loso_test_{test_subject.lower()}"
        dataset_path = destination / f"{fold_stem}.npz"
        checkpoint_path = destination / f"{fold_stem}.pt"
        export_iwr6843_fall_sequence_npz(
            source,
            dataset_path,
            split_by_subject=split_map,
            require_complete_release=require_complete_release,
        )
        arrays = _load_dataset(dataset_path)
        result, test_scores = _train_fold(
            arrays,
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            train_subject=train_subject,
            validation_subject=validation_subject,
            test_subject=test_subject,
            epochs=epochs,
            hidden_size=hidden_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + fold_index,
            device=torch_device,
        )
        test_mask = arrays["split"] == "test"
        test_labels = arrays["labels"][test_mask]
        aggregate_labels.append(test_labels)
        aggregate_scores.append(test_scores)
        aggregate_predictions.append(test_scores >= result.decision_threshold)
        aggregate_actions.append(arrays["action"][test_mask])
        fold_aurocs.append(float(result.test["auroc"]))
        fold_results.append(result)

    labels = np.concatenate(aggregate_labels)
    scores = np.concatenate(aggregate_scores)
    predictions = np.concatenate(aggregate_predictions)
    actions = np.concatenate(aggregate_actions)
    aggregate_metrics = _metrics_from_predictions(labels, predictions)
    aggregate_metrics["macro_fold_auroc"] = float(np.mean(fold_aurocs))
    aggregate_metrics["pooled_score_auroc_unrecalibrated"] = float(
        _auroc(labels, scores)
    )
    summary = LosoTrainingSummary(
        source_directory=str(source),
        output_directory=str(destination),
        report_file=str(destination / "iwr6843_fall_auxiliary_loso_v1.report.json"),
        folds=tuple(fold_results),
        aggregate_test=aggregate_metrics,
        aggregate_test_by_action=_metrics_by_action(
            labels, predictions, scores, actions
        ),
        epochs=epochs,
        hidden_size=hidden_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        task_type="fall_sequence_auxiliary",
        lead_time_evaluation=(
            "unavailable: source labels do not identify loss-of-balance, descent "
            "onset or impact time"
        ),
        deployment_eligible=False,
    )
    report_path = Path(summary.report_file)
    payload = asdict(summary)
    payload["evaluation_protocol"] = (
        "three-fold subject-disjoint rotation; each subject is test exactly once"
    )
    payload["interpretation"] = (
        "This measures terminal fall-sequence discrimination on the source "
        "protocol. It is not an advance-prediction result and is not loaded by "
        "the live warning path."
    )
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def evaluate_prefall_checkpoint_on_iwr6843(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    report_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Cross-task stress test of a pre-fall checkpoint on IWR6843 recordings."""

    checkpoint_file = Path(checkpoint_path).resolve()
    dataset_file = Path(dataset_path).resolve()
    destination = Path(report_path).resolve()
    payload = _safe_torch_load(checkpoint_file, device)
    arrays = _load_dataset(dataset_file)
    if payload.get("feature_version") != FEATURE_VERSION_V2:
        raise ValueError("checkpoint feature version is incompatible")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES_V2:
        raise ValueError("checkpoint feature names/order are incompatible")
    if int(payload.get("window_size", -1)) != WINDOW_SIZE_V2:
        raise ValueError("checkpoint window size is incompatible")

    mean = np.asarray(payload["normalization_mean"], dtype=np.float32)
    std = np.asarray(payload["normalization_std"], dtype=np.float32)
    normalized = (
        (arrays["features"] - mean[None, None, :]) / std[None, None, :]
    ).astype(np.float32)
    torch_device = torch.device(device)
    model = RadarLSTM(
        input_size=len(FEATURE_NAMES_V2), hidden_size=int(payload["hidden_size"])
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(torch_device)
    model.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(
            model(torch.from_numpy(normalized).to(torch_device))
        ).cpu().numpy().astype(np.float64)
    threshold = float(payload["decision_threshold"])
    predictions = scores >= threshold
    labels = arrays["labels"]
    report: dict[str, Any] = {
        "checkpoint_file": str(checkpoint_file),
        "checkpoint_sha256": _sha256(checkpoint_file),
        "dataset_file": str(dataset_file),
        "dataset_sha256": _sha256(dataset_file),
        "decision_threshold": threshold,
        "overall_cross_task": _metrics_from_predictions(labels, predictions),
        "by_action": _metrics_by_action(
            labels, predictions, scores, arrays["action"]
        ),
        "nonfall_false_positive_rate": float(
            np.mean(predictions[labels == 0])
        ),
        "fall_recording_response_rate": float(
            np.mean(predictions[labels == 1])
        ),
        "lead_time_evaluation": (
            "unavailable because the IWR6843 dataset has no temporal fall anchor"
        ),
        "deployment_validation_eligible": False,
        "interpretation": (
            "Cross-task/domain stress test only. A score on a terminal fall "
            "recording is not evidence of advance prediction."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _train_fold(
    arrays: dict[str, np.ndarray],
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    train_subject: str,
    validation_subject: str,
    test_subject: str,
    epochs: int,
    hidden_size: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[FoldTrainingResult, np.ndarray]:
    features = arrays["features"]
    labels = arrays["labels"]
    splits = arrays["split"]
    masks = {name: splits == name for name in ("train", "validation", "test")}
    mean = features[masks["train"]].mean(axis=(0, 1), dtype=np.float64).astype(
        np.float32
    )
    std = features[masks["train"]].std(axis=(0, 1), dtype=np.float64).astype(
        np.float32
    )
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    normalized = ((features - mean[None, None, :]) / std[None, None, :]).astype(
        np.float32
    )

    _set_seed(seed)
    model = RadarLSTM(input_size=len(FEATURE_NAMES_V2), hidden_size=hidden_size)
    model.to(device)
    train_labels = labels[masks["train"]]
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("training subject must contain both classes")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized[masks["train"]]),
            torch.from_numpy(train_labels.astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    best_epoch = 0
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features.to(device))
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        validation_scores, validation_loss = _scores_and_loss(
            model,
            normalized[masks["validation"]],
            labels[masks["validation"]],
            criterion,
            device,
            batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "validation_loss": validation_loss,
                "validation_auroc": _auroc(
                    labels[masks["validation"]], validation_scores
                ),
            }
        )
        if validation_loss < best_loss:
            best_epoch = epoch
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    model.eval()

    score_by_split: dict[str, np.ndarray] = {}
    loss_by_split: dict[str, float] = {}
    for split_name, mask in masks.items():
        score_by_split[split_name], loss_by_split[split_name] = _scores_and_loss(
            model,
            normalized[mask],
            labels[mask],
            criterion,
            device,
            batch_size,
        )
    threshold = _select_threshold(
        labels[masks["validation"]], score_by_split["validation"]
    )
    metrics = {
        split_name: asdict(
            _binary_metrics(
                labels[mask], score_by_split[split_name], loss_by_split[split_name], threshold
            )
        )
        for split_name, mask in masks.items()
    }
    test_mask = masks["test"]
    test_scores = score_by_split["test"]
    test_predictions = test_scores >= threshold

    checkpoint = {
        "model_version": MODEL_VERSION,
        "model_mode": MODEL_MODE,
        "task_type": "fall_sequence_auxiliary",
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
        "seed": seed,
        "dataset_sha256": _sha256(dataset_path),
        "train_subject": train_subject,
        "validation_subject": validation_subject,
        "test_subject": test_subject,
        "label_quality": "recording_level",
        "lead_time_supported": False,
        "warning": (
            "Auxiliary fall-sequence classifier only; not a pre-fall predictor "
            "and not accepted by the live warning checkpoint loader."
        ),
    }
    torch.save(checkpoint, checkpoint_path)
    fold_report = {
        "best_epoch": best_epoch,
        "decision_threshold": threshold,
        "history": history,
        "metrics": metrics,
        "test_by_action": _metrics_by_action(
            labels[test_mask],
            test_predictions,
            test_scores,
            arrays["action"][test_mask],
        ),
    }
    checkpoint_path.with_suffix(".report.json").write_text(
        json.dumps(fold_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result = FoldTrainingResult(
        test_subject=test_subject,
        validation_subject=validation_subject,
        train_subject=train_subject,
        dataset_file=str(dataset_path),
        dataset_sha256=_sha256(dataset_path),
        checkpoint_file=str(checkpoint_path),
        checkpoint_sha256=_sha256(checkpoint_path),
        best_epoch=best_epoch,
        decision_threshold=threshold,
        train=metrics["train"],
        validation=metrics["validation"],
        test=metrics["test"],
        test_by_action=fold_report["test_by_action"],
    )
    return result, test_scores


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as dataset:
        required = {
            "features",
            "labels",
            "split",
            "subject_id",
            "action",
            "feature_version",
            "feature_names",
            "dataset_mode",
            "deployment_eligible",
        }
        missing = sorted(required.difference(dataset.files))
        if missing:
            raise ValueError(f"IWR6843 dataset is incomplete: {missing}")
        if str(dataset["dataset_mode"].item()) != DATASET_MODE:
            raise ValueError("unexpected IWR6843 dataset mode")
        if str(dataset["feature_version"].item()) != FEATURE_VERSION_V2:
            raise ValueError("IWR6843 feature version is incompatible")
        if tuple(str(value) for value in dataset["feature_names"]) != FEATURE_NAMES_V2:
            raise ValueError("IWR6843 feature names/order are incompatible")
        if bool(dataset["deployment_eligible"].item()):
            raise ValueError("IWR6843 research dataset must be non-deployable")
        arrays = {
            name: np.asarray(dataset[name])
            for name in ("features", "labels", "split", "subject_id", "action")
        }
    features = np.asarray(arrays["features"], dtype=np.float32)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    if features.shape[1:] != (WINDOW_SIZE_V2, len(FEATURE_NAMES_V2)):
        raise ValueError("IWR6843 feature tensor has incompatible shape")
    if not np.isfinite(features).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("IWR6843 features or labels are invalid")
    arrays["features"] = features
    arrays["labels"] = labels
    return arrays


def _metrics_from_predictions(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, float | int]:
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.sum(predictions & positive))
    true_negative = int(np.sum(~predictions & negative))
    sensitivity = true_positive / max(int(positive.sum()), 1)
    specificity = true_negative / max(int(negative.sum()), 1)
    return {
        "sample_count": len(labels),
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "accuracy": float(np.mean(predictions == positive)),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def _metrics_by_action(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    actions: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for action in sorted(set(str(value) for value in actions)):
        mask = actions == action
        action_labels = labels[mask]
        action_predictions = predictions[mask]
        entry: dict[str, float | int] = {
            "sample_count": int(mask.sum()),
            "mean_score": float(np.mean(scores[mask])),
            "positive_prediction_rate": float(np.mean(action_predictions)),
        }
        if action in FALL_ACTIONS:
            entry["fall_sequence_sensitivity"] = float(
                np.mean(action_predictions[action_labels == 1])
            )
        elif action in NONFALL_ACTIONS:
            entry["false_positive_rate"] = float(
                np.mean(action_predictions[action_labels == 0])
            )
        result[action] = entry
    return result


def _safe_torch_load(
    path: Path, device: str | torch.device
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the subject-disjoint IWR6843 fall-sequence auxiliary model."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    summary = train_iwr6843_fall_loso(
        args.source,
        args.output_directory,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
