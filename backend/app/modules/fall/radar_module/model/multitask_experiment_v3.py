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

from radar_module.dataset.iwr6843_fall_v1 import DATASET_MODE, SUBJECTS
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
    MULTITASK_MODEL_VERSION,
    SharedMultiTaskTemporalModel,
    TemporalBinaryModel,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    WINDOW_SIZE_V2,
)


ACTION_CLASSES = ("back", "bow", "front", "side", "squat", "walk")


def train_two_subject_outer_loso(
    iwr6843_path: str | Path,
    output_directory: str | Path,
    *,
    epochs: int = 100,
    hidden_size: int = 16,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 20260809,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Outer LOSO with both non-test subjects used by the final model.

    Epoch count and threshold are selected only from inner cross-subject runs
    between the two training subjects.  The outer test subject is never used
    for fitting, normalization, stopping or threshold calibration.
    """

    source = Path(iwr6843_path).resolve()
    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    arrays = _load_iwr6843(source)
    all_fold_labels: list[np.ndarray] = []
    all_fold_scores: list[np.ndarray] = []
    all_fold_predictions: list[np.ndarray] = []
    all_fold_actions: list[np.ndarray] = []
    fold_reports: list[dict[str, object]] = []
    torch_device = torch.device(device)

    for fold_index, test_subject in enumerate(SUBJECTS):
        train_subjects = tuple(subject for subject in SUBJECTS if subject != test_subject)
        inner_epochs: list[int] = []
        inner_thresholds: list[float] = []
        inner_reports: list[dict[str, object]] = []
        for inner_index, validation_subject in enumerate(train_subjects):
            inner_train_subject = next(
                subject for subject in train_subjects if subject != validation_subject
            )
            train_mask = arrays["subject_id"] == inner_train_subject
            validation_mask = arrays["subject_id"] == validation_subject
            inner = _fit_binary_model(
                arrays["features"],
                arrays["labels"],
                train_mask=train_mask,
                validation_mask=validation_mask,
                epochs=epochs,
                hidden_size=hidden_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed + fold_index * 10 + inner_index,
                device=torch_device,
            )
            inner_epochs.append(int(inner["best_epoch"]))
            inner_thresholds.append(float(inner["threshold"]))
            inner_reports.append(
                {
                    "train_subject": inner_train_subject,
                    "validation_subject": validation_subject,
                    "best_epoch": inner["best_epoch"],
                    "threshold": inner["threshold"],
                    "validation_auroc": inner["validation_auroc"],
                }
            )

        selected_epochs = max(1, int(round(float(np.mean(inner_epochs)))))
        selected_threshold = float(np.mean(inner_thresholds))
        train_mask = np.isin(arrays["subject_id"], train_subjects)
        test_mask = arrays["subject_id"] == test_subject
        final = _fit_binary_fixed_epochs(
            arrays["features"],
            arrays["labels"],
            train_mask=train_mask,
            epochs=selected_epochs,
            hidden_size=hidden_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + 100 + fold_index,
            device=torch_device,
        )
        test_scores = _binary_scores(
            final["model"],
            arrays["features"][test_mask],
            final["mean"],
            final["std"],
            torch_device,
        )
        test_labels = arrays["labels"][test_mask]
        test_predictions = test_scores >= selected_threshold
        metrics = _prediction_metrics(test_labels, test_predictions, test_scores)
        checkpoint_path = destination / f"iwr6843_two_subject_loso_test_{test_subject.lower()}.pt"
        checkpoint = {
            "model_version": "iwr6843_two_subject_outer_loso_v2",
            "model_mode": "IWR6843_FALL_SEQUENCE_AUXILIARY",
            "model_architecture": "lstm",
            "task_type": "fall_sequence_auxiliary",
            "deployment_eligible": False,
            "shadow_only": True,
            "feature_version": FEATURE_VERSION_V2,
            "feature_names": FEATURE_NAMES_V2,
            "window_size": WINDOW_SIZE_V2,
            "input_size": len(FEATURE_NAMES_V2),
            "hidden_size": hidden_size,
            "state_dict": final["state_dict"],
            "normalization_mean": torch.from_numpy(final["mean"].copy()),
            "normalization_std": torch.from_numpy(final["std"].copy()),
            "decision_threshold": selected_threshold,
            "train_subjects": train_subjects,
            "test_subject": test_subject,
            "threshold_calibration": "inner_cross_subject_out_of_fold_mean",
            "lead_time_supported": False,
            "dataset_sha256": _sha256(source),
            "seed": seed + 100 + fold_index,
            "warning": (
                "Recording-level fall/nonfall auxiliary experiment only; "
                "does not support pre-fall lead-time claims or live inference."
            ),
        }
        torch.save(checkpoint, checkpoint_path)
        fold_reports.append(
            {
                "test_subject": test_subject,
                "train_subjects": train_subjects,
                "inner_runs": inner_reports,
                "selected_epochs": selected_epochs,
                "selected_threshold": selected_threshold,
                "test": metrics,
                "test_by_action": _metrics_by_action(
                    test_labels,
                    test_predictions,
                    test_scores,
                    arrays["action"][test_mask],
                ),
                "checkpoint_file": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
            }
        )
        all_fold_labels.append(test_labels)
        all_fold_scores.append(test_scores)
        all_fold_predictions.append(test_predictions)
        all_fold_actions.append(arrays["action"][test_mask])

    labels = np.concatenate(all_fold_labels)
    scores = np.concatenate(all_fold_scores)
    predictions = np.concatenate(all_fold_predictions)
    actions = np.concatenate(all_fold_actions)
    report: dict[str, object] = {
        "experiment": "iwr6843_two_subject_outer_loso_v2",
        "dataset_file": str(source),
        "dataset_sha256": _sha256(source),
        "protocol": (
            "outer test subject held out; both remaining subjects train final model; "
            "inner cross-subject runs choose epoch and threshold"
        ),
        "folds": fold_reports,
        "aggregate_test": _prediction_metrics(labels, predictions, scores),
        "aggregate_test_by_action": _metrics_by_action(
            labels, predictions, scores, actions
        ),
        "lead_time_evaluation": "unavailable for recording-level IWR6843 labels",
        "deployment_eligible": False,
    }
    report_path = destination / "iwr6843_two_subject_outer_loso_v2.report.json"
    report["report_file"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def train_shared_multitask_loso(
    dguha_path: str | Path,
    iwr6843_path: str | Path,
    output_directory: str | Path,
    *,
    architecture: str = "lstm",
    epochs: int = 15,
    hidden_size: int = 32,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    steps_per_epoch: int = 128,
    positive_weight_cap: float = 32.0,
    seed: int = 20260809,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Train separate pre-fall, fall-sequence and action heads on one encoder."""

    dguha_file = Path(dguha_path).resolve()
    iwr_file = Path(iwr6843_path).resolve()
    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    dguha = _load_dguha(dguha_file)
    iwr = _load_iwr6843(iwr_file)
    action_to_index = {action: index for index, action in enumerate(ACTION_CLASSES)}
    action_labels = np.asarray([action_to_index[str(value)] for value in iwr["action"]])
    torch_device = torch.device(device)
    fold_reports: list[dict[str, object]] = []
    aggregate_labels: list[np.ndarray] = []
    aggregate_scores: list[np.ndarray] = []
    aggregate_predictions: list[np.ndarray] = []
    aggregate_actions: list[np.ndarray] = []

    for fold_index, test_subject in enumerate(SUBJECTS):
        iwr_train_mask = iwr["subject_id"] != test_subject
        iwr_test_mask = iwr["subject_id"] == test_subject
        dguha_train_mask = dguha["split"] == "train"
        dguha_validation_mask = dguha["split"] == "validation"
        dguha_test_mask = dguha["split"] == "test"
        dguha_mean, dguha_std = _normalization(
            dguha["features"][dguha_train_mask]
        )
        iwr_mean, iwr_std = _normalization(iwr["features"][iwr_train_mask])
        dguha_normalized = (
            (dguha["features"] - dguha_mean[None, None]) / dguha_std[None, None]
        ).astype(np.float32)
        iwr_normalized = (
            (iwr["features"] - iwr_mean[None, None]) / iwr_std[None, None]
        ).astype(np.float32)
        train_labels = dguha["labels"][dguha_train_mask]
        positive_weight = min(
            (len(train_labels) - int(train_labels.sum())) / int(train_labels.sum()),
            positive_weight_cap,
        )

        _set_seed(seed + fold_index)
        model = SharedMultiTaskTemporalModel(
            architecture=architecture,
            input_size=len(FEATURE_NAMES_V2),
            hidden_size=hidden_size,
            action_class_count=len(ACTION_CLASSES),
        ).to(torch_device)
        prefall_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([positive_weight], device=torch_device)
        )
        fall_criterion = nn.BCEWithLogitsLoss()
        action_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        dguha_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(dguha_normalized[dguha_train_mask]),
                torch.from_numpy(train_labels.astype(np.float32)),
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + fold_index),
        )
        iwr_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(iwr_normalized[iwr_train_mask]),
                torch.from_numpy(iwr["labels"][iwr_train_mask].astype(np.float32)),
                torch.from_numpy(action_labels[iwr_train_mask].astype(np.int64)),
            ),
            batch_size=min(batch_size, int(iwr_train_mask.sum())),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + 50 + fold_index),
        )
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        history: list[dict[str, float | int]] = []
        for epoch in range(1, epochs + 1):
            model.train()
            dguha_iterator = cycle(dguha_loader)
            iwr_iterator = cycle(iwr_loader)
            for _ in range(steps_per_epoch):
                dguha_features, dguha_labels = next(dguha_iterator)
                iwr_features, iwr_fall_labels, iwr_actions = next(iwr_iterator)
                optimizer.zero_grad(set_to_none=True)
                prefall_loss = prefall_criterion(
                    model.forward_prefall(dguha_features.to(torch_device)),
                    dguha_labels.to(torch_device),
                )
                fall_logits, action_logits = model.forward_iwr6843(
                    iwr_features.to(torch_device)
                )
                fall_loss = fall_criterion(
                    fall_logits, iwr_fall_labels.to(torch_device)
                )
                action_loss = action_criterion(
                    action_logits, iwr_actions.to(torch_device)
                )
                loss = prefall_loss + 0.5 * fall_loss + 0.25 * action_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            validation_scores = _prefall_scores(
                model,
                dguha_normalized[dguha_validation_mask],
                torch_device,
            )
            validation_loss = _weighted_binary_loss(
                validation_scores,
                dguha["labels"][dguha_validation_mask],
                positive_weight,
            )
            history.append(
                {
                    "epoch": epoch,
                    "validation_prefall_loss": validation_loss,
                    "validation_prefall_auroc": _auroc(
                        dguha["labels"][dguha_validation_mask], validation_scores
                    ),
                }
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
        assert best_state is not None
        model.load_state_dict(best_state, strict=True)
        model.eval()
        validation_scores = _prefall_scores(
            model, dguha_normalized[dguha_validation_mask], torch_device
        )
        prefall_threshold = _select_threshold(
            dguha["labels"][dguha_validation_mask], validation_scores
        )
        dguha_test_scores = _prefall_scores(
            model, dguha_normalized[dguha_test_mask], torch_device
        )
        iwr_fall_scores, iwr_action_predictions = _iwr_scores(
            model, iwr_normalized[iwr_test_mask], torch_device
        )
        iwr_labels = iwr["labels"][iwr_test_mask]
        iwr_predictions = iwr_fall_scores >= 0.5
        checkpoint_path = destination / f"multitask_loso_test_{test_subject.lower()}.pt"
        checkpoint = {
            "model_version": MULTITASK_MODEL_VERSION,
            "model_mode": RESEARCH_MODEL_MODE,
            "model_architecture": architecture,
            "task_type": "shared_prefall_fall_sequence_action",
            "deployment_eligible": False,
            "shadow_only": True,
            "feature_version": FEATURE_VERSION_V2,
            "feature_names": FEATURE_NAMES_V2,
            "window_size": WINDOW_SIZE_V2,
            "input_size": len(FEATURE_NAMES_V2),
            "hidden_size": hidden_size,
            "action_class_count": len(ACTION_CLASSES),
            "action_classes": ACTION_CLASSES,
            "state_dict": best_state,
            "normalization_mean": torch.from_numpy(dguha_mean.copy()),
            "normalization_std": torch.from_numpy(dguha_std.copy()),
            "iwr6843_normalization_mean": torch.from_numpy(iwr_mean.copy()),
            "iwr6843_normalization_std": torch.from_numpy(iwr_std.copy()),
            "decision_threshold": prefall_threshold,
            "iwr6843_decision_threshold": 0.5,
            "prediction_horizon_seconds": tuple(
                float(value) for value in dguha["prediction_horizon_seconds"]
            ),
            "positive_anchor": str(dguha["positive_anchor"].item()),
            "iwr6843_test_subject": test_subject,
            "iwr6843_train_subjects": tuple(
                subject for subject in SUBJECTS if subject != test_subject
            ),
            "dguha_dataset_sha256": _sha256(dguha_file),
            "iwr6843_dataset_sha256": _sha256(iwr_file),
            "loss_weights": {
                "prefall": 1.0,
                "fall_sequence": 0.5,
                "action": 0.25,
            },
            "seed": seed + fold_index,
            "warning": (
                "Research-only shared encoder with semantically separate heads; "
                "IWR6843 recording labels are not pre-fall labels."
            ),
        }
        torch.save(checkpoint, checkpoint_path)
        dguha_test_metrics = asdict(
            _binary_metrics(
                dguha["labels"][dguha_test_mask],
                dguha_test_scores,
                0.0,
                prefall_threshold,
            )
        )
        action_accuracy = float(
            np.mean(iwr_action_predictions == action_labels[iwr_test_mask])
        )
        fold_reports.append(
            {
                "test_subject": test_subject,
                "best_epoch": best_epoch,
                "prefall_threshold": prefall_threshold,
                "dguha_test": dguha_test_metrics,
                "iwr6843_test": _prediction_metrics(
                    iwr_labels, iwr_predictions, iwr_fall_scores
                ),
                "iwr6843_action_accuracy": action_accuracy,
                "iwr6843_action_confusion": _action_confusion(
                    action_labels[iwr_test_mask], iwr_action_predictions
                ),
                "checkpoint_file": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "history": history,
            }
        )
        aggregate_labels.append(iwr_labels)
        aggregate_scores.append(iwr_fall_scores)
        aggregate_predictions.append(iwr_predictions)
        aggregate_actions.append(iwr["action"][iwr_test_mask])

    labels = np.concatenate(aggregate_labels)
    scores = np.concatenate(aggregate_scores)
    predictions = np.concatenate(aggregate_predictions)
    actions = np.concatenate(aggregate_actions)
    report: dict[str, object] = {
        "experiment": "shared_prefall_fall_sequence_action_v3",
        "architecture": architecture,
        "dguha_dataset_file": str(dguha_file),
        "dguha_dataset_sha256": _sha256(dguha_file),
        "iwr6843_dataset_file": str(iwr_file),
        "iwr6843_dataset_sha256": _sha256(iwr_file),
        "loss_weights": {"prefall": 1.0, "fall_sequence": 0.5, "action": 0.25},
        "source_specific_normalization": True,
        "folds": fold_reports,
        "aggregate_iwr6843_test": _prediction_metrics(
            labels, predictions, scores
        ),
        "aggregate_iwr6843_by_action": _metrics_by_action(
            labels, predictions, scores, actions
        ),
        "mean_dguha_test": _mean_metric_dict(
            [fold["dguha_test"] for fold in fold_reports]
        ),
        "deployment_eligible": False,
    }
    report_path = destination / "shared_multitask_loso_v3.report.json"
    report["report_file"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _fit_binary_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    epochs: int,
    hidden_size: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    mean, std = _normalization(features[train_mask])
    normalized = ((features - mean[None, None]) / std[None, None]).astype(np.float32)
    _set_seed(seed)
    model = TemporalBinaryModel(
        architecture="lstm", input_size=len(FEATURE_NAMES_V2), hidden_size=hidden_size
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized[train_mask]),
            torch.from_numpy(labels[train_mask].astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_scores = None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features.to(device)), batch_labels.to(device))
            loss.backward()
            optimizer.step()
        scores = _binary_scores(model, features[validation_mask], mean, std, device)
        loss = _weighted_binary_loss(scores, labels[validation_mask], 1.0)
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            best_scores = scores.copy()
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    assert best_state is not None and best_scores is not None
    return {
        "best_epoch": best_epoch,
        "threshold": _select_threshold(labels[validation_mask], best_scores),
        "validation_auroc": _auroc(labels[validation_mask], best_scores),
    }


def _fit_binary_fixed_epochs(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    train_mask: np.ndarray,
    epochs: int,
    hidden_size: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    mean, std = _normalization(features[train_mask])
    normalized = ((features - mean[None, None]) / std[None, None]).astype(np.float32)
    _set_seed(seed)
    model = TemporalBinaryModel(
        architecture="lstm", input_size=len(FEATURE_NAMES_V2), hidden_size=hidden_size
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized[train_mask]),
            torch.from_numpy(labels[train_mask].astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(epochs):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features.to(device)), batch_labels.to(device))
            loss.backward()
            optimizer.step()
    state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return {"model": model, "state_dict": state, "mean": mean, "std": std}


def _binary_scores(
    model: nn.Module,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    normalized = ((features - mean[None, None]) / std[None, None]).astype(np.float32)
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(normalized).to(device))).cpu().numpy()


def _prefall_scores(
    model: SharedMultiTaskTemporalModel,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(features), 512):
            batches.append(
                torch.sigmoid(
                    model.forward_prefall(
                        torch.from_numpy(features[start : start + 512]).to(device)
                    )
                ).cpu().numpy()
            )
    return np.concatenate(batches).astype(np.float64)


def _iwr_scores(
    model: SharedMultiTaskTemporalModel,
    features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.inference_mode():
        fall_logits, action_logits = model.forward_iwr6843(
            torch.from_numpy(features).to(device)
        )
    return (
        torch.sigmoid(fall_logits).cpu().numpy().astype(np.float64),
        torch.argmax(action_logits, dim=1).cpu().numpy().astype(np.int64),
    )


def _normalization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = features.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    return mean, np.where(std < 1e-6, 1.0, std).astype(np.float32)


def _weighted_binary_loss(
    scores: np.ndarray, labels: np.ndarray, positive_weight: float
) -> float:
    clipped = np.clip(scores.astype(np.float64), 1e-7, 1 - 1e-7)
    values = -(
        positive_weight * labels * np.log(clipped)
        + (1 - labels) * np.log(1 - clipped)
    )
    return float(np.mean(values))


def _prediction_metrics(
    labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    positive = labels == 1
    negative = ~positive
    sensitivity = float(np.mean(predictions[positive]))
    specificity = float(np.mean(~predictions[negative]))
    return {
        "sample_count": len(labels),
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "accuracy": float(np.mean(predictions == positive)),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": _auroc(labels, scores),
    }


def _metrics_by_action(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    actions: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    result = {}
    for action in ACTION_CLASSES:
        mask = actions == action
        if not mask.any():
            continue
        entry: dict[str, float | int] = {
            "sample_count": int(mask.sum()),
            "mean_score": float(np.mean(scores[mask])),
            "positive_prediction_rate": float(np.mean(predictions[mask])),
        }
        if int(labels[mask][0]) == 1:
            entry["fall_sequence_sensitivity"] = float(np.mean(predictions[mask]))
        else:
            entry["false_positive_rate"] = float(np.mean(predictions[mask]))
        result[action] = entry
    return result


def _action_confusion(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, dict[str, int]]:
    matrix = {
        actual: {predicted: 0 for predicted in ACTION_CLASSES}
        for actual in ACTION_CLASSES
    }
    for actual, predicted in zip(labels, predictions):
        matrix[ACTION_CLASSES[int(actual)]][ACTION_CLASSES[int(predicted)]] += 1
    return matrix


def _mean_metric_dict(items: list[dict[str, object]]) -> dict[str, float]:
    keys = (
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "auroc",
    )
    return {key: float(np.mean([float(item[key]) for item in items])) for key in keys}


def _load_iwr6843(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as dataset:
        if str(dataset["dataset_mode"].item()) != DATASET_MODE:
            raise ValueError("unexpected IWR6843 dataset mode")
        return {
            name: np.asarray(dataset[name])
            for name in ("features", "labels", "subject_id", "action")
        }


def _load_dguha(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as dataset:
        return {
            name: np.asarray(dataset[name])
            for name in (
                "features",
                "labels",
                "split",
                "prediction_horizon_seconds",
                "positive_anchor",
            )
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run radar multi-task experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    loso = subparsers.add_parser("two-subject-loso")
    loso.add_argument("--iwr6843", required=True, type=Path)
    loso.add_argument("--output-directory", required=True, type=Path)
    loso.add_argument("--epochs", type=int, default=100)
    multitask = subparsers.add_parser("multitask")
    multitask.add_argument("--dguha", required=True, type=Path)
    multitask.add_argument("--iwr6843", required=True, type=Path)
    multitask.add_argument("--output-directory", required=True, type=Path)
    multitask.add_argument("--architecture", choices=("lstm", "causal_tcn"), default="lstm")
    multitask.add_argument("--epochs", type=int, default=15)
    multitask.add_argument("--hidden-size", type=int, default=32)
    multitask.add_argument("--steps-per-epoch", type=int, default=128)
    multitask.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "two-subject-loso":
        report = train_two_subject_outer_loso(
            args.iwr6843, args.output_directory, epochs=args.epochs
        )
    else:
        report = train_shared_multitask_loso(
            args.dguha,
            args.iwr6843,
            args.output_directory,
            architecture=args.architecture,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            steps_per_epoch=args.steps_per_epoch,
            device=args.device,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
