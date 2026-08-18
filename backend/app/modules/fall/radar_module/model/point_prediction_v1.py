from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from radar_module.dataset.point_pretraining_v1 import POINT_PREDICTION_DATASET_MODE
from radar_module.model.point_pretraining_v1 import _batch, _normalization, _set_seed
from radar_module.model.point_temporal import (
    NormalDynamicsAuxiliaryHead,
    POINT_TEMPORAL_MODEL_VERSION,
    PointTemporalPredictionHead,
    PointTemporalPretrainingModel,
)
from radar_module.preprocess.pointcloud_sequence import POINT_FEATURE_NAMES, POINT_SEQUENCE_VERSION


POINT_PREDICTION_MODEL_VERSION = "pointnet_gru_prefall_v1"


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    sample_count: int
    positive_count: int
    negative_count: int
    loss: float
    sensitivity: float
    specificity: float
    balanced_accuracy: float
    precision: float
    auroc: float


@dataclass(frozen=True, slots=True)
class PointPredictionTrainingSummary:
    dataset_file: str
    pretraining_checkpoint_file: str
    checkpoint_file: str
    report_file: str
    epochs_requested: int
    best_epoch: int
    decision_threshold: float
    positive_weight: float
    parameter_count: int
    train: PredictionMetrics
    validation: PredictionMetrics
    test: PredictionMetrics | None
    test_event_count: int | None
    test_event_recall: float | None
    test_median_earliest_lead_seconds: float | None
    test_split_evaluated: bool
    deployment_eligible: bool


def train_point_prefall_v1(
    dataset_path: str | Path,
    pretraining_checkpoint_path: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    positive_weight_cap: float = 16.0,
    balance_temporal_groups: bool = True,
    evaluate_test_split: bool = False,
    normal_dynamics_weight: float = 0.0,
    seed: int = 20260808,
    device: str | torch.device = "cpu",
) -> PointPredictionTrainingSummary:
    if (
        epochs <= 0
        or batch_size <= 0
        or learning_rate <= 0
        or positive_weight_cap <= 0
        or normal_dynamics_weight < 0
    ):
        raise ValueError("training parameters must be positive")
    source = Path(dataset_path).resolve()
    pretraining_path = Path(pretraining_checkpoint_path).resolve()
    destination = Path(checkpoint_path).resolve()
    arrays = _load_prediction_dataset(source)
    pretrained = _safe_load(pretraining_path, map_location="cpu")
    _validate_pretraining_checkpoint(pretrained)
    splits = arrays["split"]
    masks = {name: splits == name for name in ("train", "validation", "test")}
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("dataset must contain train, validation and test samples")
    mean, std = _normalization(
        arrays["points"][masks["train"]], arrays["point_mask"][masks["train"]]
    )
    _set_seed(seed)
    torch_device = torch.device(device)
    pretraining_model = PointTemporalPretrainingModel(
        class_count=len(pretrained["class_names"]),
        frame_hidden_size=int(pretrained["frame_hidden_size"]),
        temporal_hidden_size=int(pretrained["temporal_hidden_size"]),
    )
    pretraining_model.load_state_dict(pretrained["state_dict"], strict=True)
    model = PointTemporalPredictionHead(pretraining_model.encoder, horizon_count=1).to(torch_device)
    dynamics_head = (
        NormalDynamicsAuxiliaryHead(
            temporal_hidden_size=model.encoder.temporal_hidden_size,
            frame_hidden_size=model.encoder.frame_encoder.hidden_size,
        ).to(torch_device)
        if normal_dynamics_weight > 0
        else None
    )

    train_labels = arrays["labels"][masks["train"]]
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("training split must contain both classes")
    positive_weight = 1.0 if balance_temporal_groups else min(
        negatives / positives, positive_weight_cap
    )
    evaluation_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=torch_device)
    )
    training_loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=torch_device),
        reduction="none",
    )
    train_sample_weights = np.ones(len(arrays["labels"]), dtype=np.float32)
    if balance_temporal_groups:
        same_recording = (
            arrays["label_source"]
            == "dguha_same_fall_recording_outside_prediction_horizon"
        )
        groups = (
            arrays["labels"] == 1,
            (arrays["labels"] == 0) & same_recording,
            (arrays["labels"] == 0) & ~same_recording,
        )
        for group in groups:
            train_group = group & masks["train"]
            count = int(train_group.sum())
            if count == 0:
                raise ValueError("each temporal training group must contain samples")
            train_sample_weights[train_group] = len(train_labels) / (3.0 * count)
    parameter_groups = [
        {"params": model.encoder.parameters(), "lr": learning_rate * 0.2},
        {"params": model.output.parameters(), "lr": learning_rate},
    ]
    if dynamics_head is not None:
        parameter_groups.append(
            {"params": dynamics_head.parameters(), "lr": learning_rate}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=1e-4,
    )
    train_indices = np.flatnonzero(masks["train"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_indices.astype(np.int64))),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_epoch = 0
    best_validation_auroc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        if dynamics_head is not None:
            dynamics_head.train()
        total_loss = 0.0
        total_dynamics_loss = 0.0
        dynamics_batches = 0
        seen = 0
        for (index_tensor,) in loader:
            indices = index_tensor.numpy()
            points, point_mask, frame_mask, labels = _batch(
                arrays, indices, mean, std, torch_device, augment=True
            )
            optimizer.zero_grad(set_to_none=True)
            if dynamics_head is None:
                logits = model(points, point_mask, frame_mask).squeeze(-1)
                dynamics_loss = None
            else:
                temporal_states, frame_embeddings = model.encoder.forward_sequence(
                    points, point_mask, frame_mask
                )
                representation = model.encoder.pool_last(temporal_states, frame_mask)
                logits = model.output(representation).squeeze(-1)
                normal_samples = torch.from_numpy(
                    (
                        (arrays["labels"][indices] == 0)
                        & (
                            arrays["label_source"][indices]
                            == "dguha_recording_activity_label"
                        )
                    )
                ).to(torch_device)
                valid_transitions = (
                    frame_mask[:, :-1].bool()
                    & frame_mask[:, 1:].bool()
                    & normal_samples[:, None]
                )
                predicted_next = dynamics_head(temporal_states[:, :-1])
                transition_losses = (
                    predicted_next - frame_embeddings[:, 1:].detach()
                ).square().mean(dim=-1)
                dynamics_loss = (
                    transition_losses[valid_transitions].mean()
                    if valid_transitions.any()
                    else transition_losses.sum() * 0.0
                )
            sample_weights = torch.from_numpy(train_sample_weights[indices]).to(torch_device)
            loss = (
                training_loss_function(logits, labels.float()) * sample_weights
            ).mean()
            if dynamics_loss is not None:
                loss = loss + normal_dynamics_weight * dynamics_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if dynamics_head is not None:
                torch.nn.utils.clip_grad_norm_(dynamics_head.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
            if dynamics_loss is not None:
                total_dynamics_loss += float(dynamics_loss.item())
                dynamics_batches += 1
            seen += len(indices)
        validation_scores, validation_loss = _scores(
            model, arrays, np.flatnonzero(masks["validation"]), mean, std,
            evaluation_criterion, torch_device, batch_size,
        )
        validation_auroc = _auroc(arrays["labels"][masks["validation"]], validation_scores)
        history.append(
            {
                "epoch": epoch,
                "train_weighted_loss": total_loss / max(seen, 1),
                "validation_weighted_loss": validation_loss,
                "validation_auroc": validation_auroc,
                "train_normal_dynamics_loss": (
                    total_dynamics_loss / dynamics_batches
                    if dynamics_batches
                    else None
                ),
            }
        )
        if validation_auroc > best_validation_auroc:
            best_validation_auroc = validation_auroc
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    score_by_split: dict[str, np.ndarray] = {}
    loss_by_split: dict[str, float] = {}
    evaluation_splits = (
        ("train", "validation", "test")
        if evaluate_test_split
        else ("train", "validation")
    )
    for name in evaluation_splits:
        mask = masks[name]
        score_by_split[name], loss_by_split[name] = _scores(
            model, arrays, np.flatnonzero(mask), mean, std, evaluation_criterion, torch_device, batch_size
        )
    threshold = _select_threshold(arrays["labels"][masks["validation"]], score_by_split["validation"])
    metrics = {
        name: _metrics(arrays["labels"][mask], score_by_split[name], loss_by_split[name], threshold)
        for name, mask in masks.items()
        if name in evaluation_splits
    }
    event_count: int | None = None
    event_recall: float | None = None
    median_lead: float | None = None
    if evaluate_test_split:
        event_count, event_recall, median_lead = _event_metrics(
            arrays["labels"][masks["test"]], score_by_split["test"],
            arrays["source_files"][masks["test"]], arrays["seconds_to_onset"][masks["test"]], threshold,
        )
    horizons = tuple(float(value) for value in arrays["prediction_horizon_seconds"])
    checkpoint: dict[str, Any] = {
        "model_version": POINT_PREDICTION_MODEL_VERSION,
        "encoder_initialization": POINT_TEMPORAL_MODEL_VERSION,
        "model_role": "weak_supervision_prefall_prediction",
        "sequence_version": POINT_SEQUENCE_VERSION,
        "feature_names": tuple(POINT_FEATURE_NAMES),
        "time_steps": int(arrays["points"].shape[1]),
        "max_points": int(arrays["points"].shape[2]),
        "frame_hidden_size": int(pretrained["frame_hidden_size"]),
        "temporal_hidden_size": int(pretrained["temporal_hidden_size"]),
        "state_dict": best_state,
        "normalization_mean": torch.from_numpy(mean),
        "normalization_std": torch.from_numpy(std),
        "decision_threshold": threshold,
        "prediction_horizon_seconds": horizons,
        "positive_anchor": "skeleton_derived_descent_onset",
        "kinect_used_as_model_input": False,
        "pretraining_checkpoint_sha256": _sha256(pretraining_path),
        "dataset_sha256": _sha256(source),
        "seed": seed,
        "best_epoch": best_epoch,
        "temporal_group_balancing": balance_temporal_groups,
        "normal_dynamics_auxiliary": {
            "enabled": dynamics_head is not None,
            "weight": normal_dynamics_weight,
            "selection_scope": "train dguha_recording_activity_label negatives only",
            "objective": "predict next observed PointNet frame embedding from causal GRU state",
            "saved_in_inference_checkpoint": False,
        },
        "test_split_evaluated": evaluate_test_split,
        "risk_head_trained": False,
        "deployment_eligible": False,
        "warning": (
            "Research-only weak prediction head. Labels are 0.1-0.6 seconds before "
            "a skeleton-derived descent onset in simulated forward falls."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    report_path = destination.with_suffix(".report.json")
    summary = PointPredictionTrainingSummary(
        dataset_file=str(source),
        pretraining_checkpoint_file=str(pretraining_path),
        checkpoint_file=str(destination),
        report_file=str(report_path),
        epochs_requested=epochs,
        best_epoch=best_epoch,
        decision_threshold=threshold,
        positive_weight=positive_weight,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        train=metrics["train"],
        validation=metrics["validation"],
        test=metrics.get("test"),
        test_event_count=event_count,
        test_event_recall=event_recall,
        test_median_earliest_lead_seconds=median_lead,
        test_split_evaluated=evaluate_test_split,
        deployment_eligible=False,
    )
    report = asdict(summary)
    report["training_history"] = history
    report["training_group_balancing"] = {
        "enabled": balance_temporal_groups,
        "groups": [
            "positive_prediction_horizon",
            "same_fall_recording_outside_horizon",
            "other_activity_negative",
        ],
        "policy": "equal total loss weight per group" if balance_temporal_groups else "disabled",
    }
    report["normal_dynamics_auxiliary"] = checkpoint["normal_dynamics_auxiliary"]
    report["prediction_horizon_seconds"] = horizons
    report["same_recording_negative_evaluation"] = {
        name: _negative_subgroup_metrics(
            score_by_split[name],
            arrays["label_source"][mask],
            threshold,
            "dguha_same_fall_recording_outside_prediction_horizon",
        )
        for name, mask in masks.items()
        if name in evaluation_splits
    }
    report["false_alarm_rate_note"] = (
        "Sparse sampled negative windows cannot support a false-alarms-per-hour estimate."
    )
    report["limitations"] = [
        "weak skeleton-derived timing, not verified loss-of-balance onset",
        "young healthy subjects and simulated forward falls only",
        "DGUHA hardware/domain differs from the live IWR6843 device",
        "event recall is not deployment validation",
    ]
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def _load_prediction_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"point prediction dataset does not exist: {path}")
    with np.load(path, allow_pickle=False) as dataset:
        required = {
            "points", "point_mask", "frame_mask", "labels", "split", "source_files",
            "seconds_to_onset", "label_source", "prediction_horizon_seconds", "sequence_version",
            "feature_names", "dataset_mode", "kinect_used_as_model_input", "deployment_eligible",
        }
        missing = sorted(required.difference(dataset.files))
        if missing:
            raise ValueError(f"point prediction dataset is incomplete: {missing}")
        if str(dataset["dataset_mode"].item()) != POINT_PREDICTION_DATASET_MODE:
            raise ValueError("point prediction dataset mode is incompatible")
        if str(dataset["sequence_version"].item()) != POINT_SEQUENCE_VERSION:
            raise ValueError("point prediction sequence version is incompatible")
        if tuple(str(value) for value in dataset["feature_names"]) != POINT_FEATURE_NAMES:
            raise ValueError("point prediction feature order is incompatible")
        if bool(dataset["kinect_used_as_model_input"].item()) or bool(dataset["deployment_eligible"].item()):
            raise ValueError("point prediction dataset metadata is unsafe")
        arrays = {name: np.asarray(dataset[name]) for name in required}
    points = np.asarray(arrays["points"], dtype=np.float32)
    point_mask = np.asarray(arrays["point_mask"], dtype=np.bool_)
    frame_mask = np.asarray(arrays["frame_mask"], dtype=np.bool_)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    if points.ndim != 4 or points.shape[-1] != len(POINT_FEATURE_NAMES):
        raise ValueError("point prediction tensor shape is incompatible")
    if point_mask.shape != points.shape[:3] or frame_mask.shape != points.shape[:2]:
        raise ValueError("point prediction masks are incompatible")
    if labels.shape != points.shape[:1] or not np.isin(labels, (0, 1)).all():
        raise ValueError("point prediction labels are invalid")
    return {
        "points": points,
        "point_mask": point_mask,
        "frame_mask": frame_mask,
        "labels": labels,
        "split": np.asarray(arrays["split"]),
        "source_files": np.asarray(arrays["source_files"]),
        "seconds_to_onset": np.asarray(arrays["seconds_to_onset"], dtype=np.float32),
        "label_source": np.asarray(arrays["label_source"]),
        "prediction_horizon_seconds": np.asarray(arrays["prediction_horizon_seconds"], dtype=np.float32),
    }


def _safe_load(path: Path, *, map_location: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"pretraining checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("pretraining checkpoint root must be a mapping")
    return payload


def _validate_pretraining_checkpoint(payload: dict[str, Any]) -> None:
    required = {
        "model_version", "model_role", "sequence_version", "feature_names", "class_names",
        "frame_hidden_size", "temporal_hidden_size", "state_dict", "fall_prediction_head_trained",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"pretraining checkpoint metadata is incomplete: {missing}")
    if payload["model_version"] != POINT_TEMPORAL_MODEL_VERSION:
        raise ValueError("pretraining model version is incompatible")
    if payload["model_role"] != "representation_pretraining":
        raise ValueError("checkpoint is not a representation pretraining artifact")
    if bool(payload["fall_prediction_head_trained"]):
        raise ValueError("pretraining checkpoint unexpectedly claims a prediction head")


def _scores(
    model: PointTemporalPredictionHead, arrays: dict[str, np.ndarray], indices: np.ndarray,
    mean: np.ndarray, std: np.ndarray, criterion: nn.Module, device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    scores: list[np.ndarray] = []
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            points, point_mask, frame_mask, labels = _batch(arrays, selected, mean, std, device)
            logits = model(points, point_mask, frame_mask).squeeze(-1)
            total_loss += float(criterion(logits, labels.float()).item()) * len(selected)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores).astype(np.float64), total_loss / max(len(indices), 1)


def _select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_threshold, best_balanced = 0.5, -1.0
    for candidate in candidates:
        metric = _metrics(labels, scores, 0.0, float(candidate))
        if metric.balanced_accuracy > best_balanced:
            best_threshold, best_balanced = float(candidate), metric.balanced_accuracy
    return best_threshold


def _metrics(labels: np.ndarray, scores: np.ndarray, loss: float, threshold: float) -> PredictionMetrics:
    predicted = scores >= threshold
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & negative))
    sensitivity = true_positive / max(int(positive.sum()), 1)
    specificity = int(np.sum(~predicted & negative)) / max(int(negative.sum()), 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    return PredictionMetrics(
        sample_count=len(labels), positive_count=int(positive.sum()), negative_count=int(negative.sum()),
        loss=float(loss), sensitivity=float(sensitivity), specificity=float(specificity),
        balanced_accuracy=float((sensitivity + specificity) / 2.0), precision=float(precision),
        auroc=float(_auroc(labels, scores)),
    )


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return 0.5
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _event_metrics(
    labels: np.ndarray, scores: np.ndarray, source_files: np.ndarray,
    seconds_to_onset: np.ndarray, threshold: float,
) -> tuple[int, float, float]:
    positive_files = np.unique(source_files[labels == 1])
    leads: list[float] = []
    detected = 0
    for source_file in positive_files:
        mask = (source_files == source_file) & (labels == 1)
        triggered = mask & (scores >= threshold)
        if triggered.any():
            detected += 1
            leads.append(float(np.nanmax(seconds_to_onset[triggered])))
    return len(positive_files), detected / max(len(positive_files), 1), float(np.median(leads)) if leads else 0.0


def _negative_subgroup_metrics(
    scores: np.ndarray,
    label_sources: np.ndarray,
    threshold: float,
    subgroup: str,
) -> dict[str, float | int]:
    selected = label_sources == subgroup
    count = int(selected.sum())
    false_positives = int(np.sum(scores[selected] >= threshold))
    return {
        "sample_count": count,
        "false_positive_count": false_positives,
        "false_positive_rate": false_positives / max(count, 1),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained point encoder for weak pre-fall prediction.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--pretraining-checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--positive-weight-cap", type=float, default=16.0)
    parser.add_argument("--evaluate-test-split", action="store_true")
    parser.add_argument("--normal-dynamics-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    summary = train_point_prefall_v1(
        args.dataset, args.pretraining_checkpoint, args.checkpoint,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        positive_weight_cap=args.positive_weight_cap,
        evaluate_test_split=args.evaluate_test_split,
        normal_dynamics_weight=args.normal_dynamics_weight,
        device=args.device,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
