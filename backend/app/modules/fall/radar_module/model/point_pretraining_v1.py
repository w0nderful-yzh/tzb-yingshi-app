from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from radar_module.dataset.point_pretraining_v1 import POINT_PRETRAIN_DATASET_MODE
from radar_module.model.point_temporal import (
    POINT_TEMPORAL_MODEL_VERSION,
    PointTemporalPretrainingModel,
)
from radar_module.preprocess.pointcloud_sequence import (
    POINT_FEATURE_NAMES,
    POINT_SEQUENCE_VERSION,
)


@dataclass(frozen=True, slots=True)
class ActivityMetrics:
    sample_count: int
    loss: float
    accuracy: float
    macro_recall: float


@dataclass(frozen=True, slots=True)
class PointPretrainingSummary:
    dataset_file: str
    checkpoint_file: str
    report_file: str
    epochs_requested: int
    best_epoch: int
    class_count: int
    parameter_count: int
    train: ActivityMetrics
    validation: ActivityMetrics
    test: ActivityMetrics
    representation_pretraining_only: bool
    deployment_eligible: bool


def train_point_activity_pretraining_v1(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    frame_hidden_size: int = 64,
    temporal_hidden_size: int = 64,
    seed: int = 20260808,
    device: str | torch.device = "cpu",
) -> PointPretrainingSummary:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    source = Path(dataset_path).resolve()
    destination = Path(checkpoint_path).resolve()
    arrays = _load_dataset(source)
    labels = arrays["labels"]
    splits = arrays["split"]
    class_count = len(arrays["action_names"])
    masks = {
        name: splits == name for name in ("train", "validation", "test")
    }
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("dataset must contain train, validation and test samples")

    mean, std = _normalization(
        arrays["points"][masks["train"]],
        arrays["point_mask"][masks["train"]],
    )
    _set_seed(seed)
    torch_device = torch.device(device)
    model = PointTemporalPretrainingModel(
        class_count=class_count,
        frame_hidden_size=frame_hidden_size,
        temporal_hidden_size=temporal_hidden_size,
    ).to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    train_indices = np.flatnonzero(masks["train"])
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_indices.astype(np.int64))),
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
        total_loss = 0.0
        seen = 0
        for (index_tensor,) in train_loader:
            indices = index_tensor.numpy()
            points, point_mask, frame_mask, batch_labels = _batch(
                arrays, indices, mean, std, torch_device, augment=True
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(points, point_mask, frame_mask)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
            seen += len(indices)
        validation = _evaluate(
            model, arrays, np.flatnonzero(masks["validation"]), mean, std, criterion,
            torch_device, batch_size, class_count,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(seen, 1),
                "validation_loss": validation.loss,
                "validation_accuracy": validation.accuracy,
                "validation_macro_recall": validation.macro_recall,
            }
        )
        if validation.loss < best_loss:
            best_loss = validation.loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    metrics = {
        name: _evaluate(
            model, arrays, np.flatnonzero(mask), mean, std, criterion,
            torch_device, batch_size, class_count,
        )
        for name, mask in masks.items()
    }
    checkpoint: dict[str, Any] = {
        "model_version": POINT_TEMPORAL_MODEL_VERSION,
        "model_role": "representation_pretraining",
        "sequence_version": POINT_SEQUENCE_VERSION,
        "feature_names": tuple(POINT_FEATURE_NAMES),
        "class_names": tuple(str(value) for value in arrays["action_names"]),
        "frame_hidden_size": frame_hidden_size,
        "temporal_hidden_size": temporal_hidden_size,
        "state_dict": best_state,
        "normalization_mean": torch.from_numpy(mean),
        "normalization_std": torch.from_numpy(std),
        "dataset_sha256": _sha256(source),
        "seed": seed,
        "best_epoch": best_epoch,
        "fall_prediction_head_trained": False,
        "fall_risk_head_trained": False,
        "deployment_eligible": False,
        "warning": (
            "Activity representation pretraining only; this checkpoint cannot "
            "produce or validate a fall-prediction probability."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    report_path = destination.with_suffix(".report.json")
    summary = PointPretrainingSummary(
        dataset_file=str(source),
        checkpoint_file=str(destination),
        report_file=str(report_path),
        epochs_requested=epochs,
        best_epoch=best_epoch,
        class_count=class_count,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        train=metrics["train"],
        validation=metrics["validation"],
        test=metrics["test"],
        representation_pretraining_only=True,
        deployment_eligible=False,
    )
    report = asdict(summary)
    report["training_history"] = history
    report["augmentation"] = {
        "z_axis_rotation_degrees": 15.0,
        "xyz_jitter_std_m": 0.005,
        "point_dropout_probability": 0.1,
    }
    report["interpretation"] = (
        "Subject-disjoint mmRadPose activity metrics measure representation "
        "pretraining only, not pre-fall prediction."
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"pretraining dataset does not exist: {path}")
    with np.load(path, allow_pickle=False) as dataset:
        required = {
            "points", "point_mask", "frame_mask", "labels", "split",
            "action_names", "sequence_version", "feature_names", "dataset_mode",
            "fall_prediction_labels_available", "deployment_eligible",
        }
        missing = sorted(required.difference(dataset.files))
        if missing:
            raise ValueError(f"pretraining dataset is incomplete: {missing}")
        if str(dataset["dataset_mode"].item()) != POINT_PRETRAIN_DATASET_MODE:
            raise ValueError("dataset_mode is incompatible")
        if str(dataset["sequence_version"].item()) != POINT_SEQUENCE_VERSION:
            raise ValueError("sequence_version is incompatible")
        if tuple(str(value) for value in dataset["feature_names"]) != POINT_FEATURE_NAMES:
            raise ValueError("point feature names/order are incompatible")
        if bool(dataset["fall_prediction_labels_available"].item()):
            raise ValueError("activity pretraining data must not claim prediction labels")
        if bool(dataset["deployment_eligible"].item()):
            raise ValueError("pretraining data must be research-only")
        arrays = {name: np.asarray(dataset[name]) for name in required}
    points = np.asarray(arrays["points"], dtype=np.float32)
    point_mask = np.asarray(arrays["point_mask"], dtype=np.bool_)
    frame_mask = np.asarray(arrays["frame_mask"], dtype=np.bool_)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    split = np.asarray(arrays["split"])
    if points.ndim != 4 or points.shape[-1] != len(POINT_FEATURE_NAMES):
        raise ValueError("points must have shape [sample, time, point, feature]")
    if point_mask.shape != points.shape[:3] or frame_mask.shape != points.shape[:2]:
        raise ValueError("point/frame masks are incompatible")
    if labels.shape != points.shape[:1] or split.shape != points.shape[:1]:
        raise ValueError("labels/split are incompatible")
    if not np.isfinite(points).all():
        raise ValueError("points contain non-finite values")
    return {
        "points": points,
        "point_mask": point_mask,
        "frame_mask": frame_mask,
        "labels": labels,
        "split": split,
        "action_names": np.asarray(arrays["action_names"]),
    }


def _normalization(points: np.ndarray, point_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(len(POINT_FEATURE_NAMES), dtype=np.float32)
    std = np.ones(len(POINT_FEATURE_NAMES), dtype=np.float32)
    for feature_index in range(5):
        valid = point_mask
        if feature_index == 4:
            valid = valid & (points[..., 5] > 0.5)
        values = points[..., feature_index][valid]
        if not len(values):
            continue
        mean[feature_index] = float(np.mean(values, dtype=np.float64))
        value_std = float(np.std(values, dtype=np.float64))
        std[feature_index] = max(value_std, 1e-6)
    return mean, std


def _batch(
    arrays: dict[str, np.ndarray], indices: np.ndarray, mean: np.ndarray, std: np.ndarray,
    device: torch.device, *, augment: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = np.asarray(arrays["points"][indices], dtype=np.float32)
    raw_tensor = torch.from_numpy(raw).to(device)
    point_mask = torch.from_numpy(arrays["point_mask"][indices]).to(device)
    if augment:
        raw_tensor, point_mask = _augment(raw_tensor, point_mask)
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)
    normalized = (raw_tensor - mean_tensor[None, None, None, :]) / std_tensor[None, None, None, :]
    normalized[..., 4] = torch.where(raw_tensor[..., 5] > 0.5, normalized[..., 4], 0.0)
    normalized[..., 5] = raw_tensor[..., 5]
    normalized *= point_mask.unsqueeze(-1)
    return (
        normalized,
        point_mask,
        torch.from_numpy(arrays["frame_mask"][indices]).to(device),
        torch.from_numpy(arrays["labels"][indices]).long().to(device),
    )


def _augment(points: torch.Tensor, point_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply PointNet-style augmentation in physical units before normalization."""
    result = points.clone()
    batch = len(result)
    angles = (torch.rand(batch, device=result.device) * 2.0 - 1.0) * math.radians(15.0)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    x, y = result[..., 0].clone(), result[..., 1].clone()
    result[..., 0] = cosine[:, None, None] * x - sine[:, None, None] * y
    result[..., 1] = sine[:, None, None] * x + cosine[:, None, None] * y
    jitter = torch.randn_like(result[..., :3]) * 0.005
    result[..., :3] += jitter * point_mask.unsqueeze(-1)
    keep = (torch.rand_like(point_mask.float()) >= 0.1) & point_mask
    empty_observed = point_mask.any(dim=2) & ~keep.any(dim=2)
    if empty_observed.any():
        first_valid = point_mask.float().argmax(dim=2)
        batch_index, time_index = torch.where(empty_observed)
        keep[batch_index, time_index, first_valid[batch_index, time_index]] = True
    result *= keep.unsqueeze(-1)
    return result, keep


def _evaluate(
    model: PointTemporalPretrainingModel, arrays: dict[str, np.ndarray], indices: np.ndarray,
    mean: np.ndarray, std: np.ndarray, criterion: nn.Module, device: torch.device,
    batch_size: int, class_count: int,
) -> ActivityMetrics:
    losses = 0.0
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            points, point_mask, frame_mask, batch_labels = _batch(
                arrays, batch_indices, mean, std, device
            )
            logits = model(points, point_mask, frame_mask)
            losses += float(criterion(logits, batch_labels).item()) * len(batch_indices)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    predicted = np.concatenate(predictions)
    observed = np.concatenate(labels)
    recalls = []
    for class_index in range(class_count):
        relevant = observed == class_index
        if relevant.any():
            recalls.append(float(np.mean(predicted[relevant] == class_index)))
    return ActivityMetrics(
        sample_count=len(indices),
        loss=losses / max(len(indices), 1),
        accuracy=float(np.mean(predicted == observed)),
        macro_recall=float(np.mean(recalls)),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mmRadPose PointNet/GRU representation encoder.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    summary = train_point_activity_pretraining_v1(
        args.dataset, args.checkpoint, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate, device=args.device,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
