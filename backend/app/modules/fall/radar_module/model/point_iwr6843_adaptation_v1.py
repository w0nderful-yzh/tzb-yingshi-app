from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from radar_module.dataset.point_iwr6843_adaptation_v1 import (
    DGUHA_MODE,
    FEATURE_NAMES,
    IWR_MODE,
    SEQUENCE_VERSION,
)
from radar_module.model.point_temporal import (
    PointTemporalEncoder,
    PointTemporalPredictionHead,
    PointTemporalPretrainingModel,
)


PRETRAIN_VERSION = "pointnet_gru_iwr6843_nonfall_pretrain_v1"
PREDICTION_VERSION = "pointnet_gru_iwr6843_adaptation_v1"


def run_first_stage_training(
    dguha_dataset: str | Path,
    iwr_dataset: str | Path,
    output_directory: str | Path,
    *,
    seeds: tuple[int, ...] = (20260808, 20260809, 20260810),
    pretrain_epochs: int = 30,
    prediction_epochs: int = 25,
    batch_size: int = 64,
    pretrain_learning_rate: float = 1e-3,
    prediction_learning_rate: float = 5e-4,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    dguha_path = Path(dguha_dataset).resolve()
    iwr_path = Path(iwr_dataset).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dguha = _load_dataset(dguha_path, DGUHA_MODE)
    iwr = _load_dataset(iwr_path, IWR_MODE)
    _assert_sample_contract(dguha)
    device_value = torch.device(device)
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        pretrain_path = output / f"IWR_encoder_seed{seed}.pt"
        pretrain = train_iwr_representation(
            iwr, iwr_path, pretrain_path, seed=seed, epochs=pretrain_epochs,
            batch_size=batch_size, learning_rate=pretrain_learning_rate, device=device_value,
        )
        b1_path = output / f"B1_seed{seed}.pt"
        b2_path = output / f"B2_seed{seed}.pt"
        b1 = train_prediction(
            dguha, dguha_path, b1_path, seed=seed, epochs=prediction_epochs,
            batch_size=batch_size, learning_rate=prediction_learning_rate,
            device=device_value, initialization="random_dguha_only", pretraining=None,
        )
        b2 = train_prediction(
            dguha, dguha_path, b2_path, seed=seed, epochs=prediction_epochs,
            batch_size=batch_size, learning_rate=prediction_learning_rate,
            device=device_value, initialization="iwr6843_pretrained_frozen_encoder",
            pretraining=pretrain_path,
        )
        runs.append({"seed": seed, "pretraining": pretrain, "B1": b1, "B2": b2})
        progress_path = output / "training_progress.json"
        progress_path.write_text(
            json.dumps({"runs": runs}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    summary = {
        "experiment": "pointnet_iwr6843_adaptation_phase1",
        "dguha_dataset": str(dguha_path),
        "dguha_sha256": _sha256(dguha_path),
        "iwr_dataset": str(iwr_path),
        "iwr_sha256": _sha256(iwr_path),
        "seeds": list(seeds),
        "pretrain_epochs": pretrain_epochs,
        "prediction_epochs": prediction_epochs,
        "batch_size": batch_size,
        "device": str(device_value),
        "fairness_contract": {
            "same_dguha_sample_id_sha256": _sample_id_sha(dguha["sample_id"]),
            "same_batch_order_seed": True,
            "same_architecture": True,
            "same_prediction_loss": True,
            "same_epoch_selection": "maximum DGUHA validation AUROC",
            "only_intended_differences": [
                "B1 random encoder and end-to-end optimization",
                "B2 IWR6843 nonfall-pretrained encoder frozen and head-only optimization",
                "B1 DGUHA-train normalization; B2 preserves IWR pretraining normalization",
            ],
        },
        "runs": runs,
        "b0_tcn_touched": False,
        "realtime_chain_touched": False,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def train_iwr_representation(
    arrays: dict[str, np.ndarray],
    dataset_path: Path,
    checkpoint_path: Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, Any]:
    _set_seed(seed)
    train = arrays["split"] == "train"
    validation = arrays["split"] == "validation"
    if not train.any() or not validation.any():
        raise ValueError("IWR representation dataset needs train and validation subjects")
    mean, std = _normalization(arrays, train)
    model = PointTemporalPretrainingModel(
        class_count=len(arrays["action_names"]), input_size=len(FEATURE_NAMES),
        frame_hidden_size=64, temporal_hidden_size=64,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_indices = np.flatnonzero(train)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_macro = -1.0
    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = _epoch_order(train_indices, seed, epoch)
        rng = np.random.default_rng(seed + epoch * 100003)
        total_loss = 0.0
        for selected in _chunks(order, batch_size):
            points, point_mask, frame_mask, labels = _batch(
                arrays, selected, mean, std, device, augment=True, rng=rng,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(points, point_mask, frame_mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(selected)
        metrics = _activity_metrics(model, arrays, np.flatnonzero(validation), mean, std, device, batch_size)
        history.append({
            "epoch": epoch, "train_loss": total_loss / len(train_indices),
            "validation_loss": metrics["loss"], "validation_accuracy": metrics["accuracy"],
            "validation_macro_recall": metrics["macro_recall"],
        })
        if metrics["macro_recall"] > best_macro or (
            metrics["macro_recall"] == best_macro and metrics["loss"] < best_loss
        ):
            best_macro, best_loss, best_epoch = metrics["macro_recall"], metrics["loss"], epoch
            best_state = _cpu_state(model)
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    final = {
        name: _activity_metrics(model, arrays, np.flatnonzero(mask), mean, std, device, batch_size)
        for name, mask in (("train", train), ("validation", validation))
    }
    checkpoint = {
        "model_version": PRETRAIN_VERSION,
        "model_role": "iwr6843_nonfall_representation_pretraining",
        "sequence_version": SEQUENCE_VERSION,
        "feature_names": FEATURE_NAMES,
        "input_size": len(FEATURE_NAMES),
        "class_names": tuple(str(v) for v in arrays["action_names"]),
        "frame_hidden_size": 64, "temporal_hidden_size": 64,
        "state_dict": best_state,
        "normalization_mean": torch.from_numpy(mean),
        "normalization_std": torch.from_numpy(std),
        "normalization_origin": "IWR6843 Fall-102 nonfall train subjects",
        "snr_missing_policy": "normalized channel forced to zero where snr_available=false",
        "dataset_sha256": _sha256(dataset_path),
        "seed": seed, "best_epoch": best_epoch,
        "fall_recordings_included": False,
        "prediction_labels_used": False,
        "deployment_eligible": False,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    report = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
        "best_epoch": best_epoch, "metrics": final, "history": history,
    }
    checkpoint_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def train_prediction(
    arrays: dict[str, np.ndarray],
    dataset_path: Path,
    checkpoint_path: Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    initialization: str,
    pretraining: Path | None,
) -> dict[str, Any]:
    _set_seed(seed)
    train = arrays["split"] == "train"
    validation = arrays["split"] == "validation"
    if initialization == "random_dguha_only":
        mean, std = _normalization(arrays, train)
        encoder = PointTemporalEncoder(input_size=len(FEATURE_NAMES), frame_hidden_size=64, temporal_hidden_size=64)
        encoder_frozen = False
        pretraining_sha = None
        normalization_origin = "DGUHA prediction train split"
    elif initialization == "iwr6843_pretrained_frozen_encoder" and pretraining is not None:
        payload = _safe_load(pretraining)
        _validate_pretraining(payload)
        pretrain_model = PointTemporalPretrainingModel(
            class_count=len(payload["class_names"]), input_size=len(FEATURE_NAMES),
            frame_hidden_size=64, temporal_hidden_size=64,
        )
        pretrain_model.load_state_dict(payload["state_dict"], strict=True)
        encoder = pretrain_model.encoder
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        mean = np.asarray(payload["normalization_mean"], dtype=np.float32)
        std = np.asarray(payload["normalization_std"], dtype=np.float32)
        encoder_frozen = True
        pretraining_sha = _sha256(pretraining)
        normalization_origin = str(payload["normalization_origin"])
    else:
        raise ValueError("invalid prediction initialization")
    model = PointTemporalPredictionHead(encoder, horizon_count=1).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    sample_weights, group_counts = _temporal_group_weights(arrays, train)
    train_indices = np.flatnonzero(train)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_auroc = -1.0
    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        if encoder_frozen:
            model.encoder.eval()
        order = _epoch_order(train_indices, seed, epoch)
        rng = np.random.default_rng(seed + epoch * 100003)
        total_loss = 0.0
        for selected in _chunks(order, batch_size):
            points, point_mask, frame_mask, labels = _batch(
                arrays, selected, mean, std, device, augment=True, rng=rng,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(points, point_mask, frame_mask).squeeze(-1)
            weights = torch.from_numpy(sample_weights[selected]).to(device)
            loss = (loss_function(logits, labels.float()) * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(selected)
        scores, val_loss = _prediction_scores(
            model, arrays, np.flatnonzero(validation), mean, std, device, batch_size,
        )
        auroc = _auroc(arrays["labels"][validation], scores)
        history.append({
            "epoch": epoch, "train_weighted_loss": total_loss / len(train_indices),
            "validation_loss": val_loss, "validation_auroc": auroc,
        })
        if auroc > best_auroc or (auroc == best_auroc and val_loss < best_loss):
            best_auroc, best_loss, best_epoch = auroc, val_loss, epoch
            best_state = _cpu_state(model)
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    split_metrics: dict[str, Any] = {}
    for name, mask in (("train", train), ("validation", validation)):
        scores, loss = _prediction_scores(model, arrays, np.flatnonzero(mask), mean, std, device, batch_size)
        split_metrics[name] = _binary_metrics(arrays["labels"][mask], scores, loss)
    checkpoint = {
        "model_version": PREDICTION_VERSION,
        "model_role": "weak_supervision_prefall_prediction_research",
        "variant": "B1" if initialization == "random_dguha_only" else "B2",
        "initialization": initialization,
        "sequence_version": SEQUENCE_VERSION,
        "feature_names": FEATURE_NAMES,
        "input_size": len(FEATURE_NAMES),
        "time_steps": int(arrays["points"].shape[1]),
        "max_points": int(arrays["points"].shape[2]),
        "frame_hidden_size": 64, "temporal_hidden_size": 64,
        "state_dict": best_state,
        "normalization_mean": torch.from_numpy(mean),
        "normalization_std": torch.from_numpy(std),
        "normalization_origin": normalization_origin,
        "snr_missing_policy": "normalized channel forced to zero where snr_available=false",
        "encoder_frozen_during_prediction_training": encoder_frozen,
        "pretraining_checkpoint_sha256": pretraining_sha,
        "dataset_sha256": _sha256(dataset_path),
        "sample_id_sha256": _sample_id_sha(arrays["sample_id"]),
        "seed": seed, "best_epoch": best_epoch,
        "prediction_horizon_seconds": tuple(float(v) for v in arrays["prediction_horizon_seconds"]),
        "positive_anchor": "skeleton_derived_descent_onset",
        "confirmation_windows_for_evaluation": 3,
        "threshold_selected_later_by_common_event_evaluator": True,
        "deployment_eligible": False,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    if encoder_frozen:
        loaded = _safe_load(checkpoint_path)
        if not bool(loaded["encoder_frozen_during_prediction_training"]):
            raise AssertionError("B2 freeze contract was not persisted")
    report = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
        "variant": checkpoint["variant"], "best_epoch": best_epoch,
        "encoder_frozen": encoder_frozen, "group_counts": group_counts,
        "sample_id_sha256": checkpoint["sample_id_sha256"],
        "metrics": split_metrics, "history": history,
    }
    checkpoint_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def _load_dataset(path: Path, expected_mode: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "points", "point_mask", "frame_mask", "snr_available", "labels", "split",
            "sample_id", "feature_names", "sequence_version", "dataset_mode",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"dataset incomplete: {missing}")
        if str(data["dataset_mode"].item()) != expected_mode:
            raise ValueError("dataset mode mismatch")
        if str(data["sequence_version"].item()) != SEQUENCE_VERSION:
            raise ValueError("sequence version mismatch")
        if tuple(str(v) for v in data["feature_names"]) != FEATURE_NAMES:
            raise ValueError("feature order mismatch")
        result = {name: np.asarray(data[name]) for name in data.files}
    points = np.asarray(result["points"], dtype=np.float32)
    point_mask = np.asarray(result["point_mask"], dtype=np.bool_)
    frame_mask = np.asarray(result["frame_mask"], dtype=np.bool_)
    snr_available = np.asarray(result["snr_available"], dtype=np.bool_)
    if points.ndim != 4 or points.shape[-1] != len(FEATURE_NAMES):
        raise ValueError("point tensor must be [sample,time,point,5]")
    if point_mask.shape != points.shape[:3] or frame_mask.shape != points.shape[:2]:
        raise ValueError("mask shapes mismatch")
    if snr_available.shape != point_mask.shape or np.any(snr_available & ~point_mask):
        raise ValueError("SNR availability mask invalid")
    result.update(points=points, point_mask=point_mask, frame_mask=frame_mask, snr_available=snr_available)
    return result


def _assert_sample_contract(arrays: dict[str, np.ndarray]) -> None:
    required = {"label_source", "prediction_horizon_seconds", "source_files", "seconds_to_onset"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"DGUHA adaptation metadata incomplete: {missing}")
    labels = np.asarray(arrays["labels"])
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("DGUHA prediction labels must be binary")
    positive_leads = np.asarray(arrays["seconds_to_onset"], float)[labels == 1]
    if len(positive_leads) == 0 or positive_leads.min() < 0.5 - 1e-5 or positive_leads.max() > 1.0 + 1e-5:
        raise ValueError("positive windows violate the 0.5-1.0 second target")
    early = arrays["label_source"] == "dguha_same_fall_recording_outside_prediction_horizon"
    early_leads = np.asarray(arrays["seconds_to_onset"], float)[early]
    if len(early_leads) == 0 or np.nanmin(early_leads) < 1.2 - 1e-5:
        raise ValueError("same-recording early negatives violate the 1.2 second boundary")


def _normalization(arrays: dict[str, np.ndarray], selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = arrays["point_mask"][selected]
    values = arrays["points"][selected]
    mean = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    std = np.ones(len(FEATURE_NAMES), dtype=np.float32)
    for feature in range(len(FEATURE_NAMES)):
        feature_mask = mask.copy()
        if feature == 4:
            feature_mask &= arrays["snr_available"][selected]
        column = values[..., feature][feature_mask]
        if len(column):
            mean[feature] = float(column.mean())
            std[feature] = max(float(column.std()), 1e-4)
    return mean, std


def _batch(
    arrays: dict[str, np.ndarray], indices: np.ndarray, mean: np.ndarray, std: np.ndarray,
    device: torch.device, *, augment: bool = False, rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    points = arrays["points"][indices].copy()
    point_mask = arrays["point_mask"][indices].copy()
    frame_mask = arrays["frame_mask"][indices].copy()
    snr_available = arrays["snr_available"][indices]
    if augment:
        assert rng is not None
        _augment(points, point_mask, rng)
    points = (points - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))
    points[..., 4][~snr_available] = 0.0
    points[~point_mask] = 0.0
    return (
        torch.from_numpy(points.astype(np.float32, copy=False)).to(device),
        torch.from_numpy(point_mask).to(device), torch.from_numpy(frame_mask).to(device),
        torch.from_numpy(arrays["labels"][indices].astype(np.int64)).to(device),
    )


def _augment(points: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    for sample in range(len(points)):
        angle = float(rng.uniform(-0.08, 0.08))
        cosine, sine = np.cos(angle), np.sin(angle)
        x, y = points[sample, ..., 0].copy(), points[sample, ..., 1].copy()
        points[sample, ..., 0] = cosine * x - sine * y
        points[sample, ..., 1] = sine * x + cosine * y
        valid = mask[sample]
        points[sample, ..., :3][valid] += rng.normal(0.0, 0.01, size=(int(valid.sum()), 3))
        points[sample, ..., 3][valid] += rng.normal(0.0, 0.015, size=int(valid.sum()))
        drop = valid & (rng.random(valid.shape) < 0.05)
        mask[sample][drop] = False
        points[sample][drop] = 0.0


def _temporal_group_weights(arrays: dict[str, np.ndarray], train: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    early = arrays["label_source"] == "dguha_same_fall_recording_outside_prediction_horizon"
    groups = {
        "positive_prediction_horizon": arrays["labels"] == 1,
        "same_recording_early_negative": (arrays["labels"] == 0) & early,
        "normal_action_negative": (arrays["labels"] == 0) & ~early,
    }
    weights = np.ones(len(arrays["labels"]), dtype=np.float32)
    counts: dict[str, int] = {}
    train_count = int(train.sum())
    for name, group in groups.items():
        selected = train & group
        count = int(selected.sum())
        if count == 0:
            raise ValueError(f"empty temporal group: {name}")
        counts[name] = count
        weights[selected] = train_count / (len(groups) * count)
    return weights, counts


def _activity_metrics(model: nn.Module, arrays: dict[str, np.ndarray], indices: np.ndarray, mean: np.ndarray,
                      std: np.ndarray, device: torch.device, batch_size: int) -> dict[str, Any]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.inference_mode():
        for selected in _chunks(indices, batch_size):
            points, point_mask, frame_mask, target = _batch(arrays, selected, mean, std, device)
            logits = model(points, point_mask, frame_mask)
            loss_total += float(criterion(logits, target).item()) * len(selected)
            scores.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(target.cpu().numpy())
    prediction = np.concatenate(scores)
    target = np.concatenate(labels)
    recalls = [float(np.mean(prediction[target == label] == label)) for label in range(len(arrays["action_names"]))]
    return {
        "sample_count": len(indices), "loss": loss_total / max(len(indices), 1),
        "accuracy": float(np.mean(prediction == target)), "macro_recall": float(np.mean(recalls)),
        "per_class_recall": {str(arrays["action_names"][i]): recalls[i] for i in range(len(recalls))},
    }


def _prediction_scores(model: nn.Module, arrays: dict[str, np.ndarray], indices: np.ndarray, mean: np.ndarray,
                       std: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, float]:
    values: list[np.ndarray] = []
    loss_total = 0.0
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    with torch.inference_mode():
        for selected in _chunks(indices, batch_size):
            points, point_mask, frame_mask, labels = _batch(arrays, selected, mean, std, device)
            logits = model(points, point_mask, frame_mask).squeeze(-1)
            loss_total += float(criterion(logits, labels.float()).item()) * len(selected)
            values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values).astype(np.float64), loss_total / max(len(indices), 1)


def _binary_metrics(labels: np.ndarray, scores: np.ndarray, loss: float) -> dict[str, Any]:
    positive, negative = scores[labels == 1], scores[labels == 0]
    return {
        "sample_count": int(len(labels)), "positive_count": int((labels == 1).sum()),
        "loss": float(loss), "auroc": _auroc(labels, scores),
        "positive_score": _score_stats(positive), "negative_score": _score_stats(negative),
    }


def _score_stats(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(
        ("min", "median", "p90", "p95", "p99", "max"),
        (np.min(values), np.median(values), np.quantile(values, .9), np.quantile(values, .95), np.quantile(values, .99), np.max(values)),
    )}


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return 0.5
    order = np.argsort(np.concatenate((positive, negative)), kind="mergesort")
    values = np.concatenate((positive, negative))[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    positive_ranks = ranks[inverse[: len(positive)]]
    return float((positive_ranks.sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def _validate_pretraining(payload: dict[str, Any]) -> None:
    if payload.get("model_version") != PRETRAIN_VERSION:
        raise ValueError("pretraining version mismatch")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES or int(payload.get("input_size", 0)) != 5:
        raise ValueError("pretraining feature contract mismatch")
    if bool(payload.get("fall_recordings_included")) or bool(payload.get("prediction_labels_used")):
        raise ValueError("pretraining label contract violated")


def _safe_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    return payload


def _epoch_order(indices: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    result = indices.copy()
    np.random.default_rng(seed + epoch * 7919).shuffle(result)
    return result


def _chunks(values: np.ndarray, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_id_sha(values: np.ndarray) -> str:
    return hashlib.sha256("\n".join(str(v) for v in values).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B1/B2 first-stage PointNet IWR6843 adaptation training.")
    parser.add_argument("--dguha-dataset", required=True, type=Path)
    parser.add_argument("--iwr-dataset", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--prediction-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = run_first_stage_training(
        args.dguha_dataset, args.iwr_dataset, args.output_directory,
        pretrain_epochs=args.pretrain_epochs, prediction_epochs=args.prediction_epochs,
        batch_size=args.batch_size, device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
