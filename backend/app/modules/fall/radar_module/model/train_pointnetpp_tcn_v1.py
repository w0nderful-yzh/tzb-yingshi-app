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

from radar_module.model.pointnetpp_tcn_v1 import (
    ARCHITECTURE,
    INPUT_FEATURES,
    MAX_POINTS,
    MODEL_VERSION,
    TIME_STEPS,
    PointNetPlusPlusTcnPrefall,
    SpatialActivityPretrainer,
)


def train_upgrade(
    fall102_dataset: str | Path,
    dguha_dataset: str | Path,
    output_directory: str | Path,
    *,
    seed: int = 20260811,
    pretrain_epochs: int = 25,
    head_epochs: int = 3,
    joint_epochs: int = 12,
    batch_size: int = 64,
    device: str = "cuda",
) -> dict[str, Any]:
    _seed(seed)
    fall102_path = Path(fall102_dataset).resolve()
    dguha_path = Path(dguha_dataset).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fall102 = _load(fall102_path)
    dguha = _load(dguha_path)
    _validate(fall102, dguha)
    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    mean, std = _normalization(fall102)

    pretraining_model, pretraining_result = _pretrain_spatial(
        fall102,
        mean,
        std,
        seed=seed,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        device=torch_device,
    )
    pretraining_path = output / "fall102_pointnetpp_spatial_pretraining.pt"
    torch.save(
        {
            "model_version": MODEL_VERSION,
            "model_role": "fall102_spatial_representation_pretraining",
            "architecture": "pointnetpp_frame_encoder",
            "state_dict": pretraining_model.frame_encoder.state_dict(),
            "normalization_mean": torch.from_numpy(mean),
            "normalization_std": torch.from_numpy(std),
            "class_names": tuple(str(value) for value in fall102["action_names"]),
            "best_epoch": pretraining_result["best_epoch"],
            "best_validation_macro_recall": pretraining_result["best_validation_macro_recall"],
            "fall102_dataset_sha256": _sha256(fall102_path),
            "fall_recordings_used_as_prediction_positive": False,
            "prediction_labels_used": False,
            "subject_split": True,
            "seed": seed,
            "deployment_eligible": False,
        },
        pretraining_path,
    )

    model, training_result = _train_dguha(
        dguha,
        pretraining_model,
        mean,
        std,
        seed=seed,
        head_epochs=head_epochs,
        joint_epochs=joint_epochs,
        batch_size=batch_size,
        device=torch_device,
    )
    candidate_path = output / "pointnetpp_tcn_dguha_candidate.pt"
    checkpoint = {
        "model_version": MODEL_VERSION,
        "model_role": "pointnetpp_tcn_short_horizon_radar_evidence",
        "architecture": ARCHITECTURE,
        "input_contract": "raw_point_cloud_5d",
        "feature_names": INPUT_FEATURES,
        "input_size": 5,
        "time_steps": TIME_STEPS,
        "sample_rate_hz": 10.0,
        "max_points": MAX_POINTS,
        "state_dict": model.state_dict(),
        "normalization_mean": torch.from_numpy(mean),
        "normalization_std": torch.from_numpy(std),
        "normalization_origin": "Fall-102 subject-isolated train split",
        "snr_missing_policy": "normalized SNR channel forced to zero",
        "prediction_horizon_seconds": (0.5, 1.0),
        "positive_anchor": "skeleton_derived_descent_onset",
        "decision_threshold": 0.5,
        "decision_threshold_policy": "candidate only; lock on DGUHA validation event protocol",
        "confirmation_windows": 3,
        "fall102_dataset_sha256": _sha256(fall102_path),
        "dguha_dataset_sha256": _sha256(dguha_path),
        "spatial_pretraining_checkpoint": str(pretraining_path),
        "spatial_pretraining_checkpoint_sha256": _sha256(pretraining_path),
        "fall102_falls_used_as_prefall_positive": False,
        "peerj_used_for_training_or_threshold": False,
        "selected_for_radar_encoder_upgrade": False,
        "seed": seed,
        "best_epoch": training_result["best_epoch"],
        "shadow_only": True,
        "deployment_eligible": False,
        "radar_evidence_contract": ("score", "quality", "timestamp"),
    }
    torch.save(checkpoint, candidate_path)
    summary = {
        "experiment": "radar_encoder_upgrade_pointnetpp_tcn_v1",
        "device": str(torch_device),
        "seed": seed,
        "fall102_pretraining": pretraining_result,
        "dguha_training": training_result,
        "pretraining_checkpoint": str(pretraining_path),
        "pretraining_checkpoint_sha256": _sha256(pretraining_path),
        "candidate_checkpoint": str(candidate_path),
        "candidate_checkpoint_sha256": _sha256(candidate_path),
        "protected_components_modified": {
            "fusion_api": False,
            "camera": False,
            "uart_tlv": False,
            "b0_tcn_checkpoint": False,
            "evidence_protocol": False,
        },
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _pretrain_spatial(
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[SpatialActivityPretrainer, dict[str, Any]]:
    class_count = len(arrays["action_names"])
    model = SpatialActivityPretrainer(class_count=class_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_indices = np.flatnonzero(arrays["split"] == "train")
    validation_indices = np.flatnonzero(arrays["split"] == "validation")
    test_indices = np.flatnonzero(arrays["split"] == "test")
    best: tuple[float, int, dict[str, torch.Tensor]] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(train_indices)
        total = 0.0
        for selected in _chunks(order, batch_size):
            points, point_mask, frame_mask = _batch(arrays, selected, mean, std, device)
            labels = torch.from_numpy(arrays["labels"][selected].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(points, point_mask, frame_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item()) * len(selected)
        validation = _activity_metrics(
            model, arrays, validation_indices, mean, std, device, batch_size, class_count
        )
        history.append({"epoch": epoch, "train_loss": total / len(train_indices), **validation})
        value = float(validation["macro_recall"])
        if best is None or value > best[0]:
            best = (value, epoch, _cpu_state(model))
    assert best is not None
    model.load_state_dict(best[2], strict=True)
    test = (
        _activity_metrics(model, arrays, test_indices, mean, std, device, batch_size, class_count)
        if len(test_indices) else None
    )
    return model, {
        "train_subjects": sorted(set(str(value) for value in arrays["subject_id"][train_indices])),
        "validation_subjects": sorted(set(str(value) for value in arrays["subject_id"][validation_indices])),
        "test_subjects": sorted(set(str(value) for value in arrays["subject_id"][test_indices])),
        "best_epoch": best[1],
        "best_validation_macro_recall": best[0],
        "test": test,
        "history": history,
    }


def _train_dguha(
    arrays: dict[str, np.ndarray],
    pretraining: SpatialActivityPretrainer,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    seed: int,
    head_epochs: int,
    joint_epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[PointNetPlusPlusTcnPrefall, dict[str, Any]]:
    model = PointNetPlusPlusTcnPrefall().to(device)
    model.frame_encoder.load_state_dict(pretraining.frame_encoder.state_dict(), strict=True)
    train_indices = np.flatnonzero(arrays["split"] == "train")
    validation_indices = np.flatnonzero(arrays["split"] == "validation")
    weights, group_counts = _sample_weights(arrays, train_indices)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    history: list[dict[str, Any]] = []
    best: tuple[tuple[float, float], int, dict[str, torch.Tensor], dict[str, Any]] | None = None
    for parameter in model.frame_encoder.parameters():
        parameter.requires_grad_(False)
    temporal_parameters = [
        *model.input_projection.parameters(),
        *model.temporal_blocks.parameters(),
        *model.output.parameters(),
    ]
    optimizer = torch.optim.AdamW(temporal_parameters, lr=3e-4, weight_decay=1e-4)
    total_epochs = head_epochs + joint_epochs
    for epoch in range(1, total_epochs + 1):
        if epoch == head_epochs + 1:
            for parameter in model.frame_encoder.parameters():
                parameter.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.frame_encoder.parameters(), "lr": 3e-5},
                    {"params": temporal_parameters, "lr": 2e-4},
                ],
                weight_decay=1e-4,
            )
        model.train()
        if epoch <= head_epochs:
            model.frame_encoder.eval()
        order = np.random.default_rng(seed + 1000 + epoch).permutation(train_indices)
        total = 0.0
        for selected in _chunks(order, batch_size):
            points, point_mask, frame_mask = _batch(arrays, selected, mean, std, device)
            labels = torch.from_numpy(arrays["labels"][selected].astype(np.float32)).to(device)
            sample_weights = torch.from_numpy(weights[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(points, point_mask, frame_mask)
            loss = (criterion(logits, labels) * sample_weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 5.0
            )
            optimizer.step()
            total += float(loss.item()) * len(selected)
        snapshot = _binary_metrics(
            model, arrays, validation_indices, mean, std, device, batch_size
        )
        history.append({
            "epoch": epoch,
            "phase": "temporal_warmup" if epoch <= head_epochs else "joint_low_lr",
            "train_loss": total / len(train_indices),
            **snapshot,
        })
        key = (-float(snapshot["auroc"]), float(snapshot["early_negative_fpr_at_sensitivity_0_6"]))
        if best is None or key < best[0]:
            best = (key, epoch, _cpu_state(model), snapshot)
    assert best is not None
    model.load_state_dict(best[2], strict=True)
    return model, {
        "train_subjects": sorted(set(str(value) for value in arrays["subject_id"][train_indices])),
        "validation_subjects": sorted(set(str(value) for value in arrays["subject_id"][validation_indices])),
        "test_subjects_not_used_for_selection": sorted(
            set(str(value) for value in arrays["subject_id"][arrays["split"] == "test"])
        ),
        "training_group_counts": group_counts,
        "best_epoch": best[1],
        "best_validation": best[3],
        "history": history,
    }


def _batch(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = arrays["points"][indices].astype(np.float32, copy=True)
    point_mask = arrays["point_mask"][indices]
    frame_mask = arrays["frame_mask"][indices]
    normalized = (raw - mean[None, None, None, :]) / std[None, None, None, :]
    if "snr_available" in arrays:
        normalized[..., 4][~arrays["snr_available"][indices]] = 0.0
    normalized[~point_mask] = 0.0
    return (
        torch.from_numpy(normalized).to(device),
        torch.from_numpy(point_mask).to(device),
        torch.from_numpy(frame_mask).to(device),
    )


def _normalization(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    train = arrays["split"] == "train"
    points = arrays["points"][train]
    mask = arrays["point_mask"][train]
    selected = points[mask]
    mean = selected.mean(axis=0).astype(np.float32)
    std = selected.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return mean, std


def _sample_weights(
    arrays: dict[str, np.ndarray], train_indices: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    labels = arrays["labels"]
    early = arrays["label_source"] == "dguha_same_fall_recording_outside_prediction_horizon"
    train_mask = np.zeros(len(labels), dtype=np.bool_)
    train_mask[train_indices] = True
    groups = {
        "positive": train_mask & (labels == 1),
        "same_recording_early_negative": train_mask & (labels == 0) & early,
        "normal_action_negative": train_mask & (labels == 0) & ~early,
    }
    weights = np.ones(len(labels), dtype=np.float32)
    counts = {name: int(mask.sum()) for name, mask in groups.items()}
    for name, mask in groups.items():
        if counts[name] == 0:
            raise ValueError(f"empty DGUHA group: {name}")
        weights[mask] = len(train_indices) / (len(groups) * counts[name])
    return weights, counts


def _activity_metrics(
    model: SpatialActivityPretrainer,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
    class_count: int,
) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for selected in _chunks(indices, batch_size):
            inputs = _batch(arrays, selected, mean, std, device)
            predictions.append(model(*inputs).argmax(dim=1).cpu().numpy())
    predicted = np.concatenate(predictions)
    labels = arrays["labels"][indices]
    recalls = [float(np.mean(predicted[labels == value] == value)) for value in range(class_count)]
    return {"accuracy": float(np.mean(predicted == labels)), "macro_recall": float(np.mean(recalls))}


def _binary_metrics(
    model: PointNetPlusPlusTcnPrefall,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    scores = _scores(model, arrays, indices, mean, std, device, batch_size)
    labels = arrays["labels"][indices]
    threshold = _threshold_at_sensitivity(labels, scores, 0.6)
    early = arrays["label_source"][indices] == "dguha_same_fall_recording_outside_prediction_horizon"
    return {
        "auroc": _auroc(labels, scores),
        "threshold_at_positive_sensitivity_0_6": threshold,
        "positive_sensitivity": float(np.mean(scores[labels == 1] >= threshold)),
        "early_negative_fpr_at_sensitivity_0_6": float(np.mean(scores[early] >= threshold)),
        "normal_negative_fpr_at_sensitivity_0_6": float(np.mean(scores[(labels == 0) & ~early] >= threshold)),
    }


def _scores(
    model: PointNetPlusPlusTcnPrefall,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for selected in _chunks(indices, batch_size):
            inputs = _batch(arrays, selected, mean, std, device)
            output.append(torch.sigmoid(model(*inputs)).cpu().numpy())
    return np.concatenate(output)


def _threshold_at_sensitivity(labels: np.ndarray, scores: np.ndarray, sensitivity: float) -> float:
    positive = np.sort(scores[labels == 1])
    rank = max(0, min(len(positive) - 1, int(np.floor((1.0 - sensitivity) * len(positive)))))
    return float(positive[rank])


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return 0.5
    order = np.argsort(np.concatenate((positive, negative)), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    combined = np.concatenate((positive, negative))
    for value in np.unique(combined):
        selected = combined == value
        ranks[selected] = ranks[selected].mean()
    rank_sum = ranks[: len(positive)].sum()
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _validate(fall102: dict[str, np.ndarray], dguha: dict[str, np.ndarray]) -> None:
    for name, arrays in (("Fall-102", fall102), ("DGUHA", dguha)):
        if arrays["points"].shape[1:] != (20, 64, 5):
            raise ValueError(f"{name} point tensor contract changed")
        required = {"train", "validation"} if name == "Fall-102" else {"train", "validation", "test"}
        if not set(np.unique(arrays["split"])) >= required:
            raise ValueError(f"{name} is missing required subject-isolated splits: {required}")
    if bool(fall102.get("prediction_labels_used", np.asarray(True))):
        raise ValueError("Fall-102 prediction labels are forbidden")
    overlap = {
        split: set(str(value) for value in fall102["subject_id"][fall102["split"] == split])
        for split in ("train", "validation")
    }
    if overlap["train"] & overlap["validation"]:
        raise ValueError("Fall-102 subject leakage")


def _chunks(indices: np.ndarray, size: int):
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _seed(seed: int) -> None:
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
    parser = argparse.ArgumentParser(description="Train PointNet++ frame encoder + causal TCN")
    parser.add_argument("--fall102", required=True, type=Path)
    parser.add_argument("--dguha", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--pretrain-epochs", type=int, default=25)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--joint-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train_upgrade(
        args.fall102, args.dguha, args.output,
        seed=args.seed, pretrain_epochs=args.pretrain_epochs,
        head_epochs=args.head_epochs, joint_epochs=args.joint_epochs,
        batch_size=args.batch_size, device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
