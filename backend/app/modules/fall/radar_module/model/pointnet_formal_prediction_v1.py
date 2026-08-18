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
from radar_module.model.point_iwr6843_adaptation_v1 import (
    PRETRAIN_VERSION,
    _batch,
    _chunks,
    _epoch_order,
    _load_dataset,
    _safe_load,
    _set_seed,
)
from radar_module.model.point_temporal import (
    PointTemporalEncoder,
    PointTemporalPredictionHead,
    PointTemporalPretrainingModel,
)


FORMAL_MODEL_VERSION = "pointnet_gru_prefall_formal_v1"
SIT_STAND_TOKEN = "/3_Sit_down_and_stand_up/"
IWR_HARD_ACTIONS = frozenset({"bow", "squat"})


def run_formal_training(
    dguha_dataset: str | Path,
    iwr_dataset: str | Path,
    pretraining_directory: str | Path,
    output_directory: str | Path,
    *,
    seeds: tuple[int, ...] = (20260808, 20260809, 20260810),
    head_warmup_epochs: int = 3,
    dguha_joint_epochs: int = 22,
    hard_negative_epochs: int = 6,
    batch_size: int = 64,
    head_learning_rate: float = 5e-4,
    encoder_learning_rate_ratio: float = 0.1,
    hard_negative_loss_weight: float = 0.1,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    dguha_path = Path(dguha_dataset).resolve()
    iwr_path = Path(iwr_dataset).resolve()
    pretrain_dir = Path(pretraining_directory).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dguha = _load_dataset(dguha_path, DGUHA_MODE)
    iwr = _load_dataset(iwr_path, IWR_MODE)
    _validate_formal_data(dguha, iwr)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        pretraining = pretrain_dir / f"IWR_encoder_seed{seed}.pt"
        if not pretraining.is_file():
            raise FileNotFoundError(pretraining)
        stage2_path = output / f"P2_dguha_joint_seed{seed}.pt"
        stage2 = _train_stage2(
            dguha, iwr, dguha_path, iwr_path, pretraining, stage2_path,
            seed=seed, head_warmup_epochs=head_warmup_epochs,
            joint_epochs=dguha_joint_epochs, batch_size=batch_size,
            head_learning_rate=head_learning_rate,
            encoder_learning_rate=head_learning_rate * encoder_learning_rate_ratio,
            device=torch.device(device),
        )
        stage3_path = output / f"P3_hard_negative_seed{seed}.pt"
        stage3 = _train_stage3(
            dguha, iwr, dguha_path, iwr_path, stage2_path, stage3_path,
            seed=seed, epochs=hard_negative_epochs, batch_size=batch_size,
            head_learning_rate=head_learning_rate * 0.2,
            encoder_learning_rate=head_learning_rate * encoder_learning_rate_ratio * 0.5,
            hard_negative_loss_weight=hard_negative_loss_weight,
            device=torch.device(device),
        )
        results.append({"seed": seed, "stage2": stage2, "stage3": stage3})
        (output / "training_progress.json").write_text(
            json.dumps({"runs": results}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    payload = {
        "experiment": "pointnet_gru_formal_prediction_v1",
        "task": "predict entry into DGUHA descent onset in 0.5-1.0 seconds",
        "input_contract": {"features": list(FEATURE_NAMES), "frames": 20, "rate_hz": 10},
        "data_roles": {
            "DGUHA": "only prediction-positive timing supervision and DGUHA negatives",
            "IWR6843_Fall102": "nonfall representation pretraining plus bow/squat hard negatives; no fall recording used",
        },
        "settings": {
            "seeds": list(seeds), "head_warmup_epochs": head_warmup_epochs,
            "dguha_joint_epochs": dguha_joint_epochs,
            "hard_negative_epochs": hard_negative_epochs,
            "batch_size": batch_size,
            "head_learning_rate": head_learning_rate,
            "encoder_learning_rate_ratio": encoder_learning_rate_ratio,
            "hard_negative_loss_weight": hard_negative_loss_weight,
        },
        "dguha_sha256": _sha256(dguha_path), "iwr_sha256": _sha256(iwr_path),
        "runs": results,
        "tcn_b0_modified": False, "realtime_chain_modified": False,
    }
    (output / "training_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return payload


def _train_stage2(
    dguha: dict[str, np.ndarray], iwr: dict[str, np.ndarray],
    dguha_path: Path, iwr_path: Path, pretraining_path: Path, destination: Path,
    *, seed: int, head_warmup_epochs: int, joint_epochs: int, batch_size: int,
    head_learning_rate: float, encoder_learning_rate: float, device: torch.device,
) -> dict[str, Any]:
    _set_seed(seed)
    pretraining = _safe_load(pretraining_path)
    _validate_pretraining(pretraining)
    model = _model_from_pretraining(pretraining).to(device)
    mean = np.asarray(pretraining["normalization_mean"], dtype=np.float32)
    std = np.asarray(pretraining["normalization_std"], dtype=np.float32)
    train = dguha["split"] == "train"
    validation = dguha["split"] == "validation"
    weights, group_counts = _dguha_weights(dguha, train, hard_focus=False)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    history: list[dict[str, Any]] = []
    best: tuple[tuple[float, ...], int, dict[str, torch.Tensor], dict[str, Any]] | None = None

    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.output.parameters(), lr=head_learning_rate, weight_decay=1e-4)
    total_epochs = head_warmup_epochs + joint_epochs
    for epoch in range(1, total_epochs + 1):
        if epoch == head_warmup_epochs + 1:
            for parameter in model.encoder.parameters():
                parameter.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.encoder.parameters(), "lr": encoder_learning_rate},
                    {"params": model.output.parameters(), "lr": head_learning_rate},
                ],
                weight_decay=1e-4,
            )
        train_loss = _train_dguha_epoch(
            model, dguha, np.flatnonzero(train), weights, mean, std, criterion,
            optimizer, device, batch_size, seed, epoch,
            encoder_frozen=epoch <= head_warmup_epochs,
        )
        validation_result = _validation_snapshot(model, dguha, iwr, mean, std, device, batch_size)
        selection_key = _selection_key(validation_result)
        history.append({"epoch": epoch, "phase": "head_warmup" if epoch <= head_warmup_epochs else "joint_finetune",
                        "train_loss": train_loss, **validation_result})
        if best is None or selection_key < best[0]:
            best = (selection_key, epoch, _cpu_state(model), validation_result)
    assert best is not None
    _, best_epoch, best_state, best_metrics = best
    return _save_prediction_checkpoint(
        destination, model, best_state, mean, std, dguha_path, iwr_path,
        seed=seed, best_epoch=best_epoch, variant="P2_DGUHA_JOINT",
        pretraining_path=pretraining_path, training_history=history,
        best_metrics=best_metrics, group_counts=group_counts,
        hard_negative_actions=(), encoder_training="head warm-up then joint low-LR fine-tuning",
    )


def _train_stage3(
    dguha: dict[str, np.ndarray], iwr: dict[str, np.ndarray],
    dguha_path: Path, iwr_path: Path, stage2_path: Path, destination: Path,
    *, seed: int, epochs: int, batch_size: int, head_learning_rate: float,
    encoder_learning_rate: float, hard_negative_loss_weight: float,
    device: torch.device,
) -> dict[str, Any]:
    _set_seed(seed + 101)
    stage2 = _safe_load(stage2_path)
    model = _model_from_prediction(stage2).to(device)
    mean = np.asarray(stage2["normalization_mean"], dtype=np.float32)
    std = np.asarray(stage2["normalization_std"], dtype=np.float32)
    train = dguha["split"] == "train"
    weights, group_counts = _dguha_weights(dguha, train, hard_focus=False)
    sit_stand_train = train & np.asarray([
        SIT_STAND_TOKEN in str(path) for path in dguha["source_files"]
    ]) & (dguha["labels"] == 0)
    weights[sit_stand_train] *= 2.0
    group_counts["dguha_sit_stand_2x_weight"] = int(sit_stand_train.sum())
    iwr_hard_train = (
        (iwr["split"] == "train")
        & np.isin(iwr["action"], tuple(IWR_HARD_ACTIONS))
    )
    hard_indices = np.flatnonzero(iwr_hard_train)
    if not len(hard_indices):
        raise ValueError("IWR bow/squat hard-negative train split is empty")
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.output.parameters(), lr=head_learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    history: list[dict[str, Any]] = []
    best: tuple[tuple[float, ...], int, dict[str, torch.Tensor], dict[str, Any]] | None = None
    dguha_indices = np.flatnonzero(train)
    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.eval()
        order = _epoch_order(dguha_indices, seed + 101, epoch)
        hard_rng = np.random.default_rng(seed + epoch * 65537)
        augment_rng = np.random.default_rng(seed + epoch * 100003)
        total_loss = 0.0
        for selected in _chunks(order, batch_size):
            points, point_mask, frame_mask, labels = _batch(
                dguha, selected, mean, std, device, augment=True, rng=augment_rng,
            )
            hard_selected = hard_rng.choice(hard_indices, size=min(16, len(hard_indices)), replace=False)
            hard_points, hard_point_mask, hard_frame_mask, _ = _batch(
                iwr, hard_selected, mean, std, device, augment=True, rng=augment_rng,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(points, point_mask, frame_mask).squeeze(-1)
            sample_weights = torch.from_numpy(weights[selected]).to(device)
            dguha_loss = (criterion(logits, labels.float()) * sample_weights).mean()
            hard_logits = model(hard_points, hard_point_mask, hard_frame_mask).squeeze(-1)
            hard_loss = criterion(hard_logits, torch.zeros_like(hard_logits)).mean()
            loss = dguha_loss + hard_negative_loss_weight * hard_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.output.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(selected)
        validation_result = _validation_snapshot(model, dguha, iwr, mean, std, device, batch_size)
        selection_key = _selection_key(validation_result)
        history.append({"epoch": epoch, "phase": "hard_negative_finetune",
                        "train_loss": total_loss / len(dguha_indices), **validation_result})
        if best is None or selection_key < best[0]:
            best = (selection_key, epoch, _cpu_state(model), validation_result)
    assert best is not None
    _, best_epoch, best_state, best_metrics = best
    return _save_prediction_checkpoint(
        destination, model, best_state, mean, std, dguha_path, iwr_path,
        seed=seed, best_epoch=best_epoch, variant="P3_HARD_NEGATIVE",
        pretraining_path=Path(stage2["pretraining_checkpoint_path"]),
        training_history=history, best_metrics=best_metrics,
        group_counts={**group_counts, "iwr_bow_squat": int(len(hard_indices))},
        hard_negative_actions=("DGUHA Sit_down_and_stand_up", "IWR6843 bow", "IWR6843 squat"),
        encoder_training="P2 encoder frozen; head-only conservative hard-negative refinement",
        parent_checkpoint_path=stage2_path,
    )


def _train_dguha_epoch(
    model: PointTemporalPredictionHead, arrays: dict[str, np.ndarray], indices: np.ndarray,
    weights: np.ndarray, mean: np.ndarray, std: np.ndarray, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, device: torch.device, batch_size: int,
    seed: int, epoch: int, *, encoder_frozen: bool,
) -> float:
    model.train()
    if encoder_frozen:
        model.encoder.eval()
    order = _epoch_order(indices, seed, epoch)
    rng = np.random.default_rng(seed + epoch * 100003)
    total = 0.0
    for selected in _chunks(order, batch_size):
        points, point_mask, frame_mask, labels = _batch(
            arrays, selected, mean, std, device, augment=True, rng=rng,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(points, point_mask, frame_mask).squeeze(-1)
        sample_weights = torch.from_numpy(weights[selected]).to(device)
        loss = (criterion(logits, labels.float()) * sample_weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()
        total += float(loss.item()) * len(selected)
    return total / len(indices)


def _validation_snapshot(
    model: PointTemporalPredictionHead, dguha: dict[str, np.ndarray], iwr: dict[str, np.ndarray],
    mean: np.ndarray, std: np.ndarray, device: torch.device, batch_size: int,
) -> dict[str, Any]:
    dguha_validation = dguha["split"] == "validation"
    selected = np.flatnonzero(dguha_validation)
    scores = _scores(model, dguha, selected, mean, std, device, batch_size)
    labels = dguha["labels"][selected]
    sources = dguha["source_files"][selected]
    label_source = dguha["label_source"][selected]
    threshold = _threshold_at_positive_sensitivity(labels, scores, minimum_sensitivity=0.5)
    early = label_source == "dguha_same_fall_recording_outside_prediction_horizon"
    sit_stand = np.asarray([SIT_STAND_TOKEN in str(path) for path in sources]) & (labels == 0)
    other_normal = (labels == 0) & ~early & ~sit_stand
    iwr_validation = iwr["split"] == "validation"
    iwr_indices = np.flatnonzero(iwr_validation)
    iwr_scores = _scores(model, iwr, iwr_indices, mean, std, device, batch_size)
    actions = iwr["action"][iwr_indices]
    iwr_hard = np.isin(actions, tuple(IWR_HARD_ACTIONS))
    return {
        "validation_auroc": _auroc(labels, scores),
        "selection_threshold_proxy": threshold,
        "positive_window_sensitivity": _rate(scores[labels == 1] >= threshold),
        "same_recording_early_fpr": _rate(scores[early] >= threshold),
        "dguha_sit_stand_fpr": _rate(scores[sit_stand] >= threshold),
        "dguha_other_normal_fpr": _rate(scores[other_normal] >= threshold),
        "iwr_bow_squat_fpr": _rate(iwr_scores[iwr_hard] >= threshold),
        "iwr_walk_fpr": _rate(iwr_scores[~iwr_hard] >= threshold),
        "positive_score": _describe(scores[labels == 1]),
        "early_score": _describe(scores[early]),
        "sit_stand_score": _describe(scores[sit_stand]),
        "iwr_bow_squat_score": _describe(iwr_scores[iwr_hard]),
        "iwr_walk_score": _describe(iwr_scores[~iwr_hard]),
    }


def _selection_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    # Sensitivity is constrained by threshold construction.  A low false alarm
    # proxy is not useful if prediction ranking has collapsed, so candidates
    # below AUROC 0.70 receive a continuous penalty before false-alarm ranking.
    false_rates = (
        metrics["same_recording_early_fpr"], metrics["dguha_sit_stand_fpr"],
        metrics["iwr_bow_squat_fpr"], metrics["iwr_walk_fpr"],
    )
    auroc = float(metrics["validation_auroc"])
    if auroc >= 0.70:
        return (0.0, max(false_rates), float(np.mean(false_rates)), -auroc)
    # If the whole run misses the ranking floor, retain the least-damaged
    # predictor rather than selecting a low-score constant classifier.
    return (1.0, -auroc, max(false_rates), float(np.mean(false_rates)))


def _dguha_weights(
    arrays: dict[str, np.ndarray], train: np.ndarray, *, hard_focus: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    labels = arrays["labels"]
    early = arrays["label_source"] == "dguha_same_fall_recording_outside_prediction_horizon"
    sit_stand = np.asarray([SIT_STAND_TOKEN in str(path) for path in arrays["source_files"]]) & (labels == 0)
    if hard_focus:
        groups = {
            "positive_prediction_horizon": labels == 1,
            "same_recording_early_negative": (labels == 0) & early,
            "dguha_sit_stand_negative": sit_stand,
            "other_normal_negative": (labels == 0) & ~early & ~sit_stand,
        }
    else:
        groups = {
            "positive_prediction_horizon": labels == 1,
            "same_recording_early_negative": (labels == 0) & early,
            "normal_action_negative": (labels == 0) & ~early,
        }
    weights = np.ones(len(labels), dtype=np.float32)
    counts: dict[str, int] = {}
    for name, group in groups.items():
        selected = train & group
        count = int(selected.sum())
        if not count:
            raise ValueError(f"empty training group: {name}")
        counts[name] = count
        weights[selected] = int(train.sum()) / (len(groups) * count)
    return weights, counts


def _save_prediction_checkpoint(
    destination: Path, model: PointTemporalPredictionHead, state: dict[str, torch.Tensor],
    mean: np.ndarray, std: np.ndarray, dguha_path: Path, iwr_path: Path,
    *, seed: int, best_epoch: int, variant: str, pretraining_path: Path,
    training_history: list[dict[str, Any]], best_metrics: dict[str, Any],
    group_counts: dict[str, int], hard_negative_actions: tuple[str, ...],
    encoder_training: str, parent_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    model.load_state_dict(state, strict=True)
    checkpoint: dict[str, Any] = {
        "model_version": FORMAL_MODEL_VERSION,
        "model_role": "pointnet_gru_short_horizon_radar_evidence",
        "variant": variant, "sequence_version": SEQUENCE_VERSION,
        "feature_names": FEATURE_NAMES, "input_size": 5,
        "time_steps": 20, "sample_rate_hz": 10.0, "max_points": 64,
        "frame_hidden_size": 64, "temporal_hidden_size": 64,
        "state_dict": state,
        "normalization_mean": torch.from_numpy(mean),
        "normalization_std": torch.from_numpy(std),
        "normalization_origin": "IWR6843 nonfall representation pretraining",
        "snr_missing_policy": "normalized channel forced to zero where snr_available=false",
        "pretraining_checkpoint_path": str(pretraining_path.resolve()),
        "pretraining_checkpoint_sha256": _sha256(pretraining_path),
        "parent_checkpoint_sha256": _sha256(parent_checkpoint_path) if parent_checkpoint_path else None,
        "dguha_dataset_sha256": _sha256(dguha_path),
        "iwr_dataset_sha256": _sha256(iwr_path),
        "prediction_horizon_seconds": (0.5, 1.0),
        "positive_anchor": "skeleton_derived_descent_onset",
        "iwr_fall_recordings_used_as_prediction_positive": False,
        "encoder_training": encoder_training,
        "hard_negative_actions": hard_negative_actions,
        "decision_threshold": float(best_metrics["selection_threshold_proxy"]),
        "decision_threshold_policy": "validation proxy; replace with locked continuous-event threshold before live use",
        "confirmation_windows": 3,
        "seed": seed, "best_epoch": best_epoch,
        "shadow_only": True, "deployment_eligible": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    report = {
        "checkpoint": str(destination), "checkpoint_sha256": _sha256(destination),
        "variant": variant, "seed": seed, "best_epoch": best_epoch,
        "best_validation": best_metrics, "training_group_counts": group_counts,
        "hard_negative_actions": list(hard_negative_actions),
        "training_history": training_history,
    }
    destination.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def _model_from_pretraining(checkpoint: dict[str, Any]) -> PointTemporalPredictionHead:
    pretrained = PointTemporalPretrainingModel(
        class_count=len(checkpoint["class_names"]), input_size=5,
        frame_hidden_size=64, temporal_hidden_size=64,
    )
    pretrained.load_state_dict(checkpoint["state_dict"], strict=True)
    return PointTemporalPredictionHead(pretrained.encoder, horizon_count=1)


def _model_from_prediction(checkpoint: dict[str, Any]) -> PointTemporalPredictionHead:
    encoder = PointTemporalEncoder(input_size=5, frame_hidden_size=64, temporal_hidden_size=64)
    model = PointTemporalPredictionHead(encoder, horizon_count=1)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def _scores(
    model: PointTemporalPredictionHead, arrays: dict[str, np.ndarray], indices: np.ndarray,
    mean: np.ndarray, std: np.ndarray, device: torch.device, batch_size: int,
) -> np.ndarray:
    result: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for selected in _chunks(indices, batch_size):
            points, point_mask, frame_mask, _ = _batch(arrays, selected, mean, std, device)
            result.append(torch.sigmoid(model(points, point_mask, frame_mask).squeeze(-1)).cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def _threshold_at_positive_sensitivity(labels: np.ndarray, scores: np.ndarray, *, minimum_sensitivity: float) -> float:
    positive = np.sort(scores[labels == 1])
    if not len(positive):
        raise ValueError("validation positives are empty")
    required = max(1, int(np.ceil(len(positive) * minimum_sensitivity)))
    return float(positive[-required])


def _validate_formal_data(dguha: dict[str, np.ndarray], iwr: dict[str, np.ndarray]) -> None:
    for required in ("label_source", "source_files", "seconds_to_onset", "prediction_horizon_seconds"):
        if required not in dguha:
            raise ValueError(f"DGUHA metadata missing: {required}")
    for required in ("action", "action_names"):
        if required not in iwr:
            raise ValueError(f"IWR metadata missing: {required}")
    if np.any(iwr["labels"] < 0) or set(str(v) for v in iwr["action_names"]) != {"bow", "squat", "walk"}:
        raise ValueError("IWR representation labels are incompatible")
    if not np.any([SIT_STAND_TOKEN in str(path) for path in dguha["source_files"]]):
        raise ValueError("DGUHA sit/stand recordings are unavailable")


def _validate_pretraining(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("model_version") != PRETRAIN_VERSION:
        raise ValueError("IWR pretraining checkpoint version mismatch")
    if bool(checkpoint.get("fall_recordings_included")) or bool(checkpoint.get("prediction_labels_used")):
        raise ValueError("IWR pretraining used forbidden labels")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("IWR pretraining feature contract mismatch")


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return 0.5
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _rate(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _describe(values: np.ndarray) -> dict[str, float | int | None]:
    if not len(values):
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {"count": int(len(values)), "min": float(values.min()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train formal PointNet-GRU short-horizon radar branch.")
    parser.add_argument("--dguha-dataset", required=True, type=Path)
    parser.add_argument("--iwr-dataset", required=True, type=Path)
    parser.add_argument("--pretraining-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--head-warmup-epochs", type=int, default=3)
    parser.add_argument("--joint-epochs", type=int, default=22)
    parser.add_argument("--hard-negative-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = run_formal_training(
        args.dguha_dataset, args.iwr_dataset, args.pretraining_directory, args.output_directory,
        head_warmup_epochs=args.head_warmup_epochs, dguha_joint_epochs=args.joint_epochs,
        hard_negative_epochs=args.hard_negative_epochs, batch_size=args.batch_size, device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
