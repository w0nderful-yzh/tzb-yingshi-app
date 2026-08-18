from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset

from radar_module.contracts import RadarFrame, Room
from radar_module.dataset.tcn_multisource_v1 import (
    DATASET_MODE,
    EXTERNAL_ORIGINS,
)
from radar_module.dataset.v2_export import _load_replay_frames
from radar_module.model.research_training_v2 import (
    RESEARCH_MODEL_MODE,
    _auroc,
    _set_seed,
)
from radar_module.model.prefall_experiment_v3 import (
    _select_specificity_priority_threshold,
)
from radar_module.model.temporal_models_v3 import (
    EXPERIMENT_MODEL_VERSION,
    TemporalBinaryModel,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


MODEL_SELECTION_POLICY = "dguha_sensitivity_gate_plus_iwr6843_normal_penalty_v1"
FIXED_DECISION_THRESHOLD = 0.35


class PositiveProtectedBatchSampler(Sampler[list[int]]):
    """Visit every negative once and inject one positive into every batch."""

    def __init__(
        self,
        labels: np.ndarray,
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        values = np.asarray(labels, dtype=np.int8)
        if values.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if batch_size < 2:
            raise ValueError("batch_size must be at least two")
        self.positive_indices = np.flatnonzero(values == 1)
        self.negative_indices = np.flatnonzero(values == 0)
        if not len(self.positive_indices) or not len(self.negative_indices):
            raise ValueError("protected batches require both classes")
        self.batch_size = batch_size
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return int(math.ceil(len(self.negative_indices) / (self.batch_size - 1)))

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        negatives = rng.permutation(self.negative_indices)
        positives = rng.permutation(self.positive_indices)
        positive_cursor = 0
        negative_batch_size = self.batch_size - 1
        for start in range(0, len(negatives), negative_batch_size):
            if positive_cursor >= len(positives):
                positives = rng.permutation(self.positive_indices)
                positive_cursor = 0
            batch = np.concatenate(
                (
                    np.asarray([positives[positive_cursor]], dtype=np.int64),
                    negatives[start : start + negative_batch_size],
                )
            )
            positive_cursor += 1
            rng.shuffle(batch)
            yield batch.astype(int).tolist()

    def audit(self) -> dict[str, int | float | str]:
        batches = len(self)
        positive_draws = batches
        oversampling_factor = positive_draws / len(self.positive_indices)
        return {
            "policy": "one_positive_per_batch_all_negatives_once",
            "batch_size": self.batch_size,
            "batch_count_per_epoch": batches,
            "unique_positive_count": int(len(self.positive_indices)),
            "unique_negative_count": int(len(self.negative_indices)),
            "positive_draws_per_epoch": positive_draws,
            "positive_oversampling_factor": float(oversampling_factor),
            "minimum_positive_per_batch": 1,
            "maximum_positive_per_batch": 1,
        }


def train_tcn_multisource_experiment(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    replay_paths: Sequence[str | Path] = (),
    epochs: int = 20,
    hidden_size: int = 24,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    positive_weight_cap: float = 32.0,
    minimum_dguha_validation_sensitivity: float = 0.60,
    fixed_threshold: float = FIXED_DECISION_THRESHOLD,
    seed: int = 20260809,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    if epochs <= 0 or hidden_size <= 0 or batch_size < 2 or learning_rate <= 0:
        raise ValueError("invalid training parameters")
    if not 0.0 < minimum_dguha_validation_sensitivity <= 1.0:
        raise ValueError("minimum validation sensitivity must be in (0, 1]")
    if not 0.0 < fixed_threshold < 1.0:
        raise ValueError("fixed_threshold must be in (0, 1)")
    source = Path(dataset_path).resolve()
    destination = Path(checkpoint_path).resolve()
    arrays = _load_multisource_dataset(source)
    features = arrays["features"]
    labels = arrays["labels"]
    splits = arrays["split"]
    origins = arrays["dataset_origin"]
    train_mask = splits == "train"
    dguha_validation_mask = (splits == "validation") & (origins == "dguha")
    external_validation_mask = splits == "external_validation"
    if not train_mask.any() or not dguha_validation_mask.any():
        raise ValueError("dataset lacks train or DGUHA validation rows")
    if not external_validation_mask.any():
        raise ValueError("dataset lacks external validation rows")
    for origin in EXTERNAL_ORIGINS:
        if not np.any(external_validation_mask & (origins == origin)):
            raise ValueError(f"external validation lacks {origin}")

    mean = np.asarray(arrays["normalization_mean"], dtype=np.float32)
    std = np.asarray(arrays["normalization_std"], dtype=np.float32)
    if mean.shape != (len(FEATURE_NAMES_V2),) or std.shape != mean.shape:
        raise ValueError("normalization contract is incompatible")
    normalized = ((features - mean[None, None]) / std[None, None]).astype(np.float32)

    train_labels = labels[train_mask]
    sampler = PositiveProtectedBatchSampler(
        train_labels,
        batch_size=batch_size,
        seed=seed,
    )
    sampling_audit = sampler.audit()
    raw_positive_weight = min(
        positive_weight_cap,
        float(np.sum(train_labels == 0) / np.sum(train_labels == 1)),
    )
    all_sample_weights = np.asarray(
        arrays.get("sample_weight", np.ones(len(labels))), dtype=np.float32
    )
    training_weights = all_sample_weights[train_mask]
    dguha_training = train_mask & (origins == "dguha")
    dguha_positive_weight_mass = float(
        all_sample_weights[dguha_training & (labels == 1)].sum()
    )
    dguha_negative_weight_mass = float(
        all_sample_weights[dguha_training & (labels == 0)].sum()
    )
    target_positive_to_negative_mass_ratio = (
        dguha_positive_weight_mass * raw_positive_weight
        / dguha_negative_weight_mass
    )
    positive_draws = int(sampling_audit["positive_draws_per_epoch"])
    mean_positive_sample_weight = float(training_weights[train_labels == 1].mean())
    positive_weight = min(
        positive_weight_cap,
        max(
            1.0,
            target_positive_to_negative_mass_ratio
            * float(training_weights[train_labels == 0].sum())
            / (positive_draws * mean_positive_sample_weight),
        ),
    )
    sampling_audit["raw_capped_positive_weight"] = raw_positive_weight
    sampling_audit["target_positive_to_negative_weight_mass_ratio"] = (
        target_positive_to_negative_mass_ratio
    )
    sampling_audit["positive_weight_policy"] = (
        "preserve clean-DGUHA weighted positive-to-negative loss mass after "
        "positive-protected oversampling and multisource negative addition"
    )
    sampling_audit["oversampling_adjusted_positive_weight"] = positive_weight
    training_dataset = TensorDataset(
        torch.from_numpy(normalized[train_mask]),
        torch.from_numpy(train_labels.astype(np.float32)),
        torch.from_numpy(training_weights),
    )
    loader = DataLoader(training_dataset, batch_sampler=sampler)

    _set_seed(seed)
    torch_device = torch.device(device)
    model = TemporalBinaryModel(
        architecture="causal_tcn",
        input_size=len(FEATURE_NAMES_V2),
        hidden_size=hidden_size,
    ).to(torch_device)
    training_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], device=torch_device),
        reduction="none",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    replay_inputs = {
        str(Path(path).resolve()): _extract_replay_features(Path(path).resolve())
        for path in replay_paths
    }
    epoch_log_path = destination.with_suffix(".epochs.jsonl")
    epoch_log_path.parent.mkdir(parents=True, exist_ok=True)
    if epoch_log_path.exists():
        raise FileExistsError(f"epoch log already exists: {epoch_log_path}")

    history: list[dict[str, object]] = []
    eligible_states: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    fallback_states: list[tuple[float, float, int, dict[str, torch.Tensor]]] = []
    with epoch_log_path.open("x", encoding="utf-8") as epoch_handle:
        for epoch in range(1, epochs + 1):
            model.train()
            weighted_loss_sum = 0.0
            weight_sum = 0.0
            observed_positive_counts: list[int] = []
            for batch_features, batch_labels, batch_weights in loader:
                positive_count = int(torch.sum(batch_labels == 1.0).item())
                observed_positive_counts.append(positive_count)
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
                weighted_loss_sum += float((losses * device_weights).sum().item())
                weight_sum += float(device_weights.sum().item())

            if min(observed_positive_counts) < 1:
                raise RuntimeError("positive-protected sampler emitted a negative-only batch")
            epoch_audit = _evaluate_epoch(
                model,
                normalized=normalized,
                labels=labels,
                splits=splits,
                origins=origins,
                event_ids=arrays["event_id"],
                window_end_seconds=arrays["window_end_seconds"],
                dguha_validation_mask=dguha_validation_mask,
                external_validation_mask=external_validation_mask,
                replay_inputs=replay_inputs,
                mean=mean,
                std=std,
                threshold=fixed_threshold,
                minimum_dguha_validation_sensitivity=(
                    minimum_dguha_validation_sensitivity
                ),
                batch_size=batch_size,
                device=torch_device,
            )
            dguha = epoch_audit["dguha_validation"]
            external = epoch_audit["external_normal"]
            iwr = external["by_origin"]["iwr6843_fall_102"]
            non_iwr = external["excluding_iwr6843"]
            selection_cost = (
                1.0
                - float(dguha["f1"])
                + 2.0 * float(iwr["above_threshold_window_fraction"])
                + 0.25 * float(iwr["score_distribution"]["p99"])
                + 0.50 * float(non_iwr["above_threshold_window_fraction"])
            )
            eligible = (
                float(dguha["sensitivity"])
                >= minimum_dguha_validation_sensitivity
            )
            epoch_record: dict[str, object] = {
                "epoch": epoch,
                "training_weighted_loss": (
                    weighted_loss_sum / weight_sum if weight_sum else math.nan
                ),
                "observed_batch_positive_min": min(observed_positive_counts),
                "observed_batch_positive_max": max(observed_positive_counts),
                **epoch_audit,
                "model_selection": {
                    "policy": MODEL_SELECTION_POLICY,
                    "eligible": eligible,
                    "minimum_dguha_validation_sensitivity": (
                        minimum_dguha_validation_sensitivity
                    ),
                    "selection_cost": selection_cost,
                    "formula": (
                        "(1-dguha_f1) + 2*iwr_normal_fpr + "
                        "0.25*iwr_normal_p99 + 0.5*other_external_normal_fpr"
                    ),
                    "real_replay_used": False,
                },
            }
            history.append(epoch_record)
            state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            if eligible:
                eligible_states.append((selection_cost, epoch, state))
            fallback_states.append(
                (-float(dguha["sensitivity"]), selection_cost, epoch, state)
            )
            serialized = json.dumps(
                epoch_record, ensure_ascii=False, allow_nan=False
            )
            epoch_handle.write(serialized + "\n")
            epoch_handle.flush()
            print(serialized, flush=True)

    if eligible_states:
        selection_cost, best_epoch, best_state = min(
            eligible_states, key=lambda item: (item[0], item[1])
        )
        selection_gate_passed = True
    else:
        _, selection_cost, best_epoch, best_state = min(
            fallback_states, key=lambda item: (item[0], item[1], item[2])
        )
        selection_gate_passed = False
    model.load_state_dict(best_state, strict=True)
    model.eval()
    best_epoch_record = next(item for item in history if item["epoch"] == best_epoch)
    selected_threshold = float(best_epoch_record["selection_threshold"])

    checkpoint = {
        "model_version": EXPERIMENT_MODEL_VERSION,
        "model_mode": RESEARCH_MODEL_MODE,
        "model_architecture": "causal_tcn",
        "task_type": "prefall_prediction",
        "deployment_eligible": False,
        "shadow_only": True,
        "feature_version": FEATURE_VERSION_V2,
        "feature_names": FEATURE_NAMES_V2,
        "window_size": int(features.shape[1]),
        "input_size": len(FEATURE_NAMES_V2),
        "hidden_size": hidden_size,
        "state_dict": best_state,
        "normalization_mean": torch.from_numpy(mean.copy()),
        "normalization_std": torch.from_numpy(std.copy()),
        "decision_threshold": selected_threshold,
        "decision_threshold_policy": (
            "best_epoch_DGUHA_validation_maximum_specificity_with_window_"
            f"sensitivity_at_least_{minimum_dguha_validation_sensitivity:.3f}"
        ),
        "prediction_horizon_seconds": tuple(
            float(value) for value in arrays["prediction_horizon_seconds"]
        ),
        "positive_anchor": str(arrays["positive_anchor"].item()),
        "positive_weight": positive_weight,
        "batch_sampling": sampling_audit,
        "model_selection": {
            "policy": MODEL_SELECTION_POLICY,
            "best_epoch": best_epoch,
            "selection_cost": selection_cost,
            "selected_threshold": selected_threshold,
            "dguha_sensitivity_gate_passed": selection_gate_passed,
            "real_replay_used": False,
            "public_iwr6843_normal_validation_used": True,
        },
        "dataset_sha256": _sha256(source),
        "test_split_evaluated": False,
        "seed": seed,
        "warning": "Research-only multisource experiment; live checkpoint is unchanged.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    torch.save(checkpoint, destination)
    report = {
        "dataset_file": str(source),
        "dataset_sha256": _sha256(source),
        "checkpoint_file": str(destination),
        "checkpoint_sha256": _sha256(destination),
        "architecture": "causal_tcn",
        "model_structure_modified": False,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "hidden_size": hidden_size,
        "fixed_threshold": fixed_threshold,
        "selected_threshold": selected_threshold,
        "batch_sampling": sampling_audit,
        "model_selection": checkpoint["model_selection"],
        "best_epoch_metrics": best_epoch_record,
        "epoch_history_file": str(epoch_log_path),
        "epoch_history_sha256": _sha256(epoch_log_path),
        "real_replay_role": "audit_only_not_model_selection",
        "test_split_evaluated": False,
        "deployment_eligible": False,
    }
    destination.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def audit_frozen_checkpoint(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    report_path: str | Path,
    *,
    replay_paths: Sequence[str | Path] = (),
    fixed_threshold: float = FIXED_DECISION_THRESHOLD,
    batch_size: int = 512,
) -> dict[str, object]:
    source = Path(dataset_path).resolve()
    checkpoint_file = Path(checkpoint_path).resolve()
    arrays = _load_multisource_dataset(source)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    model = _model_from_checkpoint(checkpoint)
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    normalized = ((arrays["features"] - mean[None, None]) / std[None, None]).astype(
        np.float32
    )
    replay_inputs = {
        str(Path(path).resolve()): _extract_replay_features(Path(path).resolve())
        for path in replay_paths
    }
    origins = arrays["dataset_origin"]
    splits = arrays["split"]
    audit = _evaluate_epoch(
        model,
        normalized=normalized,
        labels=arrays["labels"],
        splits=splits,
        origins=origins,
        event_ids=arrays["event_id"],
        window_end_seconds=arrays["window_end_seconds"],
        dguha_validation_mask=(splits == "validation") & (origins == "dguha"),
        external_validation_mask=splits == "external_validation",
        replay_inputs=replay_inputs,
        mean=mean,
        std=std,
        threshold=fixed_threshold,
        minimum_dguha_validation_sensitivity=0.60,
        batch_size=batch_size,
        device=torch.device("cpu"),
    )
    report = {
        "audit_version": "tcn_multisource_frozen_b0_v1",
        "dataset_file": str(source),
        "dataset_sha256": _sha256(source),
        "checkpoint_file": str(checkpoint_file),
        "checkpoint_sha256": _sha256(checkpoint_file),
        "fixed_threshold": fixed_threshold,
        **audit,
        "model_selection_role": "frozen_baseline_only",
        "deployment_eligible": False,
    }
    destination = Path(report_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _evaluate_epoch(
    model: TemporalBinaryModel,
    *,
    normalized: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    origins: np.ndarray,
    event_ids: np.ndarray,
    window_end_seconds: np.ndarray,
    dguha_validation_mask: np.ndarray,
    external_validation_mask: np.ndarray,
    replay_inputs: dict[str, dict[str, object]],
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    minimum_dguha_validation_sensitivity: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    dguha_scores = _predict(
        model, normalized[dguha_validation_mask], device=device, batch_size=batch_size
    )
    dguha_labels = labels[dguha_validation_mask]
    selection_threshold = _select_specificity_priority_threshold(
        dguha_labels,
        dguha_scores,
        minimum_sensitivity=minimum_dguha_validation_sensitivity,
    )
    dguha = _classification_metrics(dguha_labels, dguha_scores, selection_threshold)
    dguha_fixed = _classification_metrics(dguha_labels, dguha_scores, threshold)

    external_scores = _predict(
        model, normalized[external_validation_mask], device=device, batch_size=batch_size
    )
    external_origins = origins[external_validation_mask]
    external_events = event_ids[external_validation_mask]
    external_times = window_end_seconds[external_validation_mask]
    by_origin: dict[str, dict[str, object]] = {}
    by_origin_fixed: dict[str, dict[str, object]] = {}
    for origin in EXTERNAL_ORIGINS:
        mask = external_origins == origin
        by_origin[origin] = _normal_score_metrics(
            external_scores[mask],
            event_ids=external_events[mask],
            window_end_seconds=external_times[mask],
            threshold=selection_threshold,
        )
        by_origin_fixed[origin] = _normal_score_metrics(
            external_scores[mask],
            event_ids=external_events[mask],
            window_end_seconds=external_times[mask],
            threshold=threshold,
        )
    non_iwr = external_origins != "iwr6843_fall_102"
    external = {
        "aggregate": _normal_score_metrics(
            external_scores,
            event_ids=external_events,
            window_end_seconds=external_times,
            threshold=selection_threshold,
        ),
        "excluding_iwr6843": _normal_score_metrics(
            external_scores[non_iwr],
            event_ids=external_events[non_iwr],
            window_end_seconds=external_times[non_iwr],
            threshold=selection_threshold,
        ),
        "by_origin": by_origin,
    }
    external_fixed = {
        "aggregate": _normal_score_metrics(
            external_scores,
            event_ids=external_events,
            window_end_seconds=external_times,
            threshold=threshold,
        ),
        "excluding_iwr6843": _normal_score_metrics(
            external_scores[non_iwr],
            event_ids=external_events[non_iwr],
            window_end_seconds=external_times[non_iwr],
            threshold=threshold,
        ),
        "by_origin": by_origin_fixed,
    }

    replay_report: dict[str, object] = {}
    for path, replay in replay_inputs.items():
        raw = np.asarray(replay["features"], dtype=np.float32)
        replay_normalized = ((raw - mean[None, None]) / std[None, None]).astype(
            np.float32
        )
        scores = _predict(
            model, replay_normalized, device=device, batch_size=batch_size
        )
        replay_report[path] = {
            "replay_sha256": replay["sha256"],
            "valid_window_count": int(len(scores)),
            "quality_counts": replay["quality_counts"],
            "score_distribution": _describe(scores),
            "above_threshold_window_fraction": float(np.mean(scores >= threshold)),
            "confirmed_run_count": _confirmed_run_count(scores >= threshold, 3),
            "model_selection_used": False,
        }
    return {
        "selection_threshold": selection_threshold,
        "dguha_validation": dguha,
        "dguha_validation_fixed_0p35": dguha_fixed,
        "external_normal": external,
        "external_normal_fixed_0p35": external_fixed,
        "iwr6843_real_replay": replay_report,
    }


def _classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, object]:
    truth = np.asarray(labels, dtype=np.int8)
    prediction = np.asarray(scores >= threshold, dtype=np.int8)
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0
    )
    return {
        "sample_count": int(len(truth)),
        "positive_count": int(np.sum(truth == 1)),
        "negative_count": int(np.sum(truth == 0)),
        "threshold": threshold,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auroc": _auroc(truth, scores),
        "score_distribution": _describe(scores),
    }


def _normal_score_metrics(
    scores: np.ndarray,
    *,
    event_ids: np.ndarray,
    window_end_seconds: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    values = np.asarray(scores, dtype=np.float64)
    if not len(values):
        raise ValueError("normal score group must not be empty")
    confirmed = 0
    observed_seconds = 0.0
    groups: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(event_ids):
        groups[str(event)].append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: float(window_end_seconds[index]))
        high = values[ordered] >= threshold
        confirmed += _confirmed_run_count(high, 3)
        times = np.asarray(window_end_seconds)[ordered]
        finite = times[np.isfinite(times)]
        if len(finite) >= 2:
            observed_seconds += max(0.1, float(finite.max() - finite.min() + 0.1))
        else:
            observed_seconds += 0.1
    return {
        "window_count": int(len(values)),
        "recording_count": int(len(groups)),
        "score_distribution": _describe(values),
        "above_threshold_window_count": int(np.sum(values >= threshold)),
        "above_threshold_window_fraction": float(np.mean(values >= threshold)),
        "confirmed_run_count": int(confirmed),
        "observed_hours": observed_seconds / 3600.0,
        "confirmed_false_alarms_per_hour": (
            confirmed / (observed_seconds / 3600.0) if observed_seconds else 0.0
        ),
    }


def _predict(
    model: TemporalBinaryModel,
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            logits = model(torch.from_numpy(values[start : start + batch_size]).to(device))
            batches.append(torch.sigmoid(logits).cpu().numpy().astype(np.float64))
    if not batches:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(batches)


def _extract_replay_features(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"replay does not exist: {path}")
    frames = _load_replay_frames(path, default_room=Room.BATHROOM)
    extractor = RadarTemporalFeatureExtractorV2()
    history: deque[RadarFrame] = deque()
    values: list[np.ndarray] = []
    qualities: Counter[str] = Counter()
    last_timestamp = None
    for frame in frames:
        history.append(frame)
        while history and (frame.timestamp - history[0].timestamp).total_seconds() > 2.2:
            history.popleft()
        if last_timestamp is not None and (
            frame.timestamp - last_timestamp
        ).total_seconds() < 0.095:
            continue
        last_timestamp = frame.timestamp
        if (frame.timestamp - history[0].timestamp).total_seconds() < 1.9:
            qualities["WARMUP"] += 1
            continue
        window = extractor.transform(tuple(history), end_timestamp=frame.timestamp)
        qualities[window.data_quality.value] += 1
        if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
            continue
        values.append(np.asarray(window.values, dtype=np.float32))
    if not values:
        raise ValueError(f"replay produced no valid windows: {path}")
    return {
        "features": np.stack(values).astype(np.float32, copy=False),
        "quality_counts": dict(qualities),
        "sha256": _sha256(path),
    }


def _confirmed_run_count(high: Iterable[bool], confirmation_windows: int) -> int:
    count = 0
    length = 0
    for value in high:
        if bool(value):
            length += 1
            if length == confirmation_windows:
                count += 1
        else:
            length = 0
    return count


def _describe(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("cannot describe an empty score array")
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _load_multisource_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {path}")
    with np.load(path, allow_pickle=False) as dataset:
        required = {
            "features",
            "labels",
            "split",
            "dataset_origin",
            "event_id",
            "window_end_seconds",
            "feature_version",
            "feature_names",
            "dataset_mode",
            "positive_anchor",
            "prediction_horizon_seconds",
            "normalization_mean",
            "normalization_std",
        }
        missing = required.difference(dataset.files)
        if missing:
            raise ValueError(f"multisource dataset missing {sorted(missing)}")
        arrays = {name: np.asarray(dataset[name]) for name in dataset.files}
    if str(arrays["dataset_mode"].item()) != DATASET_MODE:
        raise ValueError("dataset mode is incompatible")
    if str(arrays["feature_version"].item()) != FEATURE_VERSION_V2:
        raise ValueError("feature version is incompatible")
    if tuple(map(str, arrays["feature_names"])) != FEATURE_NAMES_V2:
        raise ValueError("feature names/order are incompatible")
    arrays["features"] = np.asarray(arrays["features"], dtype=np.float32)
    arrays["labels"] = np.asarray(arrays["labels"], dtype=np.int8)
    return arrays


def _model_from_checkpoint(checkpoint: dict[str, object]) -> TemporalBinaryModel:
    if checkpoint.get("model_version") != EXPERIMENT_MODEL_VERSION:
        raise ValueError("checkpoint model version is incompatible")
    if checkpoint.get("feature_version") != FEATURE_VERSION_V2:
        raise ValueError("checkpoint feature version is incompatible")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES_V2:
        raise ValueError("checkpoint feature order is incompatible")
    model = TemporalBinaryModel(
        architecture=str(checkpoint["model_architecture"]),
        input_size=len(FEATURE_NAMES_V2),
        hidden_size=int(checkpoint["hidden_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fixed-architecture multisource TCN")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--replay", action="append", default=[], type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = train_tcn_multisource_experiment(
        args.dataset,
        args.checkpoint,
        replay_paths=args.replay,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
