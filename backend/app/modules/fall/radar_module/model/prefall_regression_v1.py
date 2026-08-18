"""Time-to-impact regression training for pre-fall prediction.

Motivation
----------
The previous binary window classifier (labels 0/1 for "window ends 0.1-0.6s
before descent onset") degenerates into a static "is the body low right now"
detector: shuffling or reversing the 20 input frames does not change recall.
That diagnostic proved the model is not using temporal evolution.

A regression head that predicts *time to descent onset* forces the model to
express "how far along the descent is" instead of "is it low". A correct
regressor must be temporally monotone: as the window slides closer to onset,
the predicted time-to-impact must decrease. This is a strictly harder and more
informative task.

Design
------
- Shared temporal encoder (same causal TCN / LSTM backbone as before).
- Two heads:
    * `pre_fall` classification head (logit): is this window part of a fall
      approach? Trained on all windows (positive=1, negative=0).
    * `time_to_impact` regression head: predicts seconds until descent_onset
      for positive windows. Only trained on positive windows (which carry a
      finite `seconds_to_onset`). Negative windows are excluded from the
      regression loss.
- Loss = classification BCE + lambda * regression Smooth-L1.
- Evaluation:
    * Window-level classification metrics (as before).
    * Regression metrics on positive windows: MAE, and **temporal monotonicity**
      (correlation between true time-to-impact and predicted time-to-impact
      within each event; a detector would give ~0, a real regressor > 0).
    * Continuous event evaluation is delegated to existing tooling.

Contract
--------
- Subject-isolated splits (DGUHA project_split is respected).
- `deployment_eligible=false`, `shadow_only=true`.
- The checkpoint records feature version, label definition, normalization,
  thresholds, and regression target semantics.

Version: radar_prefall_regression_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from radar_module.model.temporal_models_v3 import build_temporal_encoder
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
)


REGRESSION_MODEL_MODE = "RESEARCH_PREFALL_REGRESSION_V1"
REGRESSION_MODEL_VERSION = "radar_prefall_regression_v1"


class PreFallRegressionHead(nn.Module):
    """Two-head decoder: binary pre-fall logit + time-to-impact regression."""

    def __init__(
        self,
        encoder_hidden_size: int,
        hidden_size: int = 24,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(encoder_hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(encoder_hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, encoding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # encoding: (B, encoder_hidden)
        return self.classifier(encoding), self.regressor(encoding)


class PreFallRegressionModel(nn.Module):
    """Shared temporal encoder + classification/regression heads."""

    def __init__(
        self,
        *,
        architecture: str,
        input_size: int,
        encoder_hidden_size: int,
        head_hidden_size: int = 24,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.encoder = build_temporal_encoder(
            architecture=architecture,
            input_size=input_size,
            hidden_size=encoder_hidden_size,
        )
        output_size = self.encoder.output_size
        self.classifier = nn.Linear(output_size, 1)
        self.regressor = nn.Linear(output_size, 1)

    def encoding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        return self.classifier(enc).squeeze(-1), self.regressor(enc).squeeze(-1)


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    arrays = np.load(path, allow_pickle=True)
    features = np.asarray(arrays["features"], dtype=np.float32)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    splits = np.asarray(arrays["split"])
    if features.shape[1:] != (20, len(FEATURE_NAMES_V2)):
        raise ValueError("features shape incompatible with v2")
    if labels.shape != (features.shape[0],) or splits.shape != (
        features.shape[0],
    ):
        raise ValueError("labels/splits shape incompatible")
    seconds_to_onset = np.asarray(
        arrays["seconds_to_onset"], dtype=np.float64
    )
    if seconds_to_onset.shape != (features.shape[0],):
        raise ValueError("seconds_to_onset shape incompatible")
    return {
        "features": features,
        "labels": labels,
        "splits": splits,
        "seconds_to_onset": seconds_to_onset,
    }


def train_prefall_regression_v1(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    architecture: str = "causal_tcn",
    encoder_hidden_size: int = 24,
    head_hidden_size: int = 24,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    regression_weight: float = 1.0,
    positive_weight: float = 32.0,
    validation_split: str = "validation",
    test_split: str = "test",
    seed: int = 20260810,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_device = torch.device(device)

    dataset_file = Path(dataset_path).resolve()
    destination = Path(output_path).resolve()
    if destination.suffix.lower() != ".pt":
        raise ValueError("output_path must end with .pt")

    data = _load_dataset(dataset_file)
    features = data["features"]
    labels = data["labels"]
    splits = data["splits"]
    seconds = data["seconds_to_onset"]

    # Normalization computed on training split only.
    train_mask = splits == "train"
    mean = features[train_mask].mean(axis=(0, 1), keepdims=True)
    std = features[train_mask].std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-9, 1e-9, std)
    normalized = ((features - mean) / std).astype(np.float32)

    model = PreFallRegressionModel(
        architecture=architecture,
        input_size=len(FEATURE_NAMES_V2),
        encoder_hidden_size=encoder_hidden_size,
        head_hidden_size=head_hidden_size,
    ).to(torch_device)

    # BCE for classification; Smooth-L1 for regression across all rebalanced
    # windows (positive -> true time-to-impact, negative -> far target).
    classification_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], device=torch_device)
    )
    regression_loss = nn.SmoothL1Loss(beta=0.3)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Build per-split indices.
    split_names = np.unique(splits)
    split_indices = {name: np.flatnonzero(splits == name) for name in split_names}

    train_idx = split_indices.get("train", np.array([], dtype=np.int64))
    val_idx = split_indices.get(validation_split, np.array([], dtype=np.int64))
    test_idx = split_indices.get(test_split, np.array([], dtype=np.int64))
    if len(train_idx) == 0:
        raise ValueError("no training split found")
    if len(val_idx) == 0:
        raise ValueError(f"no validation split '{validation_split}' found")

    # ---- training ----
    # Balance: positives are extremely rare (98 of 39130 train). We rebalance
    # by capping negatives per positive to avoid the regressor being drowned.
    # Regression targets: positive -> true seconds_to_onset in [0.5, 1.0];
    # negative -> a far "no fall" target (max_lead + margin), clamped.
    max_lead = 2.5
    train_pos = np.flatnonzero((splits == "train") & (labels == 1))
    train_neg_all = np.flatnonzero((splits == "train") & (labels == 0))
    # subsample negatives: at most neg_per_pos * n_positives
    neg_per_pos = 8
    n_pos = len(train_pos)
    n_neg_target = min(len(train_neg_all), n_pos * neg_per_pos)
    rng = np.random.default_rng(seed)
    train_neg = rng.choice(train_neg_all, size=n_neg_target, replace=False)
    train_idx = np.sort(np.concatenate([train_pos, train_neg]))
    rng.shuffle(train_idx)

    train_features = torch.from_numpy(normalized[train_idx]).to(torch_device)
    train_labels = torch.from_numpy(labels[train_idx].astype(np.float32)).to(
        torch_device
    )
    # regression target: positive -> seconds_to_onset; negative -> far target
    train_seconds_raw = seconds[train_idx].copy()
    train_seconds_raw[~np.isfinite(train_seconds_raw)] = max_lead
    train_seconds_raw = np.clip(train_seconds_raw, 0.0, max_lead)
    train_seconds = torch.from_numpy(train_seconds_raw.astype(np.float32)).to(
        torch_device
    )

    dataset = TensorDataset(train_features, train_labels, train_seconds)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        cumulative_cls = 0.0
        cumulative_reg = 0.0
        seen = 0
        for batch_features, batch_labels, batch_seconds in loader:
            optimizer.zero_grad()
            cls_logit, reg_pred = model(batch_features)
            cls_loss = classification_loss(
                cls_logit.unsqueeze(1), batch_labels.unsqueeze(1)
            )
            reg_loss = regression_loss(reg_pred, batch_seconds)
            loss = cls_loss + regression_weight * reg_loss
            loss.backward()
            optimizer.step()
            cumulative_cls += float(cls_loss.item()) * len(batch_labels)
            cumulative_reg += float(reg_loss.item()) * len(batch_labels)
            seen += len(batch_labels)

        # ---- validation ----
        model.eval()
        val_metrics = _evaluate_classification(
            model,
            normalized[val_idx],
            labels[val_idx],
            device=torch_device,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_cls_loss": cumulative_cls / max(seen, 1),
                "train_reg_loss": cumulative_reg / max(seen, 1),
                "val_auroc": val_metrics["auroc"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

    # ---- final evaluation on test (subject-isolated, never used for tuning) ----
    test_metrics = _evaluate_classification(
        model,
        normalized[test_idx],
        labels[test_idx],
        device=torch_device,
    )
    regression_test = _evaluate_regression(
        model,
        normalized[test_idx],
        seconds[test_idx],
        device=torch_device,
    )
    # temporal monotonicity: within each event, corr(true seconds, predicted)
    monotonicity = _evaluate_temporal_monotonicity(
        model,
        normalized,
        seconds,
        splits,
        device=torch_device,
    )

    checkpoint = {
        "model_version": REGRESSION_MODEL_VERSION,
        "model_mode": REGRESSION_MODEL_MODE,
        "model_architecture": architecture,
        "task_type": "prefall_time_to_impact_regression",
        "deployment_eligible": False,
        "shadow_only": True,
        "feature_version": FEATURE_VERSION_V2,
        "feature_names": list(FEATURE_NAMES_V2),
        "window_size": 20,
        "input_size": len(FEATURE_NAMES_V2),
        "encoder_hidden_size": encoder_hidden_size,
        "head_hidden_size": head_hidden_size,
        "state_dict": model.state_dict(),
        "normalization_mean": mean[0, 0].tolist(),
        "normalization_std": std[0, 0].tolist(),
        "regression_weight": regression_weight,
        "positive_weight": positive_weight,
        "positive_label_definition": "time_to_descent_onset_seconds",
        "prediction_horizon_seconds": [0.5, 1.0],
        "dataset_sha256": _sha256(dataset_file),
        "seed": seed,
        "epochs_trained": epochs,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)

    report = {
        "schema_version": "radar_prefall_regression_report_v1",
        "checkpoint_file": str(destination),
        "checkpoint_sha256": _sha256(destination),
        "dataset_file": str(dataset_file),
        "architecture": architecture,
        "encoder_hidden_size": encoder_hidden_size,
        "head_hidden_size": head_hidden_size,
        "epochs": epochs,
        "regression_weight": regression_weight,
        "positive_weight": positive_weight,
        "split_counts": {name: int(len(idx)) for name, idx in split_indices.items()},
        "train_positive_count": int(labels[train_idx].sum()),
        "test_metrics": test_metrics,
        "test_regression": regression_test,
        "temporal_monotonicity": monotonicity,
        "training_history": history,
        "note": (
            "Research only. Time-to-impact regression forces temporal "
            "evolution; monotonicity indicates whether the model actually "
            "uses trajectory rather than static height."
        ),
    }
    report_path = destination.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def _evaluate_classification(
    model: PreFallRegressionModel,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    if len(features) == 0:
        return {"count": 0, "auroc": 0.0, "accuracy": 0.0, "sensitivity": 0.0, "specificity": 0.0}
    with torch.inference_mode():
        cls_logit, _ = model(torch.from_numpy(features).to(device))
        scores = torch.sigmoid(cls_logit).detach().cpu().numpy()
    pred = (scores >= 0.5).astype(np.int64)
    accuracy = float(np.mean(pred == labels))
    tp = float(np.sum((pred == 1) & (labels == 1)))
    fp = float(np.sum((pred == 1) & (labels == 0)))
    tn = float(np.sum((pred == 0) & (labels == 0)))
    fn = float(np.sum((pred == 0) & (labels == 1)))
    sensitivity = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    # AUROC via rank
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    auroc = _auroc(pos, neg)
    return {
        "count": int(len(features)),
        "auroc": auroc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _evaluate_regression(
    model: PreFallRegressionModel,
    features: np.ndarray,
    seconds: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    valid = np.isfinite(seconds)
    if not valid.any():
        return {"count": 0, "mae": 0.0, "rmse": 0.0}
    with torch.inference_mode():
        _, pred = model(torch.from_numpy(features).to(device))
        pred = pred.detach().cpu().numpy()
    pred_valid = pred[valid]
    true_valid = seconds[valid]
    mae = float(np.mean(np.abs(pred_valid - true_valid)))
    rmse = float(np.sqrt(np.mean((pred_valid - true_valid) ** 2)))
    return {"count": int(valid.sum()), "mae": mae, "rmse": rmse}


def _evaluate_temporal_monotonicity(
    model: PreFallRegressionModel,
    features: np.ndarray,
    seconds: np.ndarray,
    splits: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, float]:
    """Within each test event, correlate true time-to-onset vs predicted.

    A model that just detects "is low" gives ~0 (no temporal ordering).
    A real regressor should give positive correlation within an event
    (prediction decreases as the window approaches onset).
    """
    model.eval()
    test_mask = splits == "test"
    feat_test = features[test_mask]
    sec_test = seconds[test_mask]
    valid = np.isfinite(sec_test)
    if not valid.any():
        return {"count": 0, "spearman": 0.0, "pearson": 0.0}
    with torch.inference_mode():
        _, pred = model(torch.from_numpy(feat_test).to(device))
        pred = pred.detach().cpu().numpy()
    pred_valid = pred[valid]
    true_valid = sec_test[valid]
    if len(pred_valid) < 3:
        return {"count": int(len(pred_valid)), "spearman": 0.0, "pearson": 0.0}
    pearson = float(np.corrcoef(pred_valid, true_valid)[0, 1])
    # spearman
    from scipy.stats import spearmanr

    spearman, _ = spearmanr(pred_valid, true_valid)
    return {"count": int(len(pred_valid)), "spearman": float(spearman), "pearson": pearson}


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    all_scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(all_scores, kind="mergesort")
    labels = labels[order]
    pos_count = int(labels.sum())
    neg_count = int(len(labels) - pos_count)
    if pos_count == 0 or neg_count == 0:
        return 0.5
    rank_sum = 0.0
    count = 0.0
    for i in range(len(labels)):
        if labels[i] == 1:
            rank_sum += i + 1
            count += 1
    u = rank_sum - pos_count * (pos_count + 1) / 2
    return float(u / (pos_count * neg_count))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train time-to-impact regression.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", default="causal_tcn")
    parser.add_argument("--encoder-hidden-size", type=int, default=24)
    parser.add_argument("--head-hidden-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=32.0)
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = train_prefall_regression_v1(
        dataset_path=args.dataset,
        output_path=args.output,
        architecture=args.architecture,
        encoder_hidden_size=args.encoder_hidden_size,
        head_hidden_size=args.head_hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        regression_weight=args.regression_weight,
        positive_weight=args.positive_weight,
        validation_split=args.validation_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
