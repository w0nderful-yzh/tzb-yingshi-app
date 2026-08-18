"""Train a descent early-detection binary classifier.

The dataset ``dguha_descent_early_detection_v1.npz`` contains:
- positive windows sampled *during* the descent (descent_onset + 0.3s to
  near_floor - 0.3s), with ``seconds_to_floor`` recorded;
- negative windows from normal actions and fall recordings before onset.

The model is a temporal binary classifier (causal TCN or LSTM) identical to
the pre-fall models, but the positive task is "the body is actively falling
right now" instead of "a fall is imminent in 0.1-0.6s". Because the descent
window contains real z_p90 / z_p50 / height_range dynamics, the model can
learn temporal structure.

Evaluation:
- window AUROC / sensitivity / specificity;
- event-level: for each held-out fall recording, whether any descent window
  fires above threshold at least ``confirm_windows`` in a row, and the
  resulting lead before near_floor (from ``seconds_to_floor``).

Contract: subject-isolated splits, deployment_eligible=false, shadow_only=true.
Version: radar_descent_detection_train_v1
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

from radar_module.model.temporal_models_v3 import TemporalBinaryModel
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
)

MODEL_MODE = "RESEARCH_DESCENT_DETECTION_V1"
MODEL_VERSION = "radar_descent_detection_v1"


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    arrays = np.load(path, allow_pickle=True)
    features = np.asarray(arrays["features"], dtype=np.float32)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    splits = np.asarray(arrays["split"])
    if features.shape[1:] != (20, len(FEATURE_NAMES_V2)):
        raise ValueError("features shape incompatible with v2")
    stf = np.asarray(arrays["seconds_to_floor"], dtype=np.float64)
    return {
        "features": features,
        "labels": labels,
        "splits": splits,
        "seconds_to_floor": stf,
        "source_files": np.asarray(arrays["source_files"]),
    }


def train_descent_detection_v1(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    architecture: str = "causal_tcn",
    hidden_size: int = 24,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    positive_weight: float = 12.0,
    validation_split: str = "validation",
    test_split: str = "test",
    confirm_windows: int = 3,
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
    stf = data["seconds_to_floor"]

    train_mask = splits == "train"
    mean = features[train_mask].mean(axis=(0, 1), keepdims=True)
    std = features[train_mask].std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-9, 1e-9, std)
    normalized = ((features - mean) / std).astype(np.float32)

    model = TemporalBinaryModel(
        architecture=architecture,
        input_size=len(FEATURE_NAMES_V2),
        hidden_size=hidden_size,
    ).to(torch_device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], device=torch_device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    split_indices = {
        name: np.flatnonzero(splits == name) for name in np.unique(splits)
    }
    train_idx = split_indices.get("train", np.array([], dtype=np.int64))
    val_idx = split_indices.get(validation_split, np.array([], dtype=np.int64))
    test_idx = split_indices.get(test_split, np.array([], dtype=np.int64))
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("missing train/validation split")

    train_feat = torch.from_numpy(normalized[train_idx]).to(torch_device)
    train_lab = torch.from_numpy(labels[train_idx].astype(np.float32)).to(
        torch_device
    )
    loader = DataLoader(
        TensorDataset(train_feat, train_lab),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        cum = 0.0
        seen = 0
        for bx, by in loader:
            optimizer.zero_grad()
            logit = model(bx)
            loss = criterion(logit, by)
            loss.backward()
            optimizer.step()
            cum += float(loss.item()) * len(by)
            seen += len(by)
        model.eval()
        val = _binary_metrics(
            model, normalized[val_idx], labels[val_idx], device=torch_device
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": cum / max(seen, 1),
                "val_auroc": val["auroc"],
                "val_sensitivity": val["sensitivity"],
            }
        )

    # Threshold selection on validation: choose threshold that gives best
    # balanced accuracy.
    val_scores = _scores(model, normalized[val_idx], device=torch_device)
    val_labels = labels[val_idx]
    best_threshold, best_ba = _select_threshold_balanced(
        val_scores, val_labels
    )

    test_metrics = _binary_metrics(
        model, normalized[test_idx], labels[test_idx],
        threshold=best_threshold, device=torch_device,
    )
    # Event-level evaluation on test fall recordings.
    event_report = _event_level_evaluation(
        model,
        normalized,
        labels,
        splits,
        stf,
        data["source_files"],
        test_split=test_split,
        threshold=best_threshold,
        confirm_windows=confirm_windows,
        device=torch_device,
    )

    checkpoint = {
        "model_version": MODEL_VERSION,
        "model_mode": MODEL_MODE,
        "model_architecture": architecture,
        "task_type": "descent_early_detection",
        "deployment_eligible": False,
        "shadow_only": True,
        "feature_version": FEATURE_VERSION_V2,
        "feature_names": list(FEATURE_NAMES_V2),
        "window_size": 20,
        "input_size": len(FEATURE_NAMES_V2),
        "hidden_size": hidden_size,
        "state_dict": model.state_dict(),
        "normalization_mean": mean[0, 0].tolist(),
        "normalization_std": std[0, 0].tolist(),
        "decision_threshold": best_threshold,
        "positive_weight": positive_weight,
        "positive_label_definition": "window inside descent interval [onset+0.3, floor-0.3]",
        "dataset_sha256": _sha256(dataset_file),
        "seed": seed,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)

    report = {
        "schema_version": "radar_descent_detection_report_v1",
        "checkpoint_file": str(destination),
        "checkpoint_sha256": _sha256(destination),
        "architecture": architecture,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "positive_weight": positive_weight,
        "best_threshold": best_threshold,
        "split_counts": {name: int(len(idx)) for name, idx in split_indices.items()},
        "train_positive_count": int(labels[train_idx].sum()),
        "test_metrics": test_metrics,
        "event_level": event_report,
        "training_history": history,
        "note": (
            "Descent early detection: positive windows are inside the descent "
            "interval, so the model can learn z_p90/height dynamics. This is "
            "fall-in-progress detection, not future-prediction. Lead time is "
            "bounded by remaining descent duration."
        ),
    }
    report_path = destination.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def _scores(model, features, *, device):
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(features).to(device))).cpu().numpy()


def _binary_metrics(
    model, features, labels, *, threshold=None, device
) -> dict[str, float]:
    scores = _scores(model, features, device=device)
    if threshold is None:
        threshold, _ = _select_threshold_balanced(scores, labels)
    pred = (scores >= threshold).astype(np.int64)
    tp = float(np.sum((pred == 1) & (labels == 1)))
    fp = float(np.sum((pred == 1) & (labels == 0)))
    tn = float(np.sum((pred == 0) & (labels == 0)))
    fn = float(np.sum((pred == 0) & (labels == 1)))
    return {
        "count": int(len(labels)),
        "threshold": threshold,
        "accuracy": float(np.mean(pred == labels)),
        "sensitivity": tp / max(tp + fn, 1.0),
        "specificity": tn / max(tn + fp, 1.0),
        "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1.0)) + 0.5 * (tn / max(tn + fp, 1.0)),
        "auroc": _auroc(scores[labels == 1], scores[labels == 0]),
    }


def _select_threshold_balanced(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_t, best_ba = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (scores >= t).astype(np.int64)
        tp = float(np.sum((pred == 1) & (labels == 1)))
        fn = float(np.sum((pred == 0) & (labels == 1)))
        tn = float(np.sum((pred == 0) & (labels == 0)))
        fp = float(np.sum((pred == 1) & (labels == 0)))
        ba = 0.5 * (tp / max(tp + fn, 1.0)) + 0.5 * (tn / max(tn + fp, 1.0))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba


def _event_level_evaluation(
    model,
    features,
    labels,
    splits,
    stf,
    source_files,
    *,
    test_split,
    threshold,
    confirm_windows,
    device,
) -> dict[str, Any]:
    """For each test fall recording, detect consecutive high windows during
    the descent and report whether the event fired and the lead to floor."""
    test_mask = splits == test_split
    feat_test = features[test_mask]
    lab_test = labels[test_mask]
    src_test = source_files[test_mask]
    stf_test = stf[test_mask]
    scores = _scores(model, feat_test, device=device)

    # Group by source file (fall recording).
    records: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for i in range(len(feat_test)):
        src = str(src_test[i])
        if src not in records:
            records[src] = {"high": [], "stf": [], "label": []}
            order.append(src)
        records[src]["high"].append(bool(scores[i] >= threshold))
        records[src]["stf"].append(float(stf_test[i]))
        records[src]["label"].append(int(lab_test[i]))

    fired_events = 0
    total_fall_recordings = 0
    lead_times: list[float] = []
    per_recording: list[dict[str, Any]] = []
    for src in order:
        rec = records[src]
        is_fall = int(sum(rec["label"])) > 0
        if not is_fall:
            continue
        total_fall_recordings += 1
        # consecutive high runs
        consecutive = 0
        fired = False
        fire_stf: float | None = None
        for h, s in zip(rec["high"], rec["stf"]):
            if h:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= confirm_windows and fire_stf is None:
                fired = True
                fire_stf = s
        per_rec = {
            "source": src,
            "fall": is_fall,
            "window_count": len(rec["high"]),
            "positive_windows": sum(rec["label"]),
            "fired": fired,
            "lead_to_floor_at_first_confirm": fire_stf,
        }
        if fired:
            fired_events += 1
            if fire_stf is not None:
                lead_times.append(fire_stf)
        per_recording.append(per_rec)

    return {
        "test_fall_recordings": total_fall_recordings,
        "fired_events": fired_events,
        "event_recall": (
            fired_events / max(total_fall_recordings, 1)
            if total_fall_recordings
            else None
        ),
        "lead_to_floor_median": float(np.median(lead_times)) if lead_times else None,
        "lead_to_floor_min": float(np.min(lead_times)) if lead_times else None,
        "lead_to_floor_max": float(np.max(lead_times)) if lead_times else None,
        "per_recording": per_recording,
    }


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    all_scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    # Ascending sort: higher score -> larger rank. AUROC = P(positive > negative).
    order = np.argsort(all_scores, kind="mergesort")
    labels = labels[order]
    pos_count = int(labels.sum())
    neg_count = int(len(labels) - pos_count)
    if pos_count == 0 or neg_count == 0:
        return 0.5
    rank_sum = sum(i + 1 for i in range(len(labels)) if labels[i] == 1)
    u = rank_sum - pos_count * (pos_count + 1) / 2
    return float(u / (pos_count * neg_count))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train descent early-detection.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", default="causal_tcn")
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--positive-weight", type=float, default=12.0)
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--confirm-windows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = train_descent_detection_v1(
        dataset_path=args.dataset,
        output_path=args.output,
        architecture=args.architecture,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        positive_weight=args.positive_weight,
        validation_split=args.validation_split,
        test_split=args.test_split,
        confirm_windows=args.confirm_windows,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
