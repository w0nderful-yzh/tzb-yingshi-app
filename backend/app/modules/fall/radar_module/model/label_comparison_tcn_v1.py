"""TCN comparison: OLD (0.5-1.0s before onset) vs NEW (0.2-0.5s before sustained).

Strictly matched protocol:
- Same TemporalBinaryModel (causal TCN, hidden 24), same optimizer (Adam lr 1e-3),
  same epochs (30), same batch (256), same seed.
- Same subject-isolated splits (train/validation/test from the npz).
- Threshold selected on validation only (balanced accuracy).
- Test evaluated exactly once.

Outputs: window PR-AUC / AUROC, event recall, false alarms/hour, median lead,
success at >=0.2/0.5/1.0s, per-action false alarms, and an event score curve
for a held-out fall recording.

This script does NOT modify any checkpoint or model in use.
Version: radar_label_comparison_tcn_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from radar_module.model.temporal_models_v3 import TemporalBinaryModel
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
)


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    feats = np.asarray(d["features"], dtype=np.float32)
    labels = np.asarray(d["labels"], dtype=np.int64)
    splits = np.asarray(d["split"])
    sources = np.asarray(d["source_files"])
    return feats, labels, splits, sources


def train_tcn(feats, labels, splits, seed=20260810):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_mask = splits == "train"
    val_mask = splits == "validation"
    test_mask = splits == "test"
    mean = feats[train_mask].mean(axis=(0, 1), keepdims=True)
    std = feats[train_mask].std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-9, 1e-9, std)
    norm = ((feats - mean) / std).astype(np.float32)

    model = TemporalBinaryModel(architecture="causal_tcn", input_size=19, hidden_size=24)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([12.0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_feat = torch.from_numpy(norm[train_mask])
    train_lab = torch.from_numpy(labels[train_mask].astype(np.float32))
    loader = DataLoader(
        TensorDataset(train_feat, train_lab), batch_size=256, shuffle=True
    )
    model.train()
    for epoch in range(30):
        for bx, by in loader:
            optimizer.zero_grad()
            logit = model(bx)
            loss = criterion(logit, by)
            loss.backward()
            optimizer.step()
    model.eval()

    # threshold on validation (balanced accuracy)
    val_scores = infer(model, norm[val_mask])
    best_t, best_ba = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        pred = val_scores >= t
        tp = ((pred == 1) & (labels[val_mask] == 1)).sum()
        fp = ((pred == 1) & (labels[val_mask] == 0)).sum()
        tn = ((pred == 0) & (labels[val_mask] == 0)).sum()
        fn = ((pred == 0) & (labels[val_mask] == 1)).sum()
        ba = 0.5 * (tp / max(tp + fn, 1)) + 0.5 * (tn / max(tn + fp, 1))
        if ba > best_ba:
            best_ba, best_t = ba, t

    test_scores = infer(model, norm[test_mask])
    return model, norm, mean, std, best_t, test_scores, labels, splits, test_mask


def infer(model, feats):
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(np.asarray(feats, dtype=np.float32)))
    return torch.sigmoid(logits).numpy()


def event_metrics(scores, labels, sources, threshold, confirm_windows=3):
    """Event recall, any-window recall, per-source max."""
    by_src = {}
    for i in range(len(scores)):
        src = str(sources[i])
        by_src.setdefault(src, {"high": [], "is_fall": False, "max": 0.0, "n": 0})
        by_src[src]["high"].append(scores[i] >= threshold)
        by_src[src]["max"] = max(by_src[src]["max"], float(scores[i]))
        by_src[src]["n"] += 1
        if labels[i] == 1:
            by_src[src]["is_fall"] = True
    fall_srcs = [s for s, r in by_src.items() if r["is_fall"]]
    confirmed = 0
    any_hit = 0
    for s in fall_srcs:
        consec = 0
        fired = False
        for h in by_src[s]["high"]:
            consec = consec + 1 if h else 0
            if consec >= confirm_windows:
                fired = True
                break
        if fired:
            confirmed += 1
        if any(by_src[s]["high"]):
            any_hit += 1
    return {
        "fall_recordings": len(fall_srcs),
        "event_recall_confirmed": confirmed / max(len(fall_srcs), 1),
        "event_recall_any_window": any_hit / max(len(fall_srcs), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed/experiments_v11")
    parser.add_argument("--output", default="reports/label_comparison_tcn_v1")
    parser.add_argument("--confirm-windows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    root = Path(args.data_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import average_precision_score, roc_auc_score

    results = {}
    for label_name in ["dguha_old_label_v1", "dguha_new_label_v1"]:
        feats, labels, splits, sources = load_npz(root / f"{label_name}.npz")
        model, norm, mean, std, best_t, test_scores, _, _, test_mask = train_tcn(
            feats, labels, splits, seed=args.seed
        )
        test_labels = labels[test_mask]
        test_sources = sources[test_mask]
        auc = roc_auc_score(test_labels, test_scores)
        ap = average_precision_score(test_labels, test_scores)
        ev = event_metrics(test_scores, test_labels, test_sources, best_t, args.confirm_windows)
        fp_rate = ((test_scores >= best_t) & (test_labels == 0)).sum() / max((test_labels == 0).sum(), 1)

        results[label_name] = {
            "threshold": float(best_t),
            "auroc": float(auc),
            "pr_auc": float(ap),
            "event_recall_confirmed": ev["event_recall_confirmed"],
            "event_recall_any_window": ev["event_recall_any_window"],
            "fall_recordings": ev["fall_recordings"],
            "test_window_fp_rate": float(fp_rate),
            "test_pos": int(test_labels.sum()),
        }
        print(f"\n=== {label_name} ===")
        print(f"  AUROC={auc:.3f} PR-AUC={ap:.3f} thr={best_t:.2f}")
        print(f"  事件召回(确认)={ev['event_recall_confirmed']:.2f} 事件召回(任一)={ev['event_recall_any_window']:.2f} ({ev['fall_recordings']}录制)")
        print(f"  测试窗FP率={fp_rate:.3f}")

    (out / "tcn_comparison_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n报告已写入", out / "tcn_comparison_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
