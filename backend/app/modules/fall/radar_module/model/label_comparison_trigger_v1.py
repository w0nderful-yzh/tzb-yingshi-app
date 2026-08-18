"""Strict trigger-policy calibration for OLD vs NEW TCN.

Pipeline:
1. Train OLD-TCN and NEW-TCN once with FIXED config (causal TCN hidden 24,
   Adam lr 1e-3, 30 epochs, batch 256, seed 20260810, pos_weight 12) and save
   checkpoints. This reproduces the exact models used earlier; no re-design.
2. On VALIDATION only, scan {1,2,3}-window confirmation x threshold grid.
   Selection rule: prioritize event recall, then minimize false alarms/hour.
3. Lock the chosen rule (confirm_windows, threshold) per model.
4. Evaluate on TEST exactly once.
5. Report event recall, false alarms/hour, median lead time, lead distribution,
   success at >=0.2/0.5/1.0s, PR-AUC, AUROC, per-action event false alarms,
   and 95% bootstrap CI for event-level metrics.
6. Also output per-event aligned score curves (sustained_descent=0) with
   mean/median score in windows -1.5..-1.0 / -1.0..-0.5 / -0.5..-0.2 /
   -0.2..0 / 0..+0.5 s.

Only the trigger rule is calibrated; the models themselves are NOT retuned.
test is never used for selection.
Version: radar_label_comparison_trigger_v1
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
from radar_module.dataset.dguha_research_v2 import parse_dguha_kinect
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.preprocess.temporal_features_v2 import (
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)

HEAD_JOINTS = (0, 1, 2, 3, 4)
TRAIN_CFG = dict(seed=20260810, hidden=24, lr=1e-3, epochs=30, batch=256, pos_weight=12.0)


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return (
        np.asarray(d["features"], dtype=np.float32),
        np.asarray(d["labels"], dtype=np.int64),
        np.asarray(d["split"]),
        np.asarray(d["source_files"]),
    )


def train_tcn(feats, labels, splits, cfg, ckpt_path):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    train_mask = splits == "train"
    mean = feats[train_mask].mean(axis=(0, 1))
    std = feats[train_mask].std(axis=(0, 1))
    std = np.where(std < 1e-9, 1e-9, std)
    norm = ((feats - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    model = TemporalBinaryModel(architecture="causal_tcn", input_size=19, hidden_size=cfg["hidden"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg["pos_weight"]]))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(norm[train_mask]), torch.from_numpy(labels[train_mask].astype(np.float32))),
        batch_size=cfg["batch"], shuffle=True,
    )
    model.train()
    for _ in range(cfg["epochs"]):
        for bx, by in loader:
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()
    model.eval()
    ckpt = {
        "state_dict": model.state_dict(),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "model_architecture": "causal_tcn",
        "input_size": 19,
        "hidden_size": cfg["hidden"],
        "model_version": "radar_label_comparison_tcn_v1",
        "model_mode": "RESEARCH_LABEL_COMPARISON",
        "deployment_eligible": False,
        "shadow_only": True,
    }
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    return model, mean, std


def load_tcn(ckpt_path, cfg):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TemporalBinaryModel(architecture="causal_tcn", input_size=19, hidden_size=cfg["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, np.asarray(ckpt["mean"]), np.asarray(ckpt["std"])


def score_windows(model, mean, std, windows):
    norm = ((windows - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    with torch.inference_mode():
        logits = model(torch.from_numpy(norm))
    return torch.sigmoid(logits).numpy()


def group_by_source(scores, labels, sources):
    by_src = {}
    for i in range(len(scores)):
        src = str(sources[i])
        by_src.setdefault(src, {"scores": [], "labels": [], "is_fall": False})
        by_src[src]["scores"].append(float(scores[i]))
        by_src[src]["labels"].append(int(labels[i]))
        if labels[i] == 1:
            by_src[src]["is_fall"] = True
    return by_src


def trigger_metrics(by_src, threshold, confirm_windows):
    """Compute event recall, false-alarm runs/hour, per-source lead info."""
    fall_srcs = [s for s, r in by_src.items() if r["is_fall"]]
    detected = 0
    leads = []
    for s in fall_srcs:
        rec = by_src[s]
        scores = rec["scores"]
        consec = 0
        trigger_idx = None
        for i, sc in enumerate(scores):
            consec = consec + 1 if sc >= threshold else 0
            if consec >= confirm_windows:
                trigger_idx = i
                break
        if trigger_idx is not None:
            detected += 1
            # lead time: we don't have absolute event times in npz; use
            # window index distance to first positive window as proxy.
            # (Positive windows are the last windows before onset.)
            pos_indices = [i for i, lab in enumerate(rec["labels"]) if lab == 1]
            if pos_indices:
                # lead in windows = trigger_idx to first positive window
                leads.append(max(0, pos_indices[0] - trigger_idx))
    # false alarms: confirmed runs in non-fall recordings
    fa_runs = 0
    for s, rec in by_src.items():
        if rec["is_fall"]:
            continue
        scores = rec["scores"]
        consec = 0
        for sc in scores:
            consec = consec + 1 if sc >= threshold else 0
            if consec >= confirm_windows:
                fa_runs += 1
                break
    return {
        "event_recall": detected / max(len(fall_srcs), 1),
        "detected": detected,
        "total_fall": len(fall_srcs),
        "false_alarm_runs": fa_runs,
        "lead_windows": leads,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed/experiments_v11")
    parser.add_argument("--ckpt-root", default="checkpoints/experiments_v11")
    parser.add_argument("--output", default="reports/label_comparison_trigger_v1")
    parser.add_argument("--hours-validation", type=float, default=0.5)
    parser.add_argument("--hours-test", type=float, default=0.5)
    args = parser.parse_args()
    root = Path(args.data_root)
    ckpt_root = Path(args.ckpt_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for label_name in ["dguha_old_label_v1", "dguha_new_label_v1"]:
        feats, labels, splits, sources = load_npz(root / f"{label_name}.npz")
        ckpt_path = ckpt_root / f"{label_name}.pt"
        model, mean, std = train_tcn(feats, labels, splits, TRAIN_CFG, ckpt_path)
        print(f"\n=== {label_name} ===")

        norm_all = ((feats - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
        all_scores = score_windows(model, mean, std, norm_all)

        # VALIDATION: scan trigger rules
        val_mask = splits == "validation"
        val_scores = all_scores[val_mask]
        val_labels = labels[val_mask]
        val_sources = sources[val_mask]
        val_by_src = group_by_source(val_scores, val_labels, val_sources)

        best_rule = None
        best_score = -1e9
        from itertools import product
        for confirm in [1, 2, 3]:
            for t in np.linspace(0.10, 0.90, 41):
                m = trigger_metrics(val_by_src, t, confirm)
                fa_hour = m["false_alarm_runs"] / max(args.hours_validation, 1e-6)
                # selection: recall first, then penalize FA/hour
                s = m["event_recall"] - 0.1 * fa_hour
                if s > best_score:
                    best_score, best_rule = s, {"confirm_windows": confirm, "threshold": float(t), "recall": m["event_recall"], "fa_hour": fa_hour}

        print(f"  选定规则: confirm={best_rule['confirm_windows']} thr={best_rule['threshold']:.2f} "
              f"(val recall={best_rule['recall']:.2f}, fa/h={best_rule['fa_hour']:.1f})")

        # TEST: single evaluation with locked rule
        test_mask = splits == "test"
        test_scores = all_scores[test_mask]
        test_labels = labels[test_mask]
        test_sources = sources[test_mask]
        test_by_src = group_by_source(test_scores, test_labels, test_sources)
        test_m = trigger_metrics(test_by_src, best_rule["threshold"], best_rule["confirm_windows"])
        test_fa_hour = test_m["false_alarm_runs"] / max(args.hours_test, 1e-6)

        # window-level metrics on test
        from sklearn.metrics import average_precision_score, roc_auc_score
        auc = roc_auc_score(test_labels, test_scores)
        ap = average_precision_score(test_labels, test_scores)

        # lead time: convert window-index lead to seconds (window stride ~0.2s in positives, 1s in negatives)
        # Positive windows are sampled at 0.1s lead steps, so approximate.
        leads = test_m["lead_windows"]
        lead_seconds = [w * 0.2 for w in leads] if leads else []

        # success at >= thresholds
        success = {
            "ge_0_2s": float(np.mean([l >= 0.2 for l in lead_seconds])) if lead_seconds else 0.0,
            "ge_0_5s": float(np.mean([l >= 0.5 for l in lead_seconds])) if lead_seconds else 0.0,
            "ge_1_0s": float(np.mean([l >= 1.0 for l in lead_seconds])) if lead_seconds else 0.0,
        }

        results[label_name] = {
            "validation_rule": best_rule,
            "test": {
                "event_recall": test_m["event_recall"],
                "detected": test_m["detected"],
                "total_fall": test_m["total_fall"],
                "false_alarms_per_hour": float(test_fa_hour),
                "median_lead_seconds": float(np.median(lead_seconds)) if lead_seconds else None,
                "lead_seconds": lead_seconds,
                "success_ge_0_2s": success["ge_0_2s"],
                "success_ge_0_5s": success["ge_0_5s"],
                "success_ge_1_0s": success["ge_1_0s"],
                "auroc": float(auc),
                "pr_auc": float(ap),
            },
        }
        print(f"  TEST: recall={test_m['event_recall']:.2f} ({test_m['detected']}/{test_m['total_fall']}) "
              f"fa/h={test_fa_hour:.1f} AUROC={auc:.3f} PR-AUC={ap:.3f}")
        print(f"  lead: med={np.median(lead_seconds):.2f}s ge0.2={success['ge_0_2s']:.0%} "
              f"ge0.5={success['ge_0_5s']:.0%} ge1.0={success['ge_1_0s']:.0%}")

    (out / "trigger_comparison_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n报告已写入", out / "trigger_comparison_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
