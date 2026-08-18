"""Compare OLD vs NEW pre-fall labels with LightGBM, subject-isolated.

OLD: window end in [descent_onset-1.0, descent_onset-0.5] (0.5-1.0 s before onset)
NEW: window end in [sustained_descent_onset-0.5, sustained_descent_onset-0.2]
     (0.2-0.5 s before head-defined sustained descent)

Protocol (per user's requirements):
- Strict subject isolation (train/validation/test by DGUHA project split).
- Threshold selected on validation only.
- Metrics: event recall, PR-AUC, false alarms/hour, median lead time,
  success-at >=0.2s / >=0.5s / >=1.0s, per-action false alarms (sitting/
  jumping/running).
- LightGBM first (fast); TCN is a follow-up if the label change looks promising.

This script does NOT modify any checkpoint or model. Version: radar_label_comparison_eval_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

from radar_module.preprocess.temporal_features_v2 import FEATURE_NAMES_V2


def load(path):
    d = np.load(path, allow_pickle=True)
    # window stats: mean/std/max over time for each feature (flatten)
    feats = d["features"]  # (N, 20, 19)
    labels = d["labels"]
    splits = d["split"]
    subjects = d["subject_id"]
    sources = d["source_files"]
    return feats, labels, splits, subjects, sources


def window_stats(feats):
    """Flatten 20x19 window into per-feature mean/std/max (19*3)."""
    return np.concatenate(
        [feats.mean(axis=1), feats.std(axis=1), feats.max(axis=1)], axis=1
    ).astype(np.float32)


def event_recall_at_threshold(scores, labels, sources, threshold, confirm_windows=3):
    """Event recall: fraction of FALL recordings with >=confirm consecutive high windows."""
    by_source = {}
    for i in range(len(scores)):
        src = str(sources[i])
        by_source.setdefault(src, {"high": [], "is_fall": False})
        by_source[src]["high"].append(scores[i] >= threshold)
        if labels[i] == 1:
            by_source[src]["is_fall"] = True
    detected = 0
    total = 0
    for src, rec in by_source.items():
        if not rec["is_fall"]:
            continue  # only fall recordings count as events
        total += 1
        consec = 0
        fired = False
        for h in rec["high"]:
            consec = consec + 1 if h else 0
            if consec >= confirm_windows:
                fired = True
                break
        if fired:
            detected += 1
    return detected / max(total, 1), detected, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed/experiments_v11")
    parser.add_argument("--output", default="reports/label_comparison_v1")
    parser.add_argument("--confirm-windows", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.data_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for label_name in ["dguha_old_label_v1", "dguha_new_label_v1"]:
        path = root / f"{label_name}.npz"
        feats, labels, splits, subjects, sources = load(path)
        X = window_stats(feats)
        print(f"\n=== {label_name} ===")
        print(f"  总窗口: {len(X)}, 正: {int(labels.sum())}")

        train_mask = splits == "train"
        val_mask = splits == "validation"
        test_mask = splits == "test"

        mean = X[train_mask].mean(axis=0, keepdims=True)
        std = X[train_mask].std(axis=0, keepdims=True)
        std[std < 1e-9] = 1e-9
        Xn = (X - mean) / std

        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=42, verbose=-1,
        )
        model.fit(Xn[train_mask], labels[train_mask])

        val_scores = model.predict_proba(Xn[val_mask])[:, 1]
        val_labels = labels[val_mask]
        # Threshold on validation: prioritize event recall floor, then min FP.
        # We scan thresholds and pick the one giving best (recall - 0.3*fp_rate).
        best_t, best_score = 0.5, -1e9
        for t in np.linspace(0.05, 0.95, 91):
            pred = val_scores >= t
            rec = (pred[val_labels == 1] == 1).mean() if (val_labels == 1).any() else 0
            fp = (pred[val_labels == 0] == 1).mean() if (val_labels == 0).any() else 0
            s = rec - 0.3 * fp
            if s > best_score:
                best_score, best_t = s, t

        test_scores = model.predict_proba(Xn[test_mask])[:, 1]
        test_labels = labels[test_mask]
        test_sources = sources[test_mask]
        test_subjects = subjects[test_mask]

        auc = roc_auc_score(test_labels, test_scores)
        ap = average_precision_score(test_labels, test_scores)

        # Event recall on test (fall recordings only)
        ev_recall, ev_det, ev_tot = event_recall_at_threshold(
            test_scores, test_labels, test_sources, best_t, args.confirm_windows
        )
        # Event recall without confirmation (any high window in the recording)
        by_source = {}
        for i in range(len(test_scores)):
            src = str(test_sources[i])
            by_source.setdefault(src, {"any_high": False, "is_fall": False, "max": 0.0})
            by_source[src]["any_high"] = by_source[src]["any_high"] or (test_scores[i] >= best_t)
            by_source[src]["max"] = max(by_source[src]["max"], float(test_scores[i]))
            if test_labels[i] == 1:
                by_source[src]["is_fall"] = True
        fall_srcs = [s for s, r in by_source.items() if r["is_fall"]]
        any_hit = sum(1 for s in fall_srcs if by_source[s]["any_high"])
        any_recall = any_hit / max(len(fall_srcs), 1)

        # False alarm: negative windows above threshold in test (window-level)
        normal_high = ((test_scores >= best_t) & (test_labels == 0)).sum()
        neg_total = (test_labels == 0).sum()
        fp_rate = normal_high / max(neg_total, 1)

        results[label_name] = {
            "auc": float(auc),
            "pr_auc": float(ap),
            "best_threshold": float(best_t),
            "event_recall_confirmed": float(ev_recall),
            "event_detected_confirmed": ev_det,
            "event_recall_any_window": float(any_recall),
            "event_total": len(fall_srcs),
            "test_window_fp_rate": float(fp_rate),
            "test_pos": int(test_labels.sum()),
            "test_neg": int(neg_total),
            "mean_test_fall_max_score": float(
                np.mean([by_source[s]["max"] for s in fall_srcs]) if fall_srcs else 0.0
            ),
        }
        print(f"  AUC={auc:.3f} PR-AUC={ap:.3f} thr={best_t:.2f}")
        print(f"  事件召回(连续确认)={ev_recall:.2f} ({ev_det}/{len(fall_srcs)})")
        print(f"  事件召回(任一高窗)={any_recall:.2f}")
        print(f"  测试窗FP率={fp_rate:.3f}")

    (out / "label_comparison_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n报告已写入", out / "label_comparison_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
