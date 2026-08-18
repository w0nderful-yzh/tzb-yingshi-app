"""Strict event-level validation of OLD vs NEW labels with LightGBM.

Purpose: verify whether the NEW label (0.2-0.5s before sustained descent) is
genuinely better than OLD (0.5-1.0s before onset) at the *event* level, under
a strict protocol. LightGBM is used because it is stable on small samples;
it is NOT a deployment model and NOT a replacement for the TCN.

Protocol:
- Subject-isolated splits from the npz (train/validation/test unchanged).
- Threshold + trigger rule (confirm windows) selected on validation only.
- Test evaluated exactly once.
- OLD and NEW use the identical selection protocol.

Outputs:
- event recall, false alarms/hour, median lead time, success at >=0.2/0.5/1.0s,
  PR-AUC, AUROC, per-action event false alarms (sitting/jumping/running).

Version: radar_label_comparison_lgb_events_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return (
        np.asarray(d["features"], dtype=np.float32),
        np.asarray(d["labels"], dtype=np.int64),
        np.asarray(d["split"]),
        np.asarray(d["source_files"]),
    )


def window_stats(feats):
    """Flatten 20x19 into per-feature mean/std/max (19*3)."""
    return np.concatenate(
        [feats.mean(axis=1), feats.std(axis=1), feats.max(axis=1)], axis=1
    ).astype(np.float32)


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
    fall_srcs = [s for s, r in by_src.items() if r["is_fall"]]
    detected = 0
    lead_windows = []
    for s in fall_srcs:
        rec = by_src[s]
        consec = 0
        trig = None
        for i, sc in enumerate(rec["scores"]):
            consec = consec + 1 if sc >= threshold else 0
            if consec >= confirm_windows:
                trig = i
                break
        if trig is not None:
            detected += 1
            pos_idx = [i for i, lab in enumerate(rec["labels"]) if lab == 1]
            if pos_idx:
                lead_windows.append(max(0, pos_idx[0] - trig))
    fa_runs = 0
    for s, rec in by_src.items():
        if rec["is_fall"]:
            continue
        consec = 0
        for sc in rec["scores"]:
            consec = consec + 1 if sc >= threshold else 0
            if consec >= confirm_windows:
                fa_runs += 1
                break
    return {
        "event_recall": detected / max(len(fall_srcs), 1),
        "detected": detected,
        "total_fall": len(fall_srcs),
        "false_alarm_runs": fa_runs,
        "lead_windows": lead_windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed/experiments_v11")
    parser.add_argument("--output", default="reports/label_comparison_lgb_events_v1")
    parser.add_argument("--val-hours", type=float, default=0.5)
    parser.add_argument("--test-hours", type=float, default=0.5)
    args = parser.parse_args()
    root = Path(args.data_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for label_name in ["dguha_old_label_v1", "dguha_new_label_v1"]:
        feats, labels, splits, sources = load_npz(root / f"{label_name}.npz")
        X = window_stats(feats)
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
        val_sources = sources[val_mask]
        val_by_src = group_by_source(val_scores, val_labels, val_sources)

        # Select threshold + confirm windows on validation.
        # Priority: recall first (maximize), then among ties prefer lower FA/hour.
        # This is the same protocol for OLD and NEW.
        best = None
        best_recall = -1.0
        best_fa_hour = 1e9
        for confirm in [1, 2, 3]:
            for t in np.linspace(0.10, 0.90, 41):
                m = trigger_metrics(val_by_src, t, confirm)
                fa_hour = m["false_alarm_runs"] / max(args.val_hours, 1e-6)
                # tie-break: higher recall wins; if recall equal, lower FA/hour wins
                if m["event_recall"] > best_recall or (
                    abs(m["event_recall"] - best_recall) < 1e-9 and fa_hour < best_fa_hour
                ):
                    best_recall = m["event_recall"]
                    best_fa_hour = fa_hour
                    best = {
                        "confirm_windows": confirm, "threshold": float(t),
                        "recall": m["event_recall"], "fa_hour": float(fa_hour),
                    }

        # TEST single evaluation
        test_scores = model.predict_proba(Xn[test_mask])[:, 1]
        test_labels = labels[test_mask]
        test_sources = sources[test_mask]
        test_by_src = group_by_source(test_scores, test_labels, test_sources)
        tm = trigger_metrics(test_by_src, best["threshold"], best["confirm_windows"])
        test_fa_hour = tm["false_alarm_runs"] / max(args.test_hours, 1e-6)
        auc = roc_auc_score(test_labels, test_scores)
        ap = average_precision_score(test_labels, test_scores)

        leads = [w * 0.2 for w in tm["lead_windows"]]
        success = {
            "ge_0_2s": float(np.mean([l >= 0.2 for l in leads])) if leads else 0.0,
            "ge_0_5s": float(np.mean([l >= 0.5 for l in leads])) if leads else 0.0,
            "ge_1_0s": float(np.mean([l >= 1.0 for l in leads])) if leads else 0.0,
        }

        # per-action false alarms: use source filename to identify action
        action_fa = {"sitting": 0, "jumping": 0, "running": 0}
        action_total = {"sitting": 0, "jumping": 0, "running": 0}
        for src, rec in test_by_src.items():
            if rec["is_fall"]:
                continue
            if "3_Sit" in src:
                act = "sitting"
            elif "2_Jump" in src:
                act = "jumping"
            elif "1_Run" in src:
                act = "running"
            else:
                continue
            action_total[act] += 1
            consec = 0
            for sc in rec["scores"]:
                consec = consec + 1 if sc >= best["threshold"] else 0
                if consec >= best["confirm_windows"]:
                    action_fa[act] += 1
                    break

        results[label_name] = {
            "validation_rule": best,
            "test": {
                "event_recall": tm["event_recall"],
                "detected": tm["detected"],
                "total_fall": tm["total_fall"],
                "false_alarms_per_hour": float(test_fa_hour),
                "median_lead_seconds": float(np.median(leads)) if leads else None,
                "lead_seconds": leads,
                "success_ge_0_2s": success["ge_0_2s"],
                "success_ge_0_5s": success["ge_0_5s"],
                "success_ge_1_0s": success["ge_1_0s"],
                "auroc": float(auc),
                "pr_auc": float(ap),
                "action_false_alarms": action_fa,
                "action_recordings": action_total,
            },
        }
        print(f"\n=== {label_name} ===")
        print(f"  规则: confirm={best['confirm_windows']} thr={best['threshold']:.2f}")
        print(f"  事件召回={tm['event_recall']:.2f} ({tm['detected']}/{tm['total_fall']}) fa/h={test_fa_hour:.1f}")
        lead_med = np.median(leads) if leads else float('nan')
        print(f"  lead: med={lead_med:.2f}s ge0.2={success['ge_0_2s']:.0%} "
              f"ge0.5={success['ge_0_5s']:.0%} ge1.0={success['ge_1_0s']:.0%}")
        print(f"  AUROC={auc:.3f} PR-AUC={ap:.3f}")
        print(f"  分动作误报: {action_fa} / {action_total}")

    (out / "lgb_events_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n报告已写入", out / "lgb_events_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
