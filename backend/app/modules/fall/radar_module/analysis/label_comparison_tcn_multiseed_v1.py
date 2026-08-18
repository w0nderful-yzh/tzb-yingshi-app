"""Multi-seed OLD/NEW TCN summary: ranking, calibration, events, stability.

Splits the problem into four independent axes:
1. Ranking ability      -> AUROC, PR-AUC (threshold-free)
2. Probability calibration -> Brier score, ECE, positive/negative median/IQR
3. Event triggering     -> event recall, false alarms/hour at a validation-chosen
                            fixed threshold
4. Cross-seed stability -> spread of AUROC/PR-AUC/calibration across seeds

Important: a model can have good ranking but collapsed probabilities. We report
both explicitly rather than conflating them. A model with collapsed probs is
not suitable for fixed-threshold event triggering, even if AUROC looks high.

This is analysis only; no checkpoint or model is modified.
Version: radar_label_comparison_tcn_multiseed_v1
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

SEEDS = [20260810, 1, 42, 123, 999]
CONF = dict(hidden=24, lr=1e-3, epochs=30, batch=256, pos_weight=12.0)


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return (
        np.asarray(d["features"], dtype=np.float32),
        np.asarray(d["labels"], dtype=np.int64),
        np.asarray(d["split"]),
        np.asarray(d["source_files"]),
    )


def train_tcn(feats, labels, splits, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_mask = splits == "train"
    mean = feats[train_mask].mean(axis=(0, 1))
    std = feats[train_mask].std(axis=(0, 1))
    std = np.where(std < 1e-9, 1e-9, std)
    norm = ((feats - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    model = TemporalBinaryModel(architecture="causal_tcn", input_size=19, hidden_size=CONF["hidden"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CONF["pos_weight"]]))
    opt = torch.optim.Adam(model.parameters(), lr=CONF["lr"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(norm[train_mask]), torch.from_numpy(labels[train_mask].astype(np.float32))),
        batch_size=CONF["batch"], shuffle=True,
    )
    model.train()
    for _ in range(CONF["epochs"]):
        for bx, by in loader:
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()
    model.eval()
    return model, mean, std


def score_all(model, mean, std, feats):
    norm = ((feats - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    with torch.inference_mode():
        logits = model(torch.from_numpy(norm))
    return torch.sigmoid(logits).numpy()


def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(y_prob)) * abs(conf - acc)
    return float(ece)


def event_metrics(scores, labels, sources, threshold, confirm_windows):
    by_src = {}
    for i in range(len(scores)):
        src = str(sources[i])
        by_src.setdefault(src, {"high": [], "is_fall": False})
        by_src[src]["high"].append(scores[i] >= threshold)
        if labels[i] == 1:
            by_src[src]["is_fall"] = True
    fall_srcs = [s for s, r in by_src.items() if r["is_fall"]]
    detected = 0
    for s in fall_srcs:
        consec = 0
        for h in by_src[s]["high"]:
            consec = consec + 1 if h else 0
            if consec >= confirm_windows:
                detected += 1
                break
    # false alarms: confirmed runs in non-fall recordings
    fa_runs = 0
    for s, r in by_src.items():
        if r["is_fall"]:
            continue
        consec = 0
        for h in r["high"]:
            consec = consec + 1 if h else 0
            if consec >= confirm_windows:
                fa_runs += 1
                break
    return detected / max(len(fall_srcs), 1), fa_runs, len(fall_srcs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed/experiments_v11")
    parser.add_argument("--output", default="reports/label_comparison_tcn_multiseed_v1")
    parser.add_argument("--confirm-windows", type=int, default=3)
    parser.add_argument("--val-hours", type=float, default=0.5)
    args = parser.parse_args()
    root = Path(args.data_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import average_precision_score, roc_auc_score

    summary = {}
    for label_name in ["dguha_old_label_v1", "dguha_new_label_v1"]:
        feats, labels, splits, sources = load_npz(root / f"{label_name}.npz")
        val_mask = splits == "validation"
        test_mask = splits == "test"
        seed_results = []
        for seed in SEEDS:
            model, mean, std = train_tcn(feats, labels, splits, seed)
            all_scores = score_all(model, mean, std, feats)
            val_scores = all_scores[val_mask]
            val_labels = labels[val_mask]
            # threshold on validation: prioritize recall, penalize FA
            best_t, best_s = 0.5, -1e9
            for t in np.linspace(0.05, 0.95, 91):
                rec, fa, _ = event_metrics(val_scores, val_labels, sources[val_mask], t, args.confirm_windows)
                fa_hour = fa / max(args.val_hours, 1e-6)
                s = rec - 0.1 * fa_hour
                if s > best_s:
                    best_s, best_t = s, t

            test_scores = all_scores[test_mask]
            test_labels = labels[test_mask]
            test_sources = sources[test_mask]
            auc = roc_auc_score(test_labels, test_scores)
            ap = average_precision_score(test_labels, test_scores)
            brier = brier_score(test_labels.astype(float), test_scores)
            ece = expected_calibration_error(test_labels.astype(float), test_scores)
            pos = test_scores[test_labels == 1]
            neg = test_scores[test_labels == 0]
            rec, fa, nfall = event_metrics(test_scores, test_labels, test_sources, best_t, args.confirm_windows)
            seed_results.append({
                "seed": seed,
                "auroc": float(auc), "pr_auc": float(ap),
                "brier": brier, "ece": ece,
                "pos_median": float(np.median(pos)) if len(pos) else None,
                "pos_q1": float(np.percentile(pos, 25)) if len(pos) else None,
                "pos_q3": float(np.percentile(pos, 75)) if len(pos) else None,
                "neg_median": float(np.median(neg)) if len(neg) else None,
                "neg_q1": float(np.percentile(neg, 25)) if len(neg) else None,
                "neg_q3": float(np.percentile(neg, 75)) if len(neg) else None,
                "event_recall": float(rec), "fa_runs": fa, "n_fall": nfall,
                "threshold": float(best_t),
            })
            print(f"  {label_name} seed={seed}: AUROC={auc:.3f} PR-AUC={ap:.3f} "
                  f"Brier={brier:.3f} ECE={ece:.3f} pos_med={np.median(pos) if len(pos) else float('nan'):.4f} "
                  f"neg_med={np.median(neg) if len(neg) else float('nan'):.4f} "
                  f"recall={rec:.2f} fa={fa}")
        # aggregate
        au = [r["auroc"] for r in seed_results]
        pr = [r["pr_auc"] for r in seed_results]
        brier = [r["brier"] for r in seed_results]
        ece = [r["ece"] for r in seed_results]
        rec = [r["event_recall"] for r in seed_results]
        pos_med = [r["pos_median"] for r in seed_results if r["pos_median"] is not None]
        neg_med = [r["neg_median"] for r in seed_results if r["neg_median"] is not None]
        summary[label_name] = {
            "seeds": seed_results,
            "aggregate": {
                "auroc_mean": float(np.mean(au)), "auroc_std": float(np.std(au)),
                "pr_auc_mean": float(np.mean(pr)), "pr_auc_std": float(np.std(pr)),
                "brier_mean": float(np.mean(brier)),
                "ece_mean": float(np.mean(ece)),
                "event_recall_mean": float(np.mean(rec)),
                "pos_median_cross_seed": float(np.mean(pos_med)) if pos_med else None,
                "neg_median_cross_seed": float(np.mean(neg_med)) if neg_med else None,
                "pos_neg_gap": float(np.mean(pos_med) - np.mean(neg_med)) if pos_med and neg_med else None,
            },
        }
    (out / "tcn_multiseed_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n报告已写入", out / "tcn_multiseed_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
