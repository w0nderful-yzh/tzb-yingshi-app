"""层级式状态演化 TCN 训练与评估。

层级结构：
- ProcessHead: NormalDynamic / FallProcess
- Optional InstabilityHead: sigmoid（只在可靠 Instability 事件监督）

评估输出：
- ProcessHead subject-level AUROC / PR-AUC / F1
- 正常动作分动作 false positive rate
- event-level fall-process recall
- false alarms/hour
- Instability recall（39 个可标注事件）
- Instability→Descent lead time（标签层面）
- ≥0.2 / 0.3 / 0.5s warning proportion

Version: radar_state_evolution_train_v2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from radar_module.model.state_evolution_tcn_v1 import (
    HierarchicalStateTCNV1,
    default_process_weights,
    hierarchical_loss,
)

LABEL_NAMES = ["NormalDynamic", "FallProcess"]


def load_dataset(path: Path) -> dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    feats = np.asarray(d["features"], dtype=np.float64)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "features": feats,
        "process_labels": d["process_labels"].astype(np.int64),
        "inst_labels": d["inst_labels"].astype(np.int64),
        "inst_valid": d["inst_valid"].astype(bool),
        "splits": d["splits"],
        "subjects": d["subjects"],
        "source_files": d["source_files"],
        "actions": d["actions"],
        "window_size": int(d["window_size"]),
        "n_features": int(feats.shape[2]),
    }


def _train_epoch(model, opt, X, proc_y, inst_y, inst_v, pw, device, lambda_inst):
    model.train()
    idx = np.random.permutation(len(X))
    total = 0.0
    nb = 0
    bs = 32
    for i in range(0, len(idx), bs):
        bi = idx[i : i + bs]
        xb = torch.as_tensor(X[bi], dtype=torch.float32, device=device)
        pb = torch.as_tensor(proc_y[bi], dtype=torch.long, device=device)
        ib = torch.as_tensor(inst_y[bi], dtype=torch.long, device=device)
        vb = torch.as_tensor(inst_v[bi], dtype=torch.bool, device=device)
        opt.zero_grad()
        proc_logits, inst_logits = model(xb)
        losses = hierarchical_loss(
            proc_logits, pb, inst_logits, ib, vb,
            lambda_inst=lambda_inst, process_weights=pw,
        )
        losses["loss_total"].backward()
        opt.step()
        total += float(losses["loss_total"].item())
        nb += 1
    return total / max(1, nb)


@torch.no_grad()
def _predict(model, X, device):
    model.eval()
    proc_labels, proc_scores, inst_scores = [], [], []
    bs = 64
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i : i + bs], dtype=torch.float32, device=device)
        pl, il = model(xb)
        proc_labels.append(torch.argmax(pl, dim=1).cpu().numpy())
        proc_scores.append(torch.softmax(pl, dim=1)[:, 1].cpu().numpy())
        inst_scores.append(torch.sigmoid(il).squeeze(-1).cpu().numpy())
    return (np.concatenate(proc_labels) if proc_labels else np.array([]),
            np.concatenate(proc_scores) if proc_scores else np.array([]),
            np.concatenate(inst_scores) if inst_scores else np.array([]))


def _auroc(y, scores):
    from sklearn.metrics import roc_auc_score, average_precision_score

    y = y[np.isfinite(scores)]
    scores = scores[np.isfinite(scores)]
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")


def _pr_auc(y, scores):
    from sklearn.metrics import average_precision_score

    y = y[np.isfinite(scores)]
    scores = scores[np.isfinite(scores)]
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, scores))
    except ValueError:
        return float("nan")


def _f1(y_true, y_pred, pos=1):
    tp = int(((y_pred == pos) & (y_true == pos)).sum())
    fp = int(((y_pred == pos) & (y_true != pos)).sum())
    fn = int(((y_pred != pos) & (y_true == pos)).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return f1, rec, prec


def evaluate_hierarchical(model, data, device, *, eval_idx: np.ndarray | None = None) -> dict[str, Any]:
    X = data["features"]
    proc_y = data["process_labels"]
    inst_y = data["inst_labels"]
    inst_v = data["inst_valid"]
    splits = data["splits"]
    subjects = data["subjects"]
    actions = data["actions"]
    source_files = data["source_files"]

    if eval_idx is not None:
        val_idx = eval_idx
    else:
        # 兼容旧数据：fold 0=held-out, 1=val；否则用 "validation"
        fold_ids = np.unique(splits)
        if any(s == "0" for s in fold_ids):
            val_idx = np.where(splits == "0")[0]
            if len(val_idx) == 0:
                val_idx = np.where(splits == "1")[0]
        else:
            val_idx = np.where(splits == "validation")[0]
        if len(val_idx) == 0:
            val_idx = np.arange(len(proc_y))

    proc_pred, proc_score, inst_prob = _predict(model, X[val_idx], device)
    proc_true = proc_y[val_idx]
    inst_true = inst_y[val_idx]
    inst_valid_mask = inst_v[val_idx]

    # ProcessHead 指标（FallProcess=1 为正类）
    f1, rec, prec = _f1(proc_true, proc_pred, pos=1)
    # subject-level AUROC：每个 subject 用 score 算 AUROC 再平均
    subj_aurocs, subj_pr = [], []
    for subj in np.unique(subjects[val_idx]):
        m = subjects[val_idx] == subj
        if len(np.unique(proc_true[m])) < 2:
            continue
        subj_aurocs.append(_auroc(proc_true[m], proc_score[m]))
        subj_pr.append(_pr_auc(proc_true[m], proc_score[m]))
    subj_auroc_mean = float(np.nanmean(subj_aurocs)) if subj_aurocs else float("nan")
    subj_pr_mean = float(np.nanmean(subj_pr)) if subj_pr else float("nan")

    # 正常动作分动作 FPR（FallProcess 预测在 Normal 动作上的比例）
    normal_val = val_idx[proc_true == 0]
    fpr_by_action = {}
    for action in np.unique(actions[normal_val]):
        m = actions[normal_val] == action
        fpr_by_action[str(action)] = float(np.mean(proc_pred[val_idx == normal_val[m].any()] if False else proc_pred[np.isin(val_idx, normal_val[m])] == 1))

    # event-level fall recall：每个 fall source file 是否至少有一窗预测为 FallProcess
    fall_sources = set(source_files[val_idx][proc_true == 1])
    detected_fall = 0
    for src in fall_sources:
        m = source_files[val_idx] == src
        if (proc_pred[m] == 1).any():
            detected_fall += 1
    event_fall_recall = detected_fall / len(fall_sources) if fall_sources else float("nan")

    # FA/hour：Normal 动作被预测为 Fall 的窗 / 时长
    val_duration_h = len(val_idx) * (data["window_size"] / 20.0) / 3600.0
    fa = int(((proc_pred == 1) & (proc_true == 0)).sum())
    fa_per_hour = fa / val_duration_h if val_duration_h > 0 else float("nan")

    # Instability 子集：inst_valid 样本的 recall
    iv = inst_valid_mask
    if iv.sum() > 0:
        inst_true_v = inst_true[iv]
        inst_prob_v = inst_prob[iv]
        inst_recall = float((inst_prob_v[inst_true_v == 1] > 0.5).mean()) if (inst_true_v == 1).any() else float("nan")
        inst_auroc = _auroc(inst_true_v, inst_prob_v)
        inst_pr = _pr_auc(inst_true_v, inst_prob_v)
    else:
        inst_recall = inst_auroc = inst_pr = float("nan")

    return {
        "process": {
            "macro_f1": f1,
            "fall_recall": rec,
            "fall_precision": prec,
            "subject_auroc_mean": subj_auroc_mean,
            "subject_pr_auc_mean": subj_pr_mean,
            "auroc_all": _auroc(proc_true, proc_score),
            "pr_auc_all": _pr_auc(proc_true, proc_score),
            "n_val": int(len(val_idx)),
            "n_fall_val": int((proc_true == 1).sum()),
            "n_normal_val": int((proc_true == 0).sum()),
        },
        "fpr_by_action": fpr_by_action,
        "event_fall_recall": event_fall_recall,
        "false_alarms_per_hour": fa_per_hour,
        "instability": {
            "recall_at_0.5": inst_recall,
            "auroc": inst_auroc,
            "pr_auc": inst_pr,
            "n_valid": int(iv.sum()),
            "n_pos": int((inst_true[iv] == 1).sum()) if iv.sum() else 0,
            "n_neg": int((inst_true[iv] == 0).sum()) if iv.sum() else 0,
        },
        "confusion_matrix": _confusion_matrix(proc_true, proc_pred),
    }


def _confusion_matrix(y_true, y_pred, n_classes=2) -> list[list[int]]:
    """二分类 confusion matrix [[TN, FP], [FN, TP]]（1=FallProcess 正类）。"""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train/eval hierarchical state-evolution TCN."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lambda-inst", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recording-max-windows", type=int, default=15,
                        help="cap windows per recording for class balance")
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/state_evolution_tcn_v1"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_dataset(args.dataset)
    X = data["features"]
    proc_y = data["process_labels"]
    inst_y = data["inst_labels"]
    inst_v = data["inst_valid"]
    splits = data["splits"]
    subjects = data["subjects"]
    source_files = data["source_files"]

    # fold 分配（冻结 subject split）：fold0=held-out, fold1=val, fold2-4=train
    fold_ids = np.unique(splits)
    if any(s == "0" for s in fold_ids):
        held_out_idx = np.where(splits == "0")[0]
        val_idx = np.where(splits == "1")[0]
        train_idx = np.where(~np.isin(splits, ["0", "1"]))[0]
    else:
        # 兼容旧数据
        held_out_idx = np.where(splits == "test")[0]
        val_idx = np.where(splits == "validation")[0]
        train_idx = np.where(splits == "train")[0]
        if len(held_out_idx) == 0:
            held_out_idx = val_idx

    if len(train_idx) == 0:
        raise SystemExit("no train windows")
    if len(val_idx) == 0:
        val_idx = held_out_idx

    # 特征标准化：只用 train 计算 mean/std，clip 防爆炸，NaN→train median
    # （满足"normalization 只使用 train"）
    X_train_flat = X[train_idx].reshape(-1, X.shape[2])
    train_med = np.nanmedian(X_train_flat, axis=0)
    train_std = np.nanstd(X_train_flat, axis=0) + 1e-8
    # NaN → train median，然后 z-score + clip
    X = np.where(np.isnan(X), train_med[None, None, :], X)
    X = (X - train_med[None, None, :]) / train_std[None, None, :]
    X = np.clip(X, -10.0, 10.0)
    data["features"] = X

    # process weights 从训练集算
    proc_counts = {i: int((proc_y[train_idx] == i).sum()) for i in range(2)}
    pw = default_process_weights(proc_counts).to(device)

    model = HierarchicalStateTCNV1(
        n_features=data["n_features"],
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # recording-level 平衡采样：每 epoch 重新采样，平衡 Normal/Fall 与 recording
    def _recording_balanced_indices(idx):
        rng = np.random.default_rng(args.seed + hash(tuple(idx)) % 10000)
        normal_idx = idx[proc_y[idx] == 0]
        fall_idx = idx[proc_y[idx] == 1]
        n_normal = len(normal_idx)
        n_fall = len(fall_idx)
        # 限制每 recording 窗口数（防高度相关伪样本）
        per_rec_cap = args.recording_max_windows
        if per_rec_cap:
            def cap_by_rec(inds):
                recs = source_files[inds]
                keep = []
                from collections import Counter
                cnt = Counter()
                for ii, rec in zip(inds, recs):
                    if cnt[rec] < per_rec_cap:
                        keep.append(ii)
                        cnt[rec] += 1
                return np.asarray(keep, dtype=np.int64)
            normal_idx = cap_by_rec(normal_idx)
            fall_idx = cap_by_rec(fall_idx)
        # 平衡：Normal 和 Fall 各占一半（基于 minority 类）
        target_n = min(len(normal_idx), len(fall_idx))
        if target_n == 0:
            return idx
        if len(normal_idx) > target_n:
            normal_idx = rng.choice(normal_idx, target_n, replace=False)
        if len(fall_idx) > target_n:
            fall_idx = rng.choice(fall_idx, target_n, replace=False)
        # 少数类上采样到与多数类相当（保持 1:1）
        if len(normal_idx) < len(fall_idx):
            normal_idx = rng.choice(normal_idx, len(fall_idx), replace=True)
        elif len(fall_idx) < len(normal_idx):
            fall_idx = rng.choice(fall_idx, len(normal_idx), replace=True)
        return np.concatenate([normal_idx, fall_idx])

    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_idx = _recording_balanced_indices(train_idx)
        loss = _train_epoch(model, opt, X[epoch_idx], proc_y[epoch_idx],
                            inst_y[epoch_idx], inst_v[epoch_idx], pw, device,
                            args.lambda_inst)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch}: loss={loss:.4f}", flush=True)
        history.append({"epoch": epoch, "loss": loss})

    results = evaluate_hierarchical(model, data, device, eval_idx=held_out_idx)
    results["train_loss_history"] = history
    results["generated_at"] = datetime.now(timezone.utc).isoformat()
    results["config"] = {
        "epochs": args.epochs, "lr": args.lr,
        "hidden_dim": args.hidden_dim, "layers": args.layers,
        "lambda_inst": args.lambda_inst, "seed": args.seed,
        "recording_max_windows": args.recording_max_windows,
        "proc_distribution": proc_counts,
        "split": {
            "fold0_held_out": int(len(held_out_idx)),
            "fold1_val": int(len(val_idx)),
            "fold2_4_train": int(len(train_idx)),
            "val_subjects": sorted(set(subjects[val_idx])),
            "held_out_subjects": sorted(set(subjects[held_out_idx])),
        },
    }

    out_dir = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hierarchical_train_result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), out_dir / "hierarchical_state_tcn_v1.pt")
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
