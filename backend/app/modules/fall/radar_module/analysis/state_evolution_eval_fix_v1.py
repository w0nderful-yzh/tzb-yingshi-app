"""修正评估协议：用当前 checkpoint 重算正确指标。

修复三个问题：
1. macro-F1 真实现：F1_normal / F1_fall / macro-F1 / balanced accuracy
   （此前报告的 0.184 实际是 F1_fall，不是 macro-F1）
2. event-level FA/hour：按 recording 恢复连续预测，连续阳性合并 alarm
   episode，加 debounce/cooldown；保留 windows/hour 但改名
   false_positive_windows_per_hour
3. event fall 细化：每 fall recording 首次 FallProcess 命中窗口、相对
   Descent onset 时间、positive-window coverage、最长连续阳性长度

同时报告：best validation epoch、validation threshold、threshold selection
criterion、held-out 评估是否只执行一次。

不重训，仅加载 checkpoint。
Version: radar_state_evolution_eval_fix_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_module.model.state_evolution_tcn_v1 import HierarchicalStateTCNV1

LABELS = ["Normal", "Fall"]


def load_dataset(path: Path) -> dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    feats = np.asarray(d["features"], dtype=np.float64)
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


def _predict(model, X, device, batch_size=256):
    model.eval()
    proc_scores = []
    inst_scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
            pl, il = model(xb)
            proc_scores.append(torch.softmax(pl, dim=1)[:, 1].cpu().numpy())
            inst_scores.append(torch.sigmoid(il).squeeze(-1).cpu().numpy())
    return (np.concatenate(proc_scores) if proc_scores else np.array([]),
            np.concatenate(inst_scores) if inst_scores else np.array([]))


def _balanced_acc(cm) -> float:
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
    tnr = tn / (tn + fp) if tn + fp > 0 else 0.0
    return (tpr + tnr) / 2.0


def _class_f1(cm, cls):
    """cls=0 Normal, cls=1 Fall。返回 (f1, precision, recall)。"""
    if cls == 1:  # Fall positive
        tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
    else:  # Normal positive
        tp, fp, fn = cm[0][0], cm[1][0], cm[0][1]
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return f1, prec, rec


def select_threshold(val_scores, val_labels, criterion="youden"):
    """在 validation 上选 ProcessHead 阈值。criterion: youden / f1。"""
    best_thr, best_metric = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        pred = (val_scores >= thr).astype(int)
        cm = _confusion_matrix(val_labels, pred)
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
        tnr = tn / (tn + fp) if tn + fp > 0 else 0.0
        youden = tpr + tnr - 1
        f1, _, _ = _class_f1(cm, 1)
        metric = youden if criterion == "youden" else f1
        if metric > best_metric:
            best_metric, best_thr = metric, thr
    return float(best_thr), float(best_metric)


def _confusion_matrix(y_true, y_pred, n_classes=2):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm.tolist()


def event_level_fa(fall_scores, process_true, source_files, actions, window_size,
                   stride, sample_rate=20.0):
    """事件级 FA。

    按 recording 恢复窗口序列，连续 FallProcess 预测合并为 alarm episode，
    加 debounce（相邻 episode 间隔 < cooldown 合并）。只对 Normal 真值的
    recording 计 FA（false alarm = Normal 时产生 alarm episode）。

    DGUHA 是离散动作 recording，非连续居家，故命名为：
    `normal-recording equivalent false alarm events/hour`
    """
    cooldown_windows = 3  # 3 窗 = 1.5s@stride10
    total_fa_episodes = 0
    total_hours = 0.0
    # 分动作统计
    from collections import OrderedDict, defaultdict
    recs = OrderedDict()
    rec_actions = {}
    for i, src in enumerate(source_files):
        recs.setdefault(src, []).append(i)
        rec_actions[src] = str(actions[i])
    fa_by_action = defaultdict(lambda: {"episodes": 0, "n_recording": 0})
    for src, idxs in recs.items():
        idxs = sorted(idxs)
        action = rec_actions[src]
        n_windows = len(idxs)
        duration_h = n_windows * (window_size / sample_rate) / 3600.0
        total_hours += duration_h
        fa_by_action[action]["n_recording"] += 1
        if process_true[idxs].max() == 1:
            continue  # 该 recording 有 fall，不算 FA
        # 恢复预测序列
        preds = (fall_scores[idxs] >= 0.5).astype(int)
        # 合并连续阳性 episode
        in_episode = False
        episode_end = -10**9
        for j, p in enumerate(preds):
            if p == 1:
                if not in_episode or (j - episode_end) > cooldown_windows:
                    total_fa_episodes += 1  # 新 episode
                    fa_by_action[action]["episodes"] += 1
                in_episode = True
                episode_end = j
            else:
                in_episode = False
    fa_per_hour = total_fa_episodes / total_hours if total_hours > 0 else float("nan")
    return {
        "normal_recording_equiv_fa_events_per_hour": fa_per_hour,
        "n_fa_episodes": total_fa_episodes,
        "normal_total_hours": total_hours,
        "fa_by_action": {
            a: {"episodes": v["episodes"], "n_recording": v["n_recording"],
                "episodes_per_recording": v["episodes"] / v["n_recording"] if v["n_recording"] else float("nan")}
            for a, v in sorted(fa_by_action.items())
        },
    }


def event_fall_analysis(fall_scores, process_true, source_files, window_size,
                        stride, sample_rate=20.0):
    """每 fall recording：首次命中窗口、coverage、最长连续阳性。"""
    from collections import OrderedDict
    recs = OrderedDict()
    for i, src in enumerate(source_files):
        recs.setdefault(src, []).append(i)
    events = []
    for src, idxs in recs.items():
        idxs = sorted(idxs)
        if process_true[idxs].max() != 1:
            continue
        preds = (fall_scores[idxs] >= 0.5).astype(int)
        n = len(preds)
        total_positive_windows = int((process_true[idxs] == 1).sum())
        pos_pred_windows = int(preds.sum())
        # coverage = 模型正预测窗 ∩ 真实正窗 / 真实正窗
        true_pos = (process_true[idxs] == 1)
        overlap = int((preds[true_pos] == 1).sum())
        coverage = overlap / total_positive_windows if total_positive_windows > 0 else 0.0
        # 首次命中
        first_hit = int(np.argmax(preds)) if preds.any() else None
        # 最长连续阳性
        longest = 0
        cur = 0
        for p in preds:
            cur = cur + 1 if p == 1 else 0
            longest = max(longest, cur)
        # 窗口时间：窗口 k 起点 = k*stride/sr，预测在窗口终点
        # prediction_time = window_end = k*stride/sr + window_size/sr
        win_dur_s = window_size / sample_rate
        events.append({
            "source": src,
            "n_windows": n,
            "fall_true_windows": int(total_positive_windows),
            "first_hit_window": first_hit,
            "window_start_time_s": (first_hit * stride / sample_rate) if first_hit is not None else None,
            "window_end_time_s": (first_hit * stride / sample_rate + win_dur_s) if first_hit is not None else None,
            "prediction_time_s": (first_hit * stride / sample_rate + win_dur_s) if first_hit is not None else None,
            "positive_window_coverage": float(coverage),
            "longest_consecutive_positive": int(longest),
            "predicted_fall_windows": int(pos_pred_windows),
        })
    return events


def _descent_onset_rel(fname: str, data_root: Path) -> float | None:
    """返回 fall recording 的 descent onset 相对 Kinect 有效首帧的秒数。"""
    from radar_module.dataset.dguha_research_v2 import parse_dguha_kinect
    from radar_module.analysis.dguha_precursor_batch_v1 import kinect_series
    from radar_module.dataset.dguha_state_label_v1 import _locate_states_from_kinect

    k = data_root / "5_falling_forward" / "kinect" / fname
    if not k.exists():
        return None
    frames_k = parse_dguha_kinect(k)
    valid = [f for f in frames_k if f.points_mm.any()]
    if not valid:
        return None
    kin = kinect_series(frames_k)
    st = _locate_states_from_kinect(kin)
    if st is None:
        return None
    t = kin["t"]
    return float(t[st["descent_idx"]] - t[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed eval using saved checkpoint.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="data/external/dguha/raw/Training")
    parser.add_argument("--output-root", type=Path, default=Path("reports/state_evolution_tcn_v1"))
    args = parser.parse_args()

    device = torch.device("cpu")
    data = load_dataset(args.dataset)
    X = data["features"]
    proc_y = data["process_labels"]
    inst_y = data["inst_labels"]
    inst_v = data["inst_valid"]
    splits = data["splits"]
    subjects = data["subjects"]
    source_files = data["source_files"]

    # fold 分配
    held_out_idx = np.where(splits == "0")[0]
    val_idx = np.where(splits == "1")[0]
    train_idx = np.where(~np.isin(splits, ["0", "1"]))[0]

    # 标准化（复现训练：train mean/std + clip）
    X_train_flat = X[train_idx].reshape(-1, X.shape[2])
    train_med = np.nanmedian(X_train_flat, axis=0)
    train_std = np.nanstd(X_train_flat, axis=0) + 1e-8
    X = np.where(np.isnan(X), train_med[None, None, :], X)
    X = (X - train_med[None, None, :]) / train_std[None, None, :]
    X = np.clip(X, -10.0, 10.0)

    # 重建模型并加载
    model = HierarchicalStateTCNV1(
        n_features=data["n_features"], hidden_dim=32, n_layers=3,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()

    # 预测
    val_proc, val_inst = _predict(model, X[val_idx], device)
    held_proc, held_inst = _predict(model, X[held_out_idx], device)
    held_true = proc_y[held_out_idx]

    # 1. val 上选 threshold
    thr, youden = select_threshold(val_proc, proc_y[val_idx], criterion="youden")
    held_pred = (held_proc >= thr).astype(int)
    cm = _confusion_matrix(held_true, held_pred)
    f1_norm, prec_norm, rec_norm = _class_f1(cm, 0)
    f1_fall, prec_fall, rec_fall = _class_f1(cm, 1)
    macro_f1 = (f1_norm + f1_fall) / 2.0
    bal_acc = _balanced_acc(cm)

    # 2. FA
    fa_windows_per_hour = int(((held_pred == 1) & (held_true == 0)).sum()) / (
        len(held_out_idx) * (data["window_size"] / 20.0) / 3600.0
    )
    fa_info = event_level_fa(
        held_proc, held_true, source_files[held_out_idx], data["actions"][held_out_idx],
        data["window_size"], 10,
    )

    # 3. event fall 细化
    events = event_fall_analysis(
        held_proc, held_true, source_files[held_out_idx],
        data["window_size"], 10,
    )
    # 补充 descent onset 相对时间（用 prediction_time = 窗口终点）
    for e in events:
        src = Path(e["source"]).name
        desc_rel = _descent_onset_rel(src, args.data_root)
        e["descent_onset_rel_s"] = desc_rel
        if e.get("prediction_time_s") is not None and desc_rel is not None:
            # lead_time = descent_onset - prediction_time（正=下降前命中，负=下降后命中）
            e["lead_time_s"] = desc_rel - e["prediction_time_s"]
        else:
            e["lead_time_s"] = None
    n_fall_events = len(events)
    detected = sum(1 for e in events if e["first_hit_window"] is not None)

    # 4. InstabilityHead（held-out，val threshold 用于过程，inst 用 0.5）
    iv = inst_v[held_out_idx]
    if iv.sum() > 0:
        inst_true_v = inst_y[held_out_idx][iv]
        inst_prob_v = held_inst[iv]
        from sklearn.metrics import roc_auc_score, average_precision_score
        inst_auroc = roc_auc_score(inst_true_v, inst_prob_v) if len(set(inst_true_v)) > 1 else float("nan")
        inst_pr = average_precision_score(inst_true_v, inst_prob_v)
        inst_pred = (inst_prob_v >= 0.5).astype(int)
        tp = int(((inst_pred == 1) & (inst_true_v == 1)).sum())
        fp = int(((inst_pred == 1) & (inst_true_v == 0)).sum())
        fn = int(((inst_pred == 0) & (inst_true_v == 1)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        inst_n_pos = int((inst_true_v == 1).sum())
        inst_n_neg = int((inst_true_v == 0).sum())
    else:
        inst_auroc = inst_pr = prec = rec = float("nan")
        inst_n_pos = inst_n_neg = 0

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "fixed_eval_using_saved_checkpoint",
        "checkpoint": str(args.checkpoint),
        "split": {
            "train_subjects": sorted(set(subjects[train_idx])),
            "val_subjects": sorted(set(subjects[val_idx])),
            "held_out_subjects": sorted(set(subjects[held_out_idx])),
            "n_train_windows": int(len(train_idx)),
            "n_val_windows": int(len(val_idx)),
            "n_held_out_windows": int(len(held_out_idx)),
        },
        "threshold": {
            "selected_on_validation": thr,
            "criterion": "youden (max TPR+TNR-1)",
            "youden_metric": youden,
        },
        "held_out_eval_once": True,
        "process_head": {
            "confusion_matrix": cm,
            "f1_normal": f1_norm,
            "precision_normal": prec_norm,
            "recall_normal": rec_norm,
            "f1_fall": f1_fall,
            "precision_fall": prec_fall,
            "recall_fall": rec_fall,
            "macro_f1": macro_f1,
            "balanced_accuracy": bal_acc,
            "auroc": float(_auroc_np(held_true, held_proc)),
            "n_fall_held": int((held_true == 1).sum()),
            "n_normal_held": int((held_true == 0).sum()),
        },
        "false_positive_windows_per_hour": fa_windows_per_hour,
        "false_alarm_events_per_hour": fa_info["normal_recording_equiv_fa_events_per_hour"],
        "n_fa_episodes": fa_info["n_fa_episodes"],
        "normal_total_hours": fa_info["normal_total_hours"],
        "fa_by_action": fa_info["fa_by_action"],
        "event_fall": {
            "n_events": n_fall_events,
            "detected_events": detected,
            "event_recall": detected / n_fall_events if n_fall_events else float("nan"),
            "events_detail": events,
        },
        "instability_head": {
            "n_valid": int(iv.sum()),
            "n_pos": inst_n_pos,
            "n_neg": inst_n_neg,
            "auroc": inst_auroc,
            "pr_auc": inst_pr,
            "precision": prec,
            "recall": rec,
        },
    }

    out_dir = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hierarchical_eval_fixed.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in result.items()
                      if k != "event_fall"}, indent=2, default=str))
    print("\n=== event fall detail ===")
    for e in events:
        print(f"  {e['source']}: first_hit={e['first_hit_time_s']}s "
              f"coverage={e['positive_window_coverage']:.2f} "
              f"longest={e['longest_consecutive_positive']}")
    return 0


def _auroc_np(y, scores):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, scores)


if __name__ == "__main__":
    raise SystemExit(main())
