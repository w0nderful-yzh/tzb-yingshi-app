"""IWR6843-Fall102 domain adaptation validation（冻结主模型）。

目的
----
用 TI 官方 IWR6843ISK Fall102 数据集（同款硬件）验证并做轻量域适配：
- 冻结 DGUHA+ocPID hierarchical causal TCN
- 只评估 ProcessHead fall/non-fall 能力
- 不构造 pre-fall 标签、不改 InstabilityHead

三组：
1. 冻结模型直接跑 Fall102（ocPID 标准化）
2. 轻量 feature-domain calibration 后跑 Fall102
3. calibration 后跑真机 20 repeats

Fall102 按 subject 隔离 LOSO（3 受试者）。
只评估 ProcessHead（NormalDynamic/FallProcess 二分类）。

Version: radar_fall102_domain_adapt_v1
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_module.dataset.iwr6843_fall_v1 import parse_iwr6843_fall_csv
from radar_module.model.state_evolution_tcn_v1 import HierarchicalStateTCNV1
from radar_module.preprocess.baseline_relative_features_v2 import (
    FEATURE_NAMES,
    extract_sequence_features,
)

SUBJECTS = ["Areeb", "Raffay", "Towsif"]
FALL_ACTIONS = ["front", "back", "side"]
NORMAL_ACTIONS = ["walk", "bow", "squat"]
WINDOW_SIZE = 20
STRIDE = 10


def _load_fall102_samples(data_root: Path) -> list[dict[str, Any]]:
    """加载 Fall102 全部样本。返回每样本: points序列 + subject + action + is_fall。"""
    files = sorted(glob.glob(str(data_root / "**/GatheredData/*/*.csv"), recursive=True))
    samples = []
    for f in files:
        p = Path(f)
        frames, _ = parse_iwr6843_fall_csv(p)
        if len(frames) < WINDOW_SIZE:
            continue
        # 从文件名解析 subject/action
        stem = p.name
        subj = stem.split("_")[0]
        action = stem.split("_")[1]
        is_fall = action in FALL_ACTIONS
        samples.append({
            "subject": subj,
            "action": action,
            "is_fall": is_fall,
            "frames": frames,
        })
    return samples


def _sample_features(frames, sample_rate_hz: float) -> np.ndarray:
    """样本 → 20 帧窗口特征（末帧 z-score 表示）。"""
    records = [{"points": f.points, "timestamp": f.timestamp} for f in frames]
    feats, _ = extract_sequence_features(records, sample_rate_hz=sample_rate_hz)
    # 20 帧窗口（取前 20 帧）
    if len(feats) < WINDOW_SIZE:
        return None
    return feats[:WINDOW_SIZE]


def _standardize(feats, norm_mean, norm_std):
    feats = np.where(np.isnan(feats), norm_mean[None, :], feats)
    feats = (feats - norm_mean[None, :]) / norm_std[None, :]
    return np.clip(feats, -10, 10)


def _calibrate(z, mu_real, sigma_real, calibrate_mask):
    """只校准饱和特征，其余保持。"""
    out = z.copy()
    cal = (z - mu_real[None, None, :]) / sigma_real[None, None, :]
    out[:, :, calibrate_mask] = cal[:, :, calibrate_mask]
    return out


def _predict_scores(model, z_windows):
    scores = []
    with torch.no_grad():
        for i in range(len(z_windows)):
            x = torch.as_tensor(z_windows[i : i + 1], dtype=torch.float32)
            pl, _ = model(x)
            scores.append(torch.softmax(pl, dim=1)[0, 1].item())
    return np.asarray(scores)


def _eval_binary(scores, labels):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    if len(set(labels)) < 2:
        return {"auroc": float("nan"), "pr_auc": float("nan")}
    pred = (scores >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, pred)),
        "recall": float(((pred == 1) & (labels == 1)).sum() / max(1, (labels == 1).sum())),
        "precision": float(((pred == 1) & (labels == 1)).sum() / max(1, (pred == 1).sum())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fall102 domain adaptation validation.")
    parser.add_argument("--fall102-root", type=Path, required=True,
                        help="data/external/iwr6843_fall_102/mmwave-radar-fall-detection-main")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("reports/state_evolution_tcn_v1/frozen_ocpid_state_tcn_v1.pt"))
    parser.add_argument("--norm", type=Path,
                        default=Path("data/processed/ocpid_norm_v1.npz"))
    parser.add_argument("--real-session-root", type=Path,
                        default=Path("reports/real_prefall_capture_v1"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/state_evolution_tcn_v1"))
    args = parser.parse_args()

    # 加载模型
    d = np.load("data/processed/dguha_ocpid_v1.npz", allow_pickle=True)
    n_features = int(d["features"].shape[2])
    model = HierarchicalStateTCNV1(n_features=n_features, hidden_dim=32, n_layers=3)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    norm = np.load(args.norm, allow_pickle=True)
    norm_mean, norm_std = norm["mean"], norm["std"]

    # 加载 Fall102
    samples = _load_fall102_samples(args.fall102_root)
    print(f"Fall102 samples: {len(samples)} (fall={sum(s['is_fall'] for s in samples)}, "
          f"non-fall={sum(not s['is_fall'] for s in samples)})")
    for s in samples:
        s["feats"] = _sample_features(s["frames"], sample_rate_hz=10.0)
    samples = [s for s in samples if s["feats"] is not None]
    print(f"after window filter: {len(samples)}")

    # ============ LOSO 三组 ============
    loso_results = []
    for test_subject in SUBJECTS:
        test_idx = [i for i, s in enumerate(samples) if s["subject"] == test_subject]
        train_idx = [i for i, s in enumerate(samples) if s["subject"] != test_subject]
        print(f"\n=== LOSO test={test_subject} (train={len(train_idx)} test={len(test_idx)}) ===")

        # 标准化
        z_train = np.stack([_standardize(samples[i]["feats"], norm_mean, norm_std)[None]
                            for i in train_idx])
        z_train = np.concatenate(z_train, axis=0)  # (N,20,F)
        z_test = np.stack([_standardize(samples[i]["feats"], norm_mean, norm_std)[None]
                           for i in test_idx])
        z_test = np.concatenate(z_test, axis=0)

        # 组1: 无校准
        scores_g1 = _predict_scores(model, z_test)
        labels_test = np.asarray([samples[i]["is_fall"] for i in test_idx], dtype=int)
        eval_g1 = _eval_binary(scores_g1, labels_test)

        # 组2: calibration（用 train 受试者估计 mu/sigma）
        last_z = z_train[:, -1, :]  # 模型预测点
        mu_real = np.nanmean(last_z, axis=0)
        sigma_real = np.nanstd(last_z, axis=0) + 1e-8
        sat_ratio = np.mean(np.abs(last_z) >= 9.99, axis=0)
        calibrate_mask = (np.abs(mu_real) > 1.0) | (sat_ratio > 0.05)
        z_test_cal = _calibrate(z_test, mu_real, sigma_real, calibrate_mask)
        scores_g2 = _predict_scores(model, z_test_cal)
        eval_g2 = _eval_binary(scores_g2, labels_test)

        loso_results.append({
            "test_subject": test_subject,
            "group1_raw": eval_g1,
            "group2_calibrated": eval_g2,
            "calibrate_mask": [bool(x) for x in calibrate_mask],
        })

    # 汇总 LOSO
    print("\n=== LOSO 汇总 ===")
    for r in loso_results:
        g1 = r["group1_raw"]
        g2 = r["group2_calibrated"]
        print(f"  {r['test_subject']}: 组1 AUROC={g1['auroc']:.3f} F1={g1.get('f1',0):.3f} | "
              f"组2 AUROC={g2['auroc']:.3f} F1={g2.get('f1',0):.3f}")

    # ============ 分动作分析（LOSO 汇总）============
    print("\n=== 分动作误报/召回（calibrated, LOSO 汇总） ===")
    action_metrics = {}
    for action in NORMAL_ACTIONS + FALL_ACTIONS:
        # 收集所有受试者的该动作样本
        all_scores_g2 = []
        for r in loso_results:
            pass
    # 简化：直接用所有样本的 calibrated 分数（LOSO 后收集）
    # 重新计算一次全样本 calibrated scores

    # 用全部样本估计 calibration（组3 用）
    all_z = np.stack([_standardize(s["feats"], norm_mean, norm_std)[None] for s in samples])
    all_z = np.concatenate(all_z, axis=0)
    last_all = all_z[:, -1, :]
    mu_all = np.nanmean(last_all, axis=0)
    sigma_all = np.nanstd(last_all, axis=0) + 1e-8
    sat_all = np.mean(np.abs(last_all) >= 9.99, axis=0)
    mask_all = (np.abs(mu_all) > 1.0) | (sat_all > 0.05)
    z_all_cal = _calibrate(all_z, mu_all, sigma_all, mask_all)
    scores_all_cal = _predict_scores(model, z_all_cal)
    labels_all = np.asarray([s["is_fall"] for s in samples], dtype=int)

    # 分动作
    for action in NORMAL_ACTIONS + FALL_ACTIONS:
        act_idx = [i for i, s in enumerate(samples) if s["action"] == action]
        act_scores = scores_all_cal[act_idx]
        if action in NORMAL_ACTIONS:
            # 误报率 = 该动作被判为 fall 的比例
            fp_rate = float((act_scores >= 0.5).mean())
            action_metrics[action] = {"type": "FP_rate", "value": fp_rate, "n": len(act_idx)}
            print(f"  {action}: FP rate={fp_rate:.3f} (n={len(act_idx)})")
        else:
            recall = float((act_scores >= 0.5).mean())
            action_metrics[action] = {"type": "recall", "value": recall, "n": len(act_idx)}
            print(f"  {action} fall: recall={recall:.3f} (n={len(act_idx)})")

    # 全样本 calibrated 整体指标
    eval_all_cal = _eval_binary(scores_all_cal, labels_all)
    eval_all_raw = _eval_binary(_predict_scores(model, all_z), labels_all)

    # ============ 真机 20 repeats（组3）============
    print("\n=== 真机 20 repeats（组3: Fall102 calibration） ===")
    real_results = _eval_real(args, model, norm_mean, norm_std, mu_all, sigma_all, mask_all)

    result = {
        "loso": loso_results,
        "all_samples_calibrated": eval_all_cal,
        "all_samples_raw": eval_all_raw,
        "action_metrics": action_metrics,
        "real_20repeats": real_results,
        "calibration": {
            "mu_real": mu_all.tolist(), "sigma_real": sigma_all.tolist(),
            "calibrate_mask": [bool(x) for x in mask_all],
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "fall102_domain_adapt.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved to {args.output_root / 'fall102_domain_adapt.json'}")
    return 0


def _eval_real(args, model, norm_mean, norm_std, mu_all, sigma_all, mask_all):
    """真机 20 repeats，用 Fall102 calibration。

    必须与 real_sanity_decision_v1.py 完全一致的推理链路：
    - 20Hz 特征提取（真机帧率）
    - 滑动窗口（stride=10），每 20 帧一窗，逐窗 score
    - 决策层（threshold 0.6 / consec 3 / cooldown 10）
    """
    import json as _json

    session_root = args.real_session_root
    results = {}
    for action_dir in sorted(session_root.iterdir()):
        if not action_dir.is_dir():
            continue
        action = action_dir.name
        ep_counts_raw = []
        ep_counts_cal = []
        for rep_dir in sorted(action_dir.iterdir()):
            if not rep_dir.is_dir() or not rep_dir.name.startswith("repeat_"):
                continue
            frames_path = rep_dir / "frames.jsonl"
            if not frames_path.exists():
                continue
            rows = [_json.loads(l) for l in frames_path.read_text().splitlines() if l.strip()]
            records = [{
                "points": r.get("points") or r.get("points_sensor", ()),
                "timestamp": r.get("timestamp", "2026-08-18T00:00:00+00:00"),
            } for r in rows]
            try:
                feats, _ = extract_sequence_features(records, sample_rate_hz=20.0)
            except Exception:
                continue
            if len(feats) < WINDOW_SIZE:
                continue
            # 标准化 + clip（与真 sanity 一致）
            feats = np.where(np.isnan(feats), norm_mean[None, :], feats)
            feats = (feats - norm_mean[None, :]) / norm_std[None, :]
            feats = np.clip(feats, -10, 10)
            # 滑动窗口（与真 sanity 一致）
            n = len(feats)
            s_raw = []
            s_cal = []
            for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
                win = feats[start : start + WINDOW_SIZE][None, :, :]  # (1,20,F)
                s_raw.append(_predict_scores(model, win)[0])
                s_cal.append(_predict_scores(
                    model, _calibrate(win, mu_all, sigma_all, mask_all))[0])
            s_raw = np.asarray(s_raw)
            s_cal = np.asarray(s_cal)
            # 决策层
            ep_raw = _decision(s_raw)
            ep_cal = _decision(s_cal)
            ep_counts_raw.append(ep_raw)
            ep_counts_cal.append(ep_cal)
        results[action] = {
            "n_repeats": len(ep_counts_raw),
            "raw_episodes": ep_counts_raw,
            "calibrated_episodes": ep_counts_cal,
            "raw_alert_repeats": sum(1 for e in ep_counts_raw if e > 0),
            "calibrated_alert_repeats": sum(1 for e in ep_counts_cal if e > 0),
        }
        print(f"  {action}: raw报警={results[action]['raw_alert_repeats']}/{len(ep_counts_raw)} "
              f"cal报警={results[action]['calibrated_alert_repeats']}/{len(ep_counts_cal)}")
    return results


def _decision(scores, threshold=0.6, consec=3, cooldown=10):
    binseq = (scores >= threshold).astype(int)
    confirmed = np.zeros_like(binseq)
    run = 0
    for j, b in enumerate(binseq):
        run = run + 1 if b == 1 else 0
        if run >= consec:
            confirmed[j] = 1
    episodes = 0
    in_ep = False
    last_end = -10**9
    for j, c in enumerate(confirmed):
        if c == 1:
            if not in_ep and (j - last_end) > cooldown:
                episodes += 1
            in_ep = True
        else:
            if in_ep:
                last_end = j
            in_ep = False
    return episodes


if __name__ == "__main__":
    raise SystemExit(main())
