"""轻量 feature-domain calibration（真机 IWR6843 → 训练域对齐）。

背景
----
DGUHA(IWR1443)+ocPID → 真机 IWR6843 存在 domain shift。诊断发现
`var_doppler_0p5s` 在真机上 z-score 均值 +2.73、std 3.46、13% 饱和到 +10，
导致真机所有动作 score 偏高到 1.0。

方法
----
对真机特征的 z-score 表示做 **per-feature affine**，把每个特征分布对齐到
训练域（z-score 空间 ~N(0,1)）：
    x_cal = (x - mu_real) / sigma_real
其中 mu_real/sigma_real 是真机 20 repeats 每个特征的 z-score 均值/标准差。
（训练域 z-score 已近似 N(0,1)，故目标分布是标准正态。）

约束：
- TCN 权重完全冻结
- 不把 20 repeats 用于重新训练网络
- 只允许对输入特征做 affine calibration
- calibration 参数单独保存（`real_domain_calibration_v1.json`）

注意：此校准会抹掉真机特征中与训练域一致的部分真实信号，但目标只是
"比赛够用"（正常动作不持续报警、fall 尽量触发）。fall 的判别信号主要
在 drift/centroid 等偏移小的特征，校准影响有限。

Version: radar_domain_calibration_lite_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_module.model.state_evolution_tcn_v1 import HierarchicalStateTCNV1
from radar_module.preprocess.baseline_relative_features_v2 import FEATURE_NAMES, extract_sequence_features


def extract_real_z(
    session_root: Path,
    norm: dict[str, np.ndarray],
    window_size: int = 20,
    stride: int = 10,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """提取真机每 repeat 的窗口特征并 z-score（用训练 norm），返回各 repeat 的 z 窗口。"""
    norm_mean = norm["mean"]
    norm_std = norm["std"]
    model_inputs = {}
    meta = {}
    for action_dir in sorted(session_root.iterdir()):
        if not action_dir.is_dir():
            continue
        action = action_dir.name
        for rep_dir in sorted(action_dir.iterdir()):
            if not rep_dir.is_dir() or not rep_dir.name.startswith("repeat_"):
                continue
            frames_path = rep_dir / "frames.jsonl"
            if not frames_path.exists():
                continue
            rows = [json.loads(l) for l in frames_path.read_text().splitlines() if l.strip()]
            records = [{
                "points": r.get("points") or r.get("points_sensor", ()),
                "timestamp": r.get("timestamp", "2026-08-18T00:00:00+00:00"),
            } for r in rows]
            try:
                feats, _ = extract_sequence_features(records)
            except Exception:
                continue
            feats = np.where(np.isnan(feats), norm_mean[None, :], feats)
            feats = (feats - norm_mean[None, :]) / norm_std[None, :]
            feats = np.clip(feats, -10, 10)
            # 窗口
            windows = []
            n = len(feats)
            for start in range(0, n - window_size + 1, stride):
                windows.append(feats[start : start + window_size])
            if windows:
                key = f"{action}/{rep_dir.name}"
                model_inputs[key] = np.stack(windows)
                meta[key] = {"action": action, "repeat": rep_dir.name}
    return model_inputs, meta


def compute_calibration(
    model_inputs: dict[str, np.ndarray],
    meta: dict[str, Any],
    *,
    fit_actions: tuple[str, ...] = ("standing", "fast_sitting",
                                     "forward_instability_recovery"),
    sat_mean_threshold: float = 1.0,
    sat_ratio_threshold: float = 0.05,
) -> dict[str, np.ndarray]:
    """从真机窗口特征计算 per-feature affine 参数。

    只用**真机正常动作**(standing/fast_sitting/instability)估计分布，
    避免 fall 的信号被压掉。

    只校准"饱和/偏移严重"的特征（z 均值偏移 > sat_mean_threshold 或
    饱和到 ±10 比例 > sat_ratio_threshold），其余特征保持原值——避免
    破坏训练域学到的特征间相关结构。
    """
    # 只取正常动作窗口
    fit_keys = [k for k, m in meta.items() if m["action"] in fit_actions]
    if not fit_keys:
        fit_keys = list(model_inputs.keys())
    all_windows = [model_inputs[k] for k in fit_keys if model_inputs[k].size]
    if not all_windows:
        raise ValueError("no real windows for calibration")
    all_w = np.concatenate(all_windows, axis=0)  # (N, window, F)
    last_z = all_w[:, -1, :]  # 模型预测点
    mu_real = np.nanmean(last_z, axis=0)
    sigma_real = np.nanstd(last_z, axis=0) + 1e-8

    # 识别需校准的特征
    sat_ratio = np.mean(np.abs(last_z) >= 9.99, axis=0)
    calibrate_mask = (np.abs(mu_real) > sat_mean_threshold) | (sat_ratio > sat_ratio_threshold)
    return {
        "mu_real": mu_real,
        "sigma_real": sigma_real,
        "calibrate_mask": calibrate_mask,
    }


def apply_calibration(
    z_windows: np.ndarray,
    calib: dict[str, np.ndarray],
) -> np.ndarray:
    """对 z-score 窗口特征应用 affine 校准。

    仅对 calibrate_mask=True 的特征校准（对齐到 N(0,1)），其余保持原值。
    """
    mu = calib["mu_real"]
    sigma = calib["sigma_real"]
    mask = calib["calibrate_mask"]
    out = z_windows.copy()
    cal = (z_windows - mu[None, None, :]) / sigma[None, None, :]
    out[:, :, mask] = cal[:, :, mask]
    return out


def predict_scores(
    model,
    z_windows: np.ndarray,
    *,
    calibrated: bool,
    calib: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if calibrated:
        z_windows = apply_calibration(z_windows, calib)
    scores = []
    with torch.no_grad():
        for i in range(len(z_windows)):
            x = torch.as_tensor(z_windows[i : i + 1], dtype=torch.float32)
            pl, _ = model(x)
            scores.append(torch.softmax(pl, dim=1)[0, 1].item())
    return np.asarray(scores)


def decision_layer(scores: np.ndarray, threshold: float, consec: int, cooldown: int) -> int:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight real-domain calibration.")
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--consec", type=int, default=3)
    parser.add_argument("--cooldown", type=int, default=10)
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/state_evolution_tcn_v1"))
    args = parser.parse_args()

    norm = np.load(args.norm, allow_pickle=True)
    model_inputs, meta = extract_real_z(args.session_root, norm)

    # 计算校准参数
    calib = compute_calibration(model_inputs, meta)
    calib_json = {
        "schema": "real_domain_calibration_v1",
        "feature_names": list(FEATURE_NAMES),
        "mu_real": calib["mu_real"].tolist(),
        "sigma_real": calib["sigma_real"].tolist(),
        "calibrate_mask": calib["calibrate_mask"].tolist(),
        "calibrated_features": [
            FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES))
            if calib["calibrate_mask"][i]
        ],
        "note": "only calibrate saturating features: x_cal=(x-mu_real)/sigma_real on z-score; "
                "others keep raw",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "real_domain_calibration_v1.json").write_text(
        json.dumps(calib_json, indent=2), encoding="utf-8")

    # 加载模型
    data_npz = np.load(args.train_npz, allow_pickle=True)
    n_features = int(data_npz["features"].shape[2])
    model = HierarchicalStateTCNV1(n_features=n_features, hidden_dim=32, n_layers=3)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    # 逐 repeat 评估（校准前后对比）
    print(f"params: thr={args.threshold} consec={args.consec} cooldown={args.cooldown}")
    print(f"{'repeat':40s} {'未校准':>8s} {'校准后':>8s}")
    summary = {}
    for key, win in sorted(model_inputs.items()):
        act = meta[key]["action"]
        # 未校准
        raw_s = predict_scores(model, win, calibrated=False, calib=None)
        raw_ep = decision_layer(raw_s, args.threshold, args.consec, args.cooldown)
        # 校准
        cal_s = predict_scores(model, win, calibrated=True, calib=calib)
        cal_ep = decision_layer(cal_s, args.threshold, args.consec, args.cooldown)
        summary.setdefault(act, []).append({
            "repeat": meta[key]["repeat"], "raw_episodes": raw_ep,
            "calibrated_episodes": cal_ep,
            "raw_max_score": round(float(raw_s.max()), 3) if len(raw_s) else 0,
            "calibrated_max_score": round(float(cal_s.max()), 3) if len(cal_s) else 0,
        })
        print(f"{key:40s} {raw_ep:8d} {cal_ep:8d}")

    # 汇总达标判断
    print("\n=== 达标判断（比赛够用） ===")
    targets = {
        "standing": ("≤1/5", lambda e: sum(1 for r in e if r["calibrated_episodes"] > 0) <= 1),
        "fast_sitting": ("≤1/5", lambda e: sum(1 for r in e if r["calibrated_episodes"] > 0) <= 1),
        "controlled_forward_fall": ("≥4/5", lambda e: sum(1 for r in e if r["calibrated_episodes"] > 0) >= 4),
    }
    all_ok = True
    for act, (desc, cond) in targets.items():
        reps = summary.get(act, [])
        n_alert = sum(1 for r in reps if r["calibrated_episodes"] > 0)
        ok = cond(reps)
        all_ok &= ok
        print(f"{act}: {n_alert}/{len(reps)} 报警 (目标 {desc}) {'✅' if ok else '❌'}")
    inst = summary.get("forward_instability_recovery", [])
    inst_alert = sum(1 for r in inst if r["calibrated_episodes"] > 0)
    print(f"forward_instability_recovery: {inst_alert}/{len(inst)} 报警（允许 Watch，尽量不持续 FallProcess）")
    print(f"\n总体: {'✅ 达标，可冻结' if all_ok else '❌ 未完全达标'}")

    (args.output_root / "real_domain_calibration_result.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
