"""真机 IWR6843 vs 训练域(DGUHA+ocPID)特征偏移诊断。

目的
----
定位导致真机 score 饱和到 1.0 的 domain shift。对比：
- 训练域(DGUHA+ocPID train)特征 z-score 分布
- 真机 20 repeats 特征 z-score 分布

方法
----
- 真机每 repeat：读 frames.jsonl → extract_sequence_features(21维)
- 用训练 mean/std 标准化(z-score)
- 逐特征对比 训练域 vs 真机 的 mean/std/偏移量
- 识别偏移大的特征(真机 z-score 均值偏离 0 多的)

Version: radar_domain_shift_diag_v1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from radar_module.preprocess.baseline_relative_features_v2 import (
    FEATURE_NAMES,
    extract_sequence_features,
)


def _load_real_features(session_root: Path, window_size: int = 20):
    """提取真机所有 repeat 的逐帧特征。"""
    all_feats = []
    meta = []
    for action_dir in sorted(session_root.iterdir()):
        if not action_dir.is_dir():
            continue
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
            all_feats.append(feats)
            meta.append({"action": action_dir.name, "repeat": rep_dir.name})
    if not all_feats:
        return np.empty((0, len(FEATURE_NAMES))), []
    return np.concatenate(all_feats, axis=0), meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Domain shift diagnostic.")
    parser.add_argument("--session-root", type=Path, required=True,
                        help="reports/real_prefall_capture_v1")
    parser.add_argument("--train-npz", type=Path, required=True,
                        help="data/processed/dguha_ocpid_v1.npz")
    parser.add_argument("--norm", type=Path,
                        default=Path("data/processed/ocpid_norm_v1.npz"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/state_evolution_tcn_v1"))
    args = parser.parse_args()

    # 训练域统计
    d = np.load(args.train_npz, allow_pickle=True)
    X_train = d["features"]
    splits = d["splits"]
    train_idx = np.where(~np.isin(splits, ["0", "1"]))[0]
    train_feats = X_train[train_idx].reshape(-1, X_train.shape[2])
    # 原始特征统计（标准化前）
    train_med = np.nanmedian(train_feats, axis=0)
    train_std = np.nanstd(train_feats, axis=0)

    # 标准化参数（模型用的）
    norm = np.load(args.norm, allow_pickle=True)
    zmean = norm["mean"]
    zstd = norm["std"]

    # 真机特征
    real_feats, real_meta = _load_real_features(args.session_root)
    print(f"train windows={len(train_feats)}  real frames={len(real_feats)}")
    if len(real_feats) == 0:
        print("no real features")
        return 1

    # 逐特征对比（原始尺度）
    print("\n=== 逐特征分布偏移（原始尺度） ===")
    print(f"{'feat':18s} {'train_med':>10s} {'real_med':>10s} {'shift':>10s} {'|shift|/train_std':>16s}")
    shifts = []
    for i, name in enumerate(FEATURE_NAMES):
        tm = train_med[i]
        rm = np.nanmedian(real_feats[:, i])
        ts = train_std[i]
        shift = rm - tm
        rel = abs(shift) / (ts + 1e-9)
        shifts.append((name, tm, rm, shift, rel))
        print(f"{name:18s} {tm:10.3f} {rm:10.3f} {shift:+10.3f} {rel:16.2f}")

    # z-score 后（模型输入）
    print("\n=== 真机 z-score 后分布（模型输入视角） ===")
    real_z = (real_feats - zmean[None, :]) / zstd[None, :]
    real_z = np.clip(real_z, -10, 10)
    print(f"{'feat':18s} {'real_z_mean':>12s} {'real_z_std':>12s} {'饱和到±10比例':>12s}")
    saturating = []
    for i, name in enumerate(FEATURE_NAMES):
        col = real_z[:, i]
        mean = np.nanmean(col)
        std = np.nanstd(col)
        sat = np.mean(np.abs(col) >= 9.99)
        if abs(mean) > 2 or sat > 0.05:
            saturating.append(name)
        print(f"{name:18s} {mean:12.3f} {std:12.3f} {sat:12.3f}")

    print("\n=== 真机 z-score 饱和/偏移严重的特征 ===")
    print(saturating)

    # 按动作分组看真机 z-score（判断哪些动作 score 高）
    print("\n=== 真机各动作 z-score 均值（模型输入） ===")
    by_action = defaultdict(list)
    for feats, m in zip(np.array_split(real_feats, len(real_meta)), real_meta):
        by_action[m["action"]].append(feats)
    for action, feats_list in sorted(by_action.items()):
        all_a = np.concatenate(feats_list, axis=0)
        z = (all_a - zmean[None, :]) / zstd[None, :]
        z = np.clip(z, -10, 10)
        # 均值 z-score 的 top 特征
        top_idx = np.argsort(-np.abs(z.mean(axis=0)))[:5]
        print(f"{action}: n_frames={len(all_a)} | 最大偏移特征: "
              + ", ".join(f"{FEATURE_NAMES[i]}={z.mean(axis=0)[i]:+.2f}" for i in top_idx))

    out = {
        "train_windows": len(train_feats),
        "real_frames": len(real_feats),
        "per_feature_shift": [
            {"feature": n, "train_med": float(tm), "real_med": float(rm),
             "shift": float(sh), "rel_shift_std": float(rel)}
            for n, tm, rm, sh, rel in shifts
        ],
        "saturating_features": saturating,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "domain_shift_diag.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
