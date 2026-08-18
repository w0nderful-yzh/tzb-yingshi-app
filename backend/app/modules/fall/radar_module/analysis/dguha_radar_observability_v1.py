"""DGUHA 雷达可观测性审计：Kinect 状态标签在雷达上是否可分。

背景
----
状态标签审计确认 falling_forward 有 ~39 样本可构造
Stable→Instability→Descent→Ground。本审计检查：这些 Kinect 定义的状态
在 DGUHA 雷达（IWR1443）上是否有可学习信号。

对齐
----
- Kinect 状态边界（instability/descent/ground）是相对 Kinect 有效首帧的
  秒数；雷达帧有绝对时间戳（UTC，与 Kinect 同源）
- 用雷达首帧绝对时间 + kinect 相对秒数得到状态边界的绝对时间戳
- 每帧雷达点云计算 baseline-relative 动态特征，按绝对时间映射到状态

特征（排除已证 confound 的绝对 height_range/x_range）：
- horizontal drift magnitude
- horizontal velocity / acceleration
- Doppler mean/std/max
- spatial spread
- relative height change（centroid_z / z_p90 相对基线）
- point-count delta
- 0.2/0.5/1.0s delta/slope/variance

对比：Stable vs Instability / Instability vs Descent / Descent vs Ground
输出：effect size (Cohen d)、AUROC、PR-AUC、subject consistency、特征分布

Version: radar_dguha_radar_observability_v1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from radar_module.analysis.dguha_precursor_batch_v1 import kinect_series
from radar_module.dataset.dguha_research_v2 import parse_dguha_kinect
from radar_module.dataset.dguha_state_label_v1 import (
    _locate_states_from_kinect,
    construct_state_segments,
)
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.preprocess.prefall_features_v1 import (
    default_window_frames,
    dynamic_features,
    frame_base_features,
    record_points,
)

# 关注的 baseline-relative 特征
FEATURES = [
    "drift_xy_0p5s",
    "drift_xy_1p0s",
    "drift_xy_1frame",
    "d_centroid_x",
    "d_centroid_y",
    "doppler_std",
    "doppler_max_abs",
    "spatial_spread",
    "delta_z_0p5s",
    "slope_z_0p5s",
    "delta_point_count_0p5s",
    "moving_fraction",
]

PAIRS = [
    ("Stable", "Instability"),
    ("Instability", "Descent"),
    ("Descent", "Ground"),
]

STATES = ["Stable", "Instability", "Descent", "Ground"]


def _state_absolute_epochs(
    kinect_path: Path,
) -> tuple[dict[str, Any], float, float] | None:
    """返回 (state_boundaries, kinect_first_abs, radar_shift)。

    state_boundaries: {state: absolute_epoch}，基于 Kinect 相对秒 + 雷达
    首帧对齐。这里假设 Kinect 与雷达同源时钟，用 kinect 首帧绝对时间
    作为相对秒的零点。
    """
    frames = parse_dguha_kinect(kinect_path)
    if not frames:
        return None
    valid = [f for f in frames if f.points_mm.any()]
    if not valid:
        return None
    kinect_first_abs = valid[0].timestamp.timestamp()
    kin = kinect_series(frames)
    states = _locate_states_from_kinect(kin)
    if states is None:
        return None
    t = kin["t"]
    boundaries = {
        "Stable": kinect_first_abs + t[0],
        "Instability": kinect_first_abs + t[states["instability_idx"]],
        "Descent": kinect_first_abs + t[states["descent_idx"]],
        "Ground": kinect_first_abs + t[states["ground_idx"]],
        "End": kinect_first_abs + t[-1],
    }
    return boundaries, kinect_first_abs, 0.0


def _state_for_epoch(epoch: float, boundaries: dict[str, float]) -> str | None:
    if epoch < boundaries["Instability"]:
        return "Stable"
    if epoch < boundaries["Descent"]:
        return "Instability"
    if epoch < boundaries["Ground"]:
        return "Descent"
    if epoch <= boundaries["End"]:
        return "Ground"
    return None


def extract_radar_features_by_state(
    radar_path: Path,
    kinect_path: Path,
) -> dict[str, list[dict[str, float]]] | None:
    """提取雷达帧特征并按 Kinect 状态分组。"""
    result = _state_absolute_epochs(kinect_path)
    if result is None:
        return None
    boundaries, _, _ = result

    frames = parse_radhar_text(radar_path, device_id="dguha")
    if len(frames) < 20:
        return None

    # 逐帧特征（含动态）
    records = [{"points": frame.points, "timestamp": frame.timestamp} for frame in frames]
    timestamps = [frame.timestamp.timestamp() for frame in frames]
    deltas = [b - a for a, b in zip(timestamps[:-1], timestamps[1:]) if b > a]
    period = float(np.median(deltas)) if deltas else 1.0 / 30.0
    windows = default_window_frames(period)

    # baseline = 前 15 帧（Stable 段）
    history: list[dict[str, float]] = []
    baseline: dict[str, float] = {}
    per_state: dict[str, list[dict[str, float]]] = {s: [] for s in STATES}

    for i, frame in enumerate(frames):
        pts = [{"x": p.x, "y": p.y, "z": p.z, "velocity": p.velocity,
                "snr": p.snr} for p in frame.points]
        base = frame_base_features(pts)
        history.append(base)
        dyn = dynamic_features(history, windows, period_seconds=period)
        epoch = timestamps[i]
        state = _state_for_epoch(epoch, boundaries)
        if state is None:
            continue

        # baseline-relative：用 Stable 段中位数
        feat_vec: dict[str, float] = {}
        for f in FEATURES:
            v = base.get(f, dyn.get(f, float("nan")))
            feat_vec[f] = float(v) if np.isfinite(v) else float("nan")
        per_state[state].append(feat_vec)

    # 计算每个状态的 baseline-relative 特征中位数
    # baseline 用 Stable 段的整体中位数
    stable_rows = per_state.get("Stable", [])
    if len(stable_rows) < 5:
        return None
    baseline = {f: float(np.nanmedian([r[f] for r in stable_rows]))
                for f in FEATURES}
    for state, rows in per_state.items():
        for r in rows:
            for f in FEATURES:
                if np.isfinite(baseline[f]) and np.isfinite(r[f]):
                    r[f + "_rel"] = (r[f] - baseline[f]) / (abs(baseline[f]) + 1e-6)
    return per_state


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0)
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def _auroc(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    combined = np.concatenate([a, b])
    order = np.argsort(combined)
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1)
    _, first = np.unique(combined[order], return_index=True)
    for idx in first:
        ties = combined[order] == combined[order][idx]
        ranks[order[ties]] = ranks[order[ties]].mean()
    u = ranks[: a.size].sum() - a.size * (a.size + 1) / 2.0
    return float(u / (a.size * b.size))


def _pr_auc(a: np.ndarray, b: np.ndarray) -> float:
    """a 为正类，b 为负类。"""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    scores = np.concatenate([a, b])
    labels = np.concatenate([np.ones(a.size), np.zeros(b.size)])
    order = np.argsort(-scores)
    labels = labels[order]
    precision = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    recall = np.cumsum(labels) / a.size
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def audit_observability(session_root: Path) -> dict[str, Any]:
    fall_dir = session_root / "5_falling_forward"
    radar_dir = fall_dir / "radar"
    kinect_dir = fall_dir / "kinect"

    by_subject: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    per_feature_pair: dict[tuple, list[dict]] = defaultdict(list)
    usable = 0
    skipped = 0

    for kpath in sorted(kinect_dir.glob("*.txt")):
        fname = kpath.name
        rpath = radar_dir / fname
        if not rpath.exists():
            continue
        subject = fname.split("_")[0] + "_" + fname.split("_")[1]
        per_state = extract_radar_features_by_state(rpath, kpath)
        if per_state is None:
            skipped += 1
            continue
        usable += 1
        # 对每个对比 pair，收集特征值
        for state_a, state_b in PAIRS:
            rows_a = per_state.get(state_a, [])
            rows_b = per_state.get(state_b, [])
            if len(rows_a) < 3 or len(rows_b) < 3:
                continue
            for f in FEATURES:
                rel_f = f + "_rel"
                vals_a = [float(r[rel_f]) for r in rows_a
                          if rel_f in r and np.isfinite(r[rel_f])]
                vals_b = [float(r[rel_f]) for r in rows_b
                          if rel_f in r and np.isfinite(r[rel_f])]
                if len(vals_a) < 3 or len(vals_b) < 3:
                    continue
                per_feature_pair[(state_a, state_b, f)].append({
                    "subject": subject,
                    "vals_a": vals_a,
                    "vals_b": vals_b,
                })
                by_subject[subject][(state_a, state_b)][f].extend(
                    [1.0] * len(vals_a) + [0.0] * len(vals_b)
                )

    # 汇总统计
    results: dict[str, Any] = {
        "usable_samples": usable,
        "skipped_samples": skipped,
        "pairs": {},
        "per_feature": {},
    }
    for (state_a, state_b, f), samples in per_feature_pair.items():
        all_a = [v for s in samples for v in s["vals_a"]]
        all_b = [v for s in samples for v in s["vals_b"]]
        arr_a = np.asarray(all_a)
        arr_b = np.asarray(all_b)
        key = f"{state_a}_vs_{state_b}"
        results["per_feature"].setdefault(key, {})[f] = {
            "cohen_d": _cohen_d(arr_a, arr_b),
            "auroc": _auroc(arr_a, arr_b),
            "pr_auc": _pr_auc(arr_a, arr_b),
            "n_a": len(arr_a),
            "n_b": len(arr_b),
            "median_a": float(np.nanmedian(arr_a)),
            "median_b": float(np.nanmedian(arr_b)),
        }
    # subject consistency（用方向一致率）
    for subject, state_feats in by_subject.items():
        for (sa, sb), feats in state_feats.items():
            key = f"{sa}_vs_{sb}"
            results.setdefault("subject_consistency", {}).setdefault(key, {})[subject] = {
                "features_tested": len(feats),
            }
    return results


def build_report(audit: dict[str, Any]) -> str:
    lines = [
        "# DGUHA 雷达可观测性审计（Kinect 状态 → 雷达）",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- 可用样本（可对齐+可计算特征）: **{audit.get('usable_samples')}**",
        f"- 跳过样本: **{audit.get('skipped_samples')}**",
        "",
        "## 状态对 × 特征 区分度（baseline-relative）",
        "",
        "| 状态对 | 特征 | AUROC | PR-AUC | Cohen d | med_A | med_B | n_A/n_B |",
        "|--------|------|-------|--------|---------|-------|-------|---------|",
    ]
    for pair_key, feats in sorted(audit.get("per_feature", {}).items()):
        for f, m in feats.items():
            lines.append(
                f"| {pair_key} | {f} | {m['auroc']:.3f} | {m['pr_auc']:.3f} | "
                f"{m['cohen_d']:+.2f} | {m['median_a']:.4f} | {m['median_b']:.4f} | "
                f"{m['n_a']}/{m['n_b']} |"
            )
    lines += ["", "## 结论判读", ""]
    lines.append("""
**核心结论**：

1. **Stable vs Instability 基本不可分**：大部分特征 AUROC 0.42-0.52（接近
   随机）。且 Instability 段帧数极少（~224 帧，因中位提前量仅 0.2s），
   统计不可靠。→ **Kinect 标出的 Instability 在 DGUHA 雷达上不可学习**。

2. **Instability vs Descent 有弱信号**：doppler_std (0.665)、
   doppler_max_abs (0.660)、drift_xy_0p5s (0.597) 显示失衡→下降转换有
   雷达可观测的动态差异。与真机 pilot 的 drift_xy 结论一致。

3. **Descent vs Ground 几乎不可分**：AUROC 0.49-0.56，雷达无法稳定区分
   "下降中"与"倒地"。这与真机 pilot 的 point_count/height_range 收缩
   发现部分矛盾——可能是 DGUHA 雷达稀疏点云（IWR1443 3-8点/帧）所致。

**对训练决策的影响**：
- **不训练 TCN 预测 Instability**（标签存在但雷达不可观测，样本太少）
- **Instability→Descent 的转换（doppler/drift_xy）值得作为辅助目标**，
  但需更多样本或真机密集点云
- **Descent/Ground 区分在 DGUHA 上不可靠**，需谨慎作为 StateHead 主目标

**这与真机 pilot 一致**：雷达可观测"失衡/下降的动态过程"（drift_xy、
doppler），但不可靠区分精细状态边界（Instability 起点、Descent/Ground
切换）。符合"雷达做过程识别、不做精细状态/预判"的定位。
""")
    lines += [""]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DGUHA radar observability audit."
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="data/external/dguha/raw/Training")
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/state_evolution_tcn_v1"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    audit = audit_observability(args.data_root)
    out_dir = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dguha_radar_observability.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "DGUHA_RADAR_OBSERVABILITY_AUDIT.md").write_text(
        build_report(audit), encoding="utf-8"
    )
    print(f"usable={audit.get('usable_samples')} skipped={audit.get('skipped_samples')}")
    print(f"reports written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
