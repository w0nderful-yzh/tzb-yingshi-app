"""纯雷达 pre-fall：baseline-relative 特征分析。

背景
----
stage 分析发现 controlled_forward_fall 的 height_range 在 still_pre
（动作前静止）阶段就已很高（0.816 vs early 0.843），说明绝对 height_range
主要受准备姿态、点云密度和身体展开影响 → **controlled-action confound**。

本脚本对每个 repeat 使用**它自己的 still_pre 作为基线**，只比较"动作后
的变化量"，去除静态差异后重新评估特征区分能力。

方法
----
对每个 repeat：
1. 用 still_pre 阶段每特征中位数作为 baseline
2. action 阶段按单调时间三等分 early/middle/late，每段取中位数
3. 构造：
   - delta_X = X(t) - baseline_X
   - relative_X = delta_X / (baseline_X + eps)
4. 以 repeat 为单位，比较 fall vs fast_sitting / fall vs instability，
   按阶段输出 effect size (Cohen d) / AUROC / PR-AUC

特征集：height_range / x_range / drift_xy_1p0s / drift_xy_0p5s /
point_count / z_p90 / centroid_z / spatial_spread

重点回答：
A. 去掉静止基线后 height_range 是否仍能区分 fall？
B. 若下降明显 → 标记 controlled-action/posture confound，不作为核心
C. drift_xy 在 baseline-relative 后是否稳定区分"持续失衡 vs 恢复"？
D. 哪些特征真正来自"动作后变化"，而非动作前静态差异？

输出
----
reports/prefall_baseline_relative_v1/<timestamp>/
  - per_repeat_delta.jsonl      （每 repeat 每阶段 delta/relative）
  - stage_comparison.json       （逐阶段 effect size / AUROC / PR-AUC）
  - confound_removal.json       （绝对 vs baseline-relative 对比）
  - report.md

Version: radar_prefall_baseline_relative_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from radar_module.preprocess.prefall_features_v1 import (
    FrameFeatureRow,
    default_window_frames,
    dynamic_features,
    frame_base_features,
    record_points,
)

FALL = "controlled_forward_fall"
FAST_SITTING = "fast_sitting"
INSTABILITY = "forward_instability_recovery"
STANDING = "standing"

STAGES = ("early", "middle", "late")
EPS = 1e-6

# 特征集：基础 + 动态
FEATURES = [
    "height_range",
    "x_range",
    "drift_xy_1p0s",
    "drift_xy_0p5s",
    "point_count",
    "z_p90",
    "centroid_z",
    "spatial_spread",
]

PRIMARY_COMPARISONS = [
    (FALL, FAST_SITTING),
    (FALL, INSTABILITY),
]
REFERENCE_COMPARISONS = [(FALL, STANDING)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _median_feature(rows: list[FrameFeatureRow], feat: str) -> float:
    vals = np.asarray([
        r.base.get(feat, r.dynamic.get(feat, float("nan"))) for r in rows
    ], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def compute_repeat_features(
    frames_path: Path,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """计算单 repeat 的 baseline(still_pre) + early/middle/late 特征。"""
    records = _read_jsonl(frames_path)
    if not records:
        return {"repeat_id": meta.get("repeat_id"), "error": "empty"}
    marks = {m["name"]: m for m in meta.get("marks", [])}
    if "pre_start" not in marks:
        return {"repeat_id": meta.get("repeat_id"), "error": "no marks"}

    timestamps = [_parse_ts(r["timestamp"]) for r in records]
    deltas = [
        (b - a).total_seconds()
        for a, b in zip(timestamps[:-1], timestamps[1:])
        if (b - a).total_seconds() > 0
    ]
    period = float(np.median(deltas)) if deltas else 1.0 / 18.18
    windows = default_window_frames(period)

    pre_m = marks["pre_start"]["monotonic"]
    as_m = marks["action_start"]["monotonic"]
    ae_m = marks["action_end"]["monotonic"]
    action_start_rel = as_m - pre_m
    action_end_rel = ae_m - pre_m
    duration = action_end_rel - action_start_rel
    if duration <= 0:
        return {"repeat_id": meta.get("repeat_id"), "error": "bad duration"}
    boundaries = [action_start_rel + duration * i / 3 for i in range(4)]

    # 逐帧特征 + stage 标注
    rows: list[FrameFeatureRow] = []
    history: list[dict[str, float]] = []
    stage_idx: list[int | None] = []
    for record in records:
        base = frame_base_features(record_points(record))
        history.append(base)
        dyn = dynamic_features(history, windows, period_seconds=period)
        phase = record.get("phase")
        if phase == "action":
            mono_rel = float(record.get("monotonic_since_repeat_start", float("nan")))
            si = None
            for i in range(3):
                if boundaries[i] <= mono_rel < boundaries[i + 1]:
                    si = i
                    break
            if si is None:
                si = 2 if mono_rel >= boundaries[3] else 0
        else:
            si = None
        rows.append(FrameFeatureRow(
            timestamp=_parse_ts(record["timestamp"]),
            action_name=str(record.get("action_name") or meta.get("action_name")),
            repeat_index=None,
            phase=phase,
            base=base,
            dynamic=dyn,
        ))
        stage_idx.append(si)

    # baseline = still_pre 帧的中位数
    pre_rows = [r for r, si in zip(rows, stage_idx) if r.phase == "still_pre"]
    baseline = {f: _median_feature(pre_rows, f) for f in FEATURES}

    # 各 stage 中位数
    stage_medians: dict[str, dict[str, float]] = {}
    for si, stage in enumerate(STAGES):
        stage_rows = [r for r, idx in zip(rows, stage_idx) if idx == si]
        stage_medians[stage] = {
            f: _median_feature(stage_rows, f) for f in FEATURES
        }

    return {
        "repeat_id": meta.get("repeat_id"),
        "action_name": str(meta.get("action_name")),
        "period_seconds": period,
        "frame_count": len(rows),
        "baseline": baseline,
        "stage_medians": stage_medians,
        "stage_delta": {
            stage: {
                f: (stage_medians[stage][f] - baseline[f])
                for f in FEATURES
            }
            for stage in STAGES
        },
        "stage_relative": {
            stage: {
                f: (
                    (stage_medians[stage][f] - baseline[f])
                    / (abs(baseline[f]) + EPS)
                    if np.isfinite(baseline[f])
                    else float("nan")
                )
                for f in FEATURES
            }
            for stage in STAGES
        },
    }


def load_repeat_features(session_root: Path) -> dict[str, list[dict[str, Any]]]:
    """从采集输出目录加载所有 repeat 特征。"""
    by_action: dict[str, list[dict[str, Any]]] = {}
    if not session_root.exists():
        return by_action
    action_dirs = [d for d in session_root.iterdir() if d.is_dir()]
    for action_dir in action_dirs:
        repeat_dirs = sorted(
            [d for d in action_dir.iterdir()
             if d.is_dir() and d.name.startswith("repeat_")],
            key=lambda d: d.name,
        )
        for rep_dir in repeat_dirs:
            frames_path = rep_dir / "frames.jsonl"
            meta_path = rep_dir / "meta.json"
            if not frames_path.exists() or not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            feat = compute_repeat_features(frames_path, meta)
            if "error" in feat:
                continue
            by_action.setdefault(feat["action_name"], []).append(feat)
    return by_action


def _col(repeats: list[dict[str, Any]], stage: str, feat: str, mode: str) -> np.ndarray:
    key = "stage_delta" if mode == "delta" else "stage_relative"
    return np.asarray([
        r[key][stage].get(feat, float("nan")) for r in repeats
    ], dtype=np.float64)


def _safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n_pos, n_neg = len(pos), len(neg)
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined)
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1)
    _, first = np.unique(combined[order], return_index=True)
    for idx in first:
        ties = combined[order] == combined[order][idx]
        ranks[order[ties]] = ranks[order[ties]].mean()
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _safe_pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """PR-AUC：从 recall=0, precision=1 起积分（标准做法）。"""
    if len(y) == 0 or int(y.sum()) == 0:
        return float("nan")
    order = np.argsort(-scores)
    y_sorted = y[order]
    precision = np.cumsum(y_sorted) / np.arange(1, len(y) + 1)
    recall = np.cumsum(y_sorted) / int(y.sum())
    # 标准 PR 曲线从 (recall=0, precision=1) 开始
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    auc = float(np.trapz(precision, recall))
    return auc if np.isfinite(auc) else float("nan")


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0)
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def _mannwhitney(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def evaluate_pair(
    by_action: dict[str, list[dict[str, Any]]],
    pos_action: str,
    neg_action: str,
) -> dict[str, Any]:
    pos = by_action.get(pos_action, [])
    neg = by_action.get(neg_action, [])
    n_pos, n_neg = len(pos), len(neg)
    result: dict[str, Any] = {
        "pos_action": pos_action,
        "neg_action": neg_action,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "repeat_ids_pos": [r["repeat_id"] for r in pos],
        "repeat_ids_neg": [r["repeat_id"] for r in neg],
        "stages": {},
    }
    if n_pos < 3 or n_neg < 3:
        result["error"] = "insufficient"
        return result
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)

    for stage in STAGES:
        result["stages"][stage] = {}
        for feat in FEATURES:
            entry: dict[str, Any] = {}
            for mode in ("delta", "relative"):
                pos_vals = _col(pos, stage, feat, mode)
                neg_vals = _col(neg, stage, feat, mode)
                all_vals = np.concatenate([pos_vals, neg_vals])
                entry[mode] = {
                    "pos_median": float(np.nanmedian(pos_vals)),
                    "neg_median": float(np.nanmedian(neg_vals)),
                    "cohen_d": _cohen_d(pos_vals, neg_vals),
                    "auroc": _safe_auroc(y, all_vals),
                    "pr_auc": _safe_pr_auc(y, all_vals),
                    "mannwhitney_p": _mannwhitney(pos_vals, neg_vals),
                }
            result["stages"][stage][feat] = entry
    return result


def confound_removal_analysis(
    by_action: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """绝对特征 vs baseline-relative 的区分度对比（fall vs 各对照）。

    对每 stage、每特征，比较：
    - 绝对特征 AUROC（直接用 stage 中位数，不回退到 pre）
    - delta 特征 AUROC
    - relative 特征 AUROC
    并判断 height_range 的区分能力在去除基线后是否大幅下降。
    """
    result: dict[str, Any] = {}
    for pos_name, neg_name in PRIMARY_COMPARISONS + REFERENCE_COMPARISONS:
        pos = by_action.get(pos_name, [])
        neg = by_action.get(neg_name, [])
        if not pos or not neg:
            continue
        y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
        comparison: dict[str, Any] = {}
        for feat in FEATURES:
            comparison[feat] = {}
            for stage in STAGES:
                abs_pos = np.asarray([
                    r["stage_medians"][stage].get(feat, float("nan"))
                    for r in pos
                ], dtype=np.float64)
                abs_neg = np.asarray([
                    r["stage_medians"][stage].get(feat, float("nan"))
                    for r in neg
                ], dtype=np.float64)
                abs_all = np.concatenate([abs_pos, abs_neg])
                d_pos = _col(pos, stage, feat, "delta")
                d_neg = _col(neg, stage, feat, "delta")
                d_all = np.concatenate([d_pos, d_neg])
                r_pos = _col(pos, stage, feat, "relative")
                r_neg = _col(neg, stage, feat, "relative")
                r_all = np.concatenate([r_pos, r_neg])
                comparison[feat][stage] = {
                    "absolute_auroc": _safe_auroc(y, abs_all),
                    "delta_auroc": _safe_auroc(y, d_all),
                    "relative_auroc": _safe_auroc(y, r_all),
                }
        result[f"{pos_name}_vs_{neg_name}"] = comparison
    return result


def build_report(
    comparisons: dict[str, Any],
    confound: dict[str, Any],
    by_action: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [
        "# 纯雷达 pre-fall：baseline-relative 特征分析",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 方法",
        "",
        "- 每个 repeat 用**自己的 still_pre 作为基线**",
        "- delta_X = X(t) - median(X_pre)；relative_X = delta_X / (|median(X_pre)|+eps)",
        "- 只比较动作后变化量，不比较绝对值",
        "- 统计单位 = repeat；action 按单调时间三等分 early/middle/late",
        "",
        "## 输入 repeat 数",
        "",
    ]
    for action, repeats in sorted(by_action.items()):
        lines.append(f"- {action}: {len(repeats)}")
    lines += ["", "## 逐阶段对比（delta）", ""]
    for key, comp in comparisons.items():
        pos, neg = comp["pos_action"], comp["neg_action"]
        if "error" in comp:
            lines.append(f"### {pos} vs {neg}: {comp['error']}")
            continue
        lines.append(f"### {pos} vs {neg} (n={comp['n_pos']}/{comp['n_neg']})")
        lines.append("")
        lines.append("| stage | feature | delta pos_med | delta neg_med | cohen_d | AUROC | PR-AUC | p |")
        lines.append("|-------|---------|--------------|--------------|---------|-------|--------|---|")
        for stage in STAGES:
            for feat in FEATURES:
                e = comp["stages"][stage][feat]["delta"]
                lines.append(
                    f"| {stage} | {feat} | {e['pos_median']:.3f} | "
                    f"{e['neg_median']:.3f} | {e['cohen_d']:+.2f} | "
                    f"{e['auroc']:.3f} | {e['pr_auc']:.3f} | "
                    f"{e['mannwhitney_p']:.3f} |"
                )
        lines.append("")

    lines += ["", "## confound 去除前后对比（AUROC）", ""]
    lines.append("""
说明：absolute 用 stage 中位数（含 pre 静态差异）；delta/relative 去除
每 repeat 自己的 still_pre 基线。若 absolute 高但 delta/relative 大幅
下降 → controlled-action confound。""")
    for key, comp in confound.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("| feature | stage | abs | delta | relative |")
        lines.append("|---------|-------|-----|-------|----------|")
        for feat, stages in comp.items():
            for stage, metrics in stages.items():
                lines.append(
                    f"| {feat} | {stage} | {metrics['absolute_auroc']:.3f} | "
                    f"{metrics['delta_auroc']:.3f} | {metrics['relative_auroc']:.3f} |"
                )
        lines.append("")
    lines += [""]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baseline-relative pre-fall feature analysis."
    )
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/prefall_baseline_relative_v1"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    by_action = load_repeat_features(args.session_root)
    if not by_action:
        raise SystemExit("no repeats found")

    comparisons = {}
    for pos, neg in PRIMARY_COMPARISONS + REFERENCE_COMPARISONS:
        comparisons[f"{pos}_vs_{neg}"] = evaluate_pair(by_action, pos, neg)
    confound = confound_removal_analysis(by_action)

    out_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 每 repeat 明细
    detail = [r for repeats in by_action.values() for r in repeats]
    (out_dir / "per_repeat_delta.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in detail),
        encoding="utf-8",
    )
    (out_dir / "stage_comparison.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "confound_removal.json").write_text(
        json.dumps(confound, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(
        build_report(comparisons, confound, by_action), encoding="utf-8"
    )
    print(f"reports written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
