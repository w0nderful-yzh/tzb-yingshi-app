"""纯雷达 pre-fall 特征可观测性 pilot 分析（以 repeat 为统计单位）。

背景
----
旧分析(per-frame + 普通 bootstrap)把连续帧当独立样本，忽略了相邻雷达帧
的强时间相关性，因此旧报告中的 bootstrap P(Δ>0)=1.000 只作探索性结果。
本脚本改为**以 repeat 为统计单位**：

- 每个 repeat 是 1 个样本点（pilot 每类 5 个 repeat => 每类 5 个样本）
- 对每个 repeat 计算 pre / action / post 三阶段的中位数特征
- 阶段间差异(action-pre, post-pre)在 repeat 层面聚合，再做符号检验 /
  配对差异检验（不做普通逐帧 bootstrap）
- 逐帧时间曲线如需置信带，使用 **block bootstrap**（块长~1s=~18帧）保留
  时间相关性，作为辅助证据

输入
----
采集工具输出目录 reports/real_prefall_capture_v1/<action_name>/，
其中每个 repeat 子目录含 frames.jsonl + meta.json（meta 记录
repeat_id/action_name/pre_start/action_start/action_end/post_end）。

重点候选特征（v2/v3 为主，其余特征保留辅助）：
- x_range, drift_xy, point_count, height_range

研究问题（RQ）：
  RQ1: 同一动作 5 次重复中，候选特征的方向是否一致
       （per-repeat "action 中位 - pre 中位" 的符号一致率）
  RQ2: x_range / drift_xy 是 fall-specific 还是一般 instability 特征
       （fall vs forward_instability_recovery 的 repeat 级差异）
  RQ3: forward_instability_recovery 与 controlled_forward_fall 的关键差异
  RQ4: 是否存在"恢复趋势"：
       instability_recovery 中异常特征回到 baseline (post≈pre)，
       controlled_fall 中异常持续或恶化 (post 偏离 pre)
  RQ5: point_count 是否只是运动强度/CFAR 点数变化，而非 pre-fall 特异
       （对比 fast_sitting 与 fall）

输出
----
reports/prefall_pilot_eval_v1/<timestamp>/
  - per_repeat_features.jsonl  （每 repeat 每阶段特征中位数）
  - rq1_direction_consistency.json
  - rq2_fall_vs_instability.json
  - rq3_recovery_vs_fall.json
  - rq4_recovery_trend.json
  - rq5_pointcount_motion_artifact.json
  - block_bootstrap_curves.json
  - report.md

Version: radar_prefall_pilot_eval_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radar_module.preprocess.prefall_features_v1 import (
    FrameFeatureRow,
    default_window_frames,
    dynamic_features,
    frame_base_features,
    record_points,
)

FALL_ACTION = "controlled_forward_fall"
INSTABILITY_ACTION = "forward_instability_recovery"
FAST_SITTING_ACTION = "fast_sitting"
STANDING_ACTION = "standing"

PILOT_ACTIONS = [
    STANDING_ACTION,
    FAST_SITTING_ACTION,
    INSTABILITY_ACTION,
    FALL_ACTION,
]

# 本轮重点候选特征
CANDIDATE_FEATURES = [
    "x_range",
    "drift_xy_0p5s",
    "drift_xy_1p0s",
    "drift_xy_1frame",
    "point_count",
    "height_range",
]

PHASES = ("pre", "action", "post")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _feature_names(row: FrameFeatureRow) -> list[str]:
    return list(row.base.keys()) + list(row.dynamic.keys())


def _col(rows: Sequence[FrameFeatureRow], name: str) -> np.ndarray:
    return np.asarray(
        [
            row.base.get(name, row.dynamic.get(name, float("nan")))
            for row in rows
        ],
        dtype=np.float64,
    )


def extract_repeat_features(
    frames_path: Path,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """提取单个 repeat 的逐帧特征，并按 phase 分组成 pre/action/post。"""
    records = _read_jsonl(frames_path)
    if not records:
        return {"repeat_id": meta.get("repeat_id"), "error": "empty frames"}

    timestamps = [_parse_timestamp(r["timestamp"]) for r in records]
    deltas = [
        (b - a).total_seconds()
        for a, b in zip(timestamps[:-1], timestamps[1:])
        if (b - a).total_seconds() > 0
    ]
    period = float(np.median(deltas)) if deltas else 1.0 / 18.18
    windows = default_window_frames(period)

    rows: list[FrameFeatureRow] = []
    history: list[dict[str, float]] = []
    phase_series: dict[str, list[FrameFeatureRow]] = {p: [] for p in PHASES}
    for record in records:
        phase = record.get("phase")
        if phase == "still_pre":
            phase_key = "pre"
        elif phase == "action":
            phase_key = "action"
        elif phase == "still_post":
            phase_key = "post"
        else:
            phase_key = None
        base = frame_base_features(record_points(record))
        history.append(base)
        dyn = dynamic_features(history, windows, period_seconds=period)
        row = FrameFeatureRow(
            timestamp=_parse_timestamp(record["timestamp"]),
            action_name=str(record.get("action_name") or meta.get("action_name")),
            repeat_index=None,
            phase=phase_key,
            base=base,
            dynamic=dyn,
        )
        rows.append(row)
        if phase_key is not None:
            phase_series[phase_key].append(row)

    feature_names = _feature_names(rows[0])
    per_phase: dict[str, dict[str, float]] = {}
    for p in PHASES:
        ph_rows = phase_series[p]
        per_phase[p] = {
            name: float(np.nanmedian(_col(ph_rows, name)))
            if ph_rows else float("nan")
            for name in feature_names
        }

    return {
        "repeat_id": meta.get("repeat_id"),
        "action_name": str(meta.get("action_name")),
        "period_seconds": period,
        "frame_count": len(rows),
        "phase_frame_counts": {p: len(phase_series[p]) for p in PHASES},
        "per_phase_medians": per_phase,
        "phase_time_series": {
            p: {
                "timestamps": [r.timestamp.isoformat() for r in phase_series[p]],
                "values": {
                    name: [float(v) for v in _col(phase_series[p], name)]
                    for name in CANDIDATE_FEATURES
                    if phase_series[p]
                },
            }
            for p in PHASES
        },
    }


def load_sessions(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    """加载采集输出目录：返回 action -> repeat features 列表。

    支持两种布局：
    1. <root>/<action>/repeat_XX/frames.jsonl + meta.json（默认）
    2. <root> 本身就是单个 action 目录（含 repeat_XX/）
    """
    root = Path(root)
    result: dict[str, list[dict[str, Any]]] = {}
    if not root.exists():
        return result

    def _collect_action(action_dir: Path) -> None:
        repeats = sorted(
            [d for d in action_dir.iterdir()
             if d.is_dir() and d.name.startswith("repeat_")],
            key=lambda d: d.name,
        )
        if not repeats:
            return
        action_name = action_dir.name
        for repeat_dir in repeats:
            frames_path = repeat_dir / "frames.jsonl"
            meta_path = repeat_dir / "meta.json"
            if not frames_path.exists():
                continue
            meta = _read_json(meta_path) if meta_path.exists() else {}
            feat = extract_repeat_features(frames_path, meta)
            if "error" in feat:
                continue
            result.setdefault(action_name, []).append(feat)

    # 布局 2：root 本身是 action 目录
    if any(
        d.is_dir() and d.name.startswith("repeat_")
        for d in root.iterdir()
    ):
        _collect_action(root)
        return result

    # 布局 1：root 下有多个 action 子目录
    for action_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        _collect_action(action_dir)
    return result


# ---------------------------------------------------------------------------
# RQ 计算
# ---------------------------------------------------------------------------

def _phase_diff(
    repeat_feat: dict[str, Any],
    feature: str,
    *,
    a: str,
    b: str,
) -> float:
    va = repeat_feat["per_phase_medians"][a].get(feature, float("nan"))
    vb = repeat_feat["per_phase_medians"][b].get(feature, float("nan"))
    if np.isfinite(va) and np.isfinite(vb):
        return float(vb - va)
    return float("nan")


def _repeat_diffs(
    repeats: list[dict[str, Any]],
    feature: str,
    *,
    a: str,
    b: str,
) -> np.ndarray:
    return np.asarray(
        [_phase_diff(r, feature, a=a, b=b) for r in repeats],
        dtype=np.float64,
    )


def _sign_consistency(values: np.ndarray) -> dict[str, Any]:
    """符号一致率：返回正/负/零的比例与方向。"""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "positive": float("nan"), "negative": float("nan"),
                "zero": float("nan"), "consistent_direction": None,
                "sign_consistency": float("nan")}
    n = values.size
    pos = float(np.sum(values > 0))
    neg = float(np.sum(values < 0))
    zero = float(np.sum(values == 0))
    majority = max(pos, neg, zero)
    consistency = majority / n
    direction = "positive" if pos == majority else ("negative" if neg == majority else "zero")
    return {
        "n": int(n),
        "positive": pos / n,
        "negative": neg / n,
        "zero": zero / n,
        "consistent_direction": direction,
        "sign_consistency": consistency,
    }


def _wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> float:
    """配对 Wilcoxon 符号秩检验（repeat 为样本）。"""
    from scipy.stats import wilcoxon

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or a.size != b.size:
        return float("nan")
    diff = a - b
    diff = diff[diff != 0]
    if diff.size < 2:
        return float("nan")
    try:
        return float(wilcoxon(diff).pvalue)
    except ValueError:
        return float("nan")


def rq1_direction_consistency(
    by_action: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """RQ1: 同一动作 5 次重复中候选特征方向是否一致。

    对每个动作的每个候选特征，计算 per-repeat (action - pre) 的符号一致率。
    """
    result: dict[str, Any] = {}
    for action, repeats in sorted(by_action.items()):
        result[action] = {}
        for feat in CANDIDATE_FEATURES:
            diffs = _repeat_diffs(repeats, feat, a="pre", b="action")
            result[action][feat] = {
                "action_minus_pre": _sign_consistency(diffs),
                "repeat_diffs": [float(v) for v in diffs],
                "wilcoxon_p": _wilcoxon_paired(
                    np.asarray([r["per_phase_medians"]["pre"].get(feat, float("nan"))
                                for r in repeats]),
                    np.asarray([r["per_phase_medians"]["action"].get(feat, float("nan"))
                                for r in repeats]),
                ),
            }
    return result


def rq2_fall_vs_instability(
    by_action: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """RQ2: x_range / drift_xy 是 fall-specific 还是一般 instability。

    fall 与 instability_recovery 在 (action - pre) 上的 repeat 级比较。
    若两者都显著升高 => 一般 instability；若仅 fall 升高 => fall-specific。
    """
    fall_repeats = by_action.get(FALL_ACTION, [])
    inst_repeats = by_action.get(INSTABILITY_ACTION, [])
    if not fall_repeats or not inst_repeats:
        return {"error": "missing fall or instability action"}

    result: dict[str, Any] = {}
    for feat in CANDIDATE_FEATURES:
        fall_diff = _repeat_diffs(fall_repeats, feat, a="pre", b="action")
        inst_diff = _repeat_diffs(inst_repeats, feat, a="pre", b="action")
        result[feat] = {
            "fall_action_minus_pre": _sign_consistency(fall_diff),
            "instability_action_minus_pre": _sign_consistency(inst_diff),
            "fall_diffs": [float(v) for v in fall_diff],
            "instability_diffs": [float(v) for v in inst_diff],
            # 双样本秩和（unpaired），repeat 为单位
            "mannwhitney_p": _mannwhitney(fall_diff, inst_diff),
        }
    return result


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


def rq3_recovery_vs_fall(
    by_action: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """RQ3: recovery 与 fall 的关键差异（在 action 阶段的绝对水平对比）。

    关注 fall 相对 recovery 是否在候选特征上有更大的"失衡幅度"。
    """
    fall_repeats = by_action.get(FALL_ACTION, [])
    inst_repeats = by_action.get(INSTABILITY_ACTION, [])
    if not fall_repeats or not inst_repeats:
        return {"error": "missing fall or instability action"}

    result: dict[str, Any] = {}
    for feat in CANDIDATE_FEATURES:
        fall_action = np.asarray(
            [r["per_phase_medians"]["action"].get(feat, float("nan"))
             for r in fall_repeats], dtype=np.float64,
        )
        inst_action = np.asarray(
            [r["per_phase_medians"]["action"].get(feat, float("nan"))
             for r in inst_repeats], dtype=np.float64,
        )
        result[feat] = {
            "fall_action_median": float(np.nanmedian(fall_action)),
            "instability_action_median": float(np.nanmedian(inst_action)),
            "fall_minus_instability": float(
                np.nanmedian(fall_action) - np.nanmedian(inst_action)
            ),
            "mannwhitney_p": _mannwhitney(fall_action, inst_action),
        }
    return result


def rq4_recovery_trend(
    by_action: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """RQ4: 恢复趋势。post 相对 pre 是否回到基线。

    instability_recovery: post-pre 应接近 0（回到基线）。
    controlled_fall: post-pre 应偏离（异常持续/恶化）。
    """
    result: dict[str, Any] = {}
    for action, repeats in sorted(by_action.items()):
        result[action] = {}
        for feat in CANDIDATE_FEATURES:
            post_pre = _repeat_diffs(repeats, feat, a="pre", b="post")
            result[action][feat] = {
                "post_minus_pre": _sign_consistency(post_pre),
                "repeat_diffs": [float(v) for v in post_pre],
                "wilcoxon_p": _wilcoxon_paired(
                    np.asarray([r["per_phase_medians"]["pre"].get(feat, float("nan"))
                                for r in repeats]),
                    np.asarray([r["per_phase_medians"]["post"].get(feat, float("nan"))
                                for r in repeats]),
                ),
            }
    return result


def rq5_pointcount_motion_artifact(
    by_action: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """RQ5: point_count 是否只是运动强度伪特征。

    对比 fast_sitting(高运动但非跌)与 fall 在 point_count / doppler 相关
    特征上的 (action-pre) 差异。若 fast_sitting 与 fall 都升高且幅度相近，
    则 point_count 更可能是运动强度反映而非 pre-fall 特异。
    """
    fall_repeats = by_action.get(FALL_ACTION, [])
    fast_repeats = by_action.get(FAST_SITTING_ACTION, [])
    if not fall_repeats or not fast_repeats:
        return {"error": "missing fall or fast_sitting action"}

    feats = ["point_count", "moving_fraction", "doppler_std",
             "doppler_max_abs", "spatial_spread"]
    result: dict[str, Any] = {}
    for feat in feats:
        fall_diff = _repeat_diffs(fall_repeats, feat, a="pre", b="action")
        fast_diff = _repeat_diffs(fast_repeats, feat, a="pre", b="action")
        result[feat] = {
            "fall_action_minus_pre": _sign_consistency(fall_diff),
            "fast_sitting_action_minus_pre": _sign_consistency(fast_diff),
            "fall_diffs": [float(v) for v in fall_diff],
            "fast_diffs": [float(v) for v in fast_diff],
            "mannwhitney_p": _mannwhitney(fall_diff, fast_diff),
        }
    return result


def block_bootstrap_curve(
    repeats: list[dict[str, Any]],
    feature: str,
    *,
    block_seconds: float = 1.0,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """对单个 repeat 的动作阶段时间序列做 block bootstrap，给出中位数曲线。

    块长取 ~block_seconds 对应的帧数，保留时间相关性。对每类动作取各
    repeat 的 block-bootstrap 中位数的中位数，作为动作级代表曲线。
    """
    rng = np.random.default_rng(seed)
    curves: list[np.ndarray] = []
    for rep in repeats:
        series = rep["phase_time_series"]["action"]["values"].get(feature)
        if not series:
            continue
        vals = np.asarray(series, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size < 4:
            continue
        block_len = max(1, int(round(block_seconds / rep["period_seconds"])))
        n_blocks = max(1, int(np.ceil(vals.size / block_len)))
        boot_medians = []
        for _ in range(n_boot):
            starts = rng.integers(0, max(1, vals.size - block_len + 1), size=n_blocks)
            sample = np.concatenate([vals[s:s + block_len] for s in starts])
            boot_medians.append(float(np.median(sample)))
        curves.append(np.asarray(boot_medians))
    if not curves:
        return {"feature": feature, "error": "no series"}
    stacked = np.stack(curves)  # (n_repeats, n_boot)
    return {
        "feature": feature,
        "repeat_count": int(stacked.shape[0]),
        "boot_median_of_medians": float(np.median(stacked)),
        "boot_ci95": [
            float(np.percentile(stacked, 2.5)),
            float(np.percentile(stacked, 97.5)),
        ],
        "boot_min": float(np.percentile(stacked, 0)),
        "boot_max": float(np.percentile(stacked, 100)),
    }


def build_report(
    rq1: dict[str, Any],
    rq2: dict[str, Any],
    rq3: dict[str, Any],
    rq4: dict[str, Any],
    rq5: dict[str, Any],
    by_action: dict[str, list[dict[str, Any]]],
    bootstrap: dict[str, Any],
) -> str:
    lines = [
        "# 纯雷达 pre-fall 特征可观测性 pilot 分析（repeat 级）",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 统计口径",
        "",
        "- 统计单位 = repeat（每类 5 个 repeat = 5 个样本点），不是逐帧",
        "- 阶段差异(action-pre / post-pre)在 repeat 层面计算",
        "- 符号一致率 = 多数方向占比；Wilcoxon/Mann-Whitney 以 repeat 为样本",
        "- 时间曲线用 block bootstrap（块长~1s）作为辅助证据",
        "",
        "## 输入动作与 repeat 数",
        "",
        "| action | repeats |",
        "|--------|---------|",
    ]
    for action, repeats in sorted(by_action.items()):
        lines.append(f"| {action} | {len(repeats)} |")

    lines += ["", "## RQ1: 同类重复中候选特征方向一致性 (action - pre)", ""]
    lines.append("| action | feature | direction | sign_consistency | wilcoxon_p |")
    lines.append("|--------|---------|-----------|------------------|------------|")
    for action, feats in sorted(rq1.items()):
        for feat, info in feats.items():
            sc = info["action_minus_pre"]
            lines.append(
                f"| {action} | {feat} | {sc['consistent_direction']} | "
                f"{sc['sign_consistency']:.2f} | {info['wilcoxon_p']:.3f} |"
            )

    lines += ["", "## RQ2: fall vs instability (x_range/drift_xy 是否 fall-specific)", ""]
    lines.append("| feature | fall dir | fall consistency | inst dir | inst consistency | mw_p |")
    lines.append("|---------|----------|------------------|----------|------------------|------|")
    if "error" not in rq2:
        for feat, info in sorted(rq2.items()):
            f = info["fall_action_minus_pre"]
            i = info["instability_action_minus_pre"]
            lines.append(
                f"| {feat} | {f['consistent_direction']} | {f['sign_consistency']:.2f} | "
                f"{i['consistent_direction']} | {i['sign_consistency']:.2f} | "
                f"{info['mannwhitney_p']:.3f} |"
            )

    lines += ["", "## RQ3: recovery vs fall 动作阶段关键差异", ""]
    lines.append("| feature | fall_med | inst_med | fall_minus_inst | mw_p |")
    lines.append("|---------|----------|----------|-----------------|------|")
    if "error" not in rq3:
        for feat, info in sorted(rq3.items()):
            lines.append(
                f"| {feat} | {info['fall_action_median']:.3f} | "
                f"{info['instability_action_median']:.3f} | "
                f"{info['fall_minus_instability']:+.3f} | {info['mannwhitney_p']:.3f} |"
            )

    lines += ["", "## RQ4: 恢复趋势 (post - pre, 越接近 0 越回到基线)", ""]
    lines.append("| action | feature | direction | sign_consistency | wilcoxon_p |")
    lines.append("|--------|---------|-----------|------------------|------------|")
    for action, feats in sorted(rq4.items()):
        for feat, info in feats.items():
            sc = info["post_minus_pre"]
            lines.append(
                f"| {action} | {feat} | {sc['consistent_direction']} | "
                f"{sc['sign_consistency']:.2f} | {info['wilcoxon_p']:.3f} |"
            )

    lines += ["", "## RQ5: point_count 是否运动强度伪特征", ""]
    lines.append("| feature | fall dir | fall cons | fast_sitting dir | fast cons | mw_p |")
    lines.append("|---------|----------|-----------|------------------|-----------|------|")
    if "error" not in rq5:
        for feat, info in sorted(rq5.items()):
            f = info["fall_action_minus_pre"]
            s = info["fast_sitting_action_minus_pre"]
            lines.append(
                f"| {feat} | {f['consistent_direction']} | {f['sign_consistency']:.2f} | "
                f"{s['consistent_direction']} | {s['sign_consistency']:.2f} | "
                f"{info['mannwhitney_p']:.3f} |"
            )

    lines += ["", "## block bootstrap（动作阶段特征水平, 辅助证据）", ""]
    lines.append("| action | feature | median_of_medians | ci95 |")
    lines.append("|--------|---------|-------------------|------|")
    for action, feats in sorted(bootstrap.items()):
        for feat, info in feats.items():
            if "error" in info:
                continue
            lines.append(
                f"| {action} | {feat} | {info['boot_median_of_medians']:.3f} | "
                f"[{info['boot_ci95'][0]:.3f}, {info['boot_ci95'][1]:.3f}] |"
            )
    lines += [""]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure-radar pre-fall pilot evaluation (repeat-level stats)."
    )
    parser.add_argument("--session-root", type=Path, required=True,
                        help="reports/real_prefall_capture_v1 (or per-action dir)")
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/prefall_pilot_eval_v1"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    by_action = load_sessions(args.session_root)
    if not by_action:
        raise SystemExit("no repeats found under --session-root")

    rq1 = rq1_direction_consistency(by_action)
    rq2 = rq2_fall_vs_instability(by_action)
    rq3 = rq3_recovery_vs_fall(by_action)
    rq4 = rq4_recovery_trend(by_action)
    rq5 = rq5_pointcount_motion_artifact(by_action)

    bootstrap: dict[str, Any] = {}
    for action, repeats in sorted(by_action.items()):
        bootstrap[action] = {}
        for feat in CANDIDATE_FEATURES:
            bootstrap[action][feat] = block_bootstrap_curve(repeats, feat)

    out_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-repeat 特征明细
    detail_rows = []
    for action, repeats in sorted(by_action.items()):
        for rep in repeats:
            detail_rows.append(rep)
    (out_dir / "per_repeat_features.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in detail_rows),
        encoding="utf-8",
    )

    for name, payload in [
        ("rq1_direction_consistency.json", rq1),
        ("rq2_fall_vs_instability.json", rq2),
        ("rq3_recovery_vs_fall.json", rq3),
        ("rq4_recovery_trend.json", rq4),
        ("rq5_pointcount_motion_artifact.json", rq5),
        ("block_bootstrap_curves.json", bootstrap),
    ]:
        (out_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    (out_dir / "report.md").write_text(
        build_report(rq1, rq2, rq3, rq4, rq5, by_action, bootstrap),
        encoding="utf-8",
    )
    print(f"reports written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
