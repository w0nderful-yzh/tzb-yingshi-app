"""纯雷达 pre-fall pilot：action 阶段 early/middle/late 时间子阶段分析。

目的
----
判断候选特征（drift_xy / height_range / x_range / point_count）的 fall vs
其他动作差异，是在**真正跌倒前**（action early）出现，还是只在**下降/倒地后**
（action late）出现。

方法
----
每个 repeat 的 action 窗由 meta 四时间戳决定（action_start / action_end，
单调时钟）。按绝对时间把 action 窗三等分为 early/middle/late，每个子段用
该段帧的中位数作为该 repeat×stage 的特征样本（仍以 repeat 为统计单位，
不把连续帧当独立样本）。

对每个 stage、每个候选特征：
- 计算 fall 与其他动作的 repeat 级 Mann-Whitney p 值和方向
- 报告每类动作在该 stage 的特征中位数（跨 repeat 的 median of medians）

判读：
- 若 fall 的差异在 early 就显著（vs standing / vs fast_sitting / vs recovery），
  则是**真正的 pre-fall 信号**
- 若只在 late 显著，则是**下降/倒地后表现**，不是 pre-fall 前兆

输出
----
reports/prefall_pilot_stage_eval_v1/<timestamp>/
  - per_repeat_stage_features.jsonl
  - stage_comparison.json
  - stage_trajectories.json
  - report.md

Version: radar_prefall_pilot_stage_eval_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from radar_module.analysis.prefall_pilot_eval_v1 import (
    _read_json,
    extract_repeat_features,
    load_sessions,
)
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

STAGES = ("early", "middle", "late")

CANDIDATE_FEATURES = [
    "x_range",
    "drift_xy_0p5s",
    "drift_xy_1p0s",
    "drift_xy_1frame",
    "point_count",
    "height_range",
]


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def stage_slice_medians(
    records: list[dict[str, Any]],
    marks: dict[str, Any],
    *,
    period_seconds: float,
) -> dict[str, dict[str, float]]:
    """把 action 帧按单调时间三等分，返回每 stage 每候选特征的中位数。"""
    pre_m = marks["pre_start"]["monotonic"]
    as_m = marks["action_start"]["monotonic"]
    ae_m = marks["action_end"]["monotonic"]
    action_start_rel = as_m - pre_m
    action_end_rel = ae_m - pre_m
    duration = action_end_rel - action_start_rel
    if duration <= 0:
        return {}

    boundaries = [
        action_start_rel + duration * i / 3 for i in range(4)
    ]  # [start, start+d/3, start+2d/3, end]

    windows = default_window_frames(period_seconds)
    # 先算逐帧特征（复用 dynamic 依赖 history）
    rows: list[FrameFeatureRow] = []
    history: list[dict[str, float]] = []
    action_stage: list[int | None] = []
    for record in records:
        base = frame_base_features(record_points(record))
        history.append(base)
        if record.get("phase") != "action":
            continue
        mono_rel = float(record.get("monotonic_since_repeat_start", float("nan")))
        # 定位 stage
        si = None
        for i in range(3):
            if boundaries[i] <= mono_rel < boundaries[i + 1]:
                si = i
                break
        if si is None:
            if mono_rel >= boundaries[3]:
                si = 2
            elif mono_rel < boundaries[0]:
                si = 0
        dyn = dynamic_features(history, windows, period_seconds=period_seconds)
        rows.append(FrameFeatureRow(
            timestamp=_parse_ts(record["timestamp"]),
            action_name=str(record.get("action_name")),
            repeat_index=None,
            phase=f"action_{STAGES[si]}" if si is not None else "action",
            base=base,
            dynamic=dyn,
        ))
        action_stage.append(si)

    result: dict[str, dict[str, float]] = {}
    for si, stage in enumerate(STAGES):
        stage_rows = [r for r, idx in zip(rows, action_stage) if idx == si]
        if not stage_rows:
            result[stage] = {}
            continue
        result[stage] = {}
        for feat in CANDIDATE_FEATURES:
            vals = np.asarray([
                r.base.get(feat, r.dynamic.get(feat, float("nan")))
                for r in stage_rows
            ], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            result[stage][feat] = float(np.median(vals)) if vals.size else float("nan")
    return result


def _read_jsonl_frames(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mannwhitney(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def compare_stages(
    stage_by_action: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """逐 stage 对比 fall vs 其他动作（repeat 级 Mann-Whitney）。"""
    fall = stage_by_action.get(FALL_ACTION, [])
    result = {}
    for stage in STAGES:
        result[stage] = {}
        for feat in CANDIDATE_FEATURES:
            fall_vals = np.asarray([
                r["stages"].get(stage, {}).get(feat, float("nan"))
                for r in fall
            ], dtype=np.float64)
            per_other = {}
            for other in [STANDING_ACTION, FAST_SITTING_ACTION, INSTABILITY_ACTION]:
                other_rows = stage_by_action.get(other, [])
                other_vals = np.asarray([
                    r["stages"].get(stage, {}).get(feat, float("nan"))
                    for r in other_rows
                ], dtype=np.float64)
                per_other[other] = {
                    "fall_median": float(np.nanmedian(fall_vals)) if np.isfinite(fall_vals).any() else float("nan"),
                    "other_median": float(np.nanmedian(other_vals)) if np.isfinite(other_vals).any() else float("nan"),
                    "fall_minus_other": float(
                        np.nanmedian(fall_vals) - np.nanmedian(other_vals)
                    ) if np.isfinite(fall_vals).any() and np.isfinite(other_vals).any() else float("nan"),
                    "mannwhitney_p": _mannwhitney(fall_vals, other_vals),
                    "fall_vals": [float(v) for v in fall_vals],
                    "other_vals": [float(v) for v in other_vals],
                }
            result[stage][feat] = per_other
    return result


def build_report(
    stage_by_action: dict[str, list[dict[str, Any]]],
    comparison: dict[str, Any],
) -> str:
    lines = [
        "# 纯雷达 pre-fall pilot：early/middle/late 阶段分析",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 方法",
        "",
        "- 每个 repeat 的 action 窗按绝对时间三等分 early/middle/late",
        "- 每子段用该段帧的中位数作为 repeat×stage 样本（统计单位=repeat）",
        "- fall vs 其他动作用 repeat 级 Mann-Whitney",
        "",
        "## 每动作各阶段特征中位数（median of repeat medians）",
        "",
        "| action | stage | x_range | drift_xy_0p5s | drift_xy_1p0s | height_range | point_count |",
        "|--------|-------|---------|---------------|---------------|--------------|-------------|",
    ]
    for action, repeats in sorted(stage_by_action.items()):
        for stage in STAGES:
            meds = {}
            for feat in CANDIDATE_FEATURES:
                vals = np.asarray([
                    r["stages"].get(stage, {}).get(feat, float("nan"))
                    for r in repeats
                ], dtype=np.float64)
                meds[feat] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")
            lines.append(
                f"| {action} | {stage} | {meds['x_range']:.3f} | "
                f"{meds['drift_xy_0p5s']:.3f} | {meds['drift_xy_1p0s']:.3f} | "
                f"{meds['height_range']:.3f} | {meds['point_count']:.0f} |"
            )

    lines += ["", "## fall vs 其他动作 逐阶段对比（p 值）", ""]
    lines.append("| stage | feature | vs standing | vs fast_sitting | vs instability |")
    lines.append("|-------|---------|-------------|-----------------|----------------|")
    for stage in STAGES:
        for feat in CANDIDATE_FEATURES:
            per = comparison[stage][feat]
            rows = [
                f"{per[o]['mannwhitney_p']:.3f}" if np.isfinite(per[o]['mannwhitney_p'])
                else "n/a"
                for o in [STANDING_ACTION, FAST_SITTING_ACTION, INSTABILITY_ACTION]
            ]
            lines.append(f"| {stage} | {feat} | {rows[0]} | {rows[1]} | {rows[2]} |")

    lines += ["", "## 判读", ""]
    lines.append(
        "若 fall 的差异在 early 已显著（p<0.1 且方向稳定）=> 可能为 pre-fall 信号；"
        "若只在 late 显著 => 是下降/倒地后表现，非前兆。"
    )
    lines += [""]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Early/middle/late stage analysis for pre-fall pilot."
    )
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/prefall_pilot_stage_eval_v1"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    by_action = load_sessions(args.session_root)
    if not by_action:
        raise SystemExit("no repeats found under --session-root")

    # 重新按 repeat 目录加载原始帧（需要 marks）
    stage_by_action: dict[str, list[dict[str, Any]]] = {}
    root = Path(args.session_root)
    action_dirs = [root] if any(
        (root / d.name).is_dir() and d.name.startswith("repeat_")
        for d in root.iterdir()
    ) else [d for d in root.iterdir() if d.is_dir()]
    for action_dir in action_dirs:
        action_name = action_dir.name
        repeat_dirs = sorted(
            [d for d in action_dir.iterdir()
             if d.is_dir() and d.name.startswith("repeat_")],
            key=lambda d: d.name,
        )
        for rep_dir in repeat_dirs:
            meta_path = rep_dir / "meta.json"
            frames_path = rep_dir / "frames.jsonl"
            if not frames_path.exists() or not meta_path.exists():
                continue
            meta = _read_json(meta_path)
            marks = {m["name"]: m for m in meta.get("marks", [])}
            if "pre_start" not in marks:
                continue
            records = _read_jsonl_frames(frames_path)
            timestamps = [
                _parse_ts(r["timestamp"]) for r in records if r.get("timestamp")
            ]
            deltas = [
                (b - a).total_seconds()
                for a, b in zip(timestamps[:-1], timestamps[1:])
                if (b - a).total_seconds() > 0
            ]
            period = float(np.median(deltas)) if deltas else 1.0 / 18.18
            stage_feats = stage_slice_medians(records, marks, period_seconds=period)
            if not stage_feats:
                continue
            stage_by_action.setdefault(action_name, []).append({
                "repeat_id": meta.get("repeat_id"),
                "action_name": action_name,
                "stages": stage_feats,
            })

    comparison = compare_stages(stage_by_action)

    out_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-repeat per-stage 明细
    detail = []
    for action, reps in sorted(stage_by_action.items()):
        for rep in reps:
            detail.append(rep)
    (out_dir / "per_repeat_stage_features.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in detail),
        encoding="utf-8",
    )
    (out_dir / "stage_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(
        build_report(stage_by_action, comparison), encoding="utf-8"
    )
    print(f"reports written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
