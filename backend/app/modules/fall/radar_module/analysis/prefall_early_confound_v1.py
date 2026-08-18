"""检查 height_range early 高值是否存在 controlled-action confound。

问题
----
stage 分析显示 fall 的 height_range 在 action early 就显著高。需要检查：
- height_range early 高值是否与 point_count（点云密度/CFAR）相关？
- 是否与 x_range（身体水平展开/手臂张开代理）相关？
- 是否与动作开始瞬间的"人为准备"（fall 操作者知道要跌倒而提前
  前倾/张开手臂）相关？

方法
----
对 fall 的 5 个 repeat：
1. 计算 early 阶段 height_range / point_count / x_range / drift_xy
   中位数
2. 计算 height_range 与其余特征在 early 阶段的 Spearman 相关
   （repeat 为单位，n=5，方向性证据）
3. 检查 early 与 pre 阶段（动作前静止）的差异：若 early 相对 pre
   已大幅变化，说明动作开始就有人为准备成分

输出
----
reports/prefall_early_confound_v1/<timestamp>/report.json + report.md

Version: radar_prefall_early_confound_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

FALL = "controlled_forward_fall"
CANDIDATE = "height_range"
CORRELATES = ["point_count", "x_range", "drift_xy_1p0s", "drift_xy_0p5s"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 4 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    from scipy.stats import rankdata

    ra = rankdata(a)
    rb = rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _phase_median_features(records: list[dict[str, Any]], phase: str) -> dict[str, float]:
    """从原始帧记录计算某阶段（still_pre/still_post）各特征中位数。"""
    from radar_module.preprocess.prefall_features_v1 import (
        default_window_frames,
        dynamic_features,
        frame_base_features,
        record_points,
    )

    phase_records = [r for r in records if r.get("phase") == phase]
    if not phase_records:
        return {f: float("nan") for f in [CANDIDATE] + CORRELATES}
    timestamps = [
        _parse_ts(r["timestamp"]) for r in phase_records if r.get("timestamp")
    ]
    deltas = [
        (b - a).total_seconds()
        for a, b in zip(timestamps[:-1], timestamps[1:])
        if (b - a).total_seconds() > 0
    ]
    period = float(np.median(deltas)) if deltas else 1.0 / 18.18
    windows = default_window_frames(period)
    history: list[dict[str, float]] = []
    feats: dict[str, float] = {f: float("nan") for f in [CANDIDATE] + CORRELATES}
    for rec in phase_records:
        base = frame_base_features(record_points(rec))
        history.append(base)
        dyn = dynamic_features(history, windows, period_seconds=period)
        for f in feats:
            v = base.get(f, dyn.get(f, float("nan")))
            feats[f] = v if np.isfinite(v) else feats[f]
    # 取每特征的中位数
    values: dict[str, list[float]] = {f: [] for f in feats}
    history = []
    for rec in phase_records:
        base = frame_base_features(record_points(rec))
        history.append(base)
        dyn = dynamic_features(history, windows, period_seconds=period)
        for f in values:
            v = base.get(f, dyn.get(f, float("nan")))
            if np.isfinite(v):
                values[f].append(float(v))
    return {f: (float(np.median(values[f])) if values[f] else float("nan"))
            for f in values}


def _parse_ts(value: str):
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _find_fall_repeat_dirs(session_root: Path) -> list[Path]:
    """在 session_root 下定位 fall 的 repeat 目录。"""
    dirs = []
    for action_dir in session_root.iterdir():
        if not action_dir.is_dir():
            continue
        if action_dir.name != FALL:
            continue
        for rep in action_dir.iterdir():
            if rep.is_dir() and rep.name.startswith("repeat_") and (rep / "frames.jsonl").exists():
                dirs.append(rep)
    return sorted(dirs, key=lambda d: d.name)


def run_confound_check(stage_jsonl: Path, session_root: Path) -> dict[str, Any]:
    rows = _read_jsonl(stage_jsonl)
    fall_rows = [r for r in rows if r["action_name"] == FALL]
    if not fall_rows:
        return {"error": "no fall rows"}

    # 从原始帧算 pre 阶段（stage jsonl 无 pre）
    fall_repeat_dirs = _find_fall_repeat_dirs(session_root)
    pre_medians: dict[str, list[float]] = {f: [] for f in [CANDIDATE] + CORRELATES}
    for rep_dir in fall_repeat_dirs:
        records = _read_jsonl(rep_dir / "frames.jsonl")
        pre_feats = _phase_median_features(records, "still_pre")
        for f in pre_medians:
            pre_medians[f].append(pre_feats[f])

    report: dict[str, Any] = {
        "action": FALL,
        "n_repeats": len(fall_rows),
        "repeat_ids": [r["repeat_id"] for r in fall_rows],
        "early_medians": {},
        "pre_vs_early": {},
        "correlations": {},
    }

    for feat in [CANDIDATE] + CORRELATES:
        early_vals = np.asarray([
            r["stages"]["early"].get(feat, float("nan")) for r in fall_rows
        ], dtype=np.float64)
        report["early_medians"][feat] = {
            "median": float(np.nanmedian(early_vals)),
            "values": [float(v) for v in early_vals],
            "range": [float(np.nanmin(early_vals)), float(np.nanmax(early_vals))]
            if np.isfinite(early_vals).any() else [float("nan"), float("nan")],
        }

    # pre vs early：pre 阶段是动作前静止，若 early 已大幅偏离 pre，
    # 说明动作开始瞬间就有人为准备成分
    for feat in [CANDIDATE] + CORRELATES:
        pre_vals = np.asarray(pre_medians.get(feat, []), dtype=np.float64)
        early_vals = np.asarray([
            r["stages"]["early"].get(feat, float("nan")) for r in fall_rows
        ], dtype=np.float64)
        report["pre_vs_early"][feat] = {
            "pre_median": float(np.nanmedian(pre_vals)),
            "early_median": float(np.nanmedian(early_vals)),
            "early_minus_pre_median": float(
                np.nanmedian(early_vals) - np.nanmedian(pre_vals)
            ),
            "early_exceeds_pre_ratio": float(
                np.mean(early_vals > pre_vals)
            ) if np.isfinite(early_vals).any() and np.isfinite(pre_vals).any() else float("nan"),
        }

    # height_range 与各特征的 Spearman（repeat 为单位）
    hr = np.asarray([
        r["stages"]["early"].get(CANDIDATE, float("nan")) for r in fall_rows
    ], dtype=np.float64)
    for feat in CORRELATES:
        vals = np.asarray([
            r["stages"]["early"].get(feat, float("nan")) for r in fall_rows
        ], dtype=np.float64)
        report["correlations"][feat] = {
            "spearman_with_hr": _spearman(hr, vals),
            "hr_values": [float(v) for v in hr],
            "correlate_values": [float(v) for v in vals],
        }

    # 判读
    corr_pts = report["correlations"]["point_count"]["spearman_with_hr"]
    corr_xr = report["correlations"]["x_range"]["spearman_with_hr"]
    early_exceed = report["pre_vs_early"][CANDIDATE]["early_exceeds_pre_ratio"]

    notes = []
    if np.isfinite(corr_pts) and abs(corr_pts) > 0.7:
        notes.append(
            f"height_range early 与 point_count 强相关(rho={corr_pts:.2f})："
            "可能由 CFAR 点数变化驱动，标记 controlled-action confound"
        )
    if np.isfinite(corr_xr) and abs(corr_xr) > 0.7:
        notes.append(
            f"height_range early 与 x_range 强相关(rho={corr_xr:.2f})："
            "可能由身体水平展开/手臂张开驱动，标记 controlled-action confound"
        )
    if np.isfinite(early_exceed) and early_exceed >= 0.8:
        notes.append(
            f"fall 的 early height_range 在 {early_exceed:.0%} repeat 中超过 pre："
            "动作开始瞬间已偏离静止基线，存在人为准备成分"
        )
    if not notes:
        notes.append("未发现显著 confound 信号（n=5 证据有限）")
    report["notes"] = notes
    return report


def _stage_or_nan(row: dict[str, Any], stage: str, feat: str) -> float:
    try:
        return row["stages"][stage].get(feat, float("nan"))
    except (KeyError, TypeError):
        return float("nan")


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# height_range early 值 confound 检查",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"动作: {report.get('action')}，repeat 数: {report.get('n_repeats')}",
        "",
        "## early 阶段特征中位数",
        "",
        "| feature | median | values |",
        "|---------|--------|--------|",
    ]
    for feat, info in report.get("early_medians", {}).items():
        lines.append(
            f"| {feat} | {info['median']:.3f} | "
            f"{[round(v,2) for v in info['values']]} |"
        )
    lines += ["", "## pre vs early（动作开始瞬间是否已偏离基线）", ""]
    lines.append("| feature | pre_med | early_med | early-pre | exceed_ratio |")
    lines.append("|---------|---------|-----------|-----------|--------------|")
    for feat, info in report.get("pre_vs_early", {}).items():
        lines.append(
            f"| {feat} | {info['pre_median']:.3f} | {info['early_median']:.3f} | "
            f"{info['early_minus_pre_median']:+.3f} | {info['early_exceeds_pre_ratio']:.2f} |"
        )
    lines += ["", "## height_range early 与各特征 Spearman（repeat 级）", ""]
    lines.append("| feature | rho |")
    lines.append("|---------|-----|")
    for feat, info in report.get("correlations", {}).items():
        rho = info["spearman_with_hr"]
        lines.append(f"| {feat} | {rho:.2f} |")
    lines += ["", "## 判读", ""]
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    lines += [""]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check height_range early confound in pre-fall pilot."
    )
    parser.add_argument("--stage-jsonl", type=Path, required=True)
    parser.add_argument("--session-root", type=Path, required=True,
                        help="reports/real_prefall_capture_v1 (定位原始帧算 pre 阶段)")
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/prefall_early_confound_v1"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = run_confound_check(args.stage_jsonl, args.session_root)
    out_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confound_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(
        build_markdown(report), encoding="utf-8"
    )
    print(f"reports written to {out_dir}")
    for note in report.get("notes", []):
        print("NOTE:", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
