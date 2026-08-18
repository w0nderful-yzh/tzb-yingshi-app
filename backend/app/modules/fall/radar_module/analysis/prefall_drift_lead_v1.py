"""drift_xy 时间领先性分析（pre-fall precursor vs early fall process）。

背景
----
baseline-relative 分析确认 drift_xy 是唯一通过 confound 检验的核心动态
候选。本脚本回答：drift_xy 的升高是发生在"明显持续下降开始"（t<0），
还是只发生在下降过程中（t>=0）？即：drift_xy 是 pre-fall precursor，
还是 early fall process feature？

方法
----
1. sustained_descent_onset 定义（独立于 drift_xy，避免循环论证）：
   由于真机 IWR6843 高位安装导致 z 语义反转，centroid_z / z_p90 的绝对
   方向不可靠。采用**多指标垂直动态**定义：
   - 主条件：point_count 相对 still_pre 基线骤降（身体贴近地面→点云
     稀疏），连续 N 帧满足
   - 辅助：height_range 相对基线收窄
   - 备选：z_p90 相对基线下降（若方向一致）
   **所有条件均不含 drift_xy / 任何水平特征。**
2. 对每个 fall repeat，对齐 sustained_descent_onset = t0，分析
   drift_xy 的 baseline-relative 时间曲线在窗口
   [-1.5,-1.0), [-1.0,-0.5), [-0.5,-0.2), [-0.2,0), [0,+0.5) 的表现。
3. 对 forward_instability_recovery，以"最大水平偏移时刻"为参考，看
   drift_xy 是否随后回落。
4. 输出：对齐曲线 / 每时间段 median+IQR / repeat-level direction
   consistency / fall vs recovery 时间段差异 / earliest consistent
   precursor time。

输出
----
reports/prefall_drift_lead_v1/<timestamp>/
  - fall_aligned_curves.jsonl
  - recovery_aligned_curves.jsonl
  - timewindow_analysis.json
  - report.md

Version: radar_prefall_drift_lead_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from radar_module.preprocess.prefall_features_v1 import (
    default_window_frames,
    dynamic_features,
    frame_base_features,
    record_points,
)
from radar_module.analysis.sensor_to_world_audit_v1 import (
    euler_rot,
    to_world,
)

FALL = "controlled_forward_fall"
INSTABILITY = "forward_instability_recovery"
FAST_SITTING = "fast_sitting"

# 实际安装参数（用户报告 1m, 5°），用于 world-frame 转换
WORLD_SENSOR_HEIGHT_M = 1.0
WORLD_ELEV_TILT_DEG = 5.0
WORLD_AZI_TILT_DEG = 0.0

# drift_xy 候选（baseline-relative）
DRIFT_FEATURES = ["drift_xy_0p5s", "drift_xy_1p0s", "drift_xy_1frame"]

# 时间窗口（相对 onset t0）
WINDOWS = [
    (-1.5, -1.0),
    (-1.0, -0.5),
    (-0.5, -0.2),
    (-0.2, 0.0),
    (0.0, 0.5),
]

# onset 检测参数
POINT_COUNT_DROP_RATIO = 0.5     # point_count 相对基线下降比例
CONSECUTIVE_FRAMES = 5            # 连续帧数
MIN_POINT_COUNT_ABS = 4           # 绝对点数下限（防稀疏噪声）


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _median(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def apply_world_transform(
    records: list[dict[str, Any]],
    *,
    sensor_height_m: float = WORLD_SENSOR_HEIGHT_M,
    elev_tilt_deg: float = WORLD_ELEV_TILT_DEG,
    azi_tilt_deg: float = WORLD_AZI_TILT_DEG,
) -> list[dict[str, Any]]:
    """把点云从 sensor-frame 转到 world-frame（1m, 5°），返回新 records。"""
    out: list[dict[str, Any]] = []
    for r in records:
        pts = record_points(r)
        if not pts:
            out.append(r)
            continue
        xs, ys, zs = to_world(
            list(pts),
            sensor_height_m=sensor_height_m,
            elev_tilt_deg=elev_tilt_deg,
            azi_tilt_deg=azi_tilt_deg,
        )
        new_pts = []
        for p, (nx, ny, nz) in zip(pts, zip(xs, ys, zs)):
            new_pts.append({
                "x": float(nx),
                "y": float(ny),
                "z": float(nz),
                "velocity": p.get("velocity", 0.0),
                "snr": p.get("snr"),
            })
        out.append({**r, "points": new_pts})
    return out


def _compute_frame_features(
    records: list[dict[str, Any]],
    period: float,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """逐帧基础+动态特征。返回 (base_list, dyn_list)。"""
    windows = default_window_frames(period)
    bases: list[dict[str, float]] = []
    dyns: list[dict[str, float]] = []
    history: list[dict[str, float]] = []
    for rec in records:
        base = frame_base_features(record_points(rec))
        history.append(base)
        dyn = dynamic_features(history, windows, period_seconds=period)
        bases.append(base)
        dyns.append(dyn)
    return bases, dyns


def _baseline_from_pre(
    records: list[dict[str, Any]],
    bases: list[dict[str, float]],
    feature: str,
) -> float:
    pre_vals = [
        b.get(feature, float("nan"))
        for b, r in zip(bases, records) if r.get("phase") == "still_pre"
    ]
    pre_vals = [v for v in pre_vals if np.isfinite(v)]
    return float(np.median(pre_vals)) if pre_vals else float("nan")


def _baseline_from_pre_dyn(
    records: list[dict[str, Any]],
    dyns: list[dict[str, float]],
    feature: str,
) -> float:
    """从动态特征（如 drift_xy）计算 still_pre 基线。"""
    pre_vals = [
        d.get(feature, float("nan"))
        for d, r in zip(dyns, records) if r.get("phase") == "still_pre"
    ]
    pre_vals = [v for v in pre_vals if np.isfinite(v)]
    return float(np.median(pre_vals)) if pre_vals else float("nan")


def find_sustained_descent_onset(
    records: list[dict[str, Any]],
    bases: list[dict[str, float]],
    *,
    period: float,
) -> dict[str, Any]:
    """用垂直/点云动态定义 sustained_descent_onset（独立于 drift_xy）。

    主条件：point_count 相对 still_pre 基线骤降（连续 CONSECUTIVE_FRAMES
    帧低于基线*POINT_COUNT_DROP_RATIO 且绝对点数>=MIN）。
    返回 onset 时刻（相对 action_start 的秒数）及诊断信息。
    """
    pre_pc = _baseline_from_pre(records, bases, "point_count")
    if not np.isfinite(pre_pc) or pre_pc <= MIN_POINT_COUNT_ABS:
        return {"found": False, "reason": "pre point_count too low/NaN",
                "pre_point_count": pre_pc}

    threshold = pre_pc * POINT_COUNT_DROP_RATIO
    # 阈值本身要高于绝对噪声下限，否则骤降无意义
    if threshold < MIN_POINT_COUNT_ABS:
        return {"found": False, "reason": "threshold below noise floor",
                "pre_point_count": float(pre_pc), "threshold": float(threshold)}

    action_times = []
    action_pc = []
    for r, b in zip(records, bases):
        if r.get("phase") == "action":
            mono = float(r.get("monotonic_since_repeat_start", float("nan")))
            action_times.append(mono)
            action_pc.append(b.get("point_count", float("nan")))

    action_times = np.asarray(action_times)
    action_pc = np.asarray(action_pc)
    # 找连续帧满足 pc < threshold
    cond = action_pc < threshold
    run_start = None
    for i, c in enumerate(cond):
        if c and run_start is None:
            run_start = i
        if (not c or i == len(cond) - 1) and run_start is not None:
            run_len = i - run_start if not c else i - run_start + 1
            if run_len >= CONSECUTIVE_FRAMES:
                # onset = 连续段第一帧时刻
                t0_rel = float(action_times[run_start])
                return {
                    "found": True,
                    "onset_time_rel_action_start_s": t0_rel,
                    "pre_point_count": float(pre_pc),
                    "threshold": float(threshold),
                    "run_length": run_len,
                    "index_in_action": int(run_start),
                    "method": "point_count_sustained_drop",
                }
            run_start = None
    return {"found": False, "reason": "no sustained point_count drop",
            "pre_point_count": float(pre_pc)}


# height_range 收缩 onset 检测参数
HR_SHRINK_RATIO = 0.5        # height_range 相对基线收缩比例
HR_CONSECUTIVE = 5            # 连续帧数


def find_sustained_descent_onset_hr(
    records: list[dict[str, Any]],
    bases: list[dict[str, float]],
    *,
    period: float,
) -> dict[str, Any]:
    """用 world-frame height_range 收缩定义 sustained_descent_onset。

    与 drift_xy 独立（纯垂直动态）。主条件：height_range 相对 still_pre
    基线收缩（连续 HR_CONSECUTIVE 帧低于基线*HR_SHRINK_RATIO）。
    返回 onset 时刻（相对 action_start 秒）及诊断。
    """
    pre_hr = _baseline_from_pre(records, bases, "height_range")
    if not np.isfinite(pre_hr) or pre_hr < 0.05:
        return {"found": False, "reason": "pre height_range too small/NaN",
                "pre_height_range": pre_hr}

    threshold = pre_hr * HR_SHRINK_RATIO
    if threshold < 0.03:
        return {"found": False, "reason": "hr threshold below noise floor",
                "pre_height_range": float(pre_hr), "threshold": float(threshold)}

    action_times = []
    action_hr = []
    for r, b in zip(records, bases):
        if r.get("phase") == "action":
            mono = float(r.get("monotonic_since_repeat_start", float("nan")))
            action_times.append(mono)
            action_hr.append(b.get("height_range", float("nan")))

    action_times = np.asarray(action_times)
    action_hr = np.asarray(action_hr)
    cond = action_hr < threshold
    run_start = None
    for i, c in enumerate(cond):
        if c and run_start is None:
            run_start = i
        if (not c or i == len(cond) - 1) and run_start is not None:
            run_len = i - run_start if not c else i - run_start + 1
            if run_len >= HR_CONSECUTIVE:
                t0_rel = float(action_times[run_start])
                return {
                    "found": True,
                    "onset_time_rel_action_start_s": t0_rel,
                    "pre_height_range": float(pre_hr),
                    "threshold": float(threshold),
                    "run_length": run_len,
                    "index_in_action": int(run_start),
                    "method": "height_range_sustained_shrink",
                }
            run_start = None
    return {"found": False, "reason": "no sustained height_range shrink",
            "pre_height_range": float(pre_hr)}


def find_max_drift_time(
    records: list[dict[str, Any]],
    dyns: list[dict[str, float]],
    *,
    feature: str = "drift_xy_0p5s",
) -> dict[str, Any]:
    """instability_recovery：最大水平偏移时刻（drift_xy 峰值）。"""
    action_times = []
    drift_vals = []
    for r, d in zip(records, dyns):
        if r.get("phase") == "action":
            mono = float(r.get("monotonic_since_repeat_start", float("nan")))
            action_times.append(mono)
            drift_vals.append(d.get(feature, float("nan")))
    action_times = np.asarray(action_times)
    drift_vals = np.asarray(drift_vals)
    finite = np.isfinite(drift_vals)
    if not finite.any():
        return {"found": False}
    peak_idx = int(np.nanargmax(drift_vals))
    return {
        "found": True,
        "peak_time_rel_action_start_s": float(action_times[peak_idx]),
        "peak_drift": float(drift_vals[peak_idx]),
        "method": f"max_{feature}",
    }


def align_drift_curves(
    records: list[dict[str, Any]],
    bases: list[dict[str, float]],
    dyns: list[dict[str, float]],
    *,
    onset_rel_s: float,
) -> dict[str, dict[str, list[float]]]:
    """对齐到 onset，返回每 drift 特征的时间曲线（相对 onset 秒 → 值）。"""
    # drift baseline = still_pre 中位数（drift 是动态特征，从 dyns 取）
    drift_base = {
        f: _baseline_from_pre_dyn(records, dyns, f) for f in DRIFT_FEATURES
    }
    curves: dict[str, dict[str, list[float]]] = {}
    for f in DRIFT_FEATURES:
        curves[f] = {"time": [], "value": []}
    for r, b, d in zip(records, bases, dyns):
        if r.get("phase") != "action":
            continue
        mono = float(r.get("monotonic_since_repeat_start", float("nan")))
        t_rel = mono - onset_rel_s  # 相对 onset
        for f in DRIFT_FEATURES:
            v = d.get(f, float("nan"))
            bv = drift_base[f]
            if np.isfinite(v) and np.isfinite(bv):
                curves[f]["time"].append(float(t_rel))
                curves[f]["value"].append(float(v - bv))  # baseline-relative
    return curves


def timewindow_stats(
    curves: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """对每个 drift 特征、每个时间窗，统计 median/IQR。"""
    result: dict[str, dict[str, dict[str, float]]] = {}
    for f, c in curves.items():
        times = np.asarray(c["time"])
        values = np.asarray(c["value"])
        result[f] = {}
        for lo, hi in WINDOWS:
            mask = (times >= lo) & (times < hi)
            win_vals = values[mask]
            win_vals = win_vals[np.isfinite(win_vals)]
            if win_vals.size:
                result[f][f"[{lo},{hi})"] = {
                    "median": float(np.median(win_vals)),
                    "q25": float(np.percentile(win_vals, 25)),
                    "q75": float(np.percentile(win_vals, 75)),
                    "iqr": float(np.subtract(*np.percentile(win_vals, [75, 25]))),
                    "n": int(win_vals.size),
                    "mean": float(np.mean(win_vals)),
                }
            else:
                result[f][f"[{lo},{hi})"] = {
                    "median": float("nan"), "q25": float("nan"),
                    "q75": float("nan"), "iqr": float("nan"),
                    "n": 0, "mean": float("nan"),
                }
    return result


def repeat_direction_consistency(
    all_aligned: list[dict[str, dict[str, dict[str, list[float]]]]],
    window: tuple[float, float],
) -> dict[str, dict[str, Any]]:
    """对每 drift 特征、每窗口，统计跨 repeat 的方向一致率。"""
    result: dict[str, dict[str, Any]] = {}
    lo, hi = window
    for f in DRIFT_FEATURES:
        per_repeat_medians = []
        for aligned in all_aligned:
            if f not in aligned:
                continue
            times = np.asarray(aligned[f]["time"])
            values = np.asarray(aligned[f]["value"])
            mask = (times >= lo) & (times < hi)
            win = values[mask]
            win = win[np.isfinite(win)]
            if win.size:
                per_repeat_medians.append(float(np.median(win)))
        per_repeat_medians = np.asarray(per_repeat_medians)
        if per_repeat_medians.size == 0:
            result[f] = {"n": 0}
            continue
        pos = np.sum(per_repeat_medians > 0)
        neg = np.sum(per_repeat_medians < 0)
        n = per_repeat_medians.size
        majority = max(pos, neg)
        direction = "positive" if pos == majority else "negative"
        result[f] = {
            "n": int(n),
            "positive": int(pos),
            "negative": int(neg),
            "direction": direction,
            "consistency": float(majority / n),
            "median_of_medians": float(np.median(per_repeat_medians)),
        }
    return result


def load_session_by_action(session_root: Path) -> dict[str, list[Path]]:
    by_action: dict[str, list[Path]] = {}
    for action_dir in session_root.iterdir():
        if not action_dir.is_dir():
            continue
        repeats = sorted(
            [d for d in action_dir.iterdir()
             if d.is_dir() and d.name.startswith("repeat_")],
            key=lambda d: d.name,
        )
        if repeats:
            by_action[action_dir.name] = repeats
    return by_action


def analyze_repeat(rep_dir: Path, *, world_transform: bool = False) -> dict[str, Any]:
    frames_path = rep_dir / "frames.jsonl"
    meta_path = rep_dir / "meta.json"
    if not frames_path.exists() or not meta_path.exists():
        return {"repeat_dir": str(rep_dir), "error": "missing files"}
    records = _read_jsonl(frames_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not records:
        return {"repeat_id": meta.get("repeat_id"), "error": "empty"}
    if world_transform:
        records = apply_world_transform(records)
    timestamps = [_parse_ts(r["timestamp"]) for r in records]
    deltas = [
        (b - a).total_seconds()
        for a, b in zip(timestamps[:-1], timestamps[1:])
        if (b - a).total_seconds() > 0
    ]
    period = float(np.median(deltas)) if deltas else 1.0 / 18.18
    bases, dyns = _compute_frame_features(records, period)
    return {
        "repeat_id": meta.get("repeat_id"),
        "action_name": str(meta.get("action_name")),
        "period": period,
        "records": records,
        "bases": bases,
        "dyns": dyns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="drift_xy time-lead analysis for pre-fall pilot."
    )
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/prefall_drift_lead_v1"))
    parser.add_argument(
        "--onset-method",
        choices=("point_count", "height_range"),
        default="point_count",
        help="sustained_descent_onset 定义（point_count 骤降 或 world-frame "
             "height_range 收缩，均独立于 drift_xy）",
    )
    parser.add_argument(
        "--world-transform",
        action="store_true",
        help="先做 sensor→world 转换（1m, 5°）再算特征",
    )
    args = parser.parse_args()

    by_action = load_session_by_action(args.session_root)
    fall_repeats = by_action.get(FALL, [])
    inst_repeats = by_action.get(INSTABILITY, [])
    if not fall_repeats:
        raise SystemExit(f"no fall repeats under {args.session_root}")

    fall_aligned_all: list[dict[str, Any]] = []
    fall_onset_info = []
    for rep_dir in fall_repeats:
        rep = analyze_repeat(rep_dir, world_transform=args.world_transform)
        if "error" in rep:
            continue
        if args.onset_method == "height_range":
            onset = find_sustained_descent_onset_hr(
                rep["records"], rep["bases"], period=rep["period"])
        else:
            onset = find_sustained_descent_onset(
                rep["records"], rep["bases"], period=rep["period"])
        if not onset.get("found"):
            fall_onset_info.append({
                "repeat_id": rep["repeat_id"],
                "found": False, "reason": onset.get("reason"),
            })
            continue
        onset_rel = onset["onset_time_rel_action_start_s"]
        curves = align_drift_curves(
            rep["records"], rep["bases"], rep["dyns"], onset_rel_s=onset_rel)
        fall_aligned_all.append({
            "repeat_id": rep["repeat_id"],
            "onset_rel_s": onset_rel,
            "onset_info": onset,
            "curves": curves,
        })
        fall_onset_info.append({
            "repeat_id": rep["repeat_id"], "found": True,
            "onset_rel_s": onset_rel,
            "method": onset["method"],
        })

    # instability_recovery: 以最大水平偏移为参考
    inst_aligned_all: list[dict[str, Any]] = []
    inst_peak_info = []
    for rep_dir in inst_repeats:
        rep = analyze_repeat(rep_dir, world_transform=args.world_transform)
        if "error" in rep:
            continue
        peak = find_max_drift_time(rep["records"], rep["dyns"])
        if not peak.get("found"):
            continue
        peak_rel = peak["peak_time_rel_action_start_s"]
        curves = align_drift_curves(
            rep["records"], rep["bases"], rep["dyns"], onset_rel_s=peak_rel)
        inst_aligned_all.append({
            "repeat_id": rep["repeat_id"],
            "peak_rel_s": peak_rel,
            "curves": curves,
        })
        inst_peak_info.append({
            "repeat_id": rep["repeat_id"], "peak_rel_s": peak_rel,
        })

    # 时间窗统计 + 方向一致率
    out_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    fall_report = {
        "onset_info": fall_onset_info,
        "timewindow_stats": {},
        "direction_consistency": {},
        "aligned_curves": fall_aligned_all,
    }
    inst_report = {
        "peak_info": inst_peak_info,
        "timewindow_stats": {},
        "direction_consistency": {},
        "aligned_curves": inst_aligned_all,
    }

    # 汇总 fall 的 drift 曲线（把各 repeat 的 (time,value) 拼起来）
    for group_name, aligned_all, report in [
        ("fall", fall_aligned_all, fall_report),
        ("instability", inst_aligned_all, inst_report),
    ]:
        for f in DRIFT_FEATURES:
            all_times = []
            all_vals = []
            for aligned in aligned_all:
                all_times.extend(aligned["curves"][f]["time"])
                all_vals.extend(aligned["curves"][f]["value"])
            if not all_times:
                continue
            curves = {f: {"time": all_times, "value": all_vals}}
            report["timewindow_stats"][f] = timewindow_stats(curves)[f]
        report["direction_consistency"] = {
            f: repeat_direction_consistency(
                [a["curves"] for a in aligned_all],
                (WINDOWS[2][0], WINDOWS[2][1]),
            )[f] for f in DRIFT_FEATURES
        }

    # 保存
    for name, payload in [
        ("fall_analysis.json", fall_report),
        ("instability_analysis.json", inst_report),
    ]:
        (out_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # report.md
    lines = [
        "# drift_xy 时间领先性分析",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 方法",
        "",
        "- sustained_descent_onset 用 point_count 相对 still_pre 基线骤降定义",
        "  （独立于 drift_xy，避免循环论证）",
        "- 真机 z 语义反转使 centroid_z/z_p90 绝对方向不可靠，故用点云密度骤降",
        "- fall 对齐 onset=t0；instability 以最大 drift 时刻为参考",
        "",
        "## fall onset 检测结果",
        "",
    ]
    for o in fall_onset_info:
        lines.append(
            f"- {o['repeat_id']}: found={o.get('found')} "
            f"onset_rel={o.get('onset_rel_s', 'n/a')} method={o.get('method','n/a')}"
        )
    lines += ["", "## fall 各时间窗 drift_xy（baseline-relative median）", ""]
    lines.append("| feature | 窗口 | median | IQR |")
    lines.append("|---------|------|--------|-----|")
    for f in DRIFT_FEATURES:
        for wkey, stats in fall_report["timewindow_stats"].get(f, {}).items():
            lines.append(
                f"| {f} | {wkey} | {stats['median']:.4f} | {stats['iqr']:.4f} |"
            )
    lines += ["", "## fall 方向一致率（[-0.5,-0.2) 窗口）", ""]
    for f, d in fall_report["direction_consistency"].items():
        lines.append(
            f"- {f}: {d.get('direction','n/a')} "
            f"consistency={d.get('consistency', 0):.2f} ({d.get('positive',0)}+/{d.get('negative',0)}-)"
        )
    lines += ["", "## instability 峰值检测", ""]
    for p in inst_peak_info:
        lines.append(f"- {p['repeat_id']}: peak_rel={p['peak_rel_s']}")
    lines += ["", "## instability 各时间窗 drift_xy（对齐峰值）", ""]
    lines.append("| feature | 窗口 | median | IQR |")
    lines.append("|---------|------|--------|-----|")
    for f in DRIFT_FEATURES:
        for wkey, stats in inst_report["timewindow_stats"].get(f, {}).items():
            lines.append(
                f"| {f} | {wkey} | {stats['median']:.4f} | {stats['iqr']:.4f} |"
            )
    lines += [""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"reports written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
