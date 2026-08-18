"""纯雷达 pre-fall 特征可观测性分析。

目标（对应研究问题 RQ1-RQ5）：
  RQ1: 哪些特征在同类动作多次重复中稳定（跨 repeat 变异小）
  RQ2: 哪些特征能区分 controlled_forward_fall 与 sitting/fast_sitting/
       bending/squatting/instability_recovery
  RQ3: 哪些特征只是距离或站位差异，应排除（与 centroid 位置/距离强相关）
  RQ4: 哪些特征在恢复动作中回到基线，而跌倒过程中持续恶化
       （用 phase 标注 still_pre/action/still_post 检验恢复性）
  RQ5: 能否构造 centroid drift + height trend + Doppler spread + recovery
       作为纯雷达 pre-fall 特征组合

输入（两种来源，均可）：
  1. 采集工具输出目录 reports/real_prefall_capture_v1/<action>/session.jsonl
     （含 repeat_index / phase）
  2. 现有真机 phase session 目录（如 reports/continuous_scene_validation_v1/
     */phases/*/session.jsonl），action_name 从 manifest 读取

输出：
  reports/prefall_feature_observability_v1/<timestamp>/
    - per_frame_features.csv
    - feature_summary.json
    - rq1_repeat_stability.json
    - rq2_discrimination.json
    - rq3_position_artifact.json
    - rq4_recovery.json
    - rq5_combination.json
    - report.md

Version: radar_prefall_feature_observability_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from radar_module.preprocess.prefall_features_v1 import (
    FrameFeatureRow,
    default_window_frames,
    dynamic_features,
    frame_base_features,
    record_points,
)

# 目标动作的粗分组
FALL_ACTION = "controlled_forward_fall"
RECOVERY_ACTIONS = {
    "forward_lean_recovery",
    "forward_instability_recovery",
    "lateral_instability_recovery",
}
ADL_ACTIONS = {
    "sitting",
    "fast_sitting",
    "bending",
    "squatting",
    "walking",
    "standing",
}


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


def _action_name_from_manifest(session_dir: Path) -> str:
    for manifest_name in ("manifest.json", "session_manifest.json"):
        manifest = session_dir / manifest_name
        if manifest.exists():
            data = _read_json(manifest)
            name = data.get("action_name")
            if name:
                return str(name)
    # 从目录名推断（fallback）
    return session_dir.name


# 现有真机 phase session 的动作名 → 用户协议类别归一化
ACTION_ALIASES = {
    "standing_natural_motion": "standing",
    "standing_video_sync": "standing",
    "natural_walking": "walking",
    "fast_sit_repeated": "fast_sitting",
    "fast_squat_repeated": "squatting",
    "sudden_bend_repeated": "bending",
    "standing_fall_preparation": "fall_preparation",
    "post_fall_lying": "post_fall_lying",
    "assisted_recovery_to_standing": "assisted_recovery",
    "standing_recovery": "standing_recovery",
}


def _normalize_action(name: str) -> str:
    return ACTION_ALIASES.get(name, name)


def _period_seconds(records: Sequence[dict[str, Any]]) -> float:
    timestamps = [_parse_timestamp(r["timestamp"]) for r in records]
    if len(timestamps) < 2:
        return 1.0 / 18.18
    deltas = [
        (b - a).total_seconds()
        for a, b in zip(timestamps[:-1], timestamps[1:])
        if (b - a).total_seconds() > 0
    ]
    if not deltas:
        return 1.0 / 18.18
    return float(np.median(deltas))


def extract_frame_features(
    records: Sequence[dict[str, Any]],
    *,
    action_name: str,
) -> list[FrameFeatureRow]:
    """从 session.jsonl 提取逐帧特征。"""
    if not records:
        return []
    period = _period_seconds(records)
    windows = default_window_frames(period)

    base_series = [frame_base_features(record_points(r)) for r in records]
    rows: list[FrameFeatureRow] = []
    history: list[dict[str, float]] = []
    for idx, record in enumerate(records):
        base = base_series[idx]
        history.append(base)
        dyn = dynamic_features(history, windows, period_seconds=period)
        rows.append(
            FrameFeatureRow(
                timestamp=_parse_timestamp(record["timestamp"]),
                action_name=action_name,
                repeat_index=record.get("repeat_index"),
                phase=record.get("phase"),
                base=base,
                dynamic=dyn,
            )
        )
    return rows


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


def _effect_size_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d（有符号：a-b）。"""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0)
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann-Whitney U
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined)
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1)
    # 处理并列
    _, first = np.unique(combined[order], return_index=True)
    for idx in first:
        ties = combined[order] == combined[order][idx]
        ranks[order[ties]] = ranks[order[ties]].mean()
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _median_iqr(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"median": float("nan"), "iqr": float("nan"), "count": 0}
    return {
        "median": float(np.median(values)),
        "iqr": float(np.subtract(*np.percentile(values, [75, 25]))),
        "count": int(values.size),
    }


def rq1_repeat_stability(rows_by_action: dict[str, list[FrameFeatureRow]]) -> dict[str, Any]:
    """RQ1: 同类动作多次重复中，哪些特征稳定。

    对每个动作，把帧按 repeat_index 分组，跨 repeat 计算特征中位数的变异。
    """
    result: dict[str, Any] = {}
    feature_names: list[str] | None = None
    for action, rows in sorted(rows_by_action.items()):
        repeats: dict[int, list[FrameFeatureRow]] = {}
        for row in rows:
            if row.repeat_index is None:
                continue
            repeats.setdefault(int(row.repeat_index), []).append(row)
        if len(repeats) < 2:
            continue
        if feature_names is None:
            feature_names = _feature_names(rows[0])

        stability: dict[str, dict[str, float]] = {}
        for name in feature_names:
            medians = np.asarray(
                [np.nanmedian(_col(rp, name)) for rp in repeats.values()],
                dtype=np.float64,
            )
            medians = medians[np.isfinite(medians)]
            if medians.size < 2:
                stability[name] = {"cv": float("nan"), "range": float("nan"), "n": 0}
                continue
            mean = abs(float(medians.mean()))
            cv = float(medians.std(ddof=1) / mean) if mean > 1e-9 else float("inf")
            stability[name] = {
                "cv": cv,
                "range": float(medians.max() - medians.min()),
                "n": int(medians.size),
            }
        result[action] = {
            "repeat_count": len(repeats),
            "frames_per_repeat": {
                r: len(rp) for r, rp in sorted(repeats.items())
            },
            "feature_stability": stability,
        }
    return result


def rq2_discrimination(rows_by_action: dict[str, list[FrameFeatureRow]]) -> dict[str, Any]:
    """RQ2: fall 与 ADL/recovery 的区分度。

    用 action 阶段帧（phase=='action' 或整段），逐特征算 fall vs 其他 的
    Cohen's d 和 AUROC。
    """
    def action_rows(rows: list[FrameFeatureRow]) -> list[FrameFeatureRow]:
        with_phase = [r for r in rows if r.phase is not None]
        return with_phase if with_phase else rows

    fall_rows = action_rows(rows_by_action.get(FALL_ACTION, []))
    if not fall_rows:
        return {"error": f"no frames for {FALL_ACTION}"}

    feature_names = _feature_names(fall_rows[0])
    others = {a: action_rows(r) for a, r in rows_by_action.items()
              if a != FALL_ACTION and action_rows(r)}

    comparison: dict[str, Any] = {}
    for name in feature_names:
        fall = _col(fall_rows, name)
        per_other: dict[str, dict[str, float]] = {}
        for other_name, other_rows_i in sorted(others.items()):
            other = _col(other_rows_i, name)
            per_other[other_name] = {
                "cohen_d": _effect_size_d(fall, other),
                "auroc": _auroc(fall, other),
                "fall_median": float(np.nanmedian(fall)) if np.isfinite(fall).any() else float("nan"),
                "other_median": float(np.nanmedian(other)) if np.isfinite(other).any() else float("nan"),
            }
        comparison[name] = per_other

    # 汇总：按跨其他动作平均 |AUROC| 排序
    scores: dict[str, float] = {}
    for name, per in comparison.items():
        aurocs = [v["auroc"] for v in per.values() if np.isfinite(v["auroc"])]
        scores[name] = float(np.mean([abs(a - 0.5) for a in aurocs])) if aurocs else float("nan")
    ranking = sorted(scores.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else -1, reverse=True)

    return {
        "fall_action": FALL_ACTION,
        "comparisons": comparison,
        "ranking_by_discriminability": [
            {"feature": name, "mean_abs_auroc_offset": score}
            for name, score in ranking
        ],
    }


def rq3_position_artifact(rows_by_action: dict[str, list[FrameFeatureRow]]) -> dict[str, Any]:
    """RQ3: 哪些特征与站位/距离强相关（应排除）。

    把每帧的 centroid 位置/距离当作协变量，计算各特征与
    centroid_x / centroid_y / centroid_z / range_mean 的 Spearman 相关。
    """
    all_rows = [r for rows in rows_by_action.values() for r in rows]
    if len(all_rows) < 10:
        return {"error": "insufficient frames"}

    feature_names = _feature_names(all_rows[0])
    anchors = {
        "centroid_x": _col(all_rows, "centroid_x"),
        "centroid_y": _col(all_rows, "centroid_y"),
        "centroid_z": _col(all_rows, "centroid_z"),
        "range_mean": _col(all_rows, "range_mean"),
    }

    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]
        if a.size < 5 or np.all(a == a[0]) or np.all(b == b[0]):
            return float("nan")
        from scipy.stats import rankdata  # type: ignore

        ra = rankdata(a)
        rb = rankdata(b)
        if np.std(ra) == 0 or np.std(rb) == 0:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])

    result: dict[str, Any] = {}
    for name in feature_names:
        values = _col(all_rows, name)
        if not np.isfinite(values).any():
            continue
        corrs = {anchor: _spearman(values, anchor_vals)
                 for anchor, anchor_vals in anchors.items()}
        result[name] = {
            "max_abs_corr": float(max(
                (abs(c) for c in corrs.values() if np.isfinite(c)), default=float("nan")
            )),
            "corrs": corrs,
        }
    ranking = sorted(
        result.items(),
        key=lambda kv: kv[1]["max_abs_corr"] if np.isfinite(kv[1]["max_abs_corr"]) else -1,
        reverse=True,
    )
    return {
        "correlation_with_position": result,
        "ranking_most_position_dependent": [
            {"feature": name, "max_abs_corr": v["max_abs_corr"]}
            for name, v in ranking
        ],
    }


def rq4_recovery(rows_by_action: dict[str, list[FrameFeatureRow]]) -> dict[str, Any]:
    """RQ4: 恢复动作回到基线 vs 跌倒持续恶化。

    只对带 phase 标注（still_pre/action/still_post）的会话。
    计算每个特征在 still_pre（基线）vs still_post（动作后）的差异。
    recovery 动作应回到基线（|post-pre| 小），fall 应持续偏离（|post-pre| 大）。
    """
    result: dict[str, Any] = {}
    for action, rows in sorted(rows_by_action.items()):
        with_phase = [r for r in rows if r.phase is not None]
        if not with_phase:
            continue
        pre = [r for r in with_phase if r.phase == "still_pre"]
        post = [r for r in with_phase if r.phase == "still_post"]
        if not pre or not post:
            continue
        feature_names = _feature_names(with_phase[0])
        per_feature: dict[str, dict[str, float]] = {}
        for name in feature_names:
            pre_vals = _col(pre, name)
            post_vals = _col(post, name)
            pre_m = np.nanmedian(pre_vals) if np.isfinite(pre_vals).any() else float("nan")
            post_m = np.nanmedian(post_vals) if np.isfinite(post_vals).any() else float("nan")
            per_feature[name] = {
                "pre_median": pre_m,
                "post_median": post_m,
                "post_minus_pre": post_m - pre_m if np.isfinite(pre_m) and np.isfinite(post_m) else float("nan"),
            }
        result[action] = {
            "pre_frames": len(pre),
            "post_frames": len(post),
            "features": per_feature,
        }
    return result


def rq5_combination(
    rows_by_action: dict[str, list[FrameFeatureRow]],
) -> dict[str, Any]:
    """RQ5: 构造纯雷达 pre-fall 特征组合。

    组合 = [drift_xy, height trend, doppler spread, recovery trend]。
    计算每个候选特征（或简单组合）在 fall vs ADL 的 AUROC。
    """
    fall_rows = rows_by_action.get(FALL_ACTION, [])
    adl_rows = [r for a, r in rows_by_action.items()
                if a in ADL_ACTIONS and a != FALL_ACTION]
    adl_rows = [r for group in adl_rows for r in group]
    if not fall_rows or not adl_rows:
        return {"error": "insufficient fall/adl rows"}

    candidates = [
        "drift_xy_1frame",
        "drift_xy_0p5s",
        "drift_xy_1p0s",
        "delta_z_0p2s",
        "delta_z_0p5s",
        "delta_z_1p0s",
        "slope_z_0p5s",
        "slope_z_1p0s",
        "delta_height_0p5s",
        "var_doppler_0p5s",
        "doppler_std",
        "doppler_max_abs",
        "moving_fraction",
    ]
    result: dict[str, Any] = {}
    for name in candidates:
        fall = _col(fall_rows, name)
        adl = _col(adl_rows, name)
        result[name] = {
            "fall_median": float(np.nanmedian(fall)) if np.isfinite(fall).any() else float("nan"),
            "adl_median": float(np.nanmedian(adl)) if np.isfinite(adl).any() else float("nan"),
            "cohen_d": _effect_size_d(fall, adl),
            "auroc": _auroc(fall, adl),
        }
    return {"candidates": result}


def _csv_escape(value: Any) -> str:
    text = "NaN" if value is None or (isinstance(value, float) and not np.isfinite(value)) else str(value)
    if "," in text or '"' in text or "\n" in text:
        return '"' + text.replace('"', '""') + '"'
    return text


def _write_csv(path: Path, rows: Sequence[FrameFeatureRow]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = _feature_names(rows[0])
    header = ["timestamp", "action_name", "repeat_index", "phase"] + names
    lines = [",".join(header)]
    for row in rows:
        data = row.to_dict()
        lines.append(",".join(_csv_escape(data.get(h)) for h in header))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def collect_sessions(session_inputs: Sequence[str]) -> dict[str, list[FrameFeatureRow]]:
    """从多种输入收集 (action_name -> frame rows)。

    session_inputs 可以是：
      - session.jsonl 文件路径
      - phase 会话目录（含 session.jsonl + manifest.json）
      - 采集输出目录（reports/real_prefall_capture_v1/<action>/）
    """
    rows_by_action: dict[str, list[FrameFeatureRow]] = {}
    for item in session_inputs:
        p = Path(item)
        if p.is_dir():
            session_file = p / "session.jsonl"
            if not session_file.exists():
                continue
            action = _action_name_from_manifest(p)
        else:
            session_file = p
            action = p.parent.name

        records = _read_jsonl(session_file)
        if not records:
            print(f"  skip empty: {item}", file=sys.stderr)
            continue
        action = _normalize_action(action)
        action_rows = extract_frame_features(records, action_name=action)
        rows_by_action.setdefault(action, []).extend(action_rows)
        print(f"  {action}: {len(action_rows)} frames from {item}", file=sys.stderr)
    return rows_by_action


def build_report(
    rq1: dict[str, Any],
    rq2: dict[str, Any],
    rq3: dict[str, Any],
    rq4: dict[str, Any],
    rq5: dict[str, Any],
    rows_by_action: dict[str, list[FrameFeatureRow]],
) -> str:
    lines = [
        "# 纯雷达 pre-fall 特征可观测性分析",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 输入动作与帧数",
        "",
        "| action | frames |",
        "|--------|--------|",
    ]
    for action, rows in sorted(rows_by_action.items()):
        lines.append(f"| {action} | {len(rows)} |")
    lines += ["", "## RQ2: 区分度 (fall vs ADL)", ""]
    if "comparisons" in rq2:
        ranking = rq2.get("ranking_by_discriminability", [])[:10]
        lines.append("按平均 |AUROC-0.5| 排序的特征（越大越可区分）：")
        lines.append("")
        lines.append("| feature | mean_abs_auroc_offset |")
        lines.append("|---------|----------------------|")
        for entry in ranking:
            lines.append(
                f"| {entry['feature']} | {entry['mean_abs_auroc_offset']:.3f} |"
            )
    lines += ["", "## RQ3: 站位伪特征（max |Spearman corr| 最高=最依赖站位）", ""]
    if "ranking_most_position_dependent" in rq3:
        lines.append("| feature | max_abs_corr |")
        lines.append("|---------|-------------|")
        for entry in rq3["ranking_most_position_dependent"][:12]:
            lines.append(
                f"| {entry['feature']} | {entry['max_abs_corr']:.3f} |"
            )
    lines += ["", "## RQ4: 恢复 vs 恶化 (still_post - still_pre)", ""]
    if rq4:
        lines.append("| action | pre/post frames | 关键特征 post-pre |")
        lines.append("|--------|----------------|-------------------|")
        for action, info in sorted(rq4.items()):
            feats = info.get("features", {})
            picks = {
                k: feats[k]["post_minus_pre"]
                for k in ("delta_z_0p5s", "slope_z_0p5s", "drift_xy_0p5s")
                if k in feats
            }
            pick_str = ", ".join(f"{k}={v:.3f}" for k, v in picks.items())
            lines.append(
                f"| {action} | {info['pre_frames']}/{info['post_frames']} | {pick_str} |"
            )
    lines += ["", "## RQ5: 候选组合特征", ""]
    if "candidates" in rq5:
        lines.append("| feature | auroc | cohen_d | fall_med | adl_med |")
        lines.append("|---------|-------|---------|----------|---------|")
        cands = rq5["candidates"]
        for name in sorted(cands, key=lambda k: -abs(cands[k]["auroc"] - 0.5)):
            c = cands[name]
            lines.append(
                f"| {name} | {c['auroc']:.3f} | {c['cohen_d']:.2f} | "
                f"{c['fall_median']:.3f} | {c['adl_median']:.3f} |"
            )
    lines += [""]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure-radar pre-fall feature observability analysis."
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        required=True,
        help="session.jsonl paths, phase dirs, or capture output dirs",
    )
    parser.add_argument("--output-root", type=Path, default=Path("reports/prefall_feature_observability_v1"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    rows_by_action = collect_sessions(args.sessions)
    if not rows_by_action:
        raise SystemExit("no frames collected")

    rq1 = rq1_repeat_stability(rows_by_action)
    rq2 = rq2_discrimination(rows_by_action)
    rq3 = rq3_position_artifact(rows_by_action)
    rq4 = rq4_recovery(rows_by_action)
    rq5 = rq5_combination(rows_by_action)

    out_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = [r for rows in rows_by_action.values() for r in rows]
    _write_csv(out_dir / "per_frame_features.csv", all_rows)
    _write_json(out_dir / "rq1_repeat_stability.json", rq1)
    _write_json(out_dir / "rq2_discrimination.json", rq2)
    _write_json(out_dir / "rq3_position_artifact.json", rq3)
    _write_json(out_dir / "rq4_recovery.json", rq4)
    _write_json(out_dir / "rq5_combination.json", rq5)
    _write_json(out_dir / "feature_summary.json", {
        "actions": {a: len(r) for a, r in sorted(rows_by_action.items())},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "radar_prefall_feature_observability_v1",
    })
    (out_dir / "report.md").write_text(
        build_report(rq1, rq2, rq3, rq4, rq5, rows_by_action),
        encoding="utf-8",
    )
    print(f"reports written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
