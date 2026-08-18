"""纯雷达 pilot 采集质检工具。

对采集工具产出的单个 repeat（或整个 action 目录）做质量检查，回答：
- frames.jsonl 是否完整、每帧点云/phase 标注
- meta.json 四时间戳是否单调、阶段时长是否合理
- 时间戳：首末帧时长、间隔是否稳定（帧率）
- 帧率：有效帧率 vs 预期 ~18.18Hz
- 点云有效率：有非空点云的帧占比
- point_count 分布：中位数/p90/最大、按 phase 分布

用法：
  python -m radar_module.analysis.prefall_capture_qa_v1 \
    --session-root reports/real_prefall_capture_v1/standing
  # 或对单 repeat：
  python -m radar_module.analysis.prefall_capture_qa_v1 \
    --repeat-dir reports/real_prefall_capture_v1/standing/repeat_01

Version: radar_prefall_capture_qa_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_FRAME_RATE_HZ = 1000.0 / 55.0  # ~18.18


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def qa_repeat(repeat_dir: Path) -> dict[str, Any]:
    frames_path = repeat_dir / "frames.jsonl"
    meta_path = repeat_dir / "meta.json"
    if not frames_path.exists():
        return {"repeat_dir": str(repeat_dir), "error": "frames.jsonl missing"}
    records = _read_jsonl(frames_path)
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    n = len(records)
    if n == 0:
        return {"repeat_dir": str(repeat_dir), "error": "empty frames"}

    # 时间戳
    timestamps = [_parse_ts(r["timestamp"]) for r in records]
    deltas = [
        (b - a).total_seconds()
        for a, b in zip(timestamps[:-1], timestamps[1:])
    ]
    deltas = [d for d in deltas if d > 0]

    # 点云（兼容旧 points / 新 points_sensor）
    from radar_module.preprocess.prefall_features_v1 import record_points

    point_counts = np.asarray(
        [len(record_points(r)) for r in records], dtype=np.float64
    )
    nonzero = point_counts > 0

    # phase 分布
    phases = {}
    for r in records:
        p = r.get("phase")
        phases.setdefault(p, 0)
        phases[p] += 1

    # meta 四时间戳
    marks = {m["name"]: m for m in meta.get("marks", [])}
    mark_mono = {name: marks[name]["monotonic"] for name in marks}

    phase_point_stats = {}
    for p in phases:
        pc = np.asarray(
            [len(record_points(r)) for r in records if r.get("phase") == p],
            dtype=np.float64,
        )
        phase_point_stats[p] = {
            "frames": int(pc.size),
            "point_count_median": float(np.median(pc)) if pc.size else float("nan"),
            "point_count_p90": float(np.percentile(pc, 90)) if pc.size else float("nan"),
            "point_count_max": float(pc.max()) if pc.size else float("nan"),
            "empty_ratio": float((pc == 0).mean()) if pc.size else float("nan"),
        }

    result = {
        "repeat_dir": str(repeat_dir),
        "repeat_id": meta.get("repeat_id"),
        "action_name": meta.get("action_name"),
        "frame_count": n,
        "duration_seconds": (
            (timestamps[-1] - timestamps[0]).total_seconds() if n > 1 else 0.0
        ),
        "median_frame_interval_s": float(np.median(deltas)) if deltas else float("nan"),
        "effective_frame_rate_hz": float(np.median(1.0 / np.asarray(deltas))) if deltas else float("nan"),
        "expected_frame_rate_hz": EXPECTED_FRAME_RATE_HZ,
        "frame_rate_ratio": (float(np.median(1.0 / np.asarray(deltas))) / EXPECTED_FRAME_RATE_HZ if deltas else float("nan")),
        "timestamps_monotonic": all(
            b > a for a, b in zip(timestamps[:-1], timestamps[1:])
        ),
        "point_cloud_valid_ratio": float(nonzero.mean()),
        "point_count": {
            "median": float(np.median(point_counts)),
            "p90": float(np.percentile(point_counts, 90)),
            "max": float(point_counts.max()),
            "min": float(point_counts.min()),
        },
        "phase_frames": phases,
        "phase_point_stats": phase_point_stats,
        "marks": {
            name: {
                "monotonic": m["monotonic"],
                "utc_iso": m["utc_iso"],
            }
            for name, m in marks.items()
        },
        "mark_monotonic_increasing": (
            list(mark_mono.values()) == sorted(mark_mono.values())
            if len(mark_mono) >= 2 else True
        ),
    }

    # 阶段时长（由时间戳估算）
    if "pre_start" in mark_mono and "action_start" in mark_mono:
        result["still_pre_duration_s"] = (
            marks["action_start"]["monotonic"] - marks["pre_start"]["monotonic"]
        )
    if "action_start" in mark_mono and "action_end" in mark_mono:
        result["action_duration_s"] = (
            marks["action_end"]["monotonic"] - marks["action_start"]["monotonic"]
        )
    if "action_end" in mark_mono and "post_end" in mark_mono:
        result["still_post_duration_s"] = (
            marks["post_end"]["monotonic"] - marks["action_end"]["monotonic"]
        )
    return result


def qa_action(action_dir: Path) -> dict[str, Any]:
    repeat_dirs = sorted(
        [d for d in action_dir.iterdir() if d.is_dir() and d.name.startswith("repeat_")],
        key=lambda d: d.name,
    )
    if not repeat_dirs:
        return {"action_dir": str(action_dir), "error": "no repeat dirs"}
    results = [qa_repeat(d) for d in repeat_dirs]
    manifest = {}
    if (action_dir / "manifest.json").exists():
        manifest = json.loads((action_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = {
        "action_dir": str(action_dir),
        "action_name": manifest.get("action_name") or action_dir.name,
        "repeat_count": len(results),
        "repeats": results,
    }
    # 汇总指标
    valid = [r for r in results if "error" not in r]
    if valid:
        summary["aggregate"] = {
            "frame_rate_ratio_median": float(np.median(
                [r["frame_rate_ratio"] for r in valid]
            )),
            "point_cloud_valid_ratio_mean": float(np.mean(
                [r["point_cloud_valid_ratio"] for r in valid]
            )),
            "point_count_median_across_repeats": float(np.mean(
                [r["point_count"]["median"] for r in valid]
            )),
            "any_repeat_issue": any(
                r.get("frame_rate_ratio", 1.0) < 0.7
                or r.get("point_cloud_valid_ratio", 1.0) < 0.5
                or not r.get("timestamps_monotonic", True)
                or not r.get("mark_monotonic_increasing", True)
                for r in valid
            ),
        }
    return summary


def _print_qa(report: dict[str, Any]) -> None:
    if "error" in report:
        print(f"[ERROR] {report}")
        return
    for r in report.get("repeats", [report]):
        if "error" in r:
            print(f"[ERROR] {r['repeat_dir']}: {r['error']}")
            continue
        print(f"\n=== {r['repeat_dir']} ===")
        print(f"  action={r['action_name']} repeat_id={r['repeat_id']}")
        print(f"  帧数={r['frame_count']} 时长={r['duration_seconds']:.1f}s")
        print(
            f"  帧率={r['effective_frame_rate_hz']:.2f}Hz "
            f"(预期{EXPECTED_FRAME_RATE_HZ:.1f}Hz, 比例={r['frame_rate_ratio']:.2f})"
        )
        print(f"  时间戳单调={r['timestamps_monotonic']}")
        print(f"  点云有效率={r['point_cloud_valid_ratio']:.2f}")
        pc = r["point_count"]
        print(f"  point_count: 中位={pc['median']:.0f} p90={pc['p90']:.0f} "
              f"max={pc['max']:.0f} min={pc['min']:.0f}")
        print(f"  phase帧数={r['phase_frames']}")
        for p, st in r["phase_point_stats"].items():
            print(f"    [{p}] 帧={st['frames']} 中位点={st['point_count_median']:.0f} "
                  f"p90={st['point_count_p90']:.0f} 空帧比={st['empty_ratio']:.2f}")
        marks = r.get("marks", {})
        if marks:
            print(f"  四时间戳单调={r.get('mark_monotonic_increasing')}")
            if "still_pre_duration_s" in r:
                print(f"  still_pre={r['still_pre_duration_s']:.1f}s "
                      f"action={r['action_duration_s']:.1f}s "
                      f"still_post={r['still_post_duration_s']:.1f}s")
    if "aggregate" in report:
        agg = report["aggregate"]
        print(f"\n=== 汇总 ===")
        print(f"  平均帧率比例={agg['frame_rate_ratio_median']:.2f} "
              f"平均点云有效率={agg['point_cloud_valid_ratio_mean']:.2f}")
        print(f"  跨repeat point_count中位均值={agg['point_count_median_across_repeats']:.1f}")
        print(f"  存在链路问题={agg['any_repeat_issue']}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QA for pure-radar pilot capture repeats."
    )
    parser.add_argument("--session-root", type=Path, default=None,
                        help="action dir, e.g. reports/real_prefall_capture_v1/standing")
    parser.add_argument("--repeat-dir", type=Path, default=None,
                        help="single repeat dir")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.repeat_dir:
        report = qa_repeat(args.repeat_dir)
        _print_qa(report)
        return 0
    if not args.session_root:
        raise SystemExit("--session-root or --repeat-dir required")
    report = qa_action(args.session_root)
    _print_qa(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
