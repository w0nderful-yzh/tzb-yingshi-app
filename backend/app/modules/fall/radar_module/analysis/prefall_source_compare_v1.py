"""纯雷达三数据源对比 sanity check：sensor 点云 vs world 点云 vs TI tracking target。

目的
----
真机采集修复后（points_sensor + points_world + targets），对每个动作
（standing / fast_sitting / controlled_forward_fall）比较：
- 原始点云 centroid（sensor-frame）
- world-frame 点云 centroid
- TI tracking target pos/vel（world-frame，平滑滤波）

看 TI tracking target 的垂直位置/速度是否比原始点云 centroid 更符合
"站立→下降→低位"变化。若 tracking target 仍无法给出稳定下降时序，
再决定是否停止 radar-only pre-fall 探索。

输入
----
reports/real_prefall_capture_v1/<action>/repeat_XX/frames.jsonl
（新格式：points_sensor / points_world / targets）

输出
----
reports/prefall_source_compare_v1/<timestamp>/
  - per_repeat_summary.json
  - report.md

Version: radar_prefall_source_compare_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ACTIONS = ["standing", "fast_sitting", "controlled_forward_fall"]
PHASES = ["still_pre", "action", "still_post"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _centroid_z(points: list[dict[str, Any]]) -> float:
    if not points:
        return float("nan")
    zs = [p.get("z", float("nan")) for p in points]
    zs = [z for z in zs if np.isfinite(z)]
    return float(np.mean(zs)) if zs else float("nan")


def _target_stats(targets: list[dict[str, Any]]) -> dict[str, float]:
    if not targets:
        return {
            "n": 0, "pos_z": float("nan"), "vel_z": float("nan"),
            "pos_y": float("nan"), "vel_xy": float("nan"), "confidence": float("nan"),
        }
    pos_z = [t.get("pos_z", float("nan")) for t in targets]
    vel_z = [t.get("vel_z", float("nan")) for t in targets]
    pos_y = [t.get("pos_y", float("nan")) for t in targets]
    vel_x = [t.get("vel_x", float("nan")) for t in targets]
    vel_y = [t.get("vel_y", float("nan")) for t in targets]
    conf = [t.get("confidence", float("nan")) for t in targets]
    pos_z = [v for v in pos_z if np.isfinite(v)]
    vel_z = [v for v in vel_z if np.isfinite(v)]
    pos_y = [v for v in pos_y if np.isfinite(v)]
    vx = [v for v in vel_x if np.isfinite(v)]
    vy = [v for v in vel_y if np.isfinite(v)]
    conf = [v for v in conf if np.isfinite(v)]
    vel_xy = [
        float(np.hypot(a, b)) for a, b in zip(vx, vy) if np.isfinite(a) and np.isfinite(b)
    ]
    return {
        "n": len(targets),
        "pos_z": float(np.mean(pos_z)) if pos_z else float("nan"),
        "vel_z": float(np.mean(vel_z)) if vel_z else float("nan"),
        "pos_y": float(np.mean(pos_y)) if pos_y else float("nan"),
        "vel_xy": float(np.mean(vel_xy)) if vel_xy else float("nan"),
        "confidence": float(np.mean(conf)) if conf else float("nan"),
    }


def analyze_action(session_root: Path, action: str) -> dict[str, Any]:
    action_dir = session_root / action
    if not action_dir.exists():
        return {"action": action, "error": "no dir"}
    repeat_summaries = []
    for rep in sorted(d for d in action_dir.iterdir() if d.is_dir() and d.name.startswith("repeat_")):
        frames_path = rep / "frames.jsonl"
        if not frames_path.exists():
            continue
        records = _read_jsonl(frames_path)
        if not records:
            continue
        # 检查新格式
        has_world = "points_world" in records[0]
        has_targets = any(r.get("targets") for r in records)

        phase_stats: dict[str, dict[str, float]] = {}
        for phase in PHASES:
            phase_records = [r for r in records if r.get("phase") == phase]
            if not phase_records:
                continue
            sensor_z = []
            world_z = []
            target_pos_z = []
            target_vel_z = []
            for r in phase_records:
                sensor_z.append(_centroid_z(r.get("points", ()) or r.get("points_sensor", ())))
                world_z.append(_centroid_z(r.get("points_world", ())))
                ts = _target_stats(r.get("targets", []))
                if np.isfinite(ts["pos_z"]):
                    target_pos_z.append(ts["pos_z"])
                if np.isfinite(ts["vel_z"]):
                    target_vel_z.append(ts["vel_z"])
            phase_stats[phase] = {
                "sensor_centroid_z_median": float(np.nanmedian(sensor_z)),
                "world_centroid_z_median": float(np.nanmedian(world_z)),
                "target_pos_z_median": float(np.nanmedian(target_pos_z)) if target_pos_z else float("nan"),
                "target_vel_z_median": float(np.nanmedian(target_vel_z)) if target_vel_z else float("nan"),
                "frames": len(phase_records),
            }

        repeat_summaries.append({
            "repeat_id": rep.name,
            "has_world": has_world,
            "has_targets": has_targets,
            "phases": phase_stats,
        })
    return {
        "action": action,
        "repeat_count": len(repeat_summaries),
        "repeats": repeat_summaries,
    }


def build_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# 三数据源对比 sanity check",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 说明",
        "",
        "- sensor_centroid_z: 原始 sensor-frame 点云质心 z",
        "- world_centroid_z: world-frame（1m/5°）点云质心 z",
        "- target_pos_z: TI tracking target 位置 z（world-frame，平滑滤波）",
        "- target_vel_z: TI tracking target 垂直速度 z",
        "",
        "重点看 controlled_forward_fall 的 action/still_post 阶段：",
        "target_pos_z 是否比 sensor/world 点云 centroid 更符合下降→低位。",
        "",
    ]
    for res in results:
        if "error" in res:
            lines.append(f"## {res['action']}: {res['error']}")
            continue
        lines.append(f"## {res['action']}（{res['repeat_count']} repeats）")
        lines.append("")
        for rep in res["repeats"]:
            lines.append(f"### {rep['repeat_id']}（world={rep['has_world']} targets={rep['has_targets']}）")
            lines.append("")
            lines.append("| phase | sensor_cz | world_cz | target_pos_z | target_vel_z |")
            lines.append("|-------|-----------|----------|--------------|--------------|")
            for phase in PHASES:
                ps = rep["phases"].get(phase)
                if not ps:
                    continue
                lines.append(
                    f"| {phase} | {ps['sensor_centroid_z_median']:.3f} | "
                    f"{ps['world_centroid_z_median']:.3f} | "
                    f"{ps['target_pos_z_median']:.3f} | "
                    f"{ps['target_vel_z_median']:.3f} |"
                )
            lines.append("")
    lines += [""]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare sensor/world point cloud vs TI tracking target."
    )
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/prefall_source_compare_v1"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    results = [analyze_action(args.session_root, a) for a in ACTIONS]
    out_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_action_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(build_report(results), encoding="utf-8")
    print(f"reports written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
