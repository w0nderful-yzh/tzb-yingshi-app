"""IWR6843ISK 坐标系与高度特征审计。

背景
----
pilot 分析中 height_range / z_p90 / centroid_z 方向混乱，曾被解释为
"高位安装 z 语义反转"。本脚本审计 TI People Tracking 坐标系，判断
z 混乱的真正来源：

- 桥接层是否正确应用了 sensorHeight / tilt？
- TI 输出点云是 sensor-frame 还是 world-frame？
- 转换到 world frame 后高度特征是否符合人体"站立→下降→低位"？

审计结论（来自 TI 源码与 visualizer）：
1. TI 输出的 pointCloud 是 **sensor-frame 笛卡尔坐标**（球坐标转换，
   未补偿 sensorHeight / tilt）
2. PC visualizer 负责补偿：eulerRot(elev_tilt, az_tilt) 旋转 +
   z += sensorHeight
3. tracking 目标（Target List TLV posX/posY/posZ）内部已转 world-frame
4. 我们的 ti_official_bridge 直接透传原始点云，**未做 sensorHeight/tilt 补偿**

本脚本用不同安装参数转换点云到 world frame，验证高度语义：
- raw：无转换
- cfg：配置参数（sensorPosition 2 0 15 → 2m, 15°）
- real：实际安装（用户报告 1m, 5°）
- real_no_tilt：仅加高度 1m，不加 tilt

输出
----
reports/sensor_world_audit_v1/<timestamp>/report.json + report.md

Version: radar_sensor_to_world_audit_v1
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ACTIONS = ["standing", "fast_sitting", "controlled_forward_fall"]
PHASES = ["still_pre", "action", "still_post"]


def euler_rot(x: float, y: float, z: float, elev_tilt_deg: float,
              azi_tilt_deg: float) -> tuple[float, float, float]:
    """复刻 TI visualizer 的 eulerRot（绕 X 旋转 elevation + azimuth）。"""
    elev = math.radians(elev_tilt_deg)
    azi = math.radians(azi_tilt_deg)
    # elevAziRotMatrix（TI visualizer graph_utilities.eulerRot）
    r00 = math.cos(azi)
    r01 = math.cos(elev) * math.sin(azi)
    r02 = math.sin(elev) * math.sin(azi)
    r10 = -math.sin(azi)
    r11 = math.cos(elev) * math.cos(azi)
    r12 = math.sin(elev) * math.cos(azi)
    r20 = 0.0
    r21 = -math.sin(elev)
    r22 = math.cos(elev)
    rx = r00 * x + r01 * y + r02 * z
    ry = r10 * x + r11 * y + r12 * z
    rz = r20 * x + r21 * y + r22 * z
    return rx, ry, rz


def to_world(
    points: list[dict[str, Any]],
    *,
    sensor_height_m: float,
    elev_tilt_deg: float,
    azi_tilt_deg: float,
) -> tuple[list[float], list[float], list[float]]:
    """sensor-frame 点云 → world-frame。"""
    xs, ys, zs = [], [], []
    for p in points:
        x, y, z = p["x"], p["y"], p["z"]
        if elev_tilt_deg != 0 or azi_tilt_deg != 0:
            x, y, z = euler_rot(x, y, z, elev_tilt_deg, azi_tilt_deg)
        z = z + sensor_height_m  # sensorHeight 平移
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return xs, ys, zs


def frame_stats(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """对一组帧计算逐帧 centroid_z / z_p10/50/90 / height_range 统计。"""
    configs = {
        "raw": {"sensor_height_m": 0.0, "elev_tilt_deg": 0.0, "azi_tilt_deg": 0.0},
        "cfg_2m_15deg": {"sensor_height_m": 2.0, "elev_tilt_deg": 15.0, "azi_tilt_deg": 0.0},
        "real_1m_5deg": {"sensor_height_m": 1.0, "elev_tilt_deg": 5.0, "azi_tilt_deg": 0.0},
        "real_1m_notilt": {"sensor_height_m": 1.0, "elev_tilt_deg": 0.0, "azi_tilt_deg": 0.0},
    }
    cfg = configs[mode]
    centroid_zs = []
    z_p90s = []
    height_ranges = []
    for r in records:
        pts = r.get("points", ()) or r.get("points_sensor", ())
        if not pts:
            continue
        _, _, zs = to_world(list(pts), **cfg)
        zs = np.asarray(zs)
        centroid_zs.append(float(zs.mean()))
        z_p90s.append(float(np.percentile(zs, 90)))
        height_ranges.append(float(zs.max() - zs.min()))
    return {
        "mode": mode,
        "sensor_height_m": cfg["sensor_height_m"],
        "elev_tilt_deg": cfg["elev_tilt_deg"],
        "centroid_z": {
            "median": float(np.median(centroid_zs)) if centroid_zs else float("nan"),
            "q25": float(np.percentile(centroid_zs, 25)) if centroid_zs else float("nan"),
            "q75": float(np.percentile(centroid_zs, 75)) if centroid_zs else float("nan"),
        },
        "z_p90": {
            "median": float(np.median(z_p90s)) if z_p90s else float("nan"),
        },
        "height_range": {
            "median": float(np.median(height_ranges)) if height_ranges else float("nan"),
        },
    }


def load_repeat_frames(session_root: Path, action: str) -> dict[str, list[dict[str, Any]]]:
    """加载某动作所有 repeat 的帧，按 phase 分组。"""
    action_dir = session_root / action
    if not action_dir.exists():
        return {}
    phase_frames: dict[str, list[dict[str, Any]]] = {p: [] for p in PHASES}
    for rep in sorted(d for d in action_dir.iterdir() if d.is_dir() and d.name.startswith("repeat_")):
        fpath = rep / "frames.jsonl"
        if not fpath.exists():
            continue
        for line in fpath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ph = r.get("phase")
            if ph in phase_frames:
                phase_frames[ph].append(r)
    return phase_frames


def build_report(results: dict[str, Any], by_phase: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# IWR6843ISK 坐标系与高度特征审计",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 审计结论（来自 TI 源码与 visualizer）",
        "",
        "1. TI 输出 pointCloud 是 **sensor-frame** 笛卡尔坐标（未补偿",
        "   sensorHeight / tilt）",
        "2. PC visualizer 负责补偿：eulerRot(elev,az) + z += sensorHeight",
        "3. tracking 目标（Target List TLV posX/posY/posZ）内部已转 world-frame",
        "4. **我们的 ti_official_bridge 直接透传原始点云，未做补偿**",
        "",
        "## 安装参数",
        "",
        "- 配置 sensorPosition 2 0 15 → 2m, 15°",
        "- 实际安装（用户报告）→ 1m, 5°",
        "",
        "## 各动作各 phase 转换后高度特征",
        "",
    ]
    for action, phase_stats in results.items():
        lines.append(f"### {action}")
        lines.append("")
        lines.append("| phase | mode | centroid_z med | z_p90 med | height_range med |")
        lines.append("|-------|------|----------------|-----------|------------------|")
        for phase in PHASES:
            if phase not in phase_stats:
                continue
            for stat in phase_stats[phase]:
                lines.append(
                    f"| {phase} | {stat['mode']} | {stat['centroid_z']['median']:.3f} | "
                    f"{stat['z_p90']['median']:.3f} | {stat['height_range']['median']:.3f} |"
                )
        lines.append("")
    lines += ["", "## 判读", ""]
    lines.append("""
若 world-frame（实际 1m/5°）下 standing 的 centroid_world_z 显著高于
controlled_forward_fall 的倒地后（still_post），且 fast_sitting 介于中间，
则 z 语义正常，之前"z 反转"是缺转换导致。若仍混乱，则是反射部位/物理问题。
""")
    lines += [""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="IWR6843ISK sensor-to-world coordinate audit."
    )
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/sensor_world_audit_v1"))
    args = parser.parse_args()

    modes = ["raw", "cfg_2m_15deg", "real_1m_5deg", "real_1m_notilt"]
    results: dict[str, Any] = {}
    by_phase_summary: dict[str, dict[str, Any]] = {}
    for action in ACTIONS:
        phase_frames = load_repeat_frames(args.session_root, action)
        if not phase_frames:
            print(f"skip {action}: no frames", flush=True)
            continue
        results[action] = {}
        for phase in PHASES:
            frames = phase_frames.get(phase, [])
            if not frames:
                continue
            results[action][phase] = [
                frame_stats(frames, mode) for mode in modes
            ]
        by_phase_summary[action] = phase_frames

    out_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(
        build_report(results, by_phase_summary), encoding="utf-8"
    )
    print(f"reports written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
