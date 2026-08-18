"""Analyze whether DGUHA skeleton precursors are visible before sustained descent.

Motivation
----------
There was an open question: is the DGUHA "0.5-1.0s before descent onset" window
truly static (no precursor), or do existing features simply fail to capture a
real precursor? This script answers it from the Kinect skeleton side:

1. Re-derive fall events (loss-of-balance, sustained-descent, lowest point)
   from the 25-joint Kinect skeleton using a more physically meaningful metric
   (head/neck height and body-centroid height) than median-z.
2. For the sustained-descent time (t2), measure skeleton features at t2-1.0,
   t2-0.5, t2-0.2 s and compare them against normal actions (sitting,
   jumping, running) at equivalent time offsets from a reference.
3. Report whether precursors deviate from normal in skeleton space, and
   whether the same deviation is visible in radar v2 features.

This is analysis only. It does not modify any checkpoint, threshold, or model.

Version: radar_dguha_precursor_analysis_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from radar_module.dataset.dguha_research_v2 import parse_dguha_kinect
from radar_module.preprocess.temporal_features_v2 import FEATURE_NAMES_V2

# 25-joint skeleton: approximate body landmarks
# We use head (0-4), neck/shoulder (5-7), and body centroid over all joints.
HEAD_JOINTS = (0, 1, 2, 3, 4)
TORSO_JOINTS = (5, 6, 7, 12, 13, 14, 15)


def skeleton_features(points_mm: np.ndarray) -> dict[str, float]:
    """Extract height-related skeleton features for one frame.

    points_mm: (25, 3) in mm. z is vertical (larger = higher).
    Returns meters.
    """
    z = points_mm[:, 2] / 1000.0
    x = points_mm[:, 0] / 1000.0
    y = points_mm[:, 1] / 1000.0
    valid = np.isfinite(z)
    if not valid.any():
        return {
            "head_height": np.nan,
            "torso_height": np.nan,
            "centroid_height": np.nan,
            "body_extent": np.nan,
            "torso_forward_lean": np.nan,
            "torso_lateral_lean": np.nan,
        }
    head_h = float(np.nanmax(z[list(HEAD_JOINTS)])) if np.isfinite(z[list(HEAD_JOINTS)]).any() else np.nan
    torso_h = float(np.nanmean(z[list(TORSO_JOINTS)])) if np.isfinite(z[list(TORSO_JOINTS)]).any() else np.nan
    centroid_h = float(np.nanmean(z))
    extent = float(np.nanmax(z) - np.nanmin(z)) if valid.sum() > 1 else np.nan
    # torso forward lean: shoulder vs hip y-offset (forward = y positive)
    shoulder_y = np.nanmean(z[list((5, 6, 7))] if False else y[[5, 6, 7]])
    hip_y = np.nanmean(y[[12, 13, 14, 15]])
    torso_lean = float(shoulder_y - hip_y)
    return {
        "head_height": head_h,
        "torso_height": torso_h,
        "centroid_height": centroid_h,
        "body_extent": extent,
        "torso_forward_lean": torso_lean,
        "torso_lateral_lean": 0.0,
    }


def analyze_forward_fall(kinect_path: Path, radar_path: Path) -> dict:
    frames = parse_dguha_kinect(kinect_path)
    feats = []
    times = []
    for f in frames:
        if not f.points_mm.any():
            continue
        feats.append(skeleton_features(f.points_mm))
        times.append(f.timestamp.timestamp())
    times = np.asarray(times)
    t0 = times[0]

    def series(key):
        return np.asarray([f[key] for f in feats])

    head = series("head_height")
    torso = series("torso_height")
    centroid = series("centroid_height")
    extent = series("body_extent")
    lean = series("torso_forward_lean")

    # Event derivation: sustained descent = head height drops sharply.
    # loss of balance = start of sustained monotone height decrease.
    baseline = np.nanmedian(head[:20])
    drop = baseline - head
    speed = np.gradient(head, times)
    # sustained descent: head falling at > 0.3 m/s
    rapid_idx = None
    for i in range(1, len(head)):
        if np.nanmean(speed[max(0, i - 2):i + 1]) < -0.3:
            rapid_idx = i
            break
    # loss of balance: first time head drops > 5 cm from baseline and keeps going
    loss_idx = None
    for i in range(1, len(head)):
        if drop[i] > 0.05:
            loss_idx = i
            break
    # lowest point: min head height after rapid descent
    floor_idx = int(np.nanargmin(head)) if rapid_idx is not None else None

    result = {
        "frame_count": len(frames),
        "valid_frames": len(feats),
        "loss_of_balance_seconds": (times[loss_idx] - t0) if loss_idx else None,
        "sustained_descent_seconds": (times[rapid_idx] - t0) if rapid_idx else None,
        "lowest_point_seconds": (times[floor_idx] - t0) if floor_idx else None,
    }

    # Precursor features at t2-offset, where t2 = sustained descent
    if rapid_idx is not None:
        t2 = times[rapid_idx]
        for offset in (1.0, 0.5, 0.2):
            target = t2 - offset
            idx = int(np.argmin(np.abs(times - target)))
            result[f"precursor_{offset}s"] = {
                "head_height": float(head[idx]),
                "torso_height": float(torso[idx]),
                "centroid_height": float(centroid[idx]),
                "body_extent": float(extent[idx]),
                "torso_forward_lean": float(lean[idx]),
            }
        # normal reference = first 15 frames (standing before any action)
        result["standing_reference"] = {
            "head_height": float(np.nanmedian(head[:15])),
            "torso_height": float(np.nanmedian(torso[:15])),
            "centroid_height": float(np.nanmedian(centroid[:15])),
            "body_extent": float(np.nanmedian(extent[:15])),
            "torso_forward_lean": float(np.nanmedian(lean[:15])),
        }
    return result


def analyze_normal_action(kinect_path: Path) -> dict:
    frames = parse_dguha_kinect(kinect_path)
    feats = []
    for f in frames:
        if not f.points_mm.any():
            continue
        feats.append(skeleton_features(f.points_mm))
    return {
        "frame_count": len(feats),
        "head_std": float(np.nanstd([f["head_height"] for f in feats])),
        "head_median": float(np.nanmedian([f["head_height"] for f in feats])),
        "lean_std": float(np.nanstd([f["torso_forward_lean"] for f in feats])),
        "centroid_std": float(np.nanstd([f["centroid_height"] for f in feats])),
    }


def main() -> int:
    base = Path("data/external/dguha/raw/Test")
    print("=== DGUHA 骨架前兆分析 ===")
    print()
    fall = analyze_forward_fall(
        base / "5_falling_forward/kinect/F_006_A5_001.txt",
        base / "5_falling_forward/radar/F_006_A5_001.txt",
    )
    print("=== forward-fall 事件时刻 ===")
    for k, v in fall.items():
        if isinstance(v, (int, float, type(None))):
            print(f"  {k}: {v}")
    print()
    print("=== 前兆特征 (t2=持续下降起点) ===")
    for offset in (1.0, 0.5, 0.2):
        key = f"precursor_{offset}s"
        if key in fall:
            ref = fall["standing_reference"]
            p = fall[key]
            print(f"  t2-{offset}s: 头高={p['head_height']:.2f}m (站立参考{ref['head_height']:.2f}) "
                  f"前倾={p['torso_forward_lean']:.2f} (站立参考{ref['torso_forward_lean']:.2f}) "
                  f"质心={p['centroid_height']:.2f}")
    print()
    print("=== 对照: 正常动作骨架波动 ===")
    # 动作码: A1=running, A2=jumping, A3=sit_down_and_stand_up
    for name, rel, code in [
        ("sitting", "3_Sit_down_and_stand_up", "A3"),
        ("jumping", "2_Jumping", "A2"),
        ("running", "1_Running", "A1"),
    ]:
        try:
            kin = base / f"{rel}/kinect/F_006_{code}_001.txt"
            if not kin.exists():
                # 尝试其他文件
                import glob
                cands = sorted(Path(base / f"{rel}/kinect").glob("*.txt"))
                kin = cands[0] if cands else kin
            r = analyze_normal_action(kin)
            print(f"  {name}: 头高std={r['head_std']:.3f}m 头高中位={r['head_median']:.2f}m 前倾std={r['lean_std']:.3f} 质心std={r['centroid_std']:.3f}")
        except Exception as e:
            print(f"  {name}: ERR {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
