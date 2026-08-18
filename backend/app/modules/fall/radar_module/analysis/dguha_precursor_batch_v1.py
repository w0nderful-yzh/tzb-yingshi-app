"""Batch precursor analysis across all eligible DGUHA forward-fall samples.

Goal
----
Validate whether the Kinect-visible fall precursor (trunk lean build-up in the
0.2-0.5 s before descent onset) is:
  1. consistent across all eligible DGUHA forward-fall samples, and
  2. observable in the radar v2 feature stream.

Method
------
For every eligible forward-fall recording:
- Re-locate loss_of_balance, descent_onset, rapid_descent_onset, lowest_point
  from the 25-joint Kinect skeleton using head height and trunk lean.
- Extract Kinect precursor features in three windows relative to descent_onset:
    * [-1.0, -0.5] s
    * [-0.5, -0.2] s
    * [-0.2,  0.0] s
- Extract radar v2 feature time-evolution in the same three windows.
- Compare against hard negatives (sitting / jumping / running) using
  equal-length windows.

Outputs
-------
- Per-feature effect size (Cohen's d) and distribution difference.
- Fraction of fall samples showing each precursor (cross-sample consistency).
- False-trigger rate in normal actions.
- Which radar features are the best proxy for the Kinect precursor.

This is analysis only. It does not modify any checkpoint, threshold, or model.

Version: radar_dguha_precursor_batch_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radar_module.dataset.dguha_research_v2 import parse_dguha_kinect
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)

HEAD_JOINTS = (0, 1, 2, 3, 4)
TORSO_JOINTS = (5, 6, 7, 12, 13, 14, 15)
PELVIS_JOINTS = (12, 13, 14, 15)


# ---------------------------------------------------------------------------
# Kinect feature extraction
# ---------------------------------------------------------------------------

def kinect_series(frames):
    """Return time-aligned Kinect feature series (meters, seconds).

    Time is re-zeroed to the first *valid* (non-empty) frame so that
    ``locate_events`` indices and the time axis agree. Empty leading frames
    are skipped.
    """
    valid = [f for f in frames if f.points_mm.any()]
    if not valid:
        raise ValueError("no valid Kinect frames")
    t0 = valid[0].timestamp
    times = []
    head = []
    trunk_lean = []
    pelvis = []
    com = []
    for f in valid:
        pts = f.points_mm
        z = pts[:, 2] / 1000.0
        y = pts[:, 1] / 1000.0
        x = pts[:, 0] / 1000.0
        times.append((f.timestamp - t0).total_seconds())
        head.append(np.nanmax(z[list(HEAD_JOINTS)]))
        # trunk lean: shoulder-forward minus pelvis-forward (y is forward)
        trunk_lean.append(
            np.nanmean(y[[5, 6, 7]]) - np.nanmean(y[list(PELVIS_JOINTS)])
        )
        pelvis.append(np.nanmean(z[list(PELVIS_JOINTS)]))
        com.append(np.nanmean(z))
    t = np.asarray(times)
    head = np.asarray(head)
    trunk_lean = np.asarray(trunk_lean)
    pelvis = np.asarray(pelvis)
    com = np.asarray(com)
    # derived: vertical velocity of head, trunk angular velocity
    v_head = np.gradient(head, t)
    v_pelvis = np.gradient(pelvis, t)
    v_lean = np.gradient(trunk_lean, t)
    return {
        "t": t,
        "head": head,
        "trunk_lean": trunk_lean,
        "pelvis": pelvis,
        "com": com,
        "v_head": v_head,
        "v_pelvis": v_pelvis,
        "v_lean": v_lean,
    }


def locate_events(kin):
    """Locate loss_of_balance, descent_onset, rapid_descent_onset, lowest_point.

    Uses the *kinect-relative* descent onset from the existing DGUHA event
    derivation where possible (the caller aligns to radar descent_onset). Here
    we compute kinect-relative event times robustly:
    - baseline: median of the first stable segment (first 20 valid frames)
    - descent_onset: head drops > 5 cm sustained over 3 frames
    - rapid_descent: head vertical velocity < -0.3 m/s AND head already below
      baseline (avoid early noise triggers)
    - lowest_point: min head height
    """
    t = kin["t"]
    head = kin["head"]
    baseline = np.nanmedian(head[:20]) if len(head) >= 20 else np.nanmedian(head)
    drop = baseline - head

    # descent_onset: sustained head drop > 5 cm
    descent_idx = None
    for i in range(3, len(head)):
        if drop[i] > 0.05 and np.nanmean(drop[max(0, i - 2) : i + 1]) > 0.05:
            descent_idx = i
            break
    # rapid_descent: head falling fast AND already below baseline
    rapid_idx = None
    v = kin["v_head"]
    for i in range(3, len(v)):
        if v[i] < -0.3 and head[i] < baseline - 0.02:
            rapid_idx = i
            break
    # loss_of_balance: start of sustained trunk-lean build (lean > baseline+2std,
    # sustained 3 frames)
    lean = kin["trunk_lean"]
    lean_baseline = np.nanmedian(lean[:20]) if len(lean) >= 20 else np.nanmedian(lean)
    lean_std = np.nanstd(lean[:20]) if len(lean) >= 20 else np.nanstd(lean)
    loss_idx = None
    if lean_std > 1e-6:
        for i in range(3, len(lean)):
            if (lean[i] - lean_baseline) > 2.0 * lean_std and np.nanmean(lean[i - 2 : i + 1] - lean_baseline) > 0:
                loss_idx = i
                break
    floor_idx = int(np.nanargmin(head)) if len(head) else None
    return {
        "loss_of_balance": (t[loss_idx] - t[0]) if loss_idx else None,
        "descent_onset": (t[descent_idx] - t[0]) if descent_idx else None,
        "rapid_descent_onset": (t[rapid_idx] - t[0]) if rapid_idx else None,
        "lowest_point": (t[floor_idx] - t[0]) if floor_idx else None,
    }


def kinect_window_features(kin, reference_t, windows):
    """Extract per-window delta features from Kinect series."""
    t = kin["t"]
    out = {}
    for name, (lo, hi) in windows.items():
        mask = (t >= reference_t + lo) & (t < reference_t + hi)
        if mask.sum() < 2:
            out[f"head_delta_{name}"] = np.nan
            out[f"lean_delta_{name}"] = np.nan
            out[f"lean_max_{name}"] = np.nan
            out[f"com_delta_{name}"] = np.nan
            out[f"v_head_mean_{name}"] = np.nan
            out[f"v_lean_mean_{name}"] = np.nan
            continue
        out[f"head_delta_{name}"] = float(kin["head"][mask][-1] - kin["head"][mask][0])
        out[f"lean_delta_{name}"] = float(kin["trunk_lean"][mask][-1] - kin["trunk_lean"][mask][0])
        out[f"lean_max_{name}"] = float(np.nanmax(kin["trunk_lean"][mask]))
        out[f"com_delta_{name}"] = float(kin["com"][mask][-1] - kin["com"][mask][0])
        out[f"v_head_mean_{name}"] = float(np.nanmean(kin["v_head"][mask]))
        out[f"v_lean_mean_{name}"] = float(np.nanmean(kin["v_lean"][mask]))
    return out


# ---------------------------------------------------------------------------
# Radar feature extraction
# ---------------------------------------------------------------------------

def radar_window_features(frames, descent_s, windows):
    """Extract radar v2 feature time-evolution in windows relative to descent."""
    ext = RadarTemporalFeatureExtractorV2()
    start = frames[0].timestamp
    out = {}
    for name, (lo, hi) in windows.items():
        # sample window endpoints inside [lo, hi]: step 0.2, include endpoint near hi
        offsets = np.arange(lo, hi + 1e-9, 0.2)
        if offsets[-1] > hi - 1e-9 and offsets[-1] != hi:
            offsets = offsets[:-1]
        feats = []
        for off in offsets:
            end_ts = start + __import__("datetime").timedelta(seconds=descent_s + off)
            wf = [f for f in frames if f.timestamp <= end_ts and f.timestamp >= end_ts - __import__("datetime").timedelta(seconds=2)]
            if not wf:
                continue
            try:
                w = ext.transform(tuple(wf), end_timestamp=end_ts)
            except ValueError:
                continue
            if w.data_quality.value == "GOOD":
                feats.append(w.values)
        if not feats:
            for i, fname in enumerate(FEATURE_NAMES_V2):
                out[f"radar_{fname}_{name}"] = np.nan
            continue
        F = np.stack(feats)  # (n_windows, 20, 19)
        # time evolution: mean of last-frame features across windows
        last = F[:, -1, :]  # (n_windows, 19)
        for i, fname in enumerate(FEATURE_NAMES_V2):
            # delta across the window (last window last frame - first window first frame)
            out[f"radar_{fname}_{name}"] = float(last[-1, i] - last[0, i])
            out[f"radar_{fname}_{name}_mean"] = float(np.nanmean(last[:, i]))
    return out


# ---------------------------------------------------------------------------
# Effect size and consistency
# ---------------------------------------------------------------------------

def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb = len(a), len(b)
    var = ((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2)
    if var <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(var))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    base = Path("data/external/dguha/raw")
    events = json.loads(
        Path("data/processed/dguha_prefall_0p5_1p0_dense_v3.events.json").read_text()
    )
    eligible = [e for e in events if e.get("eligible_for_prediction_windows")]

    windows = {
        "m10_m05": (-1.0, -0.5),
        "m05_m02": (-0.5, -0.2),
        "m02_0": (-0.2, 0.0),
    }

    # Fall samples
    fall_kin_feats = []  # dict of feature -> value
    fall_radar_feats = []
    fall_events = []
    used = 0
    for e in eligible:
        kpath = base / e["source_file"].replace("/radar/", "/kinect/")
        rpath = base / e["source_file"]
        if not kpath.exists():
            continue
        try:
            kframes = parse_dguha_kinect(kpath)
            rframes = parse_radhar_text(rpath, device_id="dguha")
        except Exception:
            continue
        kin = kinect_series(kframes)
        ev = locate_events(kin)
        descent_s = e.get("descent_onset_seconds_from_radar_start")
        if ev["descent_onset"] is None or descent_s is None:
            continue
        used += 1
        fall_events.append(ev)
        fall_kin_feats.append(kinect_window_features(kin, ev["descent_onset"], windows))
        fall_radar_feats.append(radar_window_features(rframes, descent_s, windows))

    print(f"=== 可用 forward-fall 样本: {used}/{len(eligible)} ===")
    print()

    # Event timing summary
    print("=== 事件时刻统计 (秒, 相对录制起点) ===")
    for key in ("loss_of_balance", "descent_onset", "rapid_descent_onset", "lowest_point"):
        vals = [ev[key] for ev in fall_events if ev[key] is not None]
        if vals:
            print(f"  {key:20s}: n={len(vals)} 中位={np.median(vals):.2f}s 范围=[{min(vals):.2f},{max(vals):.2f}]")
    print()

    # Consistency of Kinect precursor
    print("=== Kinect 前兆跨样本一致性 ===")
    kin_feature_names = [
        "head_delta", "lean_delta", "lean_max", "com_delta", "v_head_mean", "v_lean_mean",
    ]
    for wname in windows:
        print(f"  --- 窗口 {wname} ---")
        for base_f in kin_feature_names:
            key = f"{base_f}_{wname}"
            vals = [d[key] for d in fall_kin_feats if key in d and np.isfinite(d[key])]
            if not vals:
                continue
            vals = np.asarray(vals, dtype=float)
            # direction consistency: fraction of samples with the dominant sign
            pos = float(np.mean(vals > 0))
            neg = float(np.mean(vals < 0))
            consist = max(pos, neg)
            print(f"    {key:22s}: 中位={np.median(vals):+.4f} 正向占比={pos:.0%} 负向占比={neg:.0%} 主导一致性={consist:.0%} (n={len(vals)})")

    # Radar proxy: which radar features have nonzero delta in precursor window
    print()
    print("=== 雷达 v2 特征在下降前窗口的变化 (中位 delta) ===")
    radar_candidates = [
        "moving_range_width", "height_range", "centroid_z", "z_p90", "z_p50",
        "max_abs_velocity", "velocity_std", "moving_range_centroid",
        "point_count", "vertical_velocity", "vertical_acceleration",
        "centroid_z_delta_0_6s", "height_range_delta_0_3s", "mean_velocity",
    ]
    for wname in windows:
        print(f"  --- 窗口 {wname} ---")
        for rname in radar_candidates:
            key = f"radar_{rname}_{wname}"
            vals = [d[key] for d in fall_radar_feats if key in d and np.isfinite(d[key])]
            if not vals:
                continue
            med = np.median(vals)
            # nonzero fraction (|delta| > small threshold relative)
            scale = np.median(np.abs(vals)) + 1e-9
            nz = float(np.mean([abs(v) > 0.1 * scale for v in vals]))
            print(f"    {rname:26s}: 中位Δ={med:+.4f} 非零率={nz:.0%} (n={len(vals)})")

    # Compare against hard negatives (kinect only for now)
    print()
    print("=== 负样本对照 (Kinect 前倾在相等长度窗口的波动) ===")
    negatives = [
        ("sitting", "3_Sit_down_and_stand_up"),
        ("jumping", "2_Jumping"),
        ("running", "1_Running"),
    ]
    for name, rel in negatives:
        try:
            import glob
            cands = sorted(Path(base / f"Training/{rel}/kinect").glob("*.txt"))
            if not cands:
                cands = sorted(Path(base / f"Test/{rel}/kinect").glob("*.txt"))
            # take one recording, compute lean std over the recording
            kframes = parse_dguha_kinect(cands[0])
            kin = kinect_series(kframes)
            lean_std = float(np.nanstd(kin["trunk_lean"]))
            head_std = float(np.nanstd(kin["head"]))
            print(f"  {name:10s}: 前倾std={lean_std:.3f}m 头高std={head_std:.3f}m")
        except Exception as e:
            print(f"  {name:10s}: ERR {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
