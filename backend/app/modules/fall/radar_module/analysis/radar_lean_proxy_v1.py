"""Radar posture/lean proxy features and their discriminative value.

Motivation
----------
Kinect analysis shows trunk lean is the clearest pre-fall precursor, but
radar_features_v2 has no direct body-tilt feature. This script designs and
evaluates lean proxies computed from the mmWave point cloud, using only
existing radar frames (offline; no model change).

Proxies per window (aggregated over the 2 s window):
- lean_angle: PCA principal-axis angle to vertical (deg)
- lean_angle_clean: PCA on range-filtered points (x < range_gate)
- height_width_ratio: point-cloud vertical extent / horizontal extent
- eig_ratio: ratio of largest to second PCA eigenvalue
- centroid_horizontal_drift: horizontal displacement of window centroid
- upper_lower_shift: change in upper-quartile vs lower-quartile z
- upper_point_fraction: fraction of points in upper half of z range

For each NEW pre-fall window we compute these and compare against equal-length
normal windows (sitting/jumping/running). Outputs: effect size (Cohen d),
single-feature AUROC/PR-AUC, precursor occurrence rate, per-negative
false-trigger rate, and (where available) correlation with Kinect trunk lean.

This is analysis only. No model or checkpoint is modified.
Version: radar_radar_lean_proxy_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.dataset.dguha_research_v2 import parse_dguha_kinect
from radar_module.preprocess.temporal_features_v2 import (
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)

HEAD_JOINTS = (0, 1, 2, 3, 4)
TORSO_JOINTS = (5, 6, 7, 12, 13, 14, 15)


# ---------------------------------------------------------------------------
# Per-window lean proxy features
# ---------------------------------------------------------------------------

def window_lean_proxies(window_points):
    """Compute lean proxies from a set of points (list of (x,y,z))."""
    pts = np.asarray(window_points, dtype=float)
    if len(pts) < 10:
        return None
    # range filter: drop points beyond gate (avoid clutter)
    r = np.linalg.norm(pts, axis=1)
    clean = pts[r < 6.0]
    if len(clean) < 8:
        clean = pts

    def pca_lean(data):
        centered = data - data.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        main = eigvecs[:, np.argmax(eigvals)]
        cos_ang = abs(main[2]) / (np.linalg.norm(main) + 1e-9)
        return np.degrees(np.arccos(np.clip(cos_ang, 0, 1))), eigvals, eigvecs

    lean_deg, eigvals, _ = pca_lean(clean)
    # height/width ratio: vertical extent vs max horizontal extent
    z = clean[:, 2]
    horiz = np.sqrt(clean[:, 0] ** 2 + clean[:, 1] ** 2)
    hw_ratio = (z.max() - z.min()) / max(horiz.max() - horiz.min(), 1e-6)
    # eigenvalue ratio
    eig_sorted = np.sort(eigvals)[::-1]
    eig_ratio = eig_sorted[0] / max(eig_sorted[1], 1e-9)
    # upper/lower z distribution
    upper_mask = z >= np.median(z)
    upper_frac = float(upper_mask.mean())
    # centroid
    centroid = clean.mean(axis=0)
    return {
        "lean_angle_deg": float(lean_deg),
        "height_width_ratio": float(hw_ratio),
        "eig_ratio": float(eig_ratio),
        "upper_point_fraction": upper_frac,
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "n_points": int(len(clean)),
    }


def window_deltas(proxies_series):
    """Compute deltas of proxies across a time series of windows."""
    if len(proxies_series) < 2:
        return None
    first, last = proxies_series[0], proxies_series[-1]
    return {
        "lean_delta": last["lean_angle_deg"] - first["lean_angle_deg"],
        "hw_ratio_delta": last["height_width_ratio"] - first["height_width_ratio"],
        "centroid_h_drift": np.hypot(
            last["centroid_x"] - first["centroid_x"],
            last["centroid_y"] - first["centroid_y"],
        ),
        "centroid_z_delta": last["centroid_z"] - first["centroid_z"],
        "upper_frac_delta": last["upper_point_fraction"] - first["upper_point_fraction"],
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def collect_window_proxies(rpath, extractor, sample_offsets, stride=0.2):
    """Slide windows at given absolute offsets, return proxy series."""
    frames = parse_radhar_text(Path(rpath), device_id="dguha")
    start = frames[0].timestamp
    proxies = []
    for off in sample_offsets:
        end_ts = start + __import__("datetime").timedelta(seconds=off)
        wf = [f for f in frames if f.timestamp <= end_ts and f.timestamp >= end_ts - __import__("datetime").timedelta(seconds=2)]
        pts = []
        for f in wf:
            pts.extend([(p.x, p.y, p.z) for p in f.points])
        p = window_lean_proxies(pts)
        if p is not None:
            proxies.append(p)
    return proxies


def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var = ((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2)
    if var <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(var))


def single_feature_auc(neg_vals, pos_vals):
    from sklearn.metrics import roc_auc_score, average_precision_score
    neg = np.asarray(neg_vals, dtype=float)
    pos = np.asarray(pos_vals, dtype=float)
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    x = np.concatenate([neg, pos])
    # direction: AUROC > 0.5 if pos higher
    auc = roc_auc_score(y, x)
    ap = average_precision_score(y, x)
    return float(auc), float(ap)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/external/dguha/raw")
    parser.add_argument("--output", default="reports/radar_lean_proxy_v1")
    parser.add_argument("--events", default="data/processed/dguha_prefall_0p5_1p0_dense_v3.events.json")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    events = json.loads(Path(args.events).read_text())
    test_falls = [e for e in events if e["project_split"] == "test" and e.get("eligible_for_prediction_windows")]
    extractor = RadarTemporalFeatureExtractorV2()

    # For each test fall, collect proxy series in windows before sustained descent.
    # We approximate sustained descent as descent_onset + 0.8 (per earlier finding).
    fall_feats = []  # dict of proxy -> value (window-level)
    per_fall_precursor = []
    for ev in test_falls:
        rpath = data_root / ev["source_file"]
        onset = ev["descent_onset_seconds_from_radar_start"]
        # sample windows ending at onset-1.5 .. onset (0.25s step) => pre-fall proxies
        offsets = np.arange(onset - 1.5, onset + 0.05, 0.25)
        proxies = collect_window_proxies(rpath, extractor, offsets)
        if not proxies:
            continue
        # precursor = window-level proxies (each window is a 'pre-fall' sample)
        for p in proxies:
            for k, v in p.items():
                fall_feats.append((k, v, 1))

    # Negatives: sitting/jumping/running, sample windows of equal length
    neg_feats = []
    negatives = {
        "sitting": "3_Sit_down_and_stand_up",
        "jumping": "2_Jumping",
        "running": "1_Running",
    }
    for aname, arel in negatives.items():
        import glob
        cands = sorted(Path(data_root / f"Test/{arel}/radar").glob("*.txt"))
        if not cands:
            cands = sorted(Path(data_root / f"Training/{arel}/radar").glob("*.txt"))
        for cp in cands[:2]:
            offsets = np.arange(1.0, 8.0, 0.25)
            proxies = collect_window_proxies(cp, extractor, offsets)
            for p in proxies:
                for k, v in p.items():
                    neg_feats.append((k, v, 0))

    # Aggregate per-feature
    features = [
        "lean_angle_deg", "height_width_ratio", "eig_ratio",
        "upper_point_fraction", "centroid_x", "centroid_y", "centroid_z",
    ]
    print("=== 单特征: fall vs 正常 ===")
    result = {}
    for fname in features:
        pos_vals = [v for k, v, lab in fall_feats if k == fname and lab == 1]
        neg_vals = [v for k, v, lab in neg_feats if k == fname and lab == 0]
        if not pos_vals or not neg_vals:
            continue
        d = cohens_d(pos_vals, neg_vals)
        auc, ap = single_feature_auc(neg_vals, pos_vals)
        result[fname] = {
            "cohens_d": d,
            "auroc": auc,
            "pr_auc": ap,
            "fall_median": float(np.median(pos_vals)),
            "neg_median": float(np.median(neg_vals)),
        }
        print(f"  {fname:24s}: d={d:+.3f} AUROC={auc:.3f} PR-AUC={ap:.3f} "
              f"fall_med={np.median(pos_vals):.3f} neg_med={np.median(neg_vals):.3f}")

    # Per-action negatives
    print("\n=== 各负样本误触发（单特征 lean_angle>阈值判断）===")
    # Use lean_angle median of fall as threshold
    pos_lean = [v for k, v, lab in fall_feats if k == "lean_angle_deg" and lab == 1]
    thr = np.median(pos_lean) if pos_lean else 60.0
    print(f"  阈值(lean_angle>={thr:.1f}° = fall):")
    for aname, arel in negatives.items():
        import glob
        cands = sorted(Path(data_root / f"Test/{arel}/radar").glob("*.txt"))
        if not cands:
            cands = sorted(Path(data_root / f"Training/{arel}/radar").glob("*.txt"))
        total = 0
        high = 0
        for cp in cands[:2]:
            proxies = collect_window_proxies(cp, extractor, np.arange(1.0, 8.0, 0.25))
            for p in proxies:
                total += 1
                if p["lean_angle_deg"] >= thr:
                    high += 1
        print(f"  {aname:8s}: 误触发 {high}/{total} = {high/max(total,1):.1%}")

    (out / "radar_lean_proxy_report.json").write_text(
        json.dumps({"single_features": result}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n已写入", out / "radar_lean_proxy_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
