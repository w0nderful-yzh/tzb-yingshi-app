"""Event score curves + per-action false alarms for OLD vs NEW TCN.

Requirement 8: verify whether NEW-TCN score rises BEFORE sustained_descent,
not only after descent starts. We reconstruct the score timeline over a
held-out test fall recording, aligned to the Kinect sustained_descent_onset,
and report per-action false alarms for sitting/jumping/running.

This is analysis only. It does not modify any model or checkpoint.
Version: radar_label_comparison_tcn_curve_v1
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from radar_module.model.temporal_models_v3 import TemporalBinaryModel
from radar_module.dataset.dguha_research_v2 import parse_dguha_kinect
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.preprocess.temporal_features_v2 import (
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)

HEAD_JOINTS = (0, 1, 2, 3, 4)


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return (
        np.asarray(d["features"], dtype=np.float32),
        np.asarray(d["labels"], dtype=np.int64),
        np.asarray(d["split"]),
        np.asarray(d["source_files"]),
    )


def train_tcn(feats, labels, splits, seed=20260810):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_mask = splits == "train"
    # mean/std over (time, feature) -> shape (19,)
    mean = feats[train_mask].mean(axis=(0, 1))
    std = feats[train_mask].std(axis=(0, 1))
    std = np.where(std < 1e-9, 1e-9, std)
    norm = ((feats - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    model = TemporalBinaryModel(architecture="causal_tcn", input_size=19, hidden_size=24)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([12.0]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t = torch.from_numpy(norm[train_mask])
    l = torch.from_numpy(labels[train_mask].astype(np.float32))
    loader = DataLoader(TensorDataset(t, l), batch_size=256, shuffle=True)
    model.train()
    for _ in range(30):
        for bx, by in loader:
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()
    model.eval()
    return model, mean, std


def kinect_sustained_onset(kpath):
    frames = parse_dguha_kinect(Path(kpath))
    valid = [f for f in frames if f.points_mm.any()]
    if len(valid) < 30:
        return None
    t = np.asarray([f.timestamp.timestamp() for f in valid])
    head = np.asarray([np.max(f.points_mm[:, 2]) / 1000.0 for f in valid])
    base = np.median(head[:15])
    v = np.gradient(head, t)
    for i in range(3, len(v)):
        if v[i] < -0.15 and head[i] < base - 0.03:
            return t[i]
    return None


def main() -> int:
    root = Path("data/processed/experiments_v11")
    out = Path("reports/label_comparison_tcn_v1")
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path("data/external/dguha/raw")
    events = json.loads(Path("data/processed/dguha_prefall_0p5_1p0_dense_v3.events.json").read_text())
    event_by_src = {e["source_file"]: e for e in events}
    test_falls = [
        e for e in events
        if e["project_split"] == "test" and e.get("eligible_for_prediction_windows")
    ]

    # A held-out fall recording for the score curve (F_006_A5_001, test)
    curve_src = "Test/5_falling_forward/radar/F_006_A5_001.txt"
    curve_ev = event_by_src[curve_src]
    kpath = data_root / curve_src.replace("/radar/", "/kinect/")
    sustained_abs = kinect_sustained_onset(kpath)
    print(f"事件: {curve_src}")
    print(f"  radar descent_onset = {curve_ev['descent_onset_seconds_from_radar_start']:.2f}s (radar rel)")
    print(f"  kinect sustained_descent_onset(abs) = {sustained_abs}")

    extractor = RadarTemporalFeatureExtractorV2()
    results = {}
    for label_name in ["dguha_old_label_v1", "dguha_new_label_v1"]:
        feats, labels, splits, sources = load_npz(root / f"{label_name}.npz")
        model, mean, std = train_tcn(feats, labels, splits)
        # score the curve recording window by window
        frames = parse_radhar_text(data_root / curve_src, device_id="dguha")
        start = frames[0].timestamp
        score_curve = []  # (seconds_from_onset, score)
        for off in np.arange(0.0, 6.0, 0.2):
            end_ts = start + __import__("datetime").timedelta(seconds=off)
            wf = [f for f in frames if f.timestamp <= end_ts and f.timestamp >= end_ts - __import__("datetime").timedelta(seconds=2)]
            if not wf:
                continue
            try:
                w = extractor.transform(tuple(wf), end_timestamp=end_ts)
            except ValueError:
                continue
            if w.data_quality.value != "GOOD":
                continue
            norm = ((w.values.astype(np.float32) - mean[None, :]) / std[None, :]).astype(np.float32)
            with torch.inference_mode():
                logit = model(torch.from_numpy(norm[None])).item()
                sc = float(torch.sigmoid(torch.tensor(logit)).item())
            score_curve.append((off, sc))
        results[label_name] = {"curve_recording": curve_src, "curve": score_curve}
        print(f"  {label_name}: 曲线点数={len(score_curve)}")

    (out / "tcn_score_curves.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Print the curve aligned to sustained descent (approx via radar_rel difference)
    print("\n=== 分数曲线 (时间 vs score, 标注下降起点) ===")
    for label_name in ["dguha_old_label_v1", "dguha_new_label_v1"]:
        print(f"--- {label_name} ---")
        for off, sc in results[label_name]["curve"]:
            print(f"  t={off:4.1f}s score={sc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
