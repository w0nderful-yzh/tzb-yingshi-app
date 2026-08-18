"""Per-event aligned score curves for OLD vs NEW TCN.

For each test fall recording, align to sustained_descent_onset = 0 and report
the mean/median risk score of NEW-TCN and OLD-TCN in intervals:
  -1.5..-1.0, -1.0..-0.5, -0.5..-0.2, -0.2..0, 0..+0.5 s

Goal (user requirement): confirm whether NEW-TCN risk genuinely rises before
sustained descent, rather than only producing high scores in isolated windows.

We reconstruct per-recording scores by sliding the v2 extractor over the radar
frames and aligning to the Kinect sustained_descent_onset (absolute epoch).

This is analysis only. No model is modified.
Version: radar_label_comparison_event_curve_v1
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
CONF = dict(hidden=24, lr=1e-3, epochs=30, batch=256, pos_weight=12.0, seed=20260810)


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return (
        np.asarray(d["features"], dtype=np.float32),
        np.asarray(d["labels"], dtype=np.int64),
        np.asarray(d["split"]),
        np.asarray(d["source_files"]),
    )


def train_tcn(feats, labels, splits):
    torch.manual_seed(CONF["seed"])
    np.random.seed(CONF["seed"])
    tr = splits == "train"
    mean = feats[tr].mean(axis=(0, 1))
    std = feats[tr].std(axis=(0, 1))
    std = np.where(std < 1e-9, 1e-9, std)
    norm = ((feats - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    m = TemporalBinaryModel(architecture="causal_tcn", input_size=19, hidden_size=CONF["hidden"])
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CONF["pos_weight"]]))
    opt = torch.optim.Adam(m.parameters(), lr=CONF["lr"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(norm[tr]), torch.from_numpy(labels[tr].astype(np.float32))),
        batch_size=CONF["batch"], shuffle=True,
    )
    m.train()
    for _ in range(CONF["epochs"]):
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(m(bx), by)
            loss.backward()
            opt.step()
    m.eval()
    return m, mean, std


def sustained_onset(kpath):
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
    data_root = Path("data/external/dguha/raw")
    events = json.loads(Path("data/processed/dguha_prefall_0p5_1p0_dense_v3.events.json").read_text())
    test_falls = [e for e in events if e["project_split"] == "test" and e.get("eligible_for_prediction_windows")]
    extractor = RadarTemporalFeatureExtractorV2()

    intervals = {
        "m15_m10": (-1.5, -1.0),
        "m10_m05": (-1.0, -0.5),
        "m05_m02": (-0.5, -0.2),
        "m02_0": (-0.2, 0.0),
        "p0_p05": (0.0, 0.5),
    }

    report = {}
    for label_name in ["dguha_old_label_v1", "dguha_new_label_v1"]:
        feats, labels, splits, sources = load_npz(root / f"{label_name}.npz")
        model, mean, std = train_tcn(feats, labels, splits)
        per_event = {}
        n_events = 0
        for ev in test_falls:
            src = ev["source_file"]
            radar_rel_onset = ev["descent_onset_seconds_from_radar_start"]
            rpath = data_root / src
            kpath = data_root / src.replace("/radar/", "/kinect/")
            so = sustained_onset(kpath)
            if so is None:
                continue
            frames = parse_radhar_text(rpath, device_id="dguha")
            start_abs = frames[0].timestamp.timestamp()
            # score each window at 0.1s stride from onset-2 to onset+1
            aligned = []
            for dt in np.arange(-2.0, 1.0, 0.1):
                end_abs = so + dt
                end_ts = frames[0].timestamp + __import__("datetime").timedelta(seconds=(end_abs - start_abs))
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
                    sc = float(torch.sigmoid(torch.tensor(model(torch.from_numpy(norm[None])).item())).item())
                aligned.append((dt, sc))
            if not aligned:
                continue
            n_events += 1
            arr = np.array(aligned)
            ev_interval = {}
            for iname, (lo, hi) in intervals.items():
                mask = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
                if mask.sum() == 0:
                    ev_interval[iname] = None
                else:
                    ev_interval[iname] = {
                        "mean": float(arr[mask, 1].mean()),
                        "median": float(np.median(arr[mask, 1])),
                        "n": int(mask.sum()),
                    }
            per_event[src.split("/")[-1]] = ev_interval
        report[label_name] = {
            "n_events": n_events,
            "per_event": per_event,
            # aggregate over events per interval
        }
        # aggregate mean/median across events
        agg = {}
        for iname in intervals:
            vals = [per_event[e][iname] for e in per_event if per_event[e][iname] is not None]
            if vals:
                agg[iname] = {
                    "mean_of_mean": float(np.mean([v["mean"] for v in vals])),
                    "median_of_median": float(np.median([v["median"] for v in vals])),
                }
            else:
                agg[iname] = None
        report[label_name]["aggregate"] = agg
        print(f"\n=== {label_name} (n={n_events} events) ===")
        print(f'{"区间":>12s} {"均值分数":>10s} {"中位分数":>10s}')
        for iname, v in agg.items():
            if v:
                print(f"{iname:>12s} {v['mean_of_mean']:10.3f} {v['median_of_median']:10.3f}")

    out = Path("reports/label_comparison_tcn_v1")
    out.mkdir(parents=True, exist_ok=True)
    (out / "event_aligned_curves.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n已写入 reports/label_comparison_tcn_v1/event_aligned_curves.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
