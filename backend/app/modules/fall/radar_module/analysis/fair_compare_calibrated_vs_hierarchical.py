"""公平复测：旧 calibrated TCN vs 当前 hierarchical TCN。

条件（完全相同）：
- 真机 20 repeats（reports/real_prefall_capture_v1/）
- 各自原生特征/窗口推理
- 统一事件级决策层（threshold 0.6 / consec 3 / cooldown 10）

说明
----
- calibrated TCN 用 `temporal_features_v2`（19维，内部 2.2s 窗口）+ consume 逐帧
- hierarchical 用 `baseline_relative_features_v2`（21维，20帧滑窗）
- 两者特征维度/窗口不同（模型固有），但**决策层完全一致**
- 对比 controlled fall 触发、fast_sitting/standing/instability 误报、
  持续重复报警

Version: radar_fair_compare_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_module.contracts import RadarFrame, RadarPoint, Room, SourceMode
from radar_module.inference.calibrated_tcn_live_v1 import CalibratedTcnLivePredictorV1
from radar_module.model.state_evolution_tcn_v1 import HierarchicalStateTCNV1
from radar_module.preprocess.baseline_relative_features_v2 import extract_sequence_features

WINDOW_SIZE = 20
STRIDE = 10
DECISION = {"threshold": 0.6, "consec": 3, "cooldown": 10}
# 各自冻结阈值（公平：每个模型用自身阈值）
CAL_THRESHOLD = 0.35   # calibrated TCN 自身
HIER_THRESHOLD = 0.6   # hierarchical TCN 冻结决策层


def _parse_ts(value: str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _frames_to_radarframes(rows: list[dict[str, Any]]) -> list[RadarFrame]:
    frames = []
    for r in rows:
        pts = []
        for p in r.get("points") or r.get("points_sensor", ()):
            pts.append(RadarPoint(
                x=float(p["x"]), y=float(p["y"]), z=float(p["z"]),
                velocity=float(p.get("velocity", 0.0)),
                snr=float(p["snr"]) if p.get("snr") is not None else None,
            ))
        frames.append(RadarFrame(
            timestamp=_parse_ts(r["timestamp"]),
            device_id="real-iwr6843", room=Room.BATHROOM,
            source_mode=SourceMode.REAL, points=tuple(pts),
        ))
    return frames


def _decision_episodes(scores: np.ndarray, threshold, consec, cooldown) -> int:
    thr = threshold
    binseq = (scores >= thr).astype(int)
    confirmed = np.zeros_like(binseq)
    run = 0
    for j, b in enumerate(binseq):
        run = run + 1 if b == 1 else 0
        if run >= consec:
            confirmed[j] = 1
    episodes = 0
    in_ep = False
    last_end = -10**9
    for j, c in enumerate(confirmed):
        if c == 1:
            if not in_ep and (j - last_end) > cooldown:
                episodes += 1
            in_ep = True
        else:
            if in_ep:
                last_end = j
            in_ep = False
    return episodes


def _calibrated_scores(predictor, radar_frames: list[RadarFrame]) -> np.ndarray:
    """逐帧 consume，收集有效 pre_fall_score。"""
    scores = []
    predictor.reset()
    for frame in radar_frames:
        result = predictor.consume(frame)
        if result is not None and result.score_valid:
            scores.append(result.pre_fall_score)
    return np.asarray(scores)


def _hierarchical_scores(model, norm_mean, norm_std, rows) -> np.ndarray:
    records = [{
        "points": r.get("points") or r.get("points_sensor", ()),
        "timestamp": r.get("timestamp", "2026-08-18T00:00:00+00:00"),
    } for r in rows]
    feats, _ = extract_sequence_features(records, sample_rate_hz=20.0)
    if len(feats) < WINDOW_SIZE:
        return np.array([])
    feats = np.where(np.isnan(feats), norm_mean[None, :], feats)
    feats = (feats - norm_mean[None, :]) / norm_std[None, :]
    feats = np.clip(feats, -10, 10)
    scores = []
    n = len(feats)
    with torch.no_grad():
        for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
            win = feats[start : start + WINDOW_SIZE][None, :, :]
            x = torch.as_tensor(win, dtype=torch.float32)
            pl, _ = model(x)
            scores.append(torch.softmax(pl, dim=1)[0, 1].item())
    return np.asarray(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fair compare calibrated vs hierarchical.")
    parser.add_argument("--session-root", type=Path, required=True,
                        help="reports/real_prefall_capture_v1")
    parser.add_argument("--calibrated-checkpoint", type=Path, required=True)
    parser.add_argument("--calibrated-sha256", type=str, required=True)
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument("--hierarchical-checkpoint", type=Path, required=True)
    parser.add_argument("--hierarchical-norm", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/state_evolution_tcn_v1"))
    args = parser.parse_args()

    # 加载两个模型
    cal_predictor = CalibratedTcnLivePredictorV1(
        args.calibrated_checkpoint,
        expected_checkpoint_sha256=args.calibrated_sha256,
        calibration_path=args.calibration_path,
        device="cpu",
    )
    d = np.load("data/processed/dguha_ocpid_v1.npz", allow_pickle=True)
    hier_model = HierarchicalStateTCNV1(
        n_features=int(d["features"].shape[2]), hidden_dim=32, n_layers=3)
    hier_model.load_state_dict(torch.load(args.hierarchical_checkpoint, map_location="cpu"))
    hier_model.eval()
    norm = np.load(args.hierarchical_norm, allow_pickle=True)
    norm_mean, norm_std = norm["mean"], norm["std"]

    # 逐动作逐 repeat 评估
    print(f"决策层: consec={DECISION['consec']} cooldown={DECISION['cooldown']} "
          f"| cal_thr={CAL_THRESHOLD} hier_thr={HIER_THRESHOLD}")
    print(f"{'action/repeat':38s} {'cal_ep':>6s} {'hier_ep':>7s} "
          f"{'cal_max':>7s} {'hier_max':>8s}")
    results = {}
    for action_dir in sorted(args.session_root.iterdir()):
        if not action_dir.is_dir():
            continue
        action = action_dir.name
        per_repeat = []
        for rep_dir in sorted(action_dir.iterdir()):
            if not rep_dir.is_dir() or not rep_dir.name.startswith("repeat_"):
                continue
            frames_path = rep_dir / "frames.jsonl"
            if not frames_path.exists():
                continue
            rows = [json.loads(l) for l in frames_path.read_text().splitlines() if l.strip()]
            radar_frames = _frames_to_radarframes(rows)
            cal_scores = _calibrated_scores(cal_predictor, radar_frames)
            hier_scores = _hierarchical_scores(hier_model, norm_mean, norm_std, rows)
            cal_ep = _decision_episodes(cal_scores, CAL_THRESHOLD,
                                        DECISION["consec"], DECISION["cooldown"])
            hier_ep = _decision_episodes(hier_scores, HIER_THRESHOLD,
                                         DECISION["consec"], DECISION["cooldown"])
            cal_max = float(cal_scores.max()) if len(cal_scores) else 0.0
            hier_max = float(hier_scores.max()) if len(hier_scores) else 0.0
            per_repeat.append({
                "repeat": rep_dir.name, "cal_episodes": cal_ep,
                "hier_episodes": hier_ep, "cal_max_score": round(cal_max, 3),
                "hier_max_score": round(hier_max, 3),
            })
            print(f"{action}/{rep_dir.name:26s} {cal_ep:6d} {hier_ep:7d} "
                  f"{cal_max:7.3f} {hier_max:8.3f}")
        results[action] = per_repeat

    # 汇总
    print("\n=== 汇总 ===")
    summary = {}
    for action, reps in results.items():
        cal_alerts = sum(1 for r in reps if r["cal_episodes"] > 0)
        hier_alerts = sum(1 for r in reps if r["hier_episodes"] > 0)
        cal_dup = sum(r["cal_episodes"] for r in reps)
        hier_dup = sum(r["hier_episodes"] for r in reps)
        summary[action] = {
            "cal_alert_repeats": cal_alerts, "hier_alert_repeats": hier_alerts,
            "cal_total_episodes": cal_dup, "hier_total_episodes": hier_dup,
            "n_repeats": len(reps),
        }
        print(f"{action:35s} cal={cal_alerts}/{len(reps)} hier={hier_alerts}/{len(reps)} "
              f"| cal_ep={cal_dup} hier_ep={hier_dup}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "fair_compare_calibrated_vs_hierarchical.json").write_text(
        json.dumps({
            "decision": DECISION,
            "summary": summary,
            "per_repeat": results,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
