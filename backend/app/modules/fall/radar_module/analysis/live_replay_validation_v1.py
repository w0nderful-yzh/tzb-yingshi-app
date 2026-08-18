"""Replay real-sensor sessions through the fixed descent + calibrated TCN.

After the zero-variance normalization fix, this script replays real IWR6843
sessions and reports whether the descent-detection and calibrated-TCN branches
now emit meaningful scores for real actions (normal actions low, fall/high-risk
actions with signal). It does not modify any checkpoint or threshold.

Version: radar_live_replay_validation_v1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from radar_module.contracts import Room
from radar_module.dataset.v2_export import _load_replay_frames
from radar_module.inference.descent_live_v1 import RadarDescentLivePredictorV1
from radar_module.inference.calibrated_tcn_live_v1 import CalibratedTcnLivePredictorV1


DESCENT_CKPT = "checkpoints/experiments_v10/descent_detection_tcn_v1.pt"
DESCENT_SHA = "82ba9c7dbb4862609ac36e02dd183df87fb8c966957c2c8b3f1e9cbb3df22ca4"
DESCENT_CALIB = (
    "reports/domain_calibration_v1_full/"
    "calibrated_normalization_descent_iwr6843_fall102.json"
)
TCN_CKPT = (
    "checkpoints/experiments_v5/tcn_hard_negative/"
    "tcn_0p5_1p0_specificity_operating_point_v1.pt"
)
TCN_SHA = "0792a712b57ae89875b2d57e6ba7a20763618a2718e961cf8c48acebe34970ef"
TCN_CALIB = (
    "reports/domain_calibration_v1_full/"
    "calibrated_normalization_real_gaussian.json"
)


def replay_session(session_path: Path, descent, calib_tcn):
    frames = _load_replay_frames(session_path, default_room=Room.BATHROOM)
    descent_scores = []
    descent_states = []
    calib_scores = []
    calib_states = []
    # tolerate non-monotonic timestamps (replay sessions may have small regressions)
    last_ts = None
    for frame in frames:
        if last_ts is not None and frame.timestamp <= last_ts:
            continue
        last_ts = frame.timestamp
        d = descent.consume(frame)
        if d is not None:
            descent_scores.append(d.descent_score)
            descent_states.append(d.risk_state)
        c = calib_tcn.consume(frame)
        if c is not None:
            calib_scores.append(c.pre_fall_score)
            calib_states.append(c.gate_state)
    return {
        "frame_count": len(frames),
        "descent": {
            "valid": len(descent_scores),
            "score_median": float(np.median(descent_scores)) if descent_scores else None,
            "score_p90": float(np.percentile(descent_scores, 90)) if descent_scores else None,
            "score_max": float(max(descent_scores)) if descent_scores else None,
            "states": dict(Counter(descent_states)),
        },
        "calibrated_tcn": {
            "valid": len(calib_scores),
            "score_median": float(np.median(calib_scores)) if calib_scores else None,
            "score_p90": float(np.percentile(calib_scores, 90)) if calib_scores else None,
            "score_max": float(max(calib_scores)) if calib_scores else None,
            "states": dict(Counter(calib_states)),
        },
    }


def main() -> int:
    import os

    root = Path(".").resolve()
    descent = RadarDescentLivePredictorV1(
        root / DESCENT_CKPT,
        expected_checkpoint_sha256=DESCENT_SHA,
        calibration_path=root / DESCENT_CALIB,
    )
    calib_tcn = CalibratedTcnLivePredictorV1(
        root / TCN_CKPT,
        expected_checkpoint_sha256=TCN_SHA,
        calibration_path=root / TCN_CALIB,
    )

    sessions = {
        "正常-站立": (
            "reports/continuous_scene_validation_v1/controlled_forward_fall_p01_r01"
            "/phases/baseline_standing/session.jsonl"
        ),
        "正常-行走": (
            "reports/real_scene_validation_v1/"
            "person_walking_single_start_20260809/session.jsonl"
        ),
        "快速坐": (
            "reports/continuous_scene_validation_v1/high_risk_screen_p01_r01"
            "/phases/fast_sit/session.jsonl"
        ),
        "快速蹲": (
            "reports/continuous_scene_validation_v1/high_risk_screen_p01_r01"
            "/phases/fast_squat/session.jsonl"
        ),
        "受控跌倒": (
            "reports/continuous_scene_validation_v1/controlled_forward_fall_p01_r01"
            "/phases/assisted_recovery/session.jsonl"
        ),
    }

    report = {}
    print(f'{"会话":10s} | {"下降中位":>8s} {"下降p90":>8s} {"下降max":>8s} | {"TCN中位":>8s} {"TCNp90":>8s} {"TCNmax":>8s} | {"下降状态":>12s}')
    for name, rel in sessions.items():
        descent.reset()
        calib_tcn.reset()
        result = replay_session(root / rel, descent, calib_tcn)
        report[name] = result
        d = result["descent"]
        c = result["calibrated_tcn"]
        dmed = f'{d["score_median"]:.4f}' if d["score_median"] is not None else "--"
        dp90 = f'{d["score_p90"]:.4f}' if d["score_p90"] is not None else "--"
        dmax = f'{d["score_max"]:.4f}' if d["score_max"] is not None else "--"
        cmed = f'{c["score_median"]:.4f}' if c["score_median"] is not None else "--"
        cp90 = f'{c["score_p90"]:.4f}' if c["score_p90"] is not None else "--"
        cmax = f'{c["score_max"]:.4f}' if c["score_max"] is not None else "--"
        print(f'{name:10s} | {dmed:>8s} {dp90:>8s} {dmax:>8s} | {cmed:>8s} {cp90:>8s} {cmax:>8s} | {str(d["states"]):>12s}')

    out = Path("reports/live_replay_validation_v1")
    out.mkdir(parents=True, exist_ok=True)
    (out / "live_replay_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n报告已写入 reports/live_replay_validation_v1/live_replay_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
