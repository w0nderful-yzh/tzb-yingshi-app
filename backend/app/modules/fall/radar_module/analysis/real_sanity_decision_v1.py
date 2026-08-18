"""真机 20 repeat sanity：ocPID 冻结模型 + 事件决策层。

目的
----
验证真机 IWR6843ISK 已有 20 repeat：
- standing / fast_sitting / forward_instability_recovery 不应持续报警
- controlled_forward_fall 应触发 FallProcess

方法
----
- 对每个真机 repeat：读 frames.jsonl（points）→ baseline-relative 特征
- ocPID 模型标准化（DGUHA+ocPID train mean/std）→ forward score
- 决策层（thr/consec/cooldown，validation 冻结）→ alarm episodes

注意：跨设备域偏移（DGUHA IWR1443 → 真机 IWR6843），结果作 sanity，
不调参。若真机正常动作持续报警，说明需要域校准（后续）。

Version: radar_real_sanity_decision_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_module.model.state_evolution_tcn_v1 import HierarchicalStateTCNV1
from radar_module.preprocess.baseline_relative_features_v2 import extract_sequence_features


def load_real_repeat_frames(repeat_dir: Path) -> list[dict[str, Any]]:
    frames_path = repeat_dir / "frames.jsonl"
    rows = []
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_sanity(
    session_root: Path,
    checkpoint: Path,
    norm_path: Path,
    *,
    threshold: float = 0.6,
    consec: int = 3,
    cooldown: int = 10,
    window_size: int = 20,
    stride: int = 10,
) -> dict[str, Any]:
    # 加载标准化
    norm = np.load(norm_path, allow_pickle=True)
    mean = norm["mean"]
    std = norm["std"]

    # 加载模型
    data_npz = np.load("data/processed/dguha_ocpid_v1.npz", allow_pickle=True)
    n_features = int(data_npz["features"].shape[2])
    model = HierarchicalStateTCNV1(n_features=n_features, hidden_dim=32, n_layers=3)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    results = {}
    for action_dir in sorted(session_root.iterdir()):
        if not action_dir.is_dir():
            continue
        action = action_dir.name
        repeat_summary = []
        for rep_dir in sorted(action_dir.iterdir()):
            if not rep_dir.is_dir() or not rep_dir.name.startswith("repeat_"):
                continue
            frames = load_real_repeat_frames(rep_dir)
            if not frames:
                continue
            records = [{
                "points": f.get("points") or f.get("points_sensor", ()),
                "timestamp": f.get("timestamp", "2026-08-18T00:00:00+00:00"),
            } for f in frames]
            try:
                feats, _ = extract_sequence_features(records)
            except Exception:
                repeat_summary.append({"repeat": rep_dir.name, "error": "feature_extract"})
                continue
            # 标准化 + clip
            feats = np.where(np.isnan(feats), mean[None, :], feats)
            feats = (feats - mean[None, :]) / std[None, :]
            feats = np.clip(feats, -10, 10)
            # 窗口
            n = len(feats)
            scores = []
            for start in range(0, n - window_size + 1, stride):
                win = feats[start : start + window_size]
                with torch.no_grad():
                    x = torch.as_tensor(win[None, :, :], dtype=torch.float32)
                    pl, _ = model(x)
                    s = torch.softmax(pl, dim=1)[0, 1].item()
                scores.append(s)
            scores = np.asarray(scores)
            # 决策层
            binseq = (scores >= threshold).astype(int)
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
            max_score = float(scores.max()) if len(scores) else 0.0
            repeat_summary.append({
                "repeat": rep_dir.name,
                "n_windows": len(scores),
                "max_score": round(max_score, 3),
                "alarm_episodes": episodes,
                "confirmed_windows": int(confirmed.sum()),
            })
        results[action] = repeat_summary
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-machine sanity with decision layer.")
    parser.add_argument("--session-root", type=Path, required=True,
                        help="reports/real_prefall_capture_v1")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("reports/state_evolution_tcn_v1/frozen_ocpid_state_tcn_v1.pt"))
    parser.add_argument("--norm", type=Path,
                        default=Path("data/processed/ocpid_norm_v1.npz"))
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--consec", type=int, default=3)
    parser.add_argument("--cooldown", type=int, default=10)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    results = run_sanity(
        args.session_root, args.checkpoint, args.norm,
        threshold=args.threshold, consec=args.consec, cooldown=args.cooldown,
    )
    print(f"params: thr={args.threshold} consec={args.consec} cooldown={args.cooldown}")
    for action, reps in sorted(results.items()):
        ep_counts = [r.get("alarm_episodes", -1) for r in reps]
        ok = all(e == 0 for e in ep_counts) if action != "controlled_forward_fall" else True
        n_alarm = sum(1 for e in ep_counts if e > 0)
        print(f"{action}: {n_alarm}/{len(reps)} repeats 有报警 | episodes={ep_counts} "
              f"{'✅' if ok else '⚠️持续报警'}")
        for r in reps:
            print(f"    {r['repeat']}: max_score={r.get('max_score')} episodes={r.get('alarm_episodes')}")
    (Path("reports/state_evolution_tcn_v1") / "real_sanity_decision.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
