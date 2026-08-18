"""Event-level temporal decision layer（不改神经网络）。

对模型输出的每窗 score，在 recording 层面做：
1. consecutive-window confirmation：连续 K 窗 score≥thr 才算"确认阳性"，
   抑制单窗噪声误报
2. alarm episode merge：确认阳性窗口聚合成 episode
3. cooldown / re-arm：episode 结束后 cooldown 窗内不重新报警

参数（threshold, K, cooldown）在 **validation** 上选择，目标：
- fall event recall 不明显下降
- left/right limb extension alarm episodes/recording 尽量降到 ~1 或更低
held-out 只用于最终验证（冻结参数）。

Version: radar_event_decision_layer_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def apply_decision_layer(
    scores: np.ndarray,
    labels: np.ndarray,
    source_files: np.ndarray,
    actions: np.ndarray,
    *,
    threshold: float,
    consec: int,
    cooldown: int,
) -> dict[str, Any]:
    """对每个 recording 应用决策层。

    返回：
    - fall event recall（fall recording 至少一个 episode）
    - 分动作 episodes/recording
    - 总 episodes、误报 episodes
    """
    from collections import OrderedDict, defaultdict

    recs = OrderedDict()
    rec_actions = {}
    for i, src in enumerate(source_files):
        recs.setdefault(src, []).append(i)
        rec_actions[src] = str(actions[i])

    fall_events = 0
    fall_detected = 0
    n_fa_episodes = 0
    eps_by_action = defaultdict(lambda: {"episodes": 0, "n_rec": 0})

    for src, idxs in recs.items():
        idxs = sorted(idxs)
        action = rec_actions[src]
        eps_by_action[action]["n_rec"] += 1
        s = scores[idxs]
        y = labels[idxs]
        # 连续确认：连续 K 窗 ≥ thr
        binseq = (s >= threshold).astype(int)
        confirmed = np.zeros_like(binseq)
        run = 0
        for j, b in enumerate(binseq):
            run = run + 1 if b == 1 else 0
            if run >= consec:
                confirmed[j] = 1
        # episode 合并 + cooldown
        episodes = 0
        in_ep = False
        last_ep_end = -10**9
        for j, c in enumerate(confirmed):
            if c == 1:
                if not in_ep and (j - last_ep_end) > cooldown:
                    episodes += 1
                in_ep = True
            else:
                if in_ep:
                    last_ep_end = j
                in_ep = False
        eps_by_action[action]["episodes"] += episodes
        if y.max() == 1:
            fall_events += 1
            if episodes > 0:
                fall_detected += 1
        else:
            n_fa_episodes += episodes

    return {
        "fall_event_recall": fall_detected / fall_events if fall_events else float("nan"),
        "n_fall_events": fall_events,
        "n_fall_detected": fall_detected,
        "n_fa_episodes": n_fa_episodes,
        "episodes_per_recording": {
            a: v["episodes"] / v["n_rec"] if v["n_rec"] else float("nan")
            for a, v in sorted(eps_by_action.items())
        },
        "episode_counts": {a: v["episodes"] for a, v in sorted(eps_by_action.items())},
    }


def search_params(scores, labels, source_files, actions):
    """在 validation 上网格搜索 threshold / consec / cooldown。"""
    results = []
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for consec in [1, 2, 3]:
            for cooldown in [3, 5, 10]:
                r = apply_decision_layer(
                    scores, labels, source_files, actions,
                    threshold=thr, consec=consec, cooldown=cooldown,
                )
                limb = r["episodes_per_recording"].get("6_Right_limb_extension", 0)
                llimb = r["episodes_per_recording"].get("7_Left_limb_extension", 0)
                # 目标：fall recall 高 + limb episodes 低
                score_val = r["fall_event_recall"] - 0.3 * (limb + llimb)
                results.append({
                    "threshold": thr, "consec": consec, "cooldown": cooldown,
                    "fall_event_recall": r["fall_event_recall"],
                    "right_limb_ep": limb, "left_limb_ep": llimb,
                    "n_fa_episodes": r["n_fa_episodes"],
                    "objective": score_val,
                })
    results.sort(key=lambda r: -r["objective"])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Event decision layer tuning.")
    parser.add_argument("--scores", type=Path, required=True,
                        help="npz with val/held scores+labels+src+action")
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/state_evolution_tcn_v1"))
    args = parser.parse_args()

    d = np.load(args.scores, allow_pickle=True)
    val_s, held_s = d["val_s"], d["held_s"]
    val_y, held_y = d["val_y"], d["held_y"]
    val_src, held_src = d["val_src"], d["held_src"]
    val_act, held_act = d["val_action"], d["held_action"]

    # validation 上搜索
    results = search_params(val_s, val_y, val_src, val_act)
    best = results[0]
    print("=== validation 参数搜索 top5 ===")
    for r in results[:5]:
        print(f"  thr={r['threshold']} consec={r['consec']} cd={r['cooldown']} "
              f"fall_rec={r['fall_event_recall']:.3f} right={r['right_limb_ep']:.2f} "
              f"left={r['left_limb_ep']:.2f} nfa={r['n_fa_episodes']} obj={r['objective']:.3f}")

    # 用 best 参数在 held-out 验证
    held_r = apply_decision_layer(
        held_s, held_y, held_src, held_act,
        threshold=best["threshold"], consec=best["consec"], cooldown=best["cooldown"],
    )
    print("\n=== held-out 验证（冻结参数）===")
    print(f"  params: thr={best['threshold']} consec={best['consec']} cooldown={best['cooldown']}")
    print(f"  fall_event_recall: {held_r['fall_event_recall']:.3f} "
          f"({held_r['n_fall_detected']}/{held_r['n_fall_events']})")
    print(f"  n_fa_episodes: {held_r['n_fa_episodes']}")
    for a, v in held_r["episodes_per_recording"].items():
        print(f"  {a}: ep/rec={v:.2f}")

    out = {
        "best_params": best,
        "val_search_top": results[:10],
        "held_out_validation": held_r,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "event_decision_layer.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
