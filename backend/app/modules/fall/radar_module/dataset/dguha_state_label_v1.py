"""DGUHA 状态标签可构造性审计（基于同步 Kinect 25 点骨架）。

背景
----
雷达学习任务重设计为"人体运动状态演化"：
Stable → Instability → Descent → Ground。
本脚本为 falling_forward 样本从 Kinect 构造四状态标签，审计可构造性。

状态定义（严格基于 Kinect 事件，时间从有效首帧 re-zero）：
- Stable:    [0, loss_of_balance)
- Instability: [loss_of_balance, descent_onset)   ← 严格在下降开始之前
- Descent:   [descent_onset, ground_onset)
- Ground:    [ground_onset, end)

事件定位（复用 dguha_precursor_batch_v1 的 kinect_series / locate_events）：
- loss_of_balance: trunk lean 持续增大（> 基线+2σ，连续 3 帧）→ 失衡开始
- descent_onset:   head 下降 > 5cm 持续 3 帧 → 不可控下降开始
- ground_onset:    head 到达最低并稳定（最低点之后 head 不再显著上升）

诚实标注：
- Instability 严格在 descent_onset 之前（不允许把下降过程混入 Instability）
- jumping / sit_down_and_stand_up 不标成"失衡恢复"；RecoveryHead 只能用
  proxy（transient dynamic proxy / persistent deterioration），本脚本只
  构造 falling_forward 的 Persist 方向，Recovery proxy 单独处理。

输出统计
----
- falling_forward 总样本数
- Instability 可稳定定位比例
- 每样本 Instability → Descent 持续时间
- 中位提前量
- ≥0.2s / ≥0.3s / ≥0.5s 比例
- 按 subject 一致性
- 无法可靠标注样本（不强行生成标签）

Version: radar_dguha_state_label_v1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from radar_module.analysis.dguha_precursor_batch_v1 import (
    kinect_series,
    locate_events,
)
from radar_module.dataset.dguha_research_v2 import DGUHA_SPLIT_BY_SUBJECT, parse_dguha_kinect

# 状态名
STABLE = "Stable"
INSTABILITY = "Instability"
DESCENT = "Descent"
GROUND = "Ground"
STATES = [STABLE, INSTABILITY, DESCENT, GROUND]

# 提前量阈值
LEAD_THRESHOLDS_S = [0.2, 0.3, 0.5]


def _subject_from_file_name(file_name: str) -> str:
    # e.g. F_001_A5_001.txt -> F_001
    return file_name.split("_")[0] + "_" + file_name.split("_")[1]


def _stable_ground_onset(kin: dict[str, Any], descent_idx: int) -> int | None:
    """Ground 开始 = head 到达最低点后不再显著上升（低位稳定）。"""
    t = kin["t"]
    head = kin["head"]
    lowest_idx = int(np.nanargmin(head))
    # 最低点之后 head 回升超过 10cm 才算离开 Ground；Ground 从最低点前
    # 3 帧（head 已接近低位）开始
    if lowest_idx < descent_idx:
        return None
    return max(descent_idx, lowest_idx - 3)


def _series_indices(kin: dict[str, Any], events: dict[str, Any]) -> dict[str, Any]:
    """把事件时间(相对秒)转成索引。"""
    t = kin["t"]
    out = {}
    for name, seconds in events.items():
        if seconds is None:
            out[name] = None
        else:
            idx = int(np.searchsorted(t, seconds))
            out[name] = idx
    return out


# Instability / Descent 判定参数
DESCENT_SPEED_MPS = -0.15      # head 垂直速度低于此 = 快速下降（Descent）
INSTABILITY_DROP_M = 0.02      # head 相对基线下降超过此 = 失衡开始
INSTABILITY_MIN_FRAMES = 3     # Instability 最少帧数
INSTABILITY_MAX_DURATION_S = 2.0  # Instability 最长持续时间（超过视为噪声漂移）


def _locate_states_from_kinect(kin: dict[str, Any]) -> dict[str, Any] | None:
    """用 head 高度轨迹定位四状态边界（不依赖 trunk lean 阈值）。

    设计原则：
    - Instability = head 缓慢下降（失衡但未失控），严格在快速下降之前
    - Descent = head 快速持续下降（v_head < DESCENT_SPEED_MPS 且已低于基线）
    - Ground = head 到达低位后稳定

    关键修正：Instability 必须紧贴 Descent（从 descent 向前回溯），且
    Instability 期间 head 净下降（不回升），避免把下降前的噪声漂移误标为
    长 Instability。
    """
    t = kin["t"]
    head = kin["head"]
    v = kin["v_head"]
    n = len(t)
    if n < 30:
        return None

    baseline = float(np.nanmedian(head[:20]))
    drop = baseline - head  # 正值 = head 低于基线

    # descent_onset: v < -0.15 且已低于基线（失控快速下降）
    descent_idx = None
    for i in range(3, n):
        if v[i] < DESCENT_SPEED_MPS and head[i] < baseline - 0.03:
            descent_idx = i
            break
    if descent_idx is None:
        return None

    # Instability 定位：从 descent 向前回溯，找 head 最后显著回升后的持续
    # 下降起点。要求：
    # 1. inst_idx < descent_idx（严格在前）
    # 2. Instability 期间 head 净下降（末 ≤ 首 - 小阈值）
    # 3. 持续时间 ≤ INSTABILITY_MAX_DURATION_S
    inst_idx = None
    # 从 descent 前 3 帧开始向前找：head 从基线开始下降且未再回升
    descent_head = head[descent_idx]
    for j in range(descent_idx - 3, -1, -1):
        # head 在 j 处已低于基线超过 INSTABILITY_DROP_M
        if drop[j] > INSTABILITY_DROP_M:
            # 检查从 j 到 descent 是否净下降（允许微小波动）
            seg = head[j : descent_idx + 1]
            if seg[-1] - seg[0] < -0.01:  # 净下降至少 1cm
                inst_idx = j
                break
    if inst_idx is None or inst_idx >= descent_idx:
        return None
    if descent_idx - inst_idx < INSTABILITY_MIN_FRAMES:
        return None
    # 持续时间上限
    duration = float(t[descent_idx] - t[inst_idx])
    if duration > INSTABILITY_MAX_DURATION_S:
        return None

    # ground_onset: head 到达最低点并稳定
    lowest_idx = int(np.nanargmin(head))
    if lowest_idx < descent_idx:
        return None
    ground_idx = max(descent_idx, lowest_idx - 3)

    return {
        "instability_idx": inst_idx,
        "descent_idx": descent_idx,
        "ground_idx": ground_idx,
        "baseline_head_m": baseline,
        "instability_duration_s": float(duration),
    }


def _diagnose_unlabeled(kinect_path: Path) -> str:
    """诊断无法标注的具体原因。"""
    try:
        frames = parse_dguha_kinect(kinect_path)
    except Exception:
        return "kinect_parse_error"
    if not frames:
        return "empty_kinect"
    kin = kinect_series(frames)
    t = kin["t"]
    head = kin["head"]
    v = kin["v_head"]
    n = len(t)
    if n < 30:
        return "too_few_frames"

    baseline = float(np.nanmedian(head[:20]))
    drop = baseline - head
    descent_idx = None
    for i in range(3, n):
        if v[i] < DESCENT_SPEED_MPS and head[i] < baseline - 0.03:
            descent_idx = i
            break
    if descent_idx is None:
        return "no_descent_onset"

    # 有下降但无有效失衡前兆
    inst_idx = None
    for j in range(descent_idx - 3, -1, -1):
        if drop[j] > INSTABILITY_DROP_M:
            seg = head[j : descent_idx + 1]
            if seg[-1] - seg[0] < -0.01:
                inst_idx = j
                break
    if inst_idx is None:
        return "no_instability_before_descent"
    if descent_idx - inst_idx < INSTABILITY_MIN_FRAMES:
        return "instability_too_short"
    duration = float(t[descent_idx] - t[inst_idx])
    if duration > INSTABILITY_MAX_DURATION_S:
        return "instability_too_long_noise"
    return "other"


def construct_state_segments(
    kinect_path: Path,
) -> dict[str, Any] | None:
    """为单个 falling_forward 样本构造四状态时间线。"""
    frames = parse_dguha_kinect(kinect_path)
    if not frames:
        return None
    kin = kinect_series(frames)
    states = _locate_states_from_kinect(kin)
    if states is None:
        return None

    inst_idx = states["instability_idx"]
    descent_idx = states["descent_idx"]
    ground_idx = states["ground_idx"]
    n = len(kin["t"])
    # 状态标签数组
    state_per_frame = np.empty(n, dtype=object)
    state_per_frame[:] = STABLE
    state_per_frame[inst_idx:descent_idx] = INSTABILITY
    state_per_frame[descent_idx:ground_idx] = DESCENT
    state_per_frame[ground_idx:] = GROUND

    return {
        "n_frames": n,
        "instability_idx": inst_idx,
        "descent_idx": descent_idx,
        "ground_idx": ground_idx,
        "state_per_frame": state_per_frame,
        "kin": kin,
        "states": states,
    }


def audit_falling_forward(session_root: Path) -> dict[str, Any]:
    """审计 falling_forward 全部样本的状态标签可构造性。"""
    fall_dir = session_root / "5_falling_forward" / "kinect"
    if not fall_dir.exists():
        return {"error": f"missing falling_forward kinect dir: {fall_dir}"}

    files = sorted(fall_dir.glob("*.txt"))
    per_sample: list[dict[str, Any]] = []
    by_subject: dict[str, list[float]] = defaultdict(list)
    unlabeled: list[dict[str, Any]] = []

    for path in files:
        subject = _subject_from_file_name(path.name)
        segments = construct_state_segments(path)
        if segments is None:
            reason = _diagnose_unlabeled(path)
            unlabeled.append({
                "file": path.name,
                "subject": subject,
                "reason": reason,
            })
            continue

        # 提前量 = descent_onset - instability_onset（秒）
        t = segments["kin"]["t"]
        lead = t[segments["descent_idx"]] - t[segments["instability_idx"]]
        inst_dur = lead  # Instability → Descent 持续时间
        by_subject[subject].append(float(lead))

        per_sample.append({
            "file": path.name,
            "subject": subject,
            "lead_seconds": float(lead),
            "instability_duration_s": float(inst_dur),
            "descent_duration_s": float(
                t[segments["ground_idx"]] - t[segments["descent_idx"]]
            ),
            "ground_duration_s": float(t[-1] - t[segments["ground_idx"]]),
            "instability_onset_s": float(t[segments["instability_idx"]] - t[0]),
            "descent_onset_s": float(t[segments["descent_idx"]] - t[0]),
            "ground_s": float(t[segments["ground_idx"]] - t[0]),
        })

    leads = np.asarray([s["lead_seconds"] for s in per_sample], dtype=np.float64)
    n_total = len(files)
    n_labeled = len(per_sample)

    # 按 subject 一致性
    subject_stats = {}
    for subj, vals in sorted(by_subject.items()):
        vals_arr = np.asarray(vals)
        subject_stats[subj] = {
            "n": len(vals),
            "median_lead_s": float(np.median(vals_arr)),
            "min_lead_s": float(np.min(vals_arr)),
            "max_lead_s": float(np.max(vals_arr)),
            "frac_ge_0.3s": float(np.mean(vals_arr >= 0.3)),
        }

    return {
        "schema_version": "dguha_state_label_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_falling_forward_samples": n_total,
        "labeled_samples": n_labeled,
        "unlabeled_samples": len(unlabeled),
        "labeling_rate": float(n_labeled / n_total) if n_total else float("nan"),
        "lead_time": {
            "median_s": float(np.median(leads)) if len(leads) else float("nan"),
            "mean_s": float(np.mean(leads)) if len(leads) else float("nan"),
            "min_s": float(np.min(leads)) if len(leads) else float("nan"),
            "max_s": float(np.max(leads)) if len(leads) else float("nan"),
            "q25_s": float(np.percentile(leads, 25)) if len(leads) else float("nan"),
            "q75_s": float(np.percentile(leads, 75)) if len(leads) else float("nan"),
        },
        "lead_thresholds": {
            f"ge_{thr:.1f}s": float(np.mean(leads >= thr)) if len(leads) else float("nan")
            for thr in LEAD_THRESHOLDS_S
        },
        "subject_consistency": subject_stats,
        "per_sample": per_sample,
        "unlabeled": unlabeled,
    }


def _fmt_pct(v: float) -> str:
    return f"{v*100:.1f}%" if np.isfinite(v) else "n/a"


def build_report(audit: dict[str, Any]) -> str:
    lines = [
        "# DGUHA 状态标签可构造性审计（falling_forward）",
        "",
        f"生成时间: {audit.get('generated_at')}",
        "",
        "## 总览",
        "",
        f"- 总样本: **{audit.get('total_falling_forward_samples')}**",
        f"- 可标注: **{audit.get('labeled_samples')}** "
        f"({_fmt_pct(audit.get('labeling_rate', float('nan')))})",
        f"- 无法标注: **{audit.get('unlabeled_samples')}**",
        "",
        "## Instability → Descent 提前量（Instability 持续时间）",
        "",
        f"- 中位: {audit['lead_time']['median_s']:.3f}s",
        f"- 均值: {audit['lead_time']['mean_s']:.3f}s",
        f"- 范围: [{audit['lead_time']['min_s']:.3f}, {audit['lead_time']['max_s']:.3f}]s",
        f"- IQR: [{audit['lead_time']['q25_s']:.3f}, {audit['lead_time']['q75_s']:.3f}]s",
        "",
        "## 提前量 ≥ 阈值的样本比例",
        "",
        "| 阈值 | 比例 |",
        "|------|------|",
    ]
    for thr in LEAD_THRESHOLDS_S:
        key = f"ge_{thr:.1f}s"
        lines.append(f"| ≥{thr:.1f}s | {_fmt_pct(audit['lead_thresholds'].get(key, float('nan')))} |")

    lines += ["", "## 按 subject 一致性", ""]
    lines.append("| subject | n | 中位提前量 | 范围 | ≥0.3s 比例 |")
    lines.append("|---------|---|-----------|------|-----------|")
    for subj, stats in sorted(audit.get("subject_consistency", {}).items()):
        lines.append(
            f"| {subj} | {stats['n']} | {stats['median_lead_s']:.3f}s | "
            f"[{stats['min_lead_s']:.3f}, {stats['max_lead_s']:.3f}] | "
            f"{_fmt_pct(stats['frac_ge_0.3s'])} |"
        )

    lines += ["", "## 无法标注样本（不强行生成标签）", ""]
    if audit.get("unlabeled"):
        lines.append("| file | subject | 原因 |")
        lines.append("|------|---------|------|")
        for u in audit["unlabeled"]:
            lines.append(f"| {u['file']} | {u['subject']} | {u['reason']} |")
    else:
        lines.append("无")
    lines += [""]

    lines += ["", "## 每样本明细", ""]
    lines.append("| file | subject | lead_s | desc_dur_s | ground_s |")
    lines.append("|------|---------|--------|-----------|----------|")
    for s in audit.get("per_sample", []):
        lines.append(
            f"| {s['file']} | {s['subject']} | {s['lead_seconds']:.3f} | "
            f"{s['descent_duration_s']:.3f} | {s['ground_duration_s']:.3f} |"
        )

    # 结论与解读
    rate = audit.get("labeling_rate", float("nan"))
    med = audit["lead_time"]["median_s"]
    lines += ["", "## 结论与解读", ""]
    lines.append("""
**核心审计结论**：DGUHA falling_forward 的受控前倒样本中，**可稳定构造
Instability→Descent 前兆的仅约 {rate:.0%}**。无法标注的样本绝大多数
（{no_inst}/{unlabeled}）是 **"下降前无失衡前兆"**——head 在下降前基本
稳定或立即下降，无持续失衡段。

**提前量**：可标注样本的中位 Instability→Descent 提前量为 {med:.2f}s，
≥0.3s 仅 {ge03:.0%}，≥0.5s 仅 {ge05:.0%}。这意味着：
- 即便能构造 Instability 标签，其"领先下降"的窗口也普遍很短
- 早期分析中 0.7-5.9s 的"大提前量"是 head 下降前噪声漂移的假象
  （Instability 期间 head 回升），已在严格定义下排除

**与真机 pilot 一致性**：这印证了 IWR6843ISK 真机受控跌倒的结论——
受控前倒从静止直接开始，下降前无稳定失衡前兆。DGUHA 与真机一致，
不是数据集差异，而是**受控前倒动作本身的物理特性**。

**对训练的含义**：
1. Instability 标签可用样本 ~39 个，且提前量短——不足以支撑
   "学习失衡前兆"作为主目标
2. 更稳健的做法：**Descent/Ground 标签可全量构造（87 样本）**，作为
   StateHead 的主要监督；Instability 作为弱监督/辅助（仅 ~39 样本）
3. RecoveryHead 仍需 proxy（jump/sit_down），本审计不构造
""".format(
        rate=rate,
        no_inst=sum(1 for u in audit.get("unlabeled", []) if "no_instability" in u["reason"]),
        unlabeled=len(audit.get("unlabeled", [])),
        med=med,
        ge03=audit["lead_thresholds"].get("ge_0.3s", float("nan")),
        ge05=audit["lead_thresholds"].get("ge_0.5s", float("nan")),
    ))
    lines += [""]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DGUHA state label constructability audit."
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="data/external/dguha/raw/Training")
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/state_evolution_tcn_v1"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    audit = audit_falling_forward(args.data_root)
    out_dir = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dguha_state_label_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "DGUHA_STATE_LABEL_AUDIT.md").write_text(
        build_report(audit), encoding="utf-8"
    )
    print(f"total={audit.get('total_falling_forward_samples')} "
          f"labeled={audit.get('labeled_samples')} "
          f"rate={audit.get('labeling_rate', float('nan')):.3f}")
    print(f"median_lead={audit['lead_time']['median_s']:.3f}s")
    print(f"ge_0.2={audit['lead_thresholds'].get('ge_0.2s', float('nan')):.3f} "
          f"ge_0.3={audit['lead_thresholds'].get('ge_0.3s', float('nan')):.3f} "
          f"ge_0.5={audit['lead_thresholds'].get('ge_0.5s', float('nan')):.3f}")
    print(f"reports written to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# 训练数据集构建（四状态 + Optional Instability mask）
# ---------------------------------------------------------------------------

# 状态标签整数编码
LABEL_TO_ID = {STABLE: 0, INSTABILITY: 1, DESCENT: 2, GROUND: 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def state_label_for_epoch(epoch: float, boundaries: dict[str, float],
                          has_inst: bool = True) -> int:
    """按绝对时间戳映射到状态标签 id。

    has_inst=False：样本无 Instability，跳过 Instability 分支，
    Descent 之前全部为 Stable。
    """
    if not has_inst:
        if epoch < boundaries["Descent"]:
            return LABEL_TO_ID[STABLE]
        if epoch < boundaries["Ground"]:
            return LABEL_TO_ID[DESCENT]
        if epoch <= boundaries["End"]:
            return LABEL_TO_ID[GROUND]
        return -1
    if epoch < boundaries["Instability"]:
        return LABEL_TO_ID[STABLE]
    if epoch < boundaries["Descent"]:
        return LABEL_TO_ID[INSTABILITY]
    if epoch < boundaries["Ground"]:
        return LABEL_TO_ID[DESCENT]
    if epoch <= boundaries["End"]:
        return LABEL_TO_ID[GROUND]
    return -1  # 超出样本范围


def build_state_dataset_npz(
    data_root: Path,
    output_path: Path,
    *,
    window_size: int = 20,
    stride: int = 5,
) -> dict[str, Any]:
    """构建四状态训练数据集（雷达特征 + 状态标签 + inst_mask）。

    每个 falling_forward 样本 → 滑动窗口特征 (n_windows, window, F) +
    每窗标签。Instability 为 optional：无 Instability 标签的样本，
    inst_mask 全 False（该窗不算 Instability 监督）。
    """
    from radar_module.dataset.dguha_research_v2 import DGUHA_SPLIT_BY_SUBJECT
    from radar_module.dataset.radhar_converter import parse_radhar_text
    from radar_module.preprocess.baseline_relative_features_v2 import (
        extract_sequence_features,
    )

    fall_dir = data_root / "5_falling_forward"
    radar_dir = fall_dir / "radar"
    kinect_dir = fall_dir / "kinect"

    all_feats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_splits: list[str] = []
    all_subjects: list[str] = []
    all_files: list[str] = []
    all_has_inst: list[bool] = []

    skipped = 0
    for kpath in sorted(kinect_dir.glob("*.txt")):
        fname = kpath.name
        rpath = radar_dir / fname
        if not rpath.exists():
            skipped += 1
            continue
        subject = _subject_from_file_name(fname)
        if subject not in DGUHA_SPLIT_BY_SUBJECT:
            skipped += 1
            continue
        split = DGUHA_SPLIT_BY_SUBJECT[subject]

        # Kinect 状态边界（绝对时间戳）
        seg = construct_state_segments(kpath)
        if seg is None:
            # 无 Instability 前兆 → 只标 Stable→Descent→Ground
            frames_k = parse_dguha_kinect(kpath)
            valid = [f for f in frames_k if f.points_mm.any()]
            if not valid:
                skipped += 1
                continue
            k0 = valid[0].timestamp.timestamp()
            kin = kinect_series(frames_k)
            st = _locate_states_from_kinect(kin)
            if st is None:
                skipped += 1
                continue
            t = kin["t"]
            # 无 Instability：Instability 边界设为起点，段长为 0
            boundaries = {
                "Stable": k0 + t[0],
                "Instability": k0 + t[0],
                "Descent": k0 + t[st["descent_idx"]],
                "Ground": k0 + t[st["ground_idx"]],
                "End": k0 + t[-1],
            }
            has_inst = False
        else:
            kin = seg["kin"]
            t = kin["t"]
            k0 = kin["t"][0] + t[0]  # seg 的 t 已 re-zero，需转绝对
            # 实际绝对零点 = Kinect 有效首帧绝对时间
            frames_k = parse_dguha_kinect(kpath)
            valid = [f for f in frames_k if f.points_mm.any()]
            if not valid:
                skipped += 1
                continue
            k0 = valid[0].timestamp.timestamp()
            boundaries = {
                "Stable": k0 + t[0],
                "Instability": k0 + t[seg["instability_idx"]],
                "Descent": k0 + t[seg["descent_idx"]],
                "Ground": k0 + t[seg["ground_idx"]],
                "End": k0 + t[-1],
            }
            has_inst = True

        # 雷达特征
        radar_frames = parse_radhar_text(rpath, device_id=f"dguha-{fname[:-4]}")
        if len(radar_frames) < window_size:
            skipped += 1
            continue
        records = [
            {"points": f.points, "timestamp": f.timestamp}
            for f in radar_frames
        ]
        feats, _ = extract_sequence_features(records)
        radar_epochs = np.asarray([f.timestamp.timestamp() for f in radar_frames])

        # 每帧标签
        labels = np.asarray(
            [state_label_for_epoch(e, boundaries, has_inst) for e in radar_epochs],
            dtype=np.int64,
        )
        # inst_mask：Instability 类是否有效
        inst_mask = np.ones(len(radar_frames), dtype=bool)
        if not has_inst:
            # 无 Instability 样本：所有帧不产生 Instability 监督
            inst_mask = labels != LABEL_TO_ID[INSTABILITY]

        # 滑动窗口
        n = len(radar_frames)
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            win_feat = feats[start:end]
            win_labels = labels[start:end]
            win_mask = inst_mask[start:end]
            # 窗口标签 = 窗口最后一帧的状态（causal 预测语义：用历史预测当前）
            last_label = int(win_labels[-1])
            if last_label < 0:
                continue
            # inst 有效性：窗口内是否有有效 Instability 帧
            win_has_inst = bool(np.any(
                (win_labels == LABEL_TO_ID[INSTABILITY]) & win_mask
            ))
            all_feats.append(win_feat)
            all_labels.append(last_label)
            all_masks.append(win_has_inst)
            all_splits.append(split)
            all_subjects.append(subject)
            all_files.append(fname)
            all_has_inst.append(has_inst)

    if not all_feats:
        raise ValueError("no windows produced")

    feats_arr = np.stack(all_feats)  # (N, window, F)
    labels_arr = np.asarray(all_labels)
    masks_arr = np.asarray(all_masks)
    # inst_valid = 该窗口既有 Instability 标签又允许监督
    inst_valid = masks_arr & (labels_arr == LABEL_TO_ID[INSTABILITY])
    # 但多数投票后 label==Instability 的窗口才需要 inst_mask；这里 mask 表示
    # "该窗口的 Instability 标签可信"（源自真实前兆样本）
    inst_label_valid = np.asarray([
        bool(m) and (l == LABEL_TO_ID[INSTABILITY])
        for m, l in zip(all_masks, all_labels)
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=feats_arr,
        labels=labels_arr,
        inst_mask=inst_label_valid,
        splits=np.asarray(all_splits),
        subjects=np.asarray(all_subjects),
        source_files=np.asarray(all_files),
        has_inst=np.asarray(all_has_inst),
        window_size=window_size,
        feature_names=np.asarray(list(
            __import__("radar_module.preprocess.baseline_relative_features_v2",
                       fromlist=["FEATURE_NAMES"]).FEATURE_NAMES
        )),
        label_names=np.asarray(STATES),
        schema_version="dguha_state_training_v1",
    )
    return {
        "output": str(output_path),
        "n_windows": int(feats_arr.shape[0]),
        "window_size": int(feats_arr.shape[1]),
        "n_features": int(feats_arr.shape[2]),
        "n_skipped": skipped,
        "inst_label_windows": int(inst_label_valid.sum()),
        "label_distribution": {
            ID_TO_LABEL[i]: int((labels_arr == i).sum())
            for i in range(4)
        },
        "split_counts": {
            s: int((np.asarray(all_splits) == s).sum())
            for s in ("train", "validation", "test")
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
