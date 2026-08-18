"""层级式因果 TCN 训练数据集构建。

层级结构
--------
- Main ProcessHead: NormalDynamic(0) / FallProcess(1)
  - NormalDynamic = 全部正常动作（running/jumping/sit_down/limb_ext）+ fall 的 Stable 段
  - FallProcess = fall 的 Descent + Ground 段（合并，不再细分 Descent/Ground）
- Optional InstabilityHead: Instability(1) / 其他(mask)
  - 只在有可靠 Instability 标签的事件上监督；无可靠事件用 mask，
    不强行标负样本

Ground 退出深度学习分类目标，仅作为 FallProcess 后的事件确认。

窗口标签策略
------------
- 每个窗口用"窗口末尾帧"的状态决定 process 标签
- Instability 段帧：process=0(NormalDynamic，未下降)、inst=1(正样本)
- Descent/Ground 段帧：process=1(FallProcess)、inst 不监督(mask)

Version: radar_dguha_hierarchical_dataset_v1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from radar_module.dataset.dguha_research_v2 import DGUHA_SPLIT_BY_SUBJECT, parse_dguha_kinect
from radar_module.dataset.dguha_state_label_v1 import (
    STABLE,
    INSTABILITY,
    DESCENT,
    GROUND,
    _subject_from_file_name,
    _locate_states_from_kinect,
    construct_state_segments,
    state_label_for_epoch,
)
from radar_module.analysis.dguha_precursor_batch_v1 import kinect_series
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.preprocess.baseline_relative_features_v2 import extract_sequence_features

# 非 fall 动作目录（NormalDynamic）
NORMAL_ACTION_DIRS = [
    "1_Running",
    "2_Jumping",
    "3_Sit_down_and_stand_up",
    "4_Both_upper_limb_extension",
    "6_Right_limb_extension",
    "7_Left_limb_extension",
]
FALL_ACTION_DIR = "5_falling_forward"

# process 标签
PROCESS_NORMAL = 0
PROCESS_FALL = 1


def _fall_state_boundaries(kpath: Path) -> tuple[dict[str, float], bool] | None:
    """fall 样本的 Kinect 状态边界（绝对时间戳）和 has_inst。"""
    frames_k = parse_dguha_kinect(kpath)
    valid = [f for f in frames_k if f.points_mm.any()]
    if not valid:
        return None
    k0 = valid[0].timestamp.timestamp()
    kin = kinect_series(frames_k)
    t = kin["t"]
    st = _locate_states_from_kinect(kin)
    if st is None:
        return None
    return {
        "Stable": k0 + t[0],
        "Instability": k0 + t[st["instability_idx"]],
        "Descent": k0 + t[st["descent_idx"]],
        "Ground": k0 + t[st["ground_idx"]],
        "End": k0 + t[-1],
    }, True


def _fall_no_inst_boundaries(kpath: Path) -> dict[str, float] | None:
    """无 Instability 前兆的 fall 样本：只标 Stable→Descent→Ground。"""
    frames_k = parse_dguha_kinect(kpath)
    valid = [f for f in frames_k if f.points_mm.any()]
    if not valid:
        return None
    k0 = valid[0].timestamp.timestamp()
    kin = kinect_series(frames_k)
    t = kin["t"]
    st = _locate_states_from_kinect(kin)
    if st is None:
        return None
    return {
        "Stable": k0 + t[0],
        "Instability": k0 + t[st["descent_idx"]],  # 段长为 0
        "Descent": k0 + t[st["descent_idx"]],
        "Ground": k0 + t[st["ground_idx"]],
        "End": k0 + t[-1],
    }


def _radar_feats(rpath: Path, window_size: int) -> tuple[np.ndarray, np.ndarray] | None:
    """雷达帧 → (特征序列, 绝对时间戳)。"""
    radar_frames = parse_radhar_text(rpath, device_id="dguha")
    if len(radar_frames) < window_size:
        return None
    records = [{"points": f.points, "timestamp": f.timestamp} for f in radar_frames]
    feats, _ = extract_sequence_features(records)
    epochs = np.asarray([f.timestamp.timestamp() for f in radar_frames])
    return feats, epochs


def build_hierarchical_dataset_npz(
    data_root: Path,
    output_path: Path,
    *,
    window_size: int = 20,
    stride: int = 10,
    max_normal_windows_per_recording: int = 40,
    max_samples_per_action: int | None = None,
    split_filter: str | None = None,
    split_path: Path | None = None,
    recording_max_windows: int | None = None,
    action_filter: str | None = None,
) -> dict[str, Any]:
    """构建层级数据集。

    输出字段：
    - features: (N, window, F)
    - process_labels: (N,)  0=NormalDynamic, 1=FallProcess
    - inst_labels: (N,)     Instability 正样本标记（仅 inst_valid 有效）
    - inst_valid: (N,)      Instability 是否参与监督
    - splits: fold id（0=held-out, 1-4=cv）——用冻结的 subject split
    - subjects / source_files / action

    split_path：冻结的 subject→fold 映射 json（dguha_subject_split_v1）。
    recording_max_windows：每个 recording 最多取窗口数（recording-level
    采样平衡，避免高度相关窗口造成伪样本量）。
    """
    all_feats: list[np.ndarray] = []
    all_process: list[int] = []
    all_inst: list[int] = []
    all_inst_valid: list[bool] = []
    all_splits: list[str] = []
    all_subjects: list[str] = []
    all_files: list[str] = []
    all_actions: list[str] = []

    # 冻结 split：subject → fold
    subject_to_fold: dict[str, str] | None = None
    if split_path is not None and split_path.exists():
        import json as _json

        split_data = _json.loads(split_path.read_text(encoding="utf-8"))
        subject_to_fold = {
            s: str(f) for s, f in split_data["subject_to_fold"].items()
        }
    skipped = 0

    # 1. 正常动作
    for action_dir in NORMAL_ACTION_DIRS:
        if action_filter is not None and action_dir != action_filter:
            continue
        adir = data_root / action_dir
        radar_dir = adir / "radar"
        if not radar_dir.exists():
            skipped += 1
            continue
        sample_count = 0
        for rpath in sorted(radar_dir.glob("*.txt")):
            if max_samples_per_action is not None and sample_count >= max_samples_per_action:
                break
            fname = rpath.name
            subject = _subject_from_file_name(fname)
            if subject not in DGUHA_SPLIT_BY_SUBJECT:
                continue
            split = (
                subject_to_fold.get(subject, "1")
                if subject_to_fold is not None
                else DGUHA_SPLIT_BY_SUBJECT[subject]
            )
            if split_filter is not None and split != split_filter:
                continue
            sample_count += 1
            feats, epochs = _radar_feats(rpath, window_size)
            if feats is None:
                skipped += 1
                continue
            n = len(feats)
            win_count = 0
            rec_cap = recording_max_windows or max_normal_windows_per_recording
            for start in range(0, n - window_size + 1, stride):
                if win_count >= rec_cap:
                    break
                all_feats.append(feats[start : start + window_size])
                all_process.append(PROCESS_NORMAL)
                all_inst.append(0)
                all_inst_valid.append(False)
                all_splits.append(split)
                all_subjects.append(subject)
                all_files.append(fname)
                all_actions.append(action_dir)
                win_count += 1

    # 2. fall 动作
    fdir = data_root / FALL_ACTION_DIR
    radar_dir = fdir / "radar"
    kinect_dir = fdir / "kinect"
    sample_count = 0
    for kpath in sorted(kinect_dir.glob("*.txt")):
        if action_filter is not None and action_filter != FALL_ACTION_DIR:
            continue
        # fall 动作不限制样本数（保证 39 个有 inst 的都在）
        fname = kpath.name
        rpath = radar_dir / fname
        if not rpath.exists():
            skipped += 1
            continue
        subject = _subject_from_file_name(fname)
        if subject not in DGUHA_SPLIT_BY_SUBJECT:
            continue
        split = (
            subject_to_fold.get(subject, "1")
            if subject_to_fold is not None
            else DGUHA_SPLIT_BY_SUBJECT[subject]
        )
        if split_filter is not None and split != split_filter:
            continue
        sample_count += 1

        # 状态边界
        seg = construct_state_segments(kpath)
        if seg is not None:
            frames_k = parse_dguha_kinect(kpath)
            valid = [f for f in frames_k if f.points_mm.any()]
            k0 = valid[0].timestamp.timestamp()
            t = seg["kin"]["t"]
            boundaries = {
                "Stable": k0 + t[0],
                "Instability": k0 + t[seg["instability_idx"]],
                "Descent": k0 + t[seg["descent_idx"]],
                "Ground": k0 + t[seg["ground_idx"]],
                "End": k0 + t[-1],
            }
            has_inst = True
        else:
            boundaries = _fall_no_inst_boundaries(kpath)
            if boundaries is None:
                skipped += 1
                continue
            has_inst = False

        feats, epochs = _radar_feats(rpath, window_size)
        if feats is None:
            skipped += 1
            continue

        # 每帧 process / inst 标签
        process_per_frame = np.full(len(epochs), PROCESS_NORMAL, dtype=np.int64)
        inst_per_frame = np.zeros(len(epochs), dtype=np.int64)
        inst_valid_per_frame = np.zeros(len(epochs), dtype=bool)
        for i, e in enumerate(epochs):
            state = state_label_for_epoch(e, boundaries, has_inst)
            if state == 2 or state == 3:  # Descent / Ground → FallProcess
                process_per_frame[i] = PROCESS_FALL
            if has_inst:
                # 有可靠 Instability 事件：
                # - Instability 段 = positive
                # - Instability 前 Stable 段 = negative
                # - Descent/Ground = mask（不监督）
                if state == 1:
                    inst_per_frame[i] = 1
                    inst_valid_per_frame[i] = True
                elif state == 0:
                    inst_per_frame[i] = 0
                    inst_valid_per_frame[i] = True
                # state==2/3 (Descent/Ground) 保持 inst_valid=False（mask）
            # has_inst=False：全部 mask（不强行 negative）

        # 滑动窗口
        n = len(feats)
        rec_cap = recording_max_windows if recording_max_windows else (n - window_size) // stride + 1
        win_count = 0
        for start in range(0, n - window_size + 1, stride):
            if win_count >= rec_cap:
                break
            end = start + window_size
            # 窗口 process 标签 = 末尾帧（causal）
            win_process = int(process_per_frame[end - 1])
            # inst：窗口末尾帧的 inst 状态（causal 语义）
            win_inst = int(inst_per_frame[end - 1])
            win_inst_valid = bool(inst_valid_per_frame[end - 1])
            all_feats.append(feats[start:end])
            all_process.append(win_process)
            all_inst.append(win_inst)
            all_inst_valid.append(win_inst_valid)
            all_splits.append(split)
            all_subjects.append(subject)
            all_files.append(fname)
            all_actions.append(FALL_ACTION_DIR)
            win_count += 1

    if not all_feats:
        raise ValueError("no windows produced")

    feats_arr = np.stack(all_feats)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=feats_arr,
        process_labels=np.asarray(all_process),
        inst_labels=np.asarray(all_inst),
        inst_valid=np.asarray(all_inst_valid),
        splits=np.asarray(all_splits),
        subjects=np.asarray(all_subjects),
        source_files=np.asarray(all_files),
        actions=np.asarray(all_actions),
        window_size=window_size,
        feature_names=np.asarray(
            __import__("radar_module.preprocess.baseline_relative_features_v2",
                       fromlist=["FEATURE_NAMES"]).FEATURE_NAMES
        ),
        schema_version="dguha_hierarchical_v1",
    )
    return {
        "output": str(output_path),
        "n_windows": int(feats_arr.shape[0]),
        "window_size": int(feats_arr.shape[1]),
        "n_features": int(feats_arr.shape[2]),
        "process_distribution": {
            "NormalDynamic": int(np.sum(np.asarray(all_process) == PROCESS_NORMAL)),
            "FallProcess": int(np.sum(np.asarray(all_process) == PROCESS_FALL)),
        },
        "inst_pos_windows": int(np.sum(
            (np.asarray(all_inst) == 1) & np.asarray(all_inst_valid)
        )),
        "inst_neg_windows": int(np.sum(
            (np.asarray(all_inst) == 0) & np.asarray(all_inst_valid)
        )),
        "inst_valid_windows": int(np.sum(np.asarray(all_inst_valid))),
        "fold_counts": {
            s: int(np.sum(np.asarray(all_splits) == s))
            for s in sorted(set(all_splits))
        },
        "n_skipped": skipped,
    }


def _build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build hierarchical TCN dataset (NormalDynamic/FallProcess + Instability)."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-normal-windows", type=int, default=40)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split-filter", type=str, default=None)
    parser.add_argument("--split-path", type=Path, default=None,
                        help="frozen subject split json (dguha_subject_split_v1)")
    parser.add_argument("--recording-max-windows", type=int, default=None,
                        help="cap windows per recording for class balance")
    parser.add_argument("--action-filter", type=str, default=None,
                        help="only process this action dir (e.g. 5_falling_forward)")
    return parser


if __name__ == "__main__":
    import sys

    args = _build_parser().parse_args()
    summary = build_hierarchical_dataset_npz(
        args.data_root,
        args.output,
        window_size=args.window_size,
        stride=args.stride,
        max_normal_windows_per_recording=args.max_normal_windows,
        max_samples_per_action=args.max_samples,
        split_filter=args.split_filter,
        split_path=args.split_path,
        recording_max_windows=args.recording_max_windows,
        action_filter=args.action_filter,
    )
    print(json.dumps(summary, indent=2, default=str))
    sys.exit(0)
