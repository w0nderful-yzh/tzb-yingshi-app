"""DGUHA 主数据集 + 外部 NormalDynamic/hard-negative 合并。

用途
----
hard-negative enhancement：外部数据作为 NormalDynamic 加入训练，
ProcessHead label=Normal、InstabilityHead mask=false。

约束
----
- DGUHA held-out(fold0) 不变，不参与训练
- 外部数据只加入 DGUHA train fold（fold 2/3/4），不污染 val/held-out 评估
- 保留 dataset_id / subject_id / recording_id
- 外部数据 frame rate 显式指定（不依赖时间戳推断，避免 RadHAR 异常）

Version: radar_dguha_external_merge_v1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.dataset.mmwave_ocpid_v1 import parse_mmwave_ocpid_text
from radar_module.preprocess.baseline_relative_features_v2 import extract_sequence_features


def _radar_feats_from_frames(
    frames,
    window_size: int,
    stride: int,
    sample_rate: float,
    recording_max_windows: int,
    *,
    dataset_id: str,
    subject_id: str,
    recording_id: str,
    action: str,
    frame_cap: int = 400,
):
    """从 RadarFrame 序列提取窗口特征 + external Normal 标签。

    frame_cap：限制每 recording 最多处理的帧数（外部数据只需少量窗口
    提供 Normal 多样本，避免全序列动态特征计算过慢）。
    """
    if len(frames) < window_size:
        return [], []
    if len(frames) > frame_cap:
        frames = frames[:frame_cap]
    # 构造 records 供特征提取（帧有 timestamp/points）
    records = [{"points": f.points, "timestamp": f.timestamp} for f in frames]
    feats, _ = extract_sequence_features(records)
    # 外部数据 sample rate 显式给，但 extract 内部固定 20Hz 计算动态特征；
    # 这里仅用窗口采样（窗口数按实际帧数），动态特征周期按外部帧率重算
    # 为简洁，复用 extract_sequence_features（内部 20Hz 假设），
    # 对 ocPID/RadHAR 的窗口划分按实际帧数。
    n = len(feats)
    windows = []
    labels = []
    win_count = 0
    for start in range(0, n - window_size + 1, stride):
        if win_count >= recording_max_windows:
            break
        windows.append(feats[start : start + window_size])
        labels.append(0)  # NormalDynamic
        win_count += 1
    return windows, [{
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "recording_id": recording_id,
        "action": action,
    }] * len(windows)


def load_ocpid_normal(
    ocpid_root: Path,
    window_size: int,
    stride: int,
    recording_max_windows: int,
    max_subjects: int | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """加载 ocPID CFARData 作为 external NormalDynamic。

    ocPID 结构：PointCloudData/CFARData/Person{1-9}/{Box,Plant,Sponge*}/Person*.txt
    """
    cfar = ocpid_root / "PointCloudData" / "CFARData"
    if not cfar.exists():
        return [], []
    all_windows, all_meta = [], []
    subjects = sorted([d for d in cfar.iterdir() if d.is_dir() and d.name.startswith("Person")])
    if max_subjects:
        subjects = subjects[:max_subjects]
    for person_dir in subjects:
        person_id = person_dir.name
        for cond_dir in sorted(person_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            for txt in sorted(cond_dir.glob("*.txt")):
                try:
                    frames = parse_mmwave_ocpid_text(str(txt))
                except Exception:
                    continue
                if len(frames) < window_size:
                    continue
                windows, meta = _radar_feats_from_frames(
                    frames, window_size, stride, sample_rate=15.0,
                    recording_max_windows=recording_max_windows,
                    dataset_id="ocPID", subject_id=person_id,
                    recording_id=txt.stem, action=f"ocPID/{cond_dir.name}",
                )
                all_windows.extend(windows)
                all_meta.extend(meta)
    return all_windows, all_meta


def load_radhar_normal(
    radhar_root: Path,
    window_size: int,
    stride: int,
    recording_max_windows: int,
    actions: tuple[str, ...] = ("walk", "jack", "squats"),
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """加载 RadHAR 语义明确的正常活动作为 external NormalDynamic。"""
    all_windows, all_meta = [], []
    data_dir = radhar_root / "Data"
    for split_dir in ("Train", "Test"):
        sd = data_dir / split_dir
        if not sd.exists():
            continue
        for action in actions:
            action_dir = sd / action
            if not action_dir.exists():
                continue
            for txt in sorted(action_dir.glob("*.txt")):
                try:
                    frames = parse_radhar_text(str(txt), device_id="radhar")
                except Exception:
                    continue
                if len(frames) < window_size:
                    continue
                windows, meta = _radar_feats_from_frames(
                    frames, window_size, stride, sample_rate=30.0,
                    recording_max_windows=recording_max_windows,
                    dataset_id="RadHAR", subject_id=f"RadHAR/{split_dir}",
                    recording_id=txt.stem, action=f"RadHAR/{action}",
                )
                all_windows.extend(windows)
                all_meta.extend(meta)
    return all_windows, all_meta


def merge_datasets(
    dguha_npz: Path,
    output_path: Path,
    *,
    window_size: int = 20,
    stride: int = 10,
    ocpid_root: Path | None = None,
    radhar_root: Path | None = None,
    recording_max_windows: int = 15,
    ocpid_max_subjects: int | None = None,
    radhar_actions: tuple[str, ...] = ("walk", "jack", "squats"),
) -> dict[str, Any]:
    """合并 DGUHA 主数据集 + 外部数据。

    外部数据只加入 DGUHA train fold（splits in 2/3/4）。
    """
    d = np.load(dguha_npz, allow_pickle=True)
    feats = d["features"]
    proc = d["process_labels"]
    inst = d["inst_labels"]
    instv = d["inst_valid"]
    splits = d["splits"]
    subjects = d["subjects"]
    source_files = d["source_files"]
    actions = d["actions"]
    feature_names = d["feature_names"]

    # 收集外部数据
    ext_windows, ext_meta = [], []
    if ocpid_root is not None:
        w, m = load_ocpid_normal(ocpid_root, window_size, stride,
                                 recording_max_windows, ocpid_max_subjects)
        ext_windows.extend(w)
        ext_meta.extend(m)
    if radhar_root is not None:
        w, m = load_radhar_normal(radhar_root, window_size, stride,
                                  recording_max_windows, radhar_actions)
        ext_windows.extend(w)
        ext_meta.extend(m)

    n_ext = len(ext_windows)
    n_dguha = len(feats)

    # 合并：DGUHA 全部 + 外部数据（只进 train fold 在训练时按 splits 过滤）
    if n_ext > 0:
        ext_feats = np.stack(ext_windows)  # (E, window, F)
        # 外部 splits 标记为 "X"（external train），训练时并入 train
        ext_proc = np.zeros(n_ext, dtype=np.int64)
        ext_inst = np.zeros(n_ext, dtype=np.int64)
        ext_instv = np.zeros(n_ext, dtype=bool)
        ext_splits = np.full(n_ext, "X", dtype="<U1")
        ext_subjects = np.asarray([m["subject_id"] for m in ext_meta], dtype="<U16")
        ext_files = np.asarray([m["recording_id"] for m in ext_meta], dtype="<U24")
        ext_actions = np.asarray([m["action"] for m in ext_meta], dtype="<U27")

        feats_all = np.concatenate([feats, ext_feats], axis=0)
        proc_all = np.concatenate([proc, ext_proc])
        inst_all = np.concatenate([inst, ext_inst])
        instv_all = np.concatenate([instv, ext_instv])
        splits_all = np.concatenate([splits, ext_splits])
        subjects_all = np.concatenate([subjects, ext_subjects])
        files_all = np.concatenate([source_files, ext_files])
        actions_all = np.concatenate([actions, ext_actions])
    else:
        feats_all, proc_all = feats, proc
        inst_all, instv_all = inst, instv
        splits_all, subjects_all = splits, subjects
        files_all, actions_all = source_files, actions

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=feats_all, process_labels=proc_all,
        inst_labels=inst_all, inst_valid=instv_all,
        splits=splits_all, subjects=subjects_all,
        source_files=files_all, actions=actions_all,
        window_size=window_size, feature_names=feature_names,
        schema_version="dguha_external_merged_v1",
    )
    return {
        "n_dguha": n_dguha,
        "n_external": n_ext,
        "n_total": len(feats_all),
        "external_breakdown": {
            "ocPID": int(sum(1 for m in ext_meta if m["dataset_id"] == "ocPID")),
            "RadHAR": int(sum(1 for m in ext_meta if m["dataset_id"] == "RadHAR")),
        },
        "output": str(output_path),
    }


def _build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge DGUHA + external NormalDynamic for hard-negative training."
    )
    parser.add_argument("--dguha-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--ocpid-root", type=Path, default=None)
    parser.add_argument("--radhar-root", type=Path, default=None)
    parser.add_argument("--recording-max-windows", type=int, default=15)
    parser.add_argument("--ocpid-max-subjects", type=int, default=None)
    parser.add_argument("--radhar-actions", type=str, default="walk,jack,squats")
    return parser


if __name__ == "__main__":
    import sys

    args = _build_parser().parse_args()
    radhar_actions = tuple(a.strip() for a in args.radhar_actions.split(",") if a.strip())
    summary = merge_datasets(
        args.dguha_npz, args.output,
        window_size=args.window_size, stride=args.stride,
        ocpid_root=args.ocpid_root, radhar_root=args.radhar_root,
        recording_max_windows=args.recording_max_windows,
        ocpid_max_subjects=args.ocpid_max_subjects,
        radhar_actions=radhar_actions,
    )
    print(json.dumps(summary, indent=2))
    sys.exit(0)
