"""DGUHA subject-isolated split（冻结，用于 hierarchical TCN 训练）。

问题
----
原 DGUHA 的 test subjects（F_006/M_013/M_014）没有 falling_forward，
无法评估 fall。因此不沿用原 test split。

方案
----
用 **GroupKFold（subject 为组）**，保证每个 fold 的 val 都有
falling_forward。fold 0 固定为 held-out（最终评估，不参与调参），
fold 1-4 循环作为 train/val（内部调参）。

严格约束：
- subject 不跨集合
- recording 不跨集合（recording 属于唯一 subject）
- overlapping windows 不跨集合（窗口从同一 recording 生成，随 recording）
- normalization 只用 train
- threshold 只用 validation
- held-out 不参与调参

冻结：生成 subject→fold 映射并保存，多次运行一致。

Version: radar_dguha_subject_split_v1
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from radar_module.dataset.dguha_research_v2 import DGUHA_SPLIT_BY_SUBJECT

FALL_DIR = "5_falling_forward"
N_FOLDS = 5


def _subjects_with_fall(data_root: Path) -> set[str]:
    kinect_dir = data_root / FALL_DIR / "kinect"
    subjects = set()
    if kinect_dir.exists():
        for f in kinect_dir.glob("*.txt"):
            subjects.add("_".join(f.name.split("_")[:2]))
    return subjects


def compute_groupkfold_split(
    data_root: Path,
    *,
    n_folds: int = N_FOLDS,
    seed: int = 42,
) -> dict[str, Any]:
    """按 subject 计算 GroupKFold 分配。返回 subject→fold。"""
    import numpy as np
    from sklearn.model_selection import GroupKFold

    all_subjects = sorted(DGUHA_SPLIT_BY_SUBJECT.keys())
    fall_subjects = sorted(_subjects_with_fall(data_root))
    fall_subjects_set = set(fall_subjects)
    non_fall_subjects = sorted(set(all_subjects) - fall_subjects_set)

    # 每个 subject 生成一个"组"用于 GroupKFold
    # 先对有 fall 的 subject 分配 fold，确保每个 fold 有 fall
    rng = np.random.default_rng(seed)
    subject_to_fold: dict[str, int] = {}

    # 把 fall_subjects 用 GroupKFold 分到 n_folds（保证每折有 fall）
    groups = np.asarray([f"{s}-g" for s in fall_subjects])
    X_dummy = np.zeros(len(fall_subjects))
    gkf = GroupKFold(n_splits=n_folds)
    for fold_idx, (_, val_idx) in enumerate(gkf.split(X_dummy, groups=groups)):
        for i in val_idx:
            subject_to_fold[fall_subjects[i]] = fold_idx

    # 非 fall subject 补充到 fold（按轮转，保证分布）
    for i, s in enumerate(non_fall_subjects):
        subject_to_fold[s] = i % n_folds

    # fold 0 = held-out, fold 1-4 = train/val 循环
    return {
        "schema_version": "dguha_subject_split_v1",
        "n_folds": n_folds,
        "seed": seed,
        "subject_to_fold": subject_to_fold,
        "fold_subjects": {
            str(f): [s for s, fo in subject_to_fold.items() if fo == f]
            for f in range(n_folds)
        },
        "fold_has_fall": {
            str(f): any(s in fall_subjects_set for s, fo in subject_to_fold.items() if fo == f)
            for f in range(n_folds)
        },
        "note": (
            "fold0=held-out(最终评估,不调参), fold1-4循环作train/val。"
            "每个fold都有falling_forward。"
        ),
    }


def save_split(split: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fold_to_role(fold: int) -> str:
    """fold 0 = held-out；其余 fold 在训练中循环，此函数仅标记。"""
    if fold == 0:
        return "held_out"
    return "cv"


def _build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute frozen DGUHA subject split for hierarchical TCN."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    import sys

    args = _build_parser().parse_args()
    split = compute_groupkfold_split(
        args.data_root, n_folds=args.n_folds, seed=args.seed
    )
    save_split(split, args.output)
    print(json.dumps({
        "n_folds": split["n_folds"],
        "fold_subjects": {k: len(v) for k, v in split["fold_subjects"].items()},
        "fold_has_fall": split["fold_has_fall"],
    }, indent=2))
    print(f"saved to {args.output}")
    sys.exit(0)
