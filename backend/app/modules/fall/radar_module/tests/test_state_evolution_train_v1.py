"""Tests for hierarchical training pipeline helpers."""

from __future__ import annotations

import numpy as np
import pytest

from radar_module.analysis.state_evolution_train_v1 import (
    _f1,
    load_dataset,
)
from radar_module.dataset.dguha_subject_split_v1 import compute_groupkfold_split


def _make_dataset(tmp_path):
    import json
    from pathlib import Path

    # 构造小型 npz：4 subject，2 fold
    rng = np.random.default_rng(0)
    n = 40
    feats = rng.standard_normal((n, 20, 5))
    process = rng.integers(0, 2, n)
    inst = rng.integers(0, 2, n)
    inst_valid = rng.choice([True, False], n)
    subjects = np.array([f"S{i % 4}" for i in range(n)])
    folds = np.array([str(i % 2) for i in range(n)])
    files = np.array([f"{subj}_r{i}" for i, subj in enumerate(subjects)])
    actions = np.array(["5_falling_forward"] * n)
    path = tmp_path / "test.npz"
    np.savez_compressed(
        path,
        features=feats, process_labels=process, inst_labels=inst,
        inst_valid=inst_valid, splits=folds, subjects=subjects,
        source_files=files, actions=actions, window_size=20,
        feature_names=np.array([f"f{i}" for i in range(5)]),
        schema_version="dguha_hierarchical_v1",
    )
    return path


def test_load_dataset_nan_to_zero(tmp_path) -> None:
    path = _make_dataset(tmp_path)
    # 注入 NaN
    d = np.load(path)
    feats = d["features"]
    feats[0, 0, 0] = np.nan
    np.savez_compressed(
        path, features=feats, process_labels=d["process_labels"],
        inst_labels=d["inst_labels"], inst_valid=d["inst_valid"],
        splits=d["splits"], subjects=d["subjects"],
        source_files=d["source_files"], actions=d["actions"],
        window_size=20, feature_names=d["feature_names"],
        schema_version="dguha_hierarchical_v1",
    )
    data = load_dataset(path)
    assert np.isfinite(data["features"]).all()


def test_f1() -> None:
    y_true = np.array([0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1])
    f1, rec, prec = _f1(y_true, y_pred, pos=1)
    assert rec == pytest.approx(2 / 3)
    assert prec == pytest.approx(2 / 3)


def test_groupkfold_split_every_fold_has_fall(tmp_path) -> None:
    # 用真实数据根验证（若存在）
    from pathlib import Path

    root = Path("data/external/dguha/raw/Training")
    if root.exists() and (root / "5_falling_forward" / "kinect").exists():
        split = compute_groupkfold_split(root, n_folds=5, seed=42)
        assert all(split["fold_has_fall"][str(f)] for f in range(5))
        # subject 不跨 fold
        stf = split["subject_to_fold"]
        assert len(set(stf.values())) <= 5
