"""Tests for repeat-level combination feature evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from radar_module.analysis.prefall_combination_eval_v1 import (
    HAS_LIGHTGBM,
    HAS_SKLEARN,
    build_feature_matrix,
    evaluate_pair,
    load_repeat_stages,
    zscore_combo_scores,
    zscore_combo_scores_loro,
    _safe_auroc,
    _safe_pr_auc,
)


def _make_stage_rows(seed: int = 0) -> list[dict]:
    """构造合成 repeat stage 数据：fall 的 early height_range/x_range 显著高。"""
    rng = np.random.default_rng(seed)
    actions = {
        "controlled_forward_fall": {
            "height_range": 0.8, "x_range": 1.5, "drift": 0.5,
        },
        "fast_sitting": {
            "height_range": 0.2, "x_range": 0.4, "drift": 0.3,
        },
        "forward_instability_recovery": {
            "height_range": 0.35, "x_range": 0.8, "drift": 0.25,
        },
        "standing": {
            "height_range": 0.1, "x_range": 0.1, "drift": 0.2,
        },
    }
    rows = []
    for action, ctr in actions.items():
        for i in range(1, 6):
            noise = rng.normal(0, 0.05, 3)
            rows.append({
                "repeat_id": f"{action}_r{i:02d}",
                "action_name": action,
                "stages": {
                    "early": {
                        "height_range": ctr["height_range"] + noise[0],
                        "x_range": ctr["x_range"] + noise[1],
                        "drift_xy_1p0s": ctr["drift"] + noise[2],
                    },
                    "middle": {
                        "drift_xy_1p0s": ctr["drift"] + noise[2] * 0.5,
                    },
                    "late": {
                        "drift_xy_1p0s": ctr["drift"] + noise[2] * 0.3,
                    },
                },
            })
    return rows


def test_load_repeat_stages(tmp_path: Path) -> None:
    path = tmp_path / "stages.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in _make_stage_rows()),
        encoding="utf-8",
    )
    by_action = load_repeat_stages(path)
    assert set(by_action) == {
        "controlled_forward_fall", "fast_sitting",
        "forward_instability_recovery", "standing",
    }
    assert len(by_action["controlled_forward_fall"]) == 5


def test_build_feature_matrix_shapes() -> None:
    rows = _make_stage_rows()
    by_action = {r["action_name"]: [r] for r in rows}
    by_action = {}
    for r in rows:
        by_action.setdefault(r["action_name"], []).append(r)
    X, y, rids = build_feature_matrix(
        by_action, "controlled_forward_fall", "fast_sitting"
    )
    assert X.shape == (10, 3)
    assert y.shape == (10,)
    assert y.sum() == 5
    assert len(rids) == 10
    # 组合 B 特征 = [height_range, x_range, drift]
    assert np.isfinite(X).all()


def test_evaluate_pair_separates_fall(tmp_path: Path) -> None:
    rows = _make_stage_rows()
    by_action = {}
    for r in rows:
        by_action.setdefault(r["action_name"], []).append(r)
    result = evaluate_pair(by_action, "controlled_forward_fall", "fast_sitting")
    assert "error" not in result
    assert result["n_pos"] == 5 and result["n_neg"] == 5
    hr = result["models"]["single_height_range"]
    # 合成数据 fall 的 height_range 显著高，AUROC 应 > 0.8
    assert hr["auroc"] > 0.8
    # 新组合键
    assert "HR" in result["models"]
    assert "HR_drift" in result["models"]
    assert "HR_xrange" in result["models"]
    # 无泄漏 z-score 子模型
    assert "zscore_loro" in result["models"]["HR_drift"]


def test_safe_auroc_perfect() -> None:
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert _safe_auroc(y, scores) == pytest.approx(1.0)


def test_safe_auroc_reverse() -> None:
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    assert _safe_auroc(y, scores) == pytest.approx(0.0)


def test_safe_pr_auc_perfect() -> None:
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    pr = _safe_pr_auc(y, scores)
    assert pr == pytest.approx(1.0)


def test_sklearn_lightgbm_availability() -> None:
    # 只是记录环境状态，不强制
    assert isinstance(HAS_SKLEARN, bool)
    assert isinstance(HAS_LIGHTGBM, bool)


def test_loro_zscore_no_leakage_direction() -> None:
    """LORO z-score 每折只用训练集确定方向，且 fall 侧高分。

    构造 fall 的 height_range 高于 neg 的数据。LORO 分数应 fall 高于 neg。
    """
    rng = np.random.default_rng(0)
    hr = np.concatenate([
        rng.normal(0.8, 0.05, 5),  # fall
        rng.normal(0.2, 0.05, 5),  # neg
    ])
    xr = np.concatenate([
        rng.normal(1.0, 0.1, 5),
        rng.normal(0.5, 0.1, 5),
    ])
    X = np.column_stack([hr, xr])
    y = np.array([1] * 5 + [0] * 5)
    oof = zscore_combo_scores_loro(X, y, [0, 1])
    # fall 侧应整体高于 neg 侧
    assert np.mean(oof[y == 1]) > np.mean(oof[y == 0])
    # 无泄漏：OOF 分数不应等于全数据 z-score（统计量不同）
    full = zscore_combo_scores(X, [0, 1])
    assert not np.allclose(oof, full)


def test_loro_zscore_handles_reversed_feature() -> None:
    """若某个特征在 neg 侧均值更高，LORO 方向应自动反转。

    构造 drift 特征 neg 高于 fall（反直觉），组合仍应给 fall 高分。
    """
    rng = np.random.default_rng(1)
    hr = np.concatenate([rng.normal(0.8, 0.05, 5), rng.normal(0.2, 0.05, 5)])
    drift = np.concatenate([rng.normal(0.1, 0.02, 5), rng.normal(0.9, 0.05, 5)])
    X = np.column_stack([hr, drift])
    y = np.array([1] * 5 + [0] * 5)
    oof = zscore_combo_scores_loro(X, y, [0, 1])
    assert np.mean(oof[y == 1]) > np.mean(oof[y == 0])
