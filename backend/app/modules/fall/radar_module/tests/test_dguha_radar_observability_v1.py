"""Tests for DGUHA radar observability audit."""

from __future__ import annotations

import numpy as np
import pytest

from radar_module.analysis.dguha_radar_observability_v1 import (
    _auroc,
    _cohen_d,
    _pr_auc,
    _state_for_epoch,
)


def test_state_for_epoch_boundaries() -> None:
    boundaries = {
        "Stable": 0.0,
        "Instability": 2.0,
        "Descent": 2.5,
        "Ground": 3.0,
        "End": 5.0,
    }
    assert _state_for_epoch(1.0, boundaries) == "Stable"
    assert _state_for_epoch(2.2, boundaries) == "Instability"
    assert _state_for_epoch(2.8, boundaries) == "Descent"
    assert _state_for_epoch(4.0, boundaries) == "Ground"
    assert _state_for_epoch(6.0, boundaries) is None


def test_auroc_perfect() -> None:
    a = np.array([0.9, 0.8, 0.7])
    b = np.array([0.2, 0.1, 0.0])
    assert _auroc(a, b) == pytest.approx(1.0)


def test_auroc_random() -> None:
    a = np.array([0.5, 0.5, 0.5])
    b = np.array([0.5, 0.5, 0.5])
    assert _auroc(a, b) == pytest.approx(0.5)


def test_cohen_d_sign() -> None:
    a = np.array([1.0, 1.1, 1.2])
    b = np.array([0.1, 0.2, 0.3])
    assert _cohen_d(a, b) > 0
    assert _cohen_d(b, a) < 0


def test_pr_auc_perfect() -> None:
    a = np.array([0.9, 0.8, 0.7])
    b = np.array([0.2, 0.1, 0.0])
    assert _pr_auc(a, b) > 0.9


def test_auroc_nan_handling() -> None:
    a = np.array([0.9, np.nan, 0.7])
    b = np.array([0.2, 0.1, 0.0])
    assert np.isfinite(_auroc(a, b))
