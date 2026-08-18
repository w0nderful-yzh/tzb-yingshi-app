"""Tests for baseline-relative radar features v2."""

from __future__ import annotations

import numpy as np
import pytest

from radar_module.preprocess.baseline_relative_features_v2 import (
    FEATURE_NAMES,
    SCALE_KEYS,
    DIFF_KEYS,
    baseline_from_frames,
    extract_sequence_features,
)


def _make_frames(n: int, z: float = 1.0, x_spread: float = 0.3):
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n):
        pts = [
            {"x": float(rng.normal(0, x_spread)), "y": 0.5,
             "z": float(z + rng.normal(0, 0.02)), "velocity": float(rng.normal(0, 0.05))}
            for _ in range(8)
        ]
        frames.append({"points": pts, "timestamp": f"2026-08-18T00:00:{i:02d}00+00:00"})
    return frames


def test_extract_sequence_features_shape() -> None:
    frames = _make_frames(40)
    feats, names = extract_sequence_features(frames)
    assert feats.shape == (40, len(FEATURE_NAMES))
    assert names == FEATURE_NAMES
    assert np.isfinite(feats).any()


def test_baseline_from_frames() -> None:
    frames = _make_frames(20, z=1.0)
    base = baseline_from_frames(frames)
    assert "spatial_spread" in base
    assert "doppler_std" in base


def test_relative_excludes_absolute_height() -> None:
    """绝对 height_range / x_range 不应出现在特征名中。"""
    assert "height_range" not in FEATURE_NAMES
    assert "x_range" not in FEATURE_NAMES
    assert "z_p90" not in FEATURE_NAMES  # 只有 delta_z_p90


def test_group_keys_consistent() -> None:
    for k in DIFF_KEYS | SCALE_KEYS:
        assert k in FEATURE_NAMES


def test_extract_handles_empty_points() -> None:
    frames = [{"points": [], "timestamp": "2026-08-18T00:00:00+00:00"}] * 5
    feats, _ = extract_sequence_features(frames)
    assert feats.shape[1] == len(FEATURE_NAMES)
