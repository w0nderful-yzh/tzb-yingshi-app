"""Zero-variance-safe normalization for live radar inference.

Root cause
----------
Domain-calibrated normalization files store per-feature mean/std. Features that
have zero variance in the calibration domain (e.g. the three quality masks
``observed_frame_mask`` / ``point_present_mask`` / ``interpolated_mask`` in
IWR6843-Fall102, which are constant 1/1/0) were clamped to ``std = 1e-9`` so
they would pass validation. At inference time, a single empty frame flips
``point_present_mask`` to 0, producing ``(0 - 1) / 1e-9 = -1e9``, which makes
the model logit collapse and sigmoid underflow to 0.

Correct semantics
-----------------
A feature that had zero variance during training/calibration contributes no
information: after z-scoring it is constant 0 (value == mean always). The
safe behavior is therefore to force normalized value 0 for zero-variance
features, and keep the raw z-score for all other features.

This module does not modify checkpoints, model weights, or thresholds. It only
fixes the runtime normalization contract so that zero-variance features cannot
produce extreme values from real-sensor outliers.

Version: radar_safe_normalize_v1
"""

from __future__ import annotations

import numpy as np


def zero_variance_feature_mask(std: np.ndarray, *, epsilon: float = 1e-9) -> np.ndarray:
    """Return a boolean mask of features whose std is (near) zero.

    ``epsilon`` is only used to decide "effectively zero variance"; it is NOT
    used as a divisor. Callers should keep the raw (possibly 0) std so the mask
    is meaningful.
    """
    std = np.asarray(std, dtype=np.float64)
    return np.abs(std) < epsilon


def safe_normalize(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    zero_variance_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Z-score ``values`` with ``mean``/``std``, forcing zero-variance features to 0.

    ``values`` has shape (..., F). ``mean``/``std`` have shape (F,). Features
    marked in ``zero_variance_mask`` are set to 0 in the output instead of
    dividing by their tiny std.

    If ``zero_variance_mask`` is None it is derived from ``std`` with the
    default epsilon; callers that clamped std should pass the pre-clamp mask.
    """
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != std.shape:
        raise ValueError("mean/std shape mismatch")
    if values.shape[-1] != mean.shape[0]:
        raise ValueError("values feature dim mismatch")

    if zero_variance_mask is None:
        zero_variance_mask = zero_variance_feature_mask(std)
    zero_variance_mask = np.asarray(zero_variance_mask, dtype=bool)
    if zero_variance_mask.shape != mean.shape:
        raise ValueError("zero_variance_mask shape mismatch")

    safe_std = np.where(zero_variance_mask, 1.0, std)
    normalized = ((values - mean[None, :]) / safe_std[None, :]).astype(np.float32)
    normalized[..., zero_variance_mask] = 0.0
    return normalized
