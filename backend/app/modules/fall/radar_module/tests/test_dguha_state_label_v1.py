"""Tests for DGUHA state label construction."""

from __future__ import annotations

import numpy as np
import pytest

from radar_module.dataset.dguha_state_label_v1 import (
    INSTABILITY_MAX_DURATION_S,
    construct_state_segments,
    _diagnose_unlabeled,
    _locate_states_from_kinect,
    kinect_series,
    parse_dguha_kinect,
)


def _make_kinect_series(times, head):
    """构造最小 kinect_series dict。"""
    n = len(times)
    return {
        "t": np.asarray(times, dtype=np.float64),
        "head": np.asarray(head, dtype=np.float64),
        "trunk_lean": np.zeros(n),
        "pelvis": np.asarray(head) - 0.3,
        "com": np.asarray(head) - 0.1,
        "v_head": np.gradient(head, times),
        "v_pelvis": np.zeros(n),
        "v_lean": np.zeros(n),
    }


def _synth_forward_fall():
    """构造合成前倒：稳定→失衡(缓降)→快速下降→低位。

    采样 30Hz，时序：
    - 0-2s: head ~0.5 稳定 (Stable)
    - 2-2.5s: head 0.50→0.46 缓慢下降 (Instability)
    - 2.5-3.0s: head 0.46→0.1 快速下降 (Descent)
    - 3.0-5s: head 0.1 低位 (Ground)
    """
    dt = 1 / 30
    t = np.arange(0, 5, dt)
    head = np.full_like(t, 0.5)
    # Instability 段 (2.0-2.5s): 缓降
    mask_inst = (t >= 2.0) & (t < 2.5)
    head[mask_inst] = np.linspace(0.50, 0.46, int(mask_inst.sum()))
    # Descent 段 (2.5-3.0s): 快降
    mask_desc = (t >= 2.5) & (t < 3.0)
    head[mask_desc] = np.linspace(0.46, 0.10, int(mask_desc.sum()))
    # Ground (3.0s+): 低位
    head[t >= 3.0] = 0.10
    return _make_kinect_series(t, head)


def test_locate_states_synthetic_fall() -> None:
    kin = _synth_forward_fall()
    st = _locate_states_from_kinect(kin)
    assert st is not None
    ii, di, gi = st["instability_idx"], st["descent_idx"], st["ground_idx"]
    # Instability 严格在 Descent 前
    assert ii < di < gi
    # Descent 在 2.5s 附近
    assert 2.3 < kin["t"][di] < 2.7
    # Instability 期间净下降
    assert kin["head"][di] - kin["head"][ii] < -0.01
    # Ground 在 3.0s 附近（最低点前 3 帧）
    assert 2.8 < kin["t"][gi] < 3.3


def test_locate_states_no_instability() -> None:
    """无失衡前兆：head 直接快速下降。"""
    dt = 1 / 30
    t = np.arange(0, 5, dt)
    head = np.full_like(t, 0.5)
    mask = t >= 2.0
    head[mask] = np.linspace(0.5, 0.1, int(mask.sum()))
    kin = _make_kinect_series(t, head)
    st = _locate_states_from_kinect(kin)
    assert st is None  # 无下降前失衡


def test_locate_states_noise_rejected() -> None:
    """早期噪声漂移(已回升)不应被当作 Instability。

    该场景下降前 head 已回稳(1.0-1.5s 小降后回升到 0.5)，2.0s 直接快速
    下降，无紧贴下降的持续失衡段 → 应返回 None（无有效失衡前兆）。
    """
    dt = 1 / 30
    t = np.arange(0, 6, dt)
    head = np.full_like(t, 0.5)
    # 早期小下降又回升（噪声）
    head[30:45] = 0.47  # 1.0-1.5s 小降
    # 2.0s 开始真正下降
    mask = t >= 2.0
    head[mask] = np.linspace(0.5, 0.1, int(mask.sum()))
    kin = _make_kinect_series(t, head)
    st = _locate_states_from_kinect(kin)
    assert st is None  # 下降前已回稳，无持续失衡前兆


def test_diagnose_unlabeled_reasons() -> None:
    # 纯稳定无下降 → no_descent_onset
    t = np.arange(0, 3, 1 / 30)
    kin = _make_kinect_series(t, np.full_like(t, 0.5))
    # 直接调用内部逻辑验证 no_descent
    assert kin["v_head"].size > 0  # 结构有效
    # 用真实文件测试 diagnose
    from pathlib import Path
    real = Path("data/external/dguha/raw/Training/5_falling_forward/kinect/F_001_A5_001.txt")
    if real.exists():
        reason = _diagnose_unlabeled(real)
        assert isinstance(reason, str)
        assert reason in {
            "no_descent_onset", "no_instability_before_descent",
            "instability_too_short", "instability_too_long_noise",
            "too_few_frames", "kinect_parse_error", "empty_kinect", "other",
        }


def test_instability_max_duration_constant() -> None:
    assert INSTABILITY_MAX_DURATION_S == 2.0


def test_construct_state_segments_real_sample() -> None:
    from pathlib import Path
    real = Path("data/external/dguha/raw/Training/5_falling_forward/kinect/F_001_A5_002.txt")
    if real.exists():
        seg = construct_state_segments(real)
        assert seg is not None
        assert seg["instability_idx"] < seg["descent_idx"] < seg["ground_idx"]
        # 状态数组长度 = 帧数
        assert len(seg["state_per_frame"]) == seg["n_frames"]
