"""Tests for state-evolution causal TCN model + training helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from radar_module.model.state_evolution_tcn_v1 import (
    N_PROCESS,
    N_STATES,
    HierarchicalStateTCNV1,
    StateEvolutionTCNV1,
    default_class_weights,
    default_process_weights,
    hierarchical_loss,
    masked_state_loss,
)


def test_model_forward_shape() -> None:
    model = StateEvolutionTCNV1(n_features=21, hidden_dim=16, n_layers=2)
    x = torch.randn(4, 20, 21)
    logits = model(x)
    assert logits.shape == (4, N_STATES)
    assert torch.isfinite(logits).all()


def test_model_forward_small_window() -> None:
    """窗口小于感受野也应能输出（padding 处理）。"""
    model = StateEvolutionTCNV1(n_features=8, hidden_dim=8, n_layers=2)
    x = torch.randn(2, 5, 8)  # 5 帧
    logits = model(x)
    assert logits.shape == (2, N_STATES)


def test_masked_loss_reduces_with_class_weights() -> None:
    logits = torch.randn(8, N_STATES)
    labels = torch.tensor([2, 2, 2, 2, 0, 3, 3, 1], dtype=torch.long)
    inst_mask = torch.tensor([True] * 8)
    weights = torch.ones(N_STATES)
    loss = masked_state_loss(logits, labels, inst_mask, weights)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()
    assert loss > 0


def test_masked_loss_ignores_inst_when_no_mask() -> None:
    """无 Instability 前兆样本：预测为 Instability 不惩罚。"""
    logits = torch.zeros(1, N_STATES)
    logits[0, 1] = 5.0  # 预测 Instability
    labels = torch.tensor([1], dtype=torch.long)  # 真实 Instability
    inst_mask = torch.tensor([False])  # 无前兆 → 不监督 Instability
    weights = torch.ones(N_STATES)
    loss = masked_state_loss(logits, labels, inst_mask, weights)
    assert loss.item() == 0.0  # 该样本被 mask 掉


def test_default_class_weights_inverse_freq() -> None:
    counts = {0: 100, 1: 10, 2: 800, 3: 500}
    w = default_class_weights(counts)
    assert w.shape == (N_STATES,)
    # Instability(1) 权重应 > Descent(2)
    assert w[1] > w[2]
    assert w[1] > w[0]


def test_masked_loss_class_weights_effect() -> None:
    """稀有类权重高时 loss 对稀有类更敏感。"""
    logits = torch.zeros(2, N_STATES)
    labels = torch.tensor([1, 1], dtype=torch.long)
    inst_mask = torch.tensor([True, True])
    # 预测错 Instability(1)
    logits[0, 0] = 3.0
    logits[1, 0] = 3.0
    w_low = torch.tensor([1.0, 0.1, 1.0, 1.0])
    w_high = torch.tensor([1.0, 5.0, 1.0, 1.0])
    loss_low = masked_state_loss(logits, labels, inst_mask, w_low)
    loss_high = masked_state_loss(logits, labels, inst_mask, w_high)
    assert loss_high > loss_low


def test_model_reproducible_with_seed() -> None:
    torch.manual_seed(0)
    m1 = StateEvolutionTCNV1(n_features=4, hidden_dim=8, n_layers=1)
    torch.manual_seed(0)
    m2 = StateEvolutionTCNV1(n_features=4, hidden_dim=8, n_layers=1)
    m1.eval()
    m2.eval()
    x = torch.randn(2, 10, 4)
    assert torch.allclose(m1(x), m2(x))


def test_hierarchical_model_forward() -> None:
    model = HierarchicalStateTCNV1(n_features=21, hidden_dim=16, n_layers=2)
    x = torch.randn(4, 20, 21)
    proc_logits, inst_logits = model(x)
    assert proc_logits.shape == (4, N_PROCESS)
    assert inst_logits.shape == (4, 1)
    assert torch.isfinite(proc_logits).all()


def test_hierarchical_loss_basic() -> None:
    proc_logits = torch.randn(8, N_PROCESS)
    inst_logits = torch.randn(8, 1)
    proc_labels = torch.tensor([0, 1, 1, 0, 1, 0, 1, 1], dtype=torch.long)
    inst_labels = torch.tensor([0, 1, 1, 0, 1, 0, 1, 1], dtype=torch.long)
    inst_valid = torch.tensor([False] * 8)
    losses = hierarchical_loss(
        proc_logits, proc_labels, inst_logits, inst_labels, inst_valid,
        lambda_inst=1.0,
    )
    assert torch.isfinite(losses["loss_total"]).all()
    # 无 inst_valid 样本 → inst loss = 0
    assert losses["loss_instability"].item() == 0.0
    # total == process
    assert losses["loss_total"].item() == losses["loss_process"].item()


def test_hierarchical_loss_masks_inst() -> None:
    """无 inst_valid 样本的 Instability 不参与监督。"""
    proc_logits = torch.randn(4, N_PROCESS)
    inst_logits = torch.randn(4, 1)
    proc_labels = torch.zeros(4, dtype=torch.long)
    inst_labels = torch.ones(4, dtype=torch.long)  # 全正
    inst_valid = torch.tensor([False, False, False, True])  # 只有最后1个有效
    losses = hierarchical_loss(
        proc_logits, proc_labels, inst_logits, inst_labels, inst_valid,
        lambda_inst=1.0,
    )
    # 只有1个 inst_valid 样本，loss 有限
    assert torch.isfinite(losses["loss_instability"]).all()
    assert losses["loss_instability"] > 0


def test_default_process_weights() -> None:
    counts = {0: 100, 1: 10}
    w = default_process_weights(counts)
    assert w.shape == (N_PROCESS,)
    assert w[1] > w[0]  # Fall 稀有 → 权重大
