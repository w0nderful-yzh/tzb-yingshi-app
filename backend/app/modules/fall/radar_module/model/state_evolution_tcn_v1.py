"""层级式 causal TCN：ProcessHead(NormalDynamic/FallProcess) + Optional InstabilityHead。

结构
----
- Shared causal TCN encoder（因果膨胀卷积）
- ProcessHead: 2 类（NormalDynamic / FallProcess）→ 主分类
- InstabilityHead: 独立 sigmoid 二分类 → 0.2–0.5s imminent/pre-fall warning
  - 只在有可靠 Instability 标签的事件上监督（mask）

四状态基线模型（StateEvolutionTCNV1）保留用于 baseline 对比，不删除。

损失：
L = L_process + λ * mask_instability * L_instability

Version: radar_state_evolution_tcn_v2
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_STATES = 4  # 旧四状态（保留 baseline）
N_PROCESS = 2  # NormalDynamic / FallProcess


class CausalConv1d(nn.Module):
    """因果卷积（padding 在左侧，不泄漏未来）。"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel, dilation=dilation, padding=self.pad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, : x.size(2)]


class CausalTCNEncoder(nn.Module):
    """共享因果 TCN 编码器。"""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        n_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        dilations = [2 ** (i % 3) for i in range(n_layers)]
        layers: list[nn.Module] = []
        in_ch = n_features
        for d in dilations:
            layers.append(CausalConv1d(in_ch, hidden_dim, kernel_size, dilation=d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = hidden_dim
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, window, F) → (B, H, T)"""
        x = x.transpose(1, 2)  # (B, F, T)
        return self.backbone(x)  # (B, H, T)


class HierarchicalStateTCNV1(nn.Module):
    """层级式 causal TCN：共享编码器 + ProcessHead + InstabilityHead。"""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        n_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = CausalTCNEncoder(
            n_features, hidden_dim, n_layers, kernel_size, dropout
        )
        self.process_head = nn.Linear(hidden_dim, N_PROCESS)
        self.instability_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (process_logits (B,2), inst_logits (B,1))。"""
        h = self.encoder(x)          # (B, H, T)
        last = h[:, :, -1]           # (B, H)
        process_logits = self.process_head(last)
        inst_logits = self.instability_head(last)
        return process_logits, inst_logits


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------

def hierarchical_loss(
    process_logits: torch.Tensor,
    process_labels: torch.Tensor,
    inst_logits: torch.Tensor,
    inst_labels: torch.Tensor,
    inst_valid: torch.Tensor,
    *,
    lambda_inst: float = 1.0,
    process_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """层级损失。

    L = L_process + λ * mask_instability * L_instability

    process_labels: (B,) 0=Normal, 1=FallProcess
    inst_labels: (B,) 0/1（仅 inst_valid=True 有效）
    inst_valid: (B,) bool，False 的样本不参与 inst 监督
    """
    # Process 损失
    if process_weights is not None:
        loss_process = F.cross_entropy(process_logits, process_labels, weight=process_weights)
    else:
        loss_process = F.cross_entropy(process_logits, process_labels)

    # Instability 损失（仅 inst_valid 样本）
    valid_idx = inst_valid.nonzero(as_tuple=True)[0]
    if valid_idx.numel() > 0:
        inst_logits_v = inst_logits[valid_idx].squeeze(-1)
        inst_labels_v = inst_labels[valid_idx].float()
        loss_inst = F.binary_cross_entropy_with_logits(
            inst_logits_v, inst_labels_v
        )
        loss_total = loss_process + lambda_inst * loss_inst
    else:
        loss_inst = torch.tensor(0.0, device=inst_logits.device)
        loss_total = loss_process

    return {
        "loss_total": loss_total,
        "loss_process": loss_process,
        "loss_instability": loss_inst,
    }


# ---------------------------------------------------------------------------
# 旧四状态模型（baseline，保留）
# ---------------------------------------------------------------------------

class StateEvolutionTCNV1(nn.Module):
    """四状态 causal TCN（baseline，不删除）。"""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        n_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.2,
        n_states: int = N_STATES,
    ) -> None:
        super().__init__()
        self.encoder = CausalTCNEncoder(
            n_features, hidden_dim, n_layers, kernel_size, dropout
        )
        self.head = nn.Linear(hidden_dim, n_states)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        last = h[:, :, -1]
        return self.head(last)


def masked_state_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    inst_mask: torch.Tensor,
    class_weights: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """带 Optional-Instability mask 的交叉熵（旧四状态损失）。"""
    valid = labels >= 0
    if not valid.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    per_sample_w = class_weights[labels.clamp(min=0)]
    no_inst_mask = ~inst_mask
    per_sample_w = per_sample_w * (~(no_inst_mask & (labels == 1))).float()
    per_sample_w = per_sample_w * valid.float()
    loss = F.cross_entropy(logits, labels, reduction="none", ignore_index=ignore_index)
    loss = loss * per_sample_w
    if per_sample_w.sum() > 0:
        return loss.sum() / per_sample_w.sum().clamp(min=1.0)
    return torch.tensor(0.0, device=logits.device, requires_grad=True)


def default_class_weights(label_counts: dict[int, int], n_classes: int = N_STATES) -> torch.Tensor:
    """逆频率 + 平方根平滑。"""
    total = sum(label_counts.values())
    n = n_classes
    weights = torch.zeros(n_classes)
    for i in range(n_classes):
        c = label_counts.get(i, 0)
        weights[i] = total / (n * (c + 1)) if c > 0 else 0.0
    if weights.sum() > 0:
        weights = weights / weights.sum() * n
    return weights


def default_process_weights(process_counts: dict[int, int]) -> torch.Tensor:
    """ProcessHead 类别权重（Normal/Fall 不平衡补偿）。"""
    total = sum(process_counts.values())
    n = N_PROCESS
    weights = torch.zeros(N_PROCESS)
    for i in range(N_PROCESS):
        c = process_counts.get(i, 0)
        weights[i] = total / (n * (c + 1)) if c > 0 else 0.0
    if weights.sum() > 0:
        weights = weights / weights.sum() * n
    return weights
