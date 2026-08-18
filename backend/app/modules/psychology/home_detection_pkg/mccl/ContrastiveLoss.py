# ============================================================
# 对比学习损失 (CustomSCLLoss)
#
# SimCLR风格的NT-Xent损失（Normalized Temperature-scaled
# Cross Entropy Loss）
#
# 核心思想：让同一批样本中，分支1和分支2的输出作为正样本对，
# 其他所有组合作为负样本对，拉近正样本、推开负样本。
#
# 最终效果：两个分支从不同角度提取的特征语义一致。
# ============================================================

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomSCLLoss(nn.Module):
    def __init__(self, args):
        super(CustomSCLLoss, self).__init__()
        self.args = args
        num_hidden = args.hidden_size_lstm * 2      # 128
        num_proj_hidden = args.num_proj_hidden       # 128

        self.tau = args.tau  # 温度系数，控制对负样本的惩罚力度

        # 投影头：128 → 128 → 128 (非线性映射)
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        """非线性投影头：将特征映射到对比学习空间"""
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        """计算余弦相似度矩阵"""
        z1 = F.normalize(z1)  # L2归一化
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())  # 相似度矩阵 [B, B]

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        计算单向对比损失：
        以z1为锚点，z2中的对应样本为正样本，
        z1中的其他样本和z2中的其他样本为负样本
        """
        f = lambda x: torch.exp(x / self.tau)  # 温度缩放

        # 自身相似度（z1与z1的所有样本）
        refl_sim = f(self.sim(z1, z1))
        # 交叉相似度（z1与z2的所有样本）
        between_sim = f(self.sim(z1, z2))

        # 正样本对: diag = 每个样本与自己在z2中的对应
        # 分母 = 自身负样本总和 + 交叉负样本总和
        return -torch.log(
            between_sim.diag() / (
                refl_sim.sum(1) - refl_sim.diag() + between_sim.sum(1)
            )
        )

    def forward(self, f1, f2):
        """
        输入：两个分支的输出特征 [B, 128]
        输出：对比损失标量
        """
        # 投影到对比学习空间
        f1 = self.projection(f1)
        f2 = self.projection(f2)

        # 双向对比损失（对称）
        l1 = self.semi_loss(f1, f2)  # 以f1为锚点
        l2 = self.semi_loss(f2, f1)  # 以f2为锚点

        loss = (l1 + l2) * 0.5  # 平均
        loss = loss.mean()      # 批次取平均

        return loss
