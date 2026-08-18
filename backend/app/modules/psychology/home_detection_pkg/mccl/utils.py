# ============================================================
# 工具函数模块
# 1. build_scheduler: 余弦退火学习率调度器（含warmup）
# 2. DMIN: 可学习参数的自适应归一化层
# 3. Logger: 同时输出到终端和文件的日志
# ============================================================

import sys
import numpy as np
import torch.nn as nn
import torch

from timm.scheduler.cosine_lr import CosineLRScheduler
from einops import rearrange


def min_max_scale(data, new_min=0, new_max=1):
    """最小-最大归一化"""
    old_min, old_max = np.min(data), np.max(data)
    scaled_data = new_min + (data - old_min) * (new_max - new_min) / (old_max - old_min)
    return scaled_data


def build_scheduler(args, optimizer, n_iter_per_epoch):
    """
    构建余弦退火学习率调度器
    - 先warmup（线性上升）
    - 然后余弦下降
    """
    num_steps = int(args.epochs * n_iter_per_epoch)
    warmup_steps = int(args.warmup_epochs * n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,            # 总步数
        lr_min=args.min_lr,             # 最小学习率
        warmup_lr_init=args.warmup_lr,  # warmup起始LR
        warmup_t=warmup_steps,           # warmup步数
        cycle_limit=1,                   # 只循环一次
        t_in_epochs=False,
    )

    return lr_scheduler


class DMIN(nn.Module):
    """
    可学习的自适应归一化层
    融合了 Layer Norm 和 Instance Norm，
    用可学习权重动态调整两者比例
    """

    def __init__(self, num_features, args):
        super(DMIN, self).__init__()
        self.eps = args.eps
        self.momentum = args.momentum
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))  # 缩放
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))   # 偏置
        self.mean_weight = nn.Parameter(torch.ones(2))  # IN和LN的均值权重
        self.var_weight = nn.Parameter(torch.ones(2))   # IN和LN的方差权重

        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        """输入: [B, C, N]"""
        x = rearrange(x, 'b c n -> b n c')  # 重排为 [B, N, C]

        # Instance Norm: 每个样本每个通道独立归一化
        mean_in = x.mean(-1, keepdim=True)  # [B, N, 1]
        var_in = x.var(-1, keepdim=True)    # [B, N, 1]

        # Layer Norm: 每个样本所有通道一起归一化
        mean_ln = mean_in.mean(1, keepdim=True)  # [B, 1, 1]
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2  # [B, 1, 1]

        # 动态融合 IN 和 LN（权重由softmax学习）
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)  # 2维权重
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
        var = var_weight[0] * var_in + var_weight[1] * var_ln

        # 归一化 + 仿射变换
        x = (x - mean) / (var + self.eps).sqrt()
        x = x * self.weight + self.bias

        x = rearrange(x, 'b n c -> b c n')
        return x


class Logger(object):
    """同时输出到终端和文件的日志器"""

    def __init__(self, log_file="log_file.log"):
        self.terminal = sys.stdout
        self.file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()
