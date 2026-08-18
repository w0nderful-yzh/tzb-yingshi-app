# ============================================================
# 模型分支1 (Classifier_One) — 多线索直接融合
#
# 思路：对每种线索分别用ResNet18提取特征
#       → 跨线索LSTM + 多头自注意力(MHSA)
#       → 1D卷积融合 → 输出256维特征
#
# 输入：4种线索 [B, T, F]
#   0: 3D facial keypoints (面部关键点)
#   1: gaze vectors (注视方向向量)
#   2: head pose (头部位置+旋转)
#   3: action units (面部动作单元)
# 输出：特征向量 [B, 256]
# ============================================================

import copy
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from einops import rearrange
from utils import *


class Classifier_One(nn.Module):
    def __init__(self, args):
        super(Classifier_One, self).__init__()

        self.args = args
        self.num_classes = args.num_classes
        self.device = torch.device('cuda:%d' % args.gpu_ids[0]) if args.gpu_ids and torch.cuda.is_available() else torch.device('cpu')
        self.num_frames = args.num_frames          # 总帧数
        self.instance_length = args.instance_length  # 每段帧数
        self.bag_size_video = self.num_frames // self.instance_length  # 片段数
        self.bag_size_cues = args.bag_size_cues    # 线索数（默认4）

        # ---------- 共享的ResNet18骨干网络 ----------
        model = resnet18(weights=ResNet18_Weights.DEFAULT)  # ImageNet预训练
        self.features_2d = nn.Sequential(*list(model.children())[:-1])  # 去掉最后FC层

        # ---------- 单通道ResNet18（用于AU这种单通道输入）----------
        self.model = model
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, 512)
        self.features_2d_gray = self.model

        # ---------- 时序建模：LSTM ----------
        self.lstm = nn.LSTM(
            input_size=self.args.input_size_lstm,   # 512
            hidden_size=self.args.hidden_size_lstm, # 64
            num_layers=self.args.num_layers_lstm,   # 1
            batch_first=True,
            bidirectional=True)                     # 双向 → 输出128维

        # ---------- 多头自注意力 (MHSA) ----------
        self.heads = self.args.head                 # 注意力头数，默认2
        self.dim_head = self.args.hidden_size_lstm * 2 // self.heads  # 每头维度 128//2=64
        self.scale = self.dim_head ** -0.5          # 缩放因子 1/sqrt(64)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(
            self.args.hidden_size_lstm * 2,          # 输入128
            (self.dim_head * self.heads) * 3,        # Q/K/V各64*2=128, 共384
            bias=False)

        # ---------- 归一化和融合 ----------
        self.norm = DMIN(num_features=self.args.hidden_size_lstm * 2, args=self.args)
        self.pwconv_cues = nn.Conv1d(self.bag_size_cues, 1, 3, 1, 1)  # 跨线索1D卷积融合

    def lstm_mhsa(self, x):
        """
        时序建模：LSTM + 多头自注意力
        输入: [B, bag_size, 512]  （bag_size=线索数）
        输出: [B, bag_size, 128]
        """
        # ----- LSTM时序编码 -----
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)  # [B, bag_size, 128]

        ori_x = x  # 残差连接用

        # ----- 多头自注意力（跨线索注意力）-----
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 拆成Q/K/V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 注意力分数
        attn = self.attend(dots)  # softmax归一化
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')  # 合并多头

        # ----- 残差 + sigmoid门控 -----
        if self.bag_size_cues > 1:
            x = self.norm(x)
        x = torch.sigmoid(x)
        x = ori_x * x  # 门控残差

        return x, attn

    def forward(self, x_complete, epoch, mode):
        """
        前向传播：
        1. 对每种线索分别用ResNet18提取特征
        2. LSTM+MHSA建模跨线索关系
        3. 1D卷积融合为单向量
        """
        features_list_cues = []
        features_types = [
            self.args.bag_cues_feature1,  # 0: 3D关键点
            self.args.bag_cues_feature2,  # 1: 注视
            self.args.bag_cues_feature3,  # 2: 姿态
            self.args.bag_cues_feature4,  # 3: AU
            self.args.bag_cues_feature5,  # 可选
        ]

        # ----- 对每种线索独立提取特征 -----
        for i in range(0, self.bag_size_cues):
            if features_types[i] == 3:
                # AU特征：单通道 → 用单通道ResNet
                x = x_complete[3]  # [B, 1800, 14]
                x = x.unsqueeze(-1).permute(0, 3, 1, 2)  # [B, 1, 1800, 14]
                x = self.features_2d_gray(x).squeeze().unsqueeze(1)  # [B, 1, 512]
            else:
                # 其他线索：3通道 → 用标准ResNet
                index = features_types[i]
                x = x_complete[index].permute(0, 3, 1, 2)  # [B, 3, T, F]
                x = self.features_2d(x).squeeze().unsqueeze(1)  # [B, 1, 512]
            features_list_cues.append(x)

        # ----- 拼接所有线索 → 跨线索建模 -----
        cues_input_mo1 = torch.cat(features_list_cues, dim=1)  # [B, 4, 512]

        # LSTM + MHSA
        features_tensor_cues, _ = self.lstm_mhsa(cues_input_mo1)  # [B, 4, 128]

        # 跨线索1D卷积融合
        features_cues_conv = self.pwconv_cues(features_tensor_cues).squeeze()  # 融合为[B, 128]

        # 注：最终输出维度 = hidden_size_lstm*2 = 64*2 = 128，
        # 与分支2对齐拼接后得256维

        return features_cues_conv  # [B, 128]
