# ============================================================
# 模型分支2 (Classifier_Two) — 多线索 + 多时间片段
#
# 与分支1的区别：先把每个线索在时间上分成多个片段，
# 对每个时间片段分别编码，再跨片段+跨线索融合。
#
# 这样既能捕获短时局部运动模式，也能建模长时序依赖。
# 输出：特征向量 [B, 128]
# ============================================================

import torch
import copy
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from einops import rearrange
from utils import *


class Classifier_Two(nn.Module):
    def __init__(self, args):
        super(Classifier_Two, self).__init__()

        self.args = args
        self.num_classes = args.num_classes
        self.device = torch.device('cuda:%d' % args.gpu_ids[0]) if args.gpu_ids and torch.cuda.is_available() else torch.device('cpu')
        self.num_frames = args.num_frames
        self.instance_length = args.instance_length
        self.bag_size_video = self.num_frames // self.instance_length  # 时间片段数
        self.bag_size_cues = args.bag_size_cues

        # ---------- 共享的ResNet18骨干（同分支1）----------
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features_2d = nn.Sequential(*list(model.children())[:-1])

        self.model = model
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, 512)
        self.features_2d_gray = self.model

        # ---------- LSTM（注意层数不同：num_layers_lstm1）----------
        self.lstm = nn.LSTM(
            input_size=self.args.input_size_lstm,
            hidden_size=self.args.hidden_size_lstm,
            num_layers=self.args.num_layers_lstm1,  # 默认也是1
            batch_first=True,
            bidirectional=True)

        # ---------- 多头自注意力 ----------
        self.heads = self.args.head
        self.dim_head = self.args.hidden_size_lstm * 2 // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(
            self.args.hidden_size_lstm * 2,
            (self.dim_head * self.heads) * 3, bias=False)

        self.norm = DMIN(num_features=self.args.hidden_size_lstm * 2, args=self.args)

        # ---------- 两个1D卷积融合层 ----------
        self.pwconv_video = nn.Conv1d(self.bag_size_video, 1, 3, 1, 1)   # 跨时间片段融合
        self.pwconv_second = nn.Conv1d(self.bag_size_cues, 1, 3, 1, 1)   # 跨线索融合

    def lstm_mhsa(self, x):
        """同分支1：LSTM + 多头自注意力"""
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)  # [B, bag_size, 512] → [B, bag_size, 128]

        ori_x = x

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        if self.bag_size_video > 1:
            x = self.norm(x)
        x = torch.sigmoid(x)
        x = ori_x * x

        return x, attn

    def forward(self, x_complete, epoch, mode):
        """
        前向传播：
        1. 对每种线索，先把时间维度切分成多个片段
        2. 对每个片段用ResNet18提取特征
        3. 跨片段LSTM+MHSA（捕获某个线索的长时依赖）
        4. 1D卷积融合各片段
        5. 最后跨线索1D卷积融合
        """
        features_list = []
        features_types = [
            self.args.bag_cues_feature1,
            self.args.bag_cues_feature2,
            self.args.bag_cues_feature3,
            self.args.bag_cues_feature4,
            self.args.bag_cues_feature5,
        ]

        for i in range(0, self.bag_size_cues):
            if features_types[i] == 3:
                # ===== AU特征（单通道）=====
                x = x_complete[3]  # [B, 1440, 14]
                # 时间维度切分：B × (片段数×每段帧数) × F → (B×片段数) × 每段帧数 × F
                x = rearrange(x, 'b (t1 t2) f-> (b t1) t2 f',
                              t1=self.bag_size_video, t2=self.instance_length)
                x = x.unsqueeze(-1).permute(0, 3, 1, 2)  # [B*10, 1, 144, 14]
                x = self.features_2d_gray(x).squeeze(1)   # [B*10, 512]

                # 恢复批次维度 → 跨片段建模
                x_be_mo1 = rearrange(x, '(b t1) f-> b t1 f', t1=self.bag_size_video)  # [B, 10, 512]
                af_mo1, _ = self.lstm_mhsa(x_be_mo1)      # [B, 10, 128]
                conv = self.pwconv_video(af_mo1).squeeze().unsqueeze(1)  # 融合→[B, 1, 128]

            else:
                # ===== 其他特征（3通道）=====
                index = features_types[i]
                x = x_complete[index].permute(0, 1, 3, 2)  # [B, 1260, 3, 68]
                # 时间切分
                x = rearrange(x, 'b (t1 t2) c h-> (b t1) c t2 h',
                              t1=self.bag_size_video, t2=self.instance_length)
                x = self.features_2d(x).squeeze()           # [B*3, 512]
                x_be_mo1 = rearrange(x, '(b t) c -> b t c', t=self.bag_size_video)  # [B, 3, 512]
                af_mo1, _ = self.lstm_mhsa(x_be_mo1)        # [B, 3, 128]
                conv = self.pwconv_video(af_mo1).squeeze().unsqueeze(1)  # [B, 1, 128]

            features_list.append(conv)

        # ----- 跨线索融合 -----
        features_tensor = torch.cat(features_list, dim=1)  # [B, 4, 128]
        features_tensor = self.pwconv_second(features_tensor).squeeze()  # [B, 128]

        return features_tensor  # [B, 128]
