# ============================================================
# MCCL 参数配置模块
# 所有可调超参数集中在这里
# ============================================================

import argparse
import os
import datetime
import sys

from utils import Logger

class Options(object):
    def __init__(self):
        super(Options, self).__init__()

    def initialize(self):
        parser = argparse.ArgumentParser()

        # ==================== 基本设置 ====================
        parser.add_argument('--mode', type=str, default="train")     # 运行模式
        parser.add_argument('--dataset', type=str, default="DAIC")   # 数据集名称
        # 自动检测CUDA
        try:
            import torch
            _has_cuda = torch.cuda.is_available()
        except:
            _has_cuda = False
        _default_gpu = '0' if _has_cuda else '-1'
        _default_dev = 'cuda:0' if _has_cuda else 'cpu'
        parser.add_argument('--gpu_ids', type=str, default=_default_gpu, help='gpu ids, eg. 0,1,2; -1 for cpu.')
        parser.add_argument('--device', type=str, default=_default_dev)  # CUDA设备
        parser.add_argument('--inference', type=str, default='0')    # 1=仅推理，0=训练
        parser.add_argument('--seed', default='42', type=int)        # 随机种子
        parser.add_argument('--regressor_model', default='xgboost', type=str)  # 回归器: rf, xgboost, mlp
        parser.add_argument('--start_epoch', default=0, type=int)    # 起始epoch
        parser.add_argument('--epochs', default=300, type=int)       # 总训练轮数
        parser.add_argument('-b', '--batch_size', default=32, type=int)
        parser.add_argument('--num_classes', default=1, type=int)    # 回归任务，输出1个值

        # ==================== 特征提取器设置 ====================
        parser.add_argument('--time_input', default=2048, type=int)          # 时间维度输入大小
        parser.add_argument('--efeature_dl', default='resnet', type=str)    # 深度学习特征: resnet, vgg
        parser.add_argument('--efeature_tr', default='openface', type=str)  # 传统特征: bow, openface
        parser.add_argument('--efeature_audio', default='bow_mfcc', type=str)  # 音频特征类型
        parser.add_argument('--efeature_text', default='text', type=str)    # 文本特征
        parser.add_argument('--norm', default=0, type=int)                  # 是否归一化

        # ==================== XGBoost 参数 ====================
        parser.add_argument('--n_estimators', default='50', type=int)       # 树的数量
        parser.add_argument('--xg_lr', default='0.08', type=float)         # 学习率
        parser.add_argument('--subsample', default='0.75', type=float)     # 样本采样率
        parser.add_argument('--colsample_bytree', default='1', type=float) # 特征采样率
        parser.add_argument('--max_depth', default='4', type=int)          # 树最大深度
        parser.add_argument('--gamma', default='0', type=float)            # 分裂最小loss减少
        parser.add_argument('--n_jobs', default='-1', type=int)            # 并行线程数
        parser.add_argument('--tree_method', default='hist', type=str)     # 树构建方法

        # ==================== LSTM 编码器维度 ====================
        parser.add_argument('--input_size_lstm', default=512, type=int)    # LSTM输入维度
        parser.add_argument('--hidden_size_lstm', default=64, type=int)    # LSTM隐藏层维度（最终输出=×2）
        parser.add_argument('--num_layers_lstm', default=1, type=int)
        parser.add_argument('--hidden_size_lstm1', default=32, type=int)
        parser.add_argument('--num_layers_lstm1', default=1, type=int)
        parser.add_argument('--num_layers_lstm2', default=1, type=int)
        # 对比学习投影头维度
        parser.add_argument('--num_proj_hidden', default=128, type=int)

        # ==================== 多线索(cue)设置 ====================
        parser.add_argument('--bag_size_cues', default=4, type=int)  # 使用的线索数量（最多5个）
        parser.add_argument('--bag_cues_feature1', default=0, type=int)  # 线索1: 0=facial_keypoints(3D关键点)
        parser.add_argument('--bag_cues_feature2', default=1, type=int)  # 线索2: 1=gaze(注视向量)
        parser.add_argument('--bag_cues_feature3', default=2, type=int)  # 线索3: 2=pose(头部姿态)
        parser.add_argument('--bag_cues_feature4', default=3, type=int)  # 线索4: 3=AUs(面部动作单元)
        parser.add_argument('--bag_cues_feature5', default=4, type=int)  # 线索5: 可选

        # ==================== 模型/temporal设置 ====================
        parser.add_argument('--sample_num', default=10, type=int)     # 帧采样间隔
        parser.add_argument('--num_frames', default=12600//10, type=int)  # 总帧数
        parser.add_argument('--instance_length', default=420, type=int)   # 每个片段的帧数
        parser.add_argument('--model', default='multi_cues_clip_fix', type=str)
        parser.add_argument('--main_mode', default='train_sep', type=str) # 'train_sep'=两阶段训练
        parser.add_argument('--learn_mode', default='cl', type=str)       # 'cl'=对比学习

        # ==================== 对比学习超参数 ====================
        parser.add_argument('--head', default=2, type=int)       # MHSA头数
        parser.add_argument('--scale', default=-0.5, type=float) # 注意力缩放
        parser.add_argument('--tau', default=0.09, type=float)   # 对比学习温度系数

        # ==================== 优化器 ====================
        parser.add_argument('-o', '--optimizer', default="AdamW", type=str)
        parser.add_argument('--lr', '--learning_rate', default=5e-4, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--weight_decay', '--wd', default=0.05, type=float)
        parser.add_argument('--eps', default=1e-1, type=float)
        parser.add_argument('--label_smoothing', default=0.1, type=float)

        # ==================== 学习率调度 ====================
        parser.add_argument('--lr_scheduler', default="cosine", type=str)
        parser.add_argument('--warmup_epochs', default=10, type=int)
        parser.add_argument('--min_lr', default=5e-6, type=float)
        parser.add_argument('--warmup_lr', default=0, type=float)

        # ==================== 数据路径 ====================
        parser.add_argument('--dataset_path',
                    default='',
                    help='DAIC-WOZ 训练数据根目录；训练时必须通过参数传入，不绑定本机盘符')
        parser.add_argument('--output_path', default='./outputs', help='日志和模型保存路径')
        parser.add_argument('--output_name', default='e1-h2-b8-s1-w0', help='实验名称')
        parser.add_argument('--num_workers', default=0)
        parser.add_argument('--light', default=0, type=int)

        return parser

    def parse(self):
        """解析参数并初始化"""
        parser = self.initialize()
        args = parser.parse_args()

        # 解析GPU ID
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            cur_id = int(str_id)
            if cur_id >= 0:
                args.gpu_ids.append(cur_id)

        # 创建输出目录
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)

        # 设置日志输出（同时输出到终端和文件）
        file_name = os.path.join(
            args.output_path, '{}-{}.log'.format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), args.output_name))
        sys.stdout = Logger(file_name)

        # 打印所有参数
        for k in args.__dict__:
            print(k + ": " + str(args.__dict__[k]))

        return args
