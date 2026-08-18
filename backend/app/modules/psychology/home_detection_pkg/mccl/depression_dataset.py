# ============================================================
# DAIC-WOZ 数据集加载器
#
# 从预处理好的文件夹中加载5种特征：
#   0: facial_keypoints — 3D面部关键点 (68个点 × 3)
#   1: gaze_vectors — 注视方向向量
#   2: position_rotation — 头部位置+旋转
#   3: action_units — 面部动作单元 (AU 0-13, 共14维)
#   4: audio_features — COVAREP统计特征 (148维, 均值+标准差)
#
# 注：索引0-3喂给对比学习模型，索引4(音频统计)只在XGBoost回归阶段使用
# 每个样本被切成7个clip（0-6），每个clip约1-2分钟
# 标签：PHQ-8得分（0-24） + 二分类标签
# ============================================================

import os
import pandas as pd
import torch
import numpy as np
import argparse

from torch.utils.data import Dataset, DataLoader


class DepressionDataset(Dataset):
    def __init__(self, args, mode):
        """
        args.dataset_path 下的目录结构:
            └── <mode>/original_data/
                ├── facial_keypoints/    *.npy
                ├── gaze_vectors/        *.npy
                ├── position_rotation/   *.npy
                ├── action_units/        *.npy
                └── audio_features/      *.npy
            <mode>_split_Depression_AVEC2017.csv  ← 标签文件
        """
        self.args = args
        self.root_dir = os.path.join(args.dataset_path, mode, 'original_data')
        self.sample_num = args.sample_num  # 帧采样间隔，默认每10帧取1帧

        # 读取CSV标签：Participant_ID, PHQ8_Score, PHQ8_Binary
        self.label = pd.read_csv(os.path.join(args.dataset_path, mode + '_split_Depression_AVEC2017.csv'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 每隔sample_num帧采样一次，1800//10=180帧
        sample_index = [j for j in range(0, 1800, self.sample_num)]

        ID = int(self.label.loc[idx, 'Participant_ID'])
        label = self.label.loc[idx, 'PHQ8_Score']
        complete_label = self.label.loc[idx, :]
        binary = self.label.loc[idx, 'PHQ8_Binary']

        # 5种特征的路径
        fkps3d_path = os.path.join(self.root_dir, 'facial_keypoints')
        gaze_path = os.path.join(self.root_dir, 'gaze_vectors')
        pose_path = os.path.join(self.root_dir, 'position_rotation')
        AUs_path = os.path.join(self.root_dir, 'action_units')
        audio_path = os.path.join(self.root_dir, 'audio_features')

        # 共7个clip编号：0,1,2,3,4,5,6
        clip_num = [0, 1, 2, 3, 4, 5, 6]

        # 分别存储每种特征的所有clip
        one_cue_fkps_3d = []
        one_cue_gaze = []
        one_cue_pose = []
        one_cue_au = []
        one_cue_audio = []

        for j in range(0, len(clip_num)):
            # 加载.npy文件并采样
            fkps_3d = torch.from_numpy(
                np.load(os.path.join(fkps3d_path, str(ID) + '-0' + str(clip_num[j]) + '_kps.npy'))
                [sample_index, :]).type(torch.FloatTensor)
            gaze = torch.from_numpy(
                np.load(os.path.join(gaze_path, str(ID) + '-0' + str(clip_num[j]) + '_gaze.npy'))
                [sample_index, :]).type(torch.FloatTensor)
            pose = torch.from_numpy(
                np.load(os.path.join(pose_path, str(ID) + '-0' + str(clip_num[j]) + '_pose.npy'))
                [sample_index, :]).type(torch.FloatTensor)
            AUs = torch.from_numpy(
                np.load(os.path.join(AUs_path, str(ID) + '-0' + str(clip_num[j]) + '_AUs.npy'))
                [sample_index, :]).type(torch.FloatTensor)

            # COVAREP统计特征：每个clip对应各自窗口的统计量(148,)
            audio_stats = torch.from_numpy(
                np.load(os.path.join(audio_path, str(ID) + '-0' + str(clip_num[j]) + '_covstats.npy'))
            ).type(torch.FloatTensor)

            one_cue_fkps_3d.append(fkps_3d)
            one_cue_gaze.append(gaze)
            one_cue_pose.append(pose)
            one_cue_au.append(AUs)
            one_cue_audio.append(audio_stats)

        # 将7个clip沿时间维度拼接
        one_cue_fkps_3d = torch.cat(one_cue_fkps_3d, dim=0)
        one_cue_gaze = torch.cat(one_cue_gaze, dim=0)
        one_cue_pose = torch.cat(one_cue_pose, dim=0)
        one_cue_au = torch.cat(one_cue_au, dim=0)
        # 音频：7个clip的统计量堆叠 → (7*148,)，保留每个时段的音频特征
        one_cue_audio = torch.cat(one_cue_audio, dim=0)

        return ([one_cue_fkps_3d,   # 线索0: 3D关键点
                 one_cue_gaze,      # 线索1: 注视
                 one_cue_pose,      # 线索2: 姿态
                 one_cue_au,        # 线索3: 面部AU
                 one_cue_audio],    # 线索4: COVAREP统计特征
                torch.tensor(complete_label.values),  # 完整标签
                binary)               # 二分类标签

    def __len__(self):
        return len(self.label)


def get_dataloaders(args):
    """创建训练和验证数据加载器"""
    dataloaders = {}
    for mode in ['train', 'validation']:
        dataset = DepressionDataset(args, mode)
        dataloaders[mode] = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=(mode == 'train'),
            num_workers=args.num_workers,
            drop_last=False)
    return dataloaders


if __name__ == '__main__':
    # 独立测试用
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default=r'/your_data_path/2017/wpingcheng/DAIC_WOZ-generated_database_V2/')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--num_workers', default=0)
    parser.add_argument('--sample_num', default=10)
    args = parser.parse_args()
    dataloaders = get_dataloaders(args)
    for i in dataloaders['train']:
        print('loading')
