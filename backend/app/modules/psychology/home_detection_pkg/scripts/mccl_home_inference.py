# -*- coding: utf-8 -*-
"""
MCCL 居家推理：对 OpenFace 转出的 clip 特征输出抑郁分数。
打包版：模块路径指向本目录的 ../mccl 和 ../checkpoint。
"""
import os
import sys
import glob
import pickle
import argparse
import re
import warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')

# 打包目录结构：scripts/ 是脚本，../mccl 是MCCL模块，../checkpoint 是模型
PKG_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MCCL_DIR = os.path.join(PKG_BASE, 'mccl')
sys.path.insert(0, MCCL_DIR)

import options  # noqa
from options import Options

CKPT_DIR = os.path.join(PKG_BASE, 'checkpoint', 'DAIC')

# 与训练一致的参数
SAMPLE_INDEX = list(range(0, 1800, 10))  # 每clip采样180帧
NUM_CLIPS_PER_SAMPLE = 7                  # 每7个clip拼一个样本
CLIP_NAMES = ['kps', 'gaze', 'pose', 'AUs']


def resolve_device(requested_device='cpu'):
    requested = requested_device.strip().lower()
    if requested == 'cpu':
        return 'cpu', []
    match = re.fullmatch(r'cuda:(\d+)', requested)
    if match is None:
        raise ValueError("MCCL device must be 'cpu' or an explicit CUDA device such as 'cuda:0'")
    if not torch.cuda.is_available():
        raise RuntimeError(f'MCCL requested {requested}, but CUDA is unavailable')
    device_index = int(match.group(1))
    device_count = torch.cuda.device_count()
    if device_index >= device_count:
        raise RuntimeError(
            f'MCCL requested {requested}, but only {device_count} CUDA device(s) are available'
        )
    return requested, [device_index]


def build_args(device='cpu'):
    args = Options().initialize().parse_args([])
    effective_device, gpu_ids = resolve_device(device)
    args.gpu_ids = gpu_ids
    args.device = effective_device
    return args


def load_models(args, ckpt_dir):
    device = torch.device(args.device)
    model1 = torch.load(os.path.join(ckpt_dir, 'current_model1'),
                        map_location=device, weights_only=False).to(device).eval()
    model2 = torch.load(os.path.join(ckpt_dir, 'current_model2'),
                        map_location=device, weights_only=False).to(device).eval()
    with open(os.path.join(ckpt_dir, 'pima.pickle.dat'), 'rb') as f:
        regressor = pickle.load(f)
    return model1, model2, regressor, device


def load_clip(clip_dir, prefix, i):
    """加载单clip，采样到180帧，返回4种特征 (180, F)"""
    def load(name):
        path = os.path.join(clip_dir, f'{prefix}-{i:02}_{name}.npy')
        arr = np.load(path)
        return torch.from_numpy(arr[SAMPLE_INDEX]).float()
    return [load(n) for n in CLIP_NAMES]


def infer_7clips(model1, model2, regressor, clip_feats, args, device):
    """对7个clip拼接的样本推理 → PHQ-8分数。
    Train_One/Two 的 .squeeze() 在 batch=1 会丢batch维，用 batch=2(dummy) 规避。
    """
    per_type = []
    for t in range(4):
        one = torch.cat([cf[t] for cf in clip_feats], dim=0).unsqueeze(0)
        per_type.append(torch.cat([one, one.clone()], dim=0))  # batch=2
    with torch.no_grad():
        per_type = [f.to(device) for f in per_type]
        rep1 = model1(per_type, 0, 'inference')
        rep2 = model2(per_type, 0, 'inference')
    z = torch.cat([rep1[:1], rep2[:1]], dim=1).detach().cpu().numpy()
    return float(regressor.predict(z)[0])


def main():
    parser = argparse.ArgumentParser(description='MCCL 居家心理评估推理')
    parser.add_argument('clip_dir')
    parser.add_argument('prefix', nargs='?')
    parser.add_argument('--device', default=os.environ.get('PSYCH_MCCL_DEVICE', 'cpu'))
    cli_args = parser.parse_args()
    clip_dir = cli_args.clip_dir
    prefix = cli_args.prefix
    if prefix is None:
        kps_files = sorted(glob.glob(os.path.join(clip_dir, '*_kps.npy')))
        if not kps_files:
            print('clip_dir 里没有 _kps.npy，请指定 prefix')
            sys.exit(2)
        prefix = os.path.basename(kps_files[0]).rsplit('-', 1)[0]

    print(f'推理 clip 目录: {clip_dir}, prefix: {prefix}')
    requested_device = cli_args.device.strip().lower()
    args = build_args(requested_device)
    model1, model2, regressor, device = load_models(args, CKPT_DIR)
    print(
        f'模型加载完成, requested_device={requested_device}, '
        f'effective_device={args.device}, cuda_available={torch.cuda.is_available()}, '
        f'XGB_device=cpu, XGB特征数={getattr(regressor, "n_features_in_", "?")}'
    )

    kps_files = sorted(glob.glob(os.path.join(clip_dir, f'{prefix}-*_kps.npy')))
    if not kps_files:
        print('没有找到该 prefix 的 clip 文件')
        sys.exit(2)
    clip_indices = [int(os.path.basename(kf).rsplit('-', 1)[1][:2]) for kf in kps_files]
    print(f'共 {len(clip_indices)} 个 clip: {clip_indices}')

    scores = []
    print('\n逐段推理（每7个clip = 一段）:')
    for g in range(0, len(clip_indices), NUM_CLIPS_PER_SAMPLE):
        group = clip_indices[g:g + NUM_CLIPS_PER_SAMPLE]
        if len(group) < NUM_CLIPS_PER_SAMPLE:
            print(f'  跳过末尾 {len(group)} 个clip（不足7个）')
            break
        clip_feats = [load_clip(clip_dir, prefix, i) for i in group]
        score = infer_7clips(model1, model2, regressor, clip_feats, args, device)
        scores.append(score)
        print(f'  段 {g//NUM_CLIPS_PER_SAMPLE}: clip{group[0]:02}-{group[-1]:02} → PHQ-8 = {score:.2f}')

    if scores:
        print(f'\n=== 各段分数: {[round(s,2) for s in scores]} ===')
        print(f'=== 平均抑郁分数: {np.mean(scores):.2f} (PHQ-8, 0-24) ===')
    else:
        print('\n可用clip不足7个，无法构成完整推理样本')


if __name__ == '__main__':
    main()
