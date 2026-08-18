# -*- coding: utf-8 -*-
"""
把 OpenFace 老人轨迹 CSV 转成 MCCL 推理需要的 4 种 npy 特征。

输入: extract_elderly.py 输出的老人轨迹 CSV（OpenFace 列，含 X_/Y_/Z_ 3D关键点、
      gaze_0/1、pose_T/R、AU01_r..AU26_r）
输出: 每 clip 60秒(1800帧) 的 4 种 npy，与 MCCL 训练格式一致：
      <out>/<prefix>-{i:02}_kps.npy   (1800, 68, 3)
      <out>/<prefix>-{i:02}_gaze.npy  (1800, 4, 3)
      <out>/<prefix>-{i:02}_pose.npy  (1800, 2, 3)
      <out>/<prefix>-{i:02}_AUs.npy   (1800, 14)

无人帧处理: 老人不在画面时该帧全部填0（保留不在场，由信号特征体现）。
clip 内不足1800帧时补0（视频末尾）。
"""
import os
import sys
import numpy as np
import pandas as pd

FPS = 30
WINDOW_FRAMES = 1800      # 60秒
HOP_FRAMES = 1500         # 50秒步长，与训练一致

# DAIC 原版用的14个AU（顺序与训练一致）
AU_LIST = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r',
           'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r',
           'AU25_r', 'AU26_r']

# OpenFace 3D关键点列
KP_X = [f'X_{i}' for i in range(68)]
KP_Y = [f'Y_{i}' for i in range(68)]
KP_Z = [f'Z_{i}' for i in range(68)]

GAZE_COLS = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']
POSE_COLS = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']


def load_elderly_csv(csv_path):
    """读老人轨迹CSV，按frame建立逐帧字典，返回 {frame_index: row_df}"""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # OpenFace frame 从1开始，统一为0-indexed
    frames = df['frame'].to_numpy(dtype=np.int64)
    # 去掉重复帧(同一帧出现多张脸时保留第一张)
    df = df.drop_duplicates(subset='frame', keep='first')
    df = df.set_index('frame')
    return df, frames.min(), frames.max()


def build_feature_matrix(df, frame_min, frame_max, num_total_frames):
    """
    对整段视频构建连续特征矩阵 (num_total_frames, F)。
    老人不在的帧填0。
    """
    n = num_total_frames
    # 关键点 (n, 68, 3)
    kps = np.zeros((n, 68, 3), dtype=np.float32)
    # 注视 (n, 4, 3)
    gaze = np.zeros((n, 4, 3), dtype=np.float32)
    # 姿态 (n, 2, 3)
    pose = np.zeros((n, 2, 3), dtype=np.float32)
    # AU (n, 14)
    aus = np.zeros((n, 14), dtype=np.float32)

    for frame_idx, row in df.iterrows():
        if frame_idx < 1 or frame_idx > n:
            continue
        i = int(frame_idx) - 1  # 1-indexed → 0-indexed
        # 3D关键点
        try:
            kps[i] = np.stack([
                row[KP_X].to_numpy(dtype=np.float32),
                row[KP_Y].to_numpy(dtype=np.float32),
                row[KP_Z].to_numpy(dtype=np.float32),
            ], axis=-1)
        except Exception:
            pass
        # 注视：OpenFace只有双眼(gaze_0/1)，放进前2组；
        # 后2组(头部向量)补0以对齐MCCL训练时的4×3结构
        try:
            eyes = row[GAZE_COLS].to_numpy(dtype=np.float32).reshape(2, 3)
            gaze[i, :2] = eyes
        except Exception:
            pass
        # 姿态
        try:
            pose[i] = row[POSE_COLS].to_numpy(dtype=np.float32).reshape(2, 3)
        except Exception:
            pass
        # AU（只取DAIC的14个）
        try:
            aus[i] = row[AU_LIST].to_numpy(dtype=np.float32)
        except Exception:
            pass

    return kps, gaze, pose, aus


def split_clips(kps, gaze, pose, aus):
    """按滑动窗口切成 clips，每clip1800帧，不足补0"""
    T = kps.shape[0]
    if T < WINDOW_FRAMES:
        num_frame = 0
    else:
        num_frame = (T - WINDOW_FRAMES) // HOP_FRAMES + 1

    clips = []
    for i in range(num_frame):
        s = i * HOP_FRAMES
        e = s + WINDOW_FRAMES
        ck = np.zeros((WINDOW_FRAMES, 68, 3), dtype=np.float32)
        cg = np.zeros((WINDOW_FRAMES, 4, 3), dtype=np.float32)
        cp = np.zeros((WINDOW_FRAMES, 2, 3), dtype=np.float32)
        ca = np.zeros((WINDOW_FRAMES, 14), dtype=np.float32)
        L = min(WINDOW_FRAMES, T - s)
        ck[:L] = kps[s:s+L]
        cg[:L] = gaze[s:s+L]
        cp[:L] = pose[s:s+L]
        ca[:L] = aus[s:s+L]
        clips.append((ck, cg, cp, ca))
    return clips


def main():
    if len(sys.argv) < 3:
        print('用法: python openface_to_mccl.py <老人轨迹csv> <输出目录> [prefix]')
        sys.exit(1)
    csv_path = sys.argv[1]
    out_dir = sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(os.path.basename(csv_path))[0]

    df, frame_min, frame_max = load_elderly_csv(csv_path)
    print(f'读入 {len(df)} 帧, 范围 {frame_min}-{frame_max}, 帧数={frame_max}')

    # 视频总帧数 = 最大帧号（OpenFace连续帧）
    num_total = int(frame_max)
    kps, gaze, pose, aus = build_feature_matrix(df, frame_min, frame_max, num_total)
    print(f'连续特征矩阵: kps={kps.shape} gaze={gaze.shape} pose={pose.shape} aus={aus.shape}')

    clips = split_clips(kps, gaze, pose, aus)
    print(f'切成 {len(clips)} 个clip')
    if not clips:
        print('视频太短，不足60秒')
        sys.exit(2)

    os.makedirs(out_dir, exist_ok=True)
    for i, (ck, cg, cp, ca) in enumerate(clips):
        np.save(os.path.join(out_dir, f'{prefix}-{i:02}_kps.npy'), ck)
        np.save(os.path.join(out_dir, f'{prefix}-{i:02}_gaze.npy'), cg)
        np.save(os.path.join(out_dir, f'{prefix}-{i:02}_pose.npy'), cp)
        np.save(os.path.join(out_dir, f'{prefix}-{i:02}_AUs.npy'), ca)
    print(f'已输出 {len(clips)} 个clip到 {out_dir}')


if __name__ == '__main__':
    main()
