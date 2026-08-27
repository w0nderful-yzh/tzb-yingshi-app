# -*- coding: utf-8 -*-
"""
居家摄像头抑郁检测 —— 端到端入口（打包版）

一条命令: 视频 → OpenFace特征 → MCCL推理 → PHQ-8抑郁分数

用法:
   python home_detect.py <video_path>
   输出: PHQ-8分数

环境配置（换机器需改）:
   OPENFACE_EXE — OpenFace FeatureExtraction.exe 路径
   PYTHON       — python 可执行文件路径
"""
import os
import sys
import subprocess
import argparse

# ============ 脚本/包路径 ============
BASE = os.path.dirname(os.path.abspath(__file__))
PKG_BASE = os.path.dirname(BASE)

# ============ 配置 ============
# 默认使用随心理模块一同提供的 OpenFace；部署时仍可通过环境变量覆盖。
OPENFACE_EXE = os.environ.get(
    'OPENFACE_EXE',
    os.path.join(os.path.dirname(PKG_BASE), 'OpenFace_2.2.0_win_x64', 'FeatureExtraction.exe'),
)
PYTHON = sys.executable  # 用当前python，不写死路径

# ============ 脚本路径（相对本文件）============
EXTRACT_ELDERLY = os.path.join(BASE, 'extract_elderly.py')
OPENFACE_TO_MCCL = os.path.join(BASE, 'openface_to_mccl.py')
MCCL_INFER = os.path.join(BASE, 'mccl_home_inference.py')

OPENFACE_ARGS = ['-multi_view', '1', '-track', '1']


def run(cmd, desc, echo_output=False):
    print(f'\n[{desc}]')
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'  !! {desc} 失败')
        print(r.stdout[-2000:] if r.stdout else '')
        print(r.stderr[-2000:] if r.stderr else '')
        return False
    if echo_output and r.stdout:
        print(r.stdout.rstrip())
    return True


def main():
    parser = argparse.ArgumentParser(description='居家摄像头抑郁检测端到端')
    parser.add_argument('video', help='输入视频路径')
    parser.add_argument('--outdir', default=os.path.join(os.path.dirname(BASE), 'home_out'),
                        help='输出目录')
    parser.add_argument('--mccl-device', default=os.environ.get('PSYCH_MCCL_DEVICE', 'cpu'),
                        help="MCCL device，默认 cpu；GPU 必须显式指定 cuda:0")
    args = parser.parse_args()

    video = args.video
    if not os.path.exists(video):
        print(f'视频不存在: {video}')
        sys.exit(1)
    name = os.path.splitext(os.path.basename(video))[0]
    outdir = args.outdir
    of_dir = os.path.join(outdir, name, 'openface')
    os.makedirs(of_dir, exist_ok=True)

    # ---------- 1. OpenFace 特征提取 ----------
    if not run([OPENFACE_EXE, '-f', video, '-out_dir', of_dir,
                *OPENFACE_ARGS], '1/4 OpenFace 特征提取'):
        sys.exit(1)
    csvs = [f for f in os.listdir(of_dir) if f.endswith('.csv')]
    if not csvs:
        print('OpenFace 未生成CSV，失败')
        sys.exit(1)
    of_csv = os.path.join(of_dir, csvs[0])
    print(f'  生成特征: {of_csv}')

    # ---------- 判断单人/多人 ----------
    import pandas as pd
    df = pd.read_csv(of_csv)
    df.columns = [c.strip() for c in df.columns]
    max_faces = df.groupby('frame')['face_id'].nunique().max() if 'face_id' in df.columns else 1
    print(f'  检测到每帧最多 {max_faces} 张脸')

    if max_faces <= 1:
        print('  [单人场景] 跳过老人轨迹提取')
        src_csv = of_csv
    else:
        elder_csv = os.path.join(outdir, name, f'{name}_elderly.csv')
        if not run([PYTHON, EXTRACT_ELDERLY, of_csv, elder_csv], '2/4 识别老人轨迹'):
            print('  未识别出老人')
            sys.exit(2)
        print(f'  老人轨迹: {elder_csv}')
        src_csv = elder_csv

    # ---------- 3. 转为 MCCL clip 特征 ----------
    clip_dir = os.path.join(outdir, name, 'clips')
    if not run([PYTHON, OPENFACE_TO_MCCL, src_csv, clip_dir, name], '3/4 格式转换'):
        sys.exit(1)
    print(f'  clip特征: {clip_dir}')

    # ---------- 4. MCCL 推理 ----------
    if not run(
        [PYTHON, MCCL_INFER, clip_dir, name, '--device', args.mccl_device],
        '4/4 MCCL 抑郁推理',
        echo_output=True,
    ):
        sys.exit(1)

    print('\n=== 完成: 居家检测流程已跑通 ===')


if __name__ == '__main__':
    main()
