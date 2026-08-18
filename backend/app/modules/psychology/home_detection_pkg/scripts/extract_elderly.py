# -*- coding: utf-8 -*-
"""从 OpenFace 多脸输出中识别并提取"老年人"特征。

居家摄像头场景假设：
- 老年人长时间坐在固定位置，面部位移小、基本全程在场
- 护工/家人走动多、或中途进出画面

策略：
1. 读多脸版 CSV，按帧分组，每帧 0~N 张脸
2. 用连续帧的人脸中心点距离做"轨迹拼接"，把同一物理人连成 track
3. 按 track 统计：总在场帧数、平均帧间位移
4. 判定"老年人" = 在场时间最长 + 平均位移最小的 track
5. 输出该 track 的逐帧特征（只保留 frame 的列）

用法:
    python extract_elderly.py <multi_csv> <out_csv>
"""
import pandas as pd
import numpy as np
import sys


def read_openface_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def face_center(df):
    """计算每行(每张脸)的中心点坐标."""
    x_cols = [c for c in df.columns if c.startswith('x_')][:68]
    y_cols = [c for c in df.columns if c.startswith('y_')][:68]
    cx = df[x_cols].mean(axis=1)
    cy = df[y_cols].mean(axis=1)
    return cx.values, cy.values


def build_tracks(df, thr=150.0):
    """按时间顺序把每帧检测到的脸拼成轨迹.

    策略: 对当前帧的每张脸, 找上一帧中距离最近且 <thr 像素的脸续接;
    否则开启新轨迹. thr 需按画面大小调整.
    """
    frames = sorted(df['frame'].unique())
    df = df.set_index('frame')
    cx, cy = face_center(df.reset_index())

    # face_id 不可靠, 自己分配轨迹 id
    df = df.reset_index()
    df['track'] = -1
    last_faces = {}  # track -> (cx, cy)

    tracks = {}  # track -> list of (frame, row_index)
    next_track = 0

    for f in frames:
        rows = df[df['frame'] == f].index.tolist()
        candidates = []
        for i in rows:
            # 中心
            xcols = [c for c in df.columns if c.startswith('x_')][:68]
            ycols = [c for c in df.columns if c.startswith('y_')][:68]
            xi = df.loc[i, xcols].mean()
            yi = df.loc[i, ycols].mean()
            candidates.append((i, xi, yi))

        # 上一帧脸按最近距离匹配
        used_tracks = set()
        for i, xi, yi in sorted(candidates, key=lambda t: t[0]):
            best_t, best_d = None, float('inf')
            for t, (lx, ly) in last_faces.items():
                if t in used_tracks:
                    continue
                d = np.hypot(xi - lx, yi - ly)
                if d < best_d:
                    best_d, best_t = d, t
            if best_t is not None and best_d < thr:
                df.loc[i, 'track'] = best_t
                used_tracks.add(best_t)
                last_faces[best_t] = (xi, yi)
                tracks.setdefault(best_t, []).append(i)
            else:
                # 新轨迹
                df.loc[i, 'track'] = next_track
                last_faces[next_track] = (xi, yi)
                tracks.setdefault(next_track, []).append(i)
                next_track += 1

        # 清理太久没更新的轨迹(2秒没出现视为结束)
        for t in list(last_faces.keys()):
            if t not in used_tracks and t in tracks:
                if tracks[t] and (f - df.loc[tracks[t][-1], 'frame']) > 120:
                    del last_faces[t]

    return df, tracks


def pick_elderly(df, tracks, fps=60.0):
    """按 在场帧数多 + 平均位移小 选老年人 track."""
    stats = []
    for t, idxs in tracks.items():
        if len(idxs) < 30:  # 太短的轨迹忽略
            continue
        sub = df.loc[idxs].sort_values('frame')
        n_frames = len(sub)
        xcols = [c for c in df.columns if c.startswith('x_')][:68]
        ycols = [c for c in df.columns if c.startswith('y_')][:68]
        c = sub[xcols].mean(axis=1)
        r = sub[ycols].mean(axis=1)
        d = np.hypot(c.diff().fillna(0), r.diff().fillna(0))
        mean_move = d.mean()
        stats.append((t, n_frames, mean_move, c.mean(), r.mean()))
        print("track=%d frames=%d mean_move=%.2f center=(%.0f,%.0f)" % (
            t, n_frames, mean_move, c.mean(), r.mean()))
    if not stats:
        return None
    # 按 在场帧数降序, 位移升序
    stats.sort(key=lambda s: (-s[1], s[2]))
    return stats[0][0]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    multi_csv = sys.argv[1]
    out_csv = sys.argv[2]

    df = read_openface_csv(multi_csv)
    print("读入 %d 行, %d 帧" % (len(df), df['frame'].nunique()))

    df, tracks = build_tracks(df)
    print("拼出 %d 条轨迹" % len(tracks))

    elder_track = pick_elderly(df, tracks)
    if elder_track is None:
        print("未找到足够长的轨迹")
        sys.exit(2)
    print("判定老年人 track =", elder_track)

    sub = df[df['track'] == elder_track].sort_values('frame')
    sub = sub.drop(columns=['track'])
    sub.to_csv(out_csv, index=False)
    print("已输出 %d 帧到 %s" % (len(sub), out_csv))


if __name__ == '__main__':
    main()
