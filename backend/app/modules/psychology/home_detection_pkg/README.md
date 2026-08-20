# 居家抑郁检测系统

输入一段居家摄像头视频，输出 PHQ-8 抑郁分数（0-24）。

## 快速使用

```bash
python scripts/home_detect.py <视频路径.mp4>
```

一条命令自动完成：OpenFace 人脸特征提取 → 转成模型特征 → MCCL 模型推理 → 输出 PHQ-8 分数。

## 目录结构

```
home_detection_pkg/
├── scripts/                 # 入口和工具脚本
│   ├── home_detect.py       # ★ 一键入口：视频 → PHQ-8 分数
│   ├── openface_to_mccl.py  # OpenFace CSV → 模型clip特征
│   ├── mccl_home_inference.py # 模型推理 → 抑郁分数
│   └── extract_elderly.py   # 多人场景识别老人轨迹（单人自动跳过）
├── mccl/                    # MCCL 模型代码（推理用）
│   ├── options.py, utils.py, ContrastiveLoss.py
│   └── train_seprate/       # 两个对比学习分支模型
└── checkpoint/
    └── DAIC/                # 训练好的模型
        ├── current_model1   # 对比学习分支1
        ├── current_model2   # 对比学习分支2
        └── pima.pickle.dat  # XGBoost 回归器
```

## 环境依赖（换机器需安装）

1. **OpenFace 2.2**：人脸特征提取工具。
   - 路径在 `scripts/home_detect.py` 顶部 `OPENFACE_EXE` 配置
   - 官方下载：https://github.com/TadasBaltrusaitis/OpenFace/releases
2. **Python 3 + PyTorch(CUDA)**：模型推理
   - `pip install torch torchvision pandas numpy xgboost scikit-learn`
3. **至少 7 分钟视频**：模型结构需要 7 个 60 秒片段

## 已知限制

- 分数为**参考值，非诊断**（模型在 DAIC 临床数据训练，居家场景绝对值仅参考）
- OpenFace 对**小脸/侧脸有漏检**，会影响精度（已计划用 RetinaFace 人脸检测改进）
- 输出在终端显示，中间产物在 `home_out/视频名/`

## 流式评估 worker（实时，可选算法进程）

离线版是"给一段视频出分"；流式 worker 改为**常驻进程实时累积，攒满 7 个 clip
（每个严格 1800 帧 ≈ 1 分钟）判断一次**，并把最新结果写入 `home_out/latest/`
快照，Backend 的 LocalPsychologySource 直接读取——**Backend/App 零改动**。

- 普通项目启动仍是 `docker compose up`；**worker 不随 docker compose 启动**，
  需要在需要实时评估的机器上单独运行。
- worker 不跑时，Backend 照常读已有 latest 快照 / unavailable，不影响
  Fraud/Fall/Backend。
- 视频不长期保存：临时 CSV/chunk 位于本包 `.runtime/`（已被根 `.gitignore`
  排除），单窗口处理完立即删除，worker 启动时清理超时残留。

### 启动

在本目录（`home_detection_pkg/`）下：

```bash
python -m service.psychology_worker_main \
    --subject-key u-elder-001 --loop --capture-mode opensdk
```

- `--subject-key` **必须与 Backend 该 elder 的 `external_subject` 一致**，否则
  Backend 找不到快照（演示老人默认 `u-elder-001`）。
- YS7 凭据取 `APP_YS7_APP_KEY / APP_YS7_APP_SECRET / APP_YS7_DEVICE_SERIAL`
  环境变量或对应 CLI 参数；`--openface-exe` 指外置 OpenFace 路径
  （或设 `OPENFACE_EXE`）。
- 取流链路：本设备 C6c 经 `v2/live/address/get` 只返回**单片段占位 HLS**
  （`ErrCode/9053_0.ts`，标准 openlive 流仅支持 H.264，而设备码流是 HEVC），
  标准 url/tempfile 模式都拿不到真实画面。`--capture-mode opensdk`（配好
  `PSYCH_YS7_SDK_ROOT` 后 `auto` 也会优先它）走 EZVIZ OpenSDK（app/浏览器同款
  `ezopen://` 链路，实测 ~13fps），经系统 ffmpeg 转临时 H.264 文件喂 OpenFace，
  处理完即删。OpenFace 直读直播 URL 的模式在本设备同样崩溃，仅为通用兜底保留。
- 每轮约 7×(1800/fps 秒 + 余量)，OpenSDK 约 15fps 时约 **15-16 分钟**一次评估。

### 只做编排，不改算法

worker（`service/psychology_worker_main.py`）只复用既有算法代码：
OpenFace 特征提取、`openface_to_mccl.py`、`mccl_home_inference.py`、
MCCL/XGBoost/checkpoint 均不改动。模型输入语义不变：1 clip = 1800 帧、
7 clips = 1 次评估。

## 仓库内运行说明（迁移后）

本工程位于 `tzb-yingshi-app/backend/app/modules/psychology/home_detection_pkg/`，与后端 Psychology Adapter 同层（Adapter 为 `../mapping.py`、`../service.py` 等，两者职责不同：本目录是算法工程，Adapter 是 App 后端只读适配层）。

**OpenFace 外部放置方式（729MB 完整发行包不入 Git）**：
- 二选一：
  1. 将 OpenFace 完整包放在本包**同级目录** `backend/app/modules/psychology/OpenFace_2.2.0_win_x64/`（内含 `FeatureExtraction.exe`），这样 `scripts/home_detect.py` 顶部的默认相对路径自动生效；
  2. 或设置环境变量 `OPENFACE_EXE` 指向 `FeatureExtraction.exe` 的任意位置，例如：
     ```bash
     export OPENFACE_EXE=/path/to/OpenFace_2.2.0_win_x64/FeatureExtraction.exe
     ```
- 官方下载：https://github.com/TadasBaltrusaitis/OpenFace/releases （2.2.0）

**运行期产物不入 Git**：中间产物写入本包 `home_out/`（latest snapshots、OpenFace CSV、clip 特征），该目录已被仓库根 `.gitignore` 排除。

**只读 FastAPI 服务**（供 App Backend Psychology Adapter 读取 latest snapshot）：
```bash
# 在本目录下启动
uvicorn service.api:app --host 127.0.0.1 --port 8001
# GET /health
# GET /api/psychology/assessments/latest?subject_key=u-elder-001
```
最新快照存储目录默认 `home_out/latest/`，可用环境变量 `PSYCHOLOGY_LATEST_STORE` 覆盖；该服务只读快照，**不会触发视频推理**。
