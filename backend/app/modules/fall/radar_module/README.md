# IWR6843ISK 风险推理框架 MVP

本目录实现比赛第一阶段的毫米波技术链路：

```text
TI官方已解码输出 / JSONL Replay
→ RadarSourceAdapter
→ RadarFrame
→ FeatureVector (radar_features_v1)
→ 30帧窗口
→ LSTM risk_logit
→ Sigmoid risk_score
→ FastAPI
```

当前是“风险推理模型框架”，不是经过真实数据训练和独立验证的跌倒预测
模型。默认自动创建的 `TEST_CHECKPOINT` 只用于联调，输出必须显示：

> 当前为DEMO风险推理框架结果，不能代表真实跌倒预测能力

## 目录

```text
radar_module/
├── contracts.py
├── acquisition/ti_reader.py
├── preprocess/
│   ├── pointcloud_processing.py
│   ├── feature_extraction.py
│   └── window_generation.py
├── model/radar_lstm.py
├── inference/risk_prediction.py
├── service/radar_api.py
└── integration/ezviz_backend.py
```

`FeatureVector` 是雷达帧与模型之间的唯一接口。窗口和 LSTM 都不会读取
`RadarFrame` 或点云。

## 快速启动 Replay Demo

在本目录执行：

```powershell
python -m pip install -r requirements.txt
python -m uvicorn radar_module.service.radar_api:app --env-file .env --host 127.0.0.1 --port 8010
```

启动循环回放：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8010/api/radar/replay `
  -ContentType application/json `
  -Body '{"file_name":"demo_session.jsonl","speed":1.0,"loop":true}'
```

读取状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8010/health
Invoke-RestMethod http://127.0.0.1:8010/api/radar/latest
```

停止：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8010/api/radar/stop
```

默认 checkpoint 不存在时会确定性生成：

```text
checkpoints/radar_lstm_test.pt
model_mode=TEST_CHECKPOINT
```

它不是训练产物。固定权重仅让满 30 帧后的风险分数稳定，便于验证 HTTP
和事件路由。

## JSONL格式

每行一个已经标准化的雷达帧：

```json
{"timestamp":"2026-07-24T12:30:05.123+08:00","device_id":"iwr6843isk-01","room":"living_room","points":[{"x":0.1,"y":1.8,"z":1.2,"velocity":-0.2}]}
```

回放只允许访问 `RADAR_REPLAY_ROOT` 下的 `.jsonl` 文件，绝对路径和目录
穿越会被拒绝。REAL 与 REPLAY 数据源互斥。

## 将mmFall DS1转换为Replay JSONL

mmFall的DS1原始点不是`[x, y, z, velocity]`。每个点共有15列，
其中第9至12列依次是`range、azimuth、elevation、Doppler`。
转换器按照mmFall仓库自身的坐标算法生成笛卡尔坐标，并沿用其
`-10°`安装倾角与`1.8 m`雷达高度设置：

```powershell
python -m radar_module.dataset.mmfall_converter `
  --input "data/external/mmfall/data/DS1/DS1_4falls.npy" `
  --output "data/replay/mmfall_ds1_4falls.jsonl" `
  --device-id "mmfall-ds1-falls" `
  --room "living_room"

python -m radar_module.dataset.mmfall_converter `
  --input "data/external/mmfall/data/DS1/DS1_4normal.npy" `
  --output "data/replay/mmfall_ds1_4normal.jsonl" `
  --device-id "mmfall-ds1-normal" `
  --room "living_room"
```

每次转换还会生成相邻的`.manifest.json`，记录源文件SHA-256、
帧数、点数和坐标转换参数。mmFall使用NumPy对象数组，因此转换时
必须使用`allow_pickle=True`；只应转换来自可信来源的文件。

转换结果可以直接交给现有回放接口：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8010/api/radar/replay `
  -ContentType application/json `
  -Body '{"file_name":"mmfall_ds1_4falls.jsonl","speed":10.0,"loop":false}'
```

DS1只适合验证数据转换、V1特征提取和风险推理链路，不能据此证明
当前测试checkpoint具备真实跌倒预测能力。

## 研究版DS2跌倒前预测头

已下载 DS2 的24个分动作文件。导出器只把官方跌倒动作锚点前
0.2--1.5秒作为弱正样本，把同传感器的弯腰、下蹲、跳跃和坐地录制作为
困难负样本，并先按完整录制切分再滑窗：

```powershell
python -m radar_module.dataset.mmfall_research_v2 `
  --ds2-directory "data/external/mmfall/data/DS2" `
  --output "data/processed/mmfall_ds2_research_v2.npz" `
  --allow-weak-supervision

E:\python3.10.9aaa\python.exe -m radar_module.model.research_training_v2 `
  --dataset "data/processed/mmfall_ds2_research_v2.npz" `
  --checkpoint "checkpoints/radar_lstm_research_weak_v2.pt"
```

该 checkpoint 只训练 `pre_fall_score`。动作风险仍由规则计算，展示层保持
`fall_risk_score >= pre_fall_score`；两项分开展示。checkpoint 写有
`RESEARCH_WEAK_SUPERVISION`、归一化参数、特征顺序、数据校验值和
`deployment_eligible=false`，现有正式模型加载器不会接受它。

RadHAR 下蹲/跳跃困难负样本及跨传感器压力测试可复现为：

```powershell
python -m radar_module.dataset.radhar_converter `
  --data-directory "data/external/radhar/Data" `
  --output "data/processed/radhar_squat_jump_hard_negatives_v2.npz"

E:\python3.10.9aaa\python.exe -m radar_module.model.research_evaluation_v2 `
  --checkpoint "checkpoints/radar_lstm_research_weak_v2.pt" `
  --dataset "data/processed/radhar_squat_jump_hard_negatives_v2.npz" `
  --report "checkpoints/radar_lstm_research_weak_v2.radhar_report.json"
```

当前4200个 DS2 窗口上的验证 AUROC 为0.781；3190个 RadHAR 下蹲/跳跃
窗口在验证阈值下未触发误报。后一个结果很可能受到跨传感器域差异影响，
不能替代本机雷达上的每小时误报、逐事件召回和真实预警提前量测试。

离线影子推理不会调用事件路由或告警接口：

```powershell
E:\python3.10.9aaa\python.exe -m radar_module.inference.research_shadow_v2 `
  --replay "data/replay/mmfall_ds1_4normal.jsonl" `
  --checkpoint "checkpoints/radar_lstm_research_weak_v2.pt" `
  --output "data/shadow/mmfall_ds1_4normal.shadow.jsonl" `
  --recording-semantics normal
```

首轮外部影子测试没有通过可用性门槛：DS1正常动作104.2秒中产生12段
连续高分，约414.6段/小时，579/1024个窗口超过阈值；DS1跌倒片段和正常
片段的分数也明显重叠。因此研究 checkpoint 不得接入告警。RadHAR零误报与
DS1高误报并不矛盾，组合起来说明跨数据集域偏移非常明显。

## DGUHA 预测弱标签与多源困难负样本（2026-08-07）

DGUHA 官方包已完整校验并转换。模型输入始终只有雷达点云；Kinect 骨架只在
离线构建阶段定位全身持续下降起点，绝不进入特征或在线推理。正窗口结束于
下降起点前 0.1--0.6 秒；因起始历史不足等原因，110 段前向跌倒中只有 62 段
可用于该定义。19 名受试者严格隔离为训练/验证/测试，最终得到 6880 个窗口
（304 正、6576 负）：

```powershell
python -m radar_module.dataset.dguha_research_v2 `
  --data-directory "data/external/dguha/raw" `
  --output "data/processed/dguha_research_v2.npz" `
  --allow-skeleton-pseudolabels

E:\python3.10.9aaa\python.exe -m radar_module.model.research_training_v2 `
  --dataset "data/processed/dguha_research_v2.npz" `
  --checkpoint "checkpoints/radar_lstm_research_dguha_v2.pt" `
  --epochs 30
```

DGUHA 受试者隔离测试为：灵敏度 0.845、特异度 0.847、AUROC 0.928。
这些数字只说明健康年轻受试者、单一前向跌倒协议中的骨架伪标签可分，不能
外推为老人、真实失衡或本机雷达的预测能力。

mmRadPose 完整发布包也已校验。实际文件结构为 p1--p12、431 个
`targetlist_64.npy`（论文名义数量 432）；点云列顺序为
`x/y/z/velocity/SNR/noise/intensity`，15 Hz。转换器丢弃全零填充、保留真正
SNR，并按受试者切分。它没有跌倒，只能作为下蹲、弓步、躯干前屈等困难
负样本：

```powershell
python -m radar_module.dataset.mmradpose_converter `
  --pointcloud-directory "data/external/mmradpose/full_extract/mmRadPose_pointclouds" `
  --output "data/processed/mmradpose_hard_negatives_v2.npz"

python -m radar_module.dataset.composite_research_v2 `
  --mmfall "data/processed/mmfall_ds2_research_v2.npz" `
  --dguha "data/processed/dguha_research_v2.npz" `
  --mmradpose "data/processed/mmradpose_hard_negatives_v2.npz" `
  --radhar "data/processed/radhar_squat_jump_hard_negatives_v2.npz" `
  --output "data/processed/radar_composite_research_v2.npz"
```

组合产物有 17,367 个窗口，其中 mmRadPose 只抽取训练受试者的 4000 个负
窗口，RadHAR 只抽取 `external_train_pool` 的 2287 个负窗口；两者留出集没有
混入训练。直接混训失败：未均衡模型在 DGUHA 测试正样本上灵敏度为 0，
来源×标签均衡模型虽恢复部分灵敏度，但在 104.2 秒 DS1 正常回放中仍产生
15 段确认高分（约 518.2 段/小时）。因此组合 checkpoint 只保留作失败实验，
不得接入告警。

DGUHA 单域 checkpoint 在 mmRadPose 上窗口误报率 1.28%，在 RadHAR
下蹲/跳跃上为 13.86%；在 DS1 正常回放上 0 段确认误报，但在 DS1 含跌倒
回放上也 0 段确认预测。这证明当前主要阻塞是跨设备/协议域偏移，而不是已经
获得通用预测器。所有研究 checkpoint 都带有
`deployment_eligible=false`，影子推理也保持告警抑制。

### IWR6843短序列辅助实验（2026-08-08）

公开 IWR6843 数据已完整适配为 `radar_features_v2`。102段记录严格按三名
受试者轮换训练/验证/测试；每段只有23--25帧且没有跌倒时刻，因此模型任务
明确为末端2秒“跌倒过程/非跌倒”辅助分类，不是跌倒前预测：

```powershell
E:\python3.10.9aaa\python.exe -m radar_module.dataset.iwr6843_fall_v1 `
  --source "data/external/iwr6843_fall_102/mmwave-radar-fall-detection-main" `
  --output "data/processed/iwr6843_fall_sequence_auxiliary_v1.npz"

E:\python3.10.9aaa\python.exe -m radar_module.model.iwr6843_fall_auxiliary_v1 `
  --source "data/external/iwr6843_fall_102/mmwave-radar-fall-detection-main" `
  --output-directory "checkpoints/iwr6843_fall_auxiliary_loso_v1"
```

三折汇总灵敏度0.902、特异度0.451、平衡准确率0.676、宏平均AUROC 0.802。
不同测试受试者的平衡准确率为0.912、0.500和0.618，跨人波动明显。辅助
checkpoint使用独立模型模式，实时预测加载器不会接受它。

组合训练只加入固定训练受试者的17段 bow/squat/walk 负样本，排除全部51段
无时刻标签的跌倒记录。新组合模型窗口测试平衡准确率0.726，低于旧组合
基线0.734；DGUHA连续测试命中0/12个下降前事件，因此只保留为失败实验，
不替换当前 `dguha_prefall_dense_pw32_v2` 研究影子基线。

外部困难负样本复测示例：

```powershell
E:\python3.10.9aaa\python.exe -m radar_module.model.research_evaluation_v2 `
  --checkpoint "checkpoints/radar_lstm_research_dguha_v2.pt" `
  --dataset "data/processed/mmradpose_hard_negatives_v2.npz" `
  --report "checkpoints/radar_lstm_research_dguha_v2.mmradpose_report.json"
```

### 严格下降前预测候选与实时影子显示

已淘汰的 1--2 秒候选把正标签定义为骨架达到近地面状态前 1--2 秒，并强制所有正窗口仍
结束在明显下降开始至少 0.1 秒之前。正常动作改为 0.2 秒步长、每段最多
100 个窗口，共生成 58,933 个窗口（139 正、58,794 负）：

```powershell
python -m radar_module.dataset.dguha_research_v2 `
  --data-root "data/external/dguha/raw" `
  --output "data/processed/dguha_nearfloor_1to2_dense_v2.npz" `
  --allow-skeleton-pseudolabels `
  --positive-anchor near_floor_level_reached `
  --minimum-lead-seconds 1.0 `
  --maximum-lead-seconds 2.0 `
  --minimum-pre-descent-margin-seconds 0.1 `
  --negative-stride-seconds 0.2 `
  --max-negative-windows-per-recording 100
```

该专门候选在连续测试中只命中 1/12 个事件，正常确认误报约 66.1 次/小时；
原 DGUHA 候选按相同近地面前 1--2 秒标准命中 6/12，但误报约 304.6 次/小时。
阈值扫描没有找到可用折中，所以两者均不可作为正式预测模型。

当前研究默认改为严格的“全身下降起点前 0.1--0.6 秒”候选。数据集包含
59,098 个窗口（304 正、58,794 负），所有正窗结束于下降起点前至少 0.1 秒，
Kinect 骨架只用于离线时间标注。正类损失权重上限 32 的候选在受试者独立
测试回放中命中 10/12 个事件（83.3%），中位提前 0.327 秒；正常确认误报
约 190.2 次/小时，相比旧基线约 304.6 次/小时下降 37.6%。它仍不满足真实
部署要求，只能作为研究影子模型。

Radar FastAPI 可把该候选作为研究影子输出附加到既有 `/api/radar/latest`，
不会改变 V1 结果或触发正式告警：

```text
RADAR_RESEARCH_SHADOW_ENABLED=true
RADAR_RESEARCH_CHECKPOINT_PATH=./checkpoints/radar_lstm_research_dguha_prefall_dense_pw32_v2.pt
RADAR_RESEARCH_CONFIRMATION_WINDOWS=3
```

返回值的 `research` 字段分别包含 `pre_fall_score`、`fall_risk_score`、
`fall_risk_score_5s`、`prediction_state`、数据质量和预测区间，并固定带有
`shadow_only=true`、`alert_suppressed=true`。快速下蹲等动作可提高动作风险，
但不会被改成跌倒预测正类。

## TCN 实时影子推理 v1

最终因果 TCN 通过独立的 `radar_module/inference/tcn_live_v1.py` 运行，不复用
旧 `RadarLSTM` 实时类，也不改变模型和训练代码。启动时必须同时通过 checkpoint
契约、19 维特征顺序和 SHA256 校验。实时状态为 `UNKNOWN/NORMAL/WATCH/IMMINENT`；
数据不足或时间缺口超限时输出 `UNKNOWN`，不会沿用旧分数。

```text
RADAR_RESEARCH_SHADOW_ENABLED=false
RADAR_TCN_SHADOW_ENABLED=true
RADAR_TCN_CHECKPOINT_PATH=./checkpoints/experiments_v5/tcn_hard_negative/tcn_0p5_1p0_specificity_operating_point_v1.pt
RADAR_TCN_CHECKPOINT_SHA256=0792a712b57ae89875b2d57e6ba7a20763618a2718e961cf8c48acebe34970ef
RADAR_TCN_CONFIRMATION_WINDOWS=3
```

启用后，`/api/radar/latest` 只返回 `tcn_prediction`，不混合旧规则或研究头分数。
输出固定包含模型版本、checkpoint SHA256、阈值、风险状态、输入时间戳，且
`shadow_only=true`、`alert_suppressed=true`。离线逐帧/批量一致性检查命令：

```powershell
python -m radar_module.inference.tcn_replay_validation_v1 `
  --replay data/replay/demo_session.jsonl `
  --checkpoint checkpoints/experiments_v5/tcn_hard_negative/tcn_0p5_1p0_specificity_operating_point_v1.pt `
  --sha256 0792a712b57ae89875b2d57e6ba7a20763618a2718e961cf8c48acebe34970ef `
  --output reports/tcn_live_v1/demo_session.results.jsonl
```

真实动作、稳定性和 `UNKNOWN` 归因流程见
[`REAL_SCENE_VALIDATION_V1.md`](REAL_SCENE_VALIDATION_V1.md)。

## TI官方输出接入

`TiOfficialOutputAdapter` 不假设 TI parser 能作为 Python 包导入，支持：

1. 官方 Demo/导出层提供的 Python callback；
2. 启动官方或桥接进程，从 stdout 逐行接收已解码 JSON。

当前已接入 Radar Toolbox 2.20.00.05 的官方 `UARTParser`。桥接进程会：

1. 通过 COM5 (115200) 发送 `ISK_6m_default.cfg`；
2. 通过 COM6 (921600) 调用 TI parser 读取 People Tracking 帧；
3. 把官方解码后的 `pointCloud[:, 0:4]` 转为
   `x/y/z/velocity` JSONL；
4. 透传官方点云已提供的可选 `SNR` 与短时 `trackIndex`（255 表示未关联）；
5. 交给现有 `TiOfficialOutputAdapter -> TiRadarReader` 链路。

安装 REAL 模式依赖：

```powershell
python -m pip install -r requirements.txt
```

在 `.env` 中配置（具体 Python 路径按本机调整）：

```text
RADAR_ROOM=bathroom
TI_OFFICIAL_OUTPUT_COMMAND_JSON=["E:\\python3.10.9aaa\\python.exe","-m","radar_module.acquisition.ti_official_bridge","--cli-port","COM5","--data-port","COM6"]
TI_OFFICIAL_OUTPUT_CWD=E:\创新实践\老人摔倒预警\雷达模块
```

桥接输出只需包含 `timestamp` 与 `points`。点字段支持：

- `x/y/z/velocity`；
- TI输出常见别名 `posX/posY/posZ/doppler`；
- `[x, y, z, velocity]` 数组。

点还可带可选 `snr` 和会话内匿名 `track_id`。它们不改变现有 v1/v2
特征顺序，也不会让旧 checkpoint 失效；当前 v2 仍按房间聚合。后续只有在
短时轨迹连续性通过实测后，才会按 `track_id` 分组运行多人预测。

然后调用：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8010/api/radar/real
```

每次板卡断电后不需要重新烧录固件，但需重新发送 `.cfg`。REAL
桥接进程会在启动时自动完成这一步。启动前不要让 Industrial
Visualizer、UniFlash 或其他串口工具占用 COM5/COM6。

Windows 实机验证要求雷达 FastAPI 在独立终端以前台命令运行。不要使用
`Start-Process -WindowStyle Hidden` 启动 8010：本机 CP2105 在该隐藏后台
派生路径下会出现 CLI 命令无响应，而同一代码和配置在前台托管时可稳定进入
`source_mode=REAL`。该结论定位到进程/串口会话层，不代表已确定是 DTR/RTS、
Windows 句柄继承或驱动时序中的哪一个底层机制。

该边界只消费“已经解码”的输出，不实现 UART、TLV、CFAR 或 ADC 处理。

## V1特征与checkpoint兼容性

固定八维顺序：

1. `centroid_x`
2. `centroid_y`
3. `centroid_z`
4. `height_range`
5. `mean_velocity`
6. `max_abs_velocity`
7. `velocity_std`
8. `point_count`

这些特征仅用于 MVP 链路验证。P2 新增 `height_delta`、
`vertical_velocity` 等特征时必须建立 `radar_features_v2`，并创建匹配
特征版本、字段顺序、窗口长度的新 checkpoint。不兼容 checkpoint 会拒绝
加载。

## 萤石后端接入

`radar_module/integration/ezviz_backend.py` 提供两类适配器：

- `RadarServiceDataSourceAdapter`：Radar HTTP → `UnifiedDataPacket(RADAR)`；
- `RadarRiskAdapter`：Packet → `AlgorithmFinding`。

它复用现有萤石原型的契约，不更改 RiskEvent 结构。事件路由为：

| 模型/数据源 | 事件 |
|---|---|
| TEST_CHECKPOINT + 任意来源 | `RADAR_DEMO_RISK` |
| TRAINED_CHECKPOINT + REPLAY | `RADAR_REPLAY_RISK` |
| TRAINED_CHECKPOINT + REAL | 默认不生成正式事件 |

只有完成 P2 验证并显式设置 `allow_formal_predictions=True`，才允许
`TRAINED_CHECKPOINT + REAL` 生成 `PRE_FALL_RISK`。

服务不可达、雷达未连接、结果时间戳未更新时，数据源适配器返回 `None`，
不会复用旧风险。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖 Adapter 可替换性、字段清洗、FeatureVector 边界、窗口重置、
checkpoint版本拒绝、logit/Sigmoid分层、DEMO事件路由、路径安全和 Replay
HTTP闭环。
