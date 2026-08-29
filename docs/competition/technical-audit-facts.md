# 比赛提交材料最终技术审计事实清单

> 审计口径：以当前仓库中的可执行代码、依赖锁、Compose/Dockerfile、Gradle、配置类、启动脚本、数据库模型、Android Retrofit 调用及现有运行产物为准。README 和历史设计稿仅用于定位，不作为功能成立依据。本文不记录任何真实密钥、口令、设备验证码、设备序列号或个人信息。

## 1. 运行状态口径

| 状态 | 判定标准 |
|---|---|
| 当前默认运行链路 | 执行仓库根目录 `docker compose up` 会自动启动的服务，或当前客户端必经且无需额外功能开关的链路 |
| 已实现但需手动开启 | 已有可执行入口和接口，但 Compose 不负责启动，或必须配置开关、外部模型/硬件后手动运行 |
| Shadow / 实验链路 | 代码明确标记为 shadow、suppressed、formal=false、affects_alerts=false，或仅写实验产物而不触发正式事件 |
| 仅有代码但当前未接入 | 有模块或接口，但当前配置、Android 调用、数据桥接或启动链路均未使用 |

## 2. 已确认真实实现

### 2.1 默认可自动启动的基础链路

1. 根目录 Compose **只编排 PostgreSQL 与主 FastAPI 后端**。PostgreSQL 使用 `postgres:16-alpine`；后端映射 `${TZB_BACKEND_PORT:-8000}:8000`，启动前自动执行 Alembic 升级，并可按 `APP_SEED_DEMO` 写入演示账户、家庭绑定与设备。证据：`compose.yaml:1`、`docker/backend-entrypoint.sh:1`。
2. 主后端容器基于 Python 3.12，使用 `uv` 锁定依赖；系统 FFmpeg 由 Debian 包管理器安装但版本未固定。证据：`docker/backend.Dockerfile:1`、`backend/pyproject.toml:1`、`backend/uv.lock`。
3. 主后端入口为 `backend/app/main.py`，API 前缀 `/api/v1`，默认端口 8000，文档入口 `/docs`。启动时若数据库启用会先检查连接。证据：`backend/app/main.py:1`、`backend/app/core/config.py:10`。
4. 当前代码默认值中，数据库、跌倒外部源、心理源、萤石媒体、SenseVoice、流式 ASR、LLM、萤石报警轮询均为关闭；Compose 会显式开启 PostgreSQL 和演示数据。根目录本机 `.env` 已打开跌倒、心理、App 音频中继、SenseVoice 与流式 ASR，但这些外部 AI 进程仍不会被 Compose 自动启动。证据：`backend/app/core/config.py:17`、`:31`、`:39`、`:55`、`:63`、`:68`、`:89`、`:98`，`compose.yaml:23`、`:39`，以及根目录 `.env` 的布尔开关（未记录敏感值）。

### 2.2 Android 家属端

1. Android 工程位于 `android/`，入口 Activity 为 `android/app/src/main/java/com/tzb/safeguard/MainActivity.kt`，应用导航在 `ui/navigation/NavGraph.kt`。
2. 已核实构建要求：Gradle Wrapper 9.6.1、Android Gradle Plugin 9.3.1、Kotlin 2.4.10、JDK 17、compileSdk/targetSdk 37、minSdk 26。证据：`android/gradle/wrapper/gradle-wrapper.properties:3`、`android/build.gradle.kts:6`、`:11`，`android/app/build.gradle.kts:13`、`:17`、`:18`、`:49`。
3. 后端地址来自 Gradle 属性 `API_BASE_URL`，未指定时为 `http://127.0.0.1:8000/`；WebSocket 由同一 Base URL 推导为 `/api/v1/ws/events`。证据：`android/app/build.gradle.kts:8`、`android/app/src/main/java/com/tzb/safeguard/data/network/NetworkModule.kt`、`android/app/src/main/java/com/tzb/safeguard/data/realtime/AlertWebSocketClient.kt`。
4. 萤石 SDK 为 `io.github.ezviz-open:ezviz-sdk:5.30.2`。App 从后端 `/devices/{id}/live-sdk-session` 获取 `app_key`、短期 `access_token`、`device_serial` 与 `channel_no`，然后在 `Ys7SdkRuntime.kt` 初始化；AppSecret 不下发到客户端。证据：`android/app/build.gradle.kts`、`android/app/src/main/java/com/tzb/safeguard/data/media/Ys7SdkRuntime.kt`、`backend/app/api/v1/routes/app_client.py:238`。
5. 原生播放回调会输出 PCM；客户端转换/批处理为 16 kHz、单声道、16-bit，并按 1 秒、32000 字节分批上传 `/devices/{id}/audio-pcm`。证据：`android/app/src/main/java/com/tzb/safeguard/data/media/CameraAudioRelay.kt`。
6. 家属端实际页面包括首页/实时视频、跌倒概览、诈骗风险事件、Care 心理与认知评估（"抑郁风险评估"+"认知状态辅助评估"两张卡片）、个人页和事件处置。事件处置实际支持 `acknowledged`、`resolved`、`false_alarm`；“稍后提醒”接口存在但后端返回 501。诈骗列表页当前只筛选 `fraud_suspected`，不会展示跌倒事件。证据：`android/app/src/main/java/com/tzb/safeguard/ui/navigation/NavGraph.kt`、`android/app/src/main/java/com/tzb/safeguard/data/network/ApiService.kt`、`android/app/src/main/java/com/tzb/safeguard/ui/screens/care/CareScreen.kt`、`android/app/src/main/java/com/tzb/safeguard/data/psychology/model/CognitiveModels.kt`、`backend/app/api/v1/routes/app_client.py:187`。
7. Android debug 构建已从当前仓库执行通过，产物为 `android/app/build/outputs/apk/debug/app-debug.apk`。审计时命令：`cd android; .\gradlew.bat assembleDebug`。

### 2.3 主 FastAPI、PostgreSQL 与正式 RiskEvent

1. 主后端采用 FastAPI、Pydantic Settings、SQLAlchemy 2、Psycopg 3、Alembic、PostgreSQL。依赖版本由 `backend/uv.lock` 锁定。
2. PostgreSQL 持久化用户、会话、家庭绑定、设备、老人安全设置、萤石信号、视觉事件、模型运行、诈骗会话、正式风险事件、事件处置与投递状态。证据：`backend/app/infrastructure/database/models.py`、`backend/alembic/versions/`。
3. Android 使用的正式风险事件模型是 PostgreSQL `RiskEventModel`，事件类型包括诈骗、跌倒等，状态为 OPEN/ACKNOWLEDGED/FALSE_ALARM/RESOLVED。证据：`backend/app/infrastructure/database/models.py:555`、`backend/app/api/v1/routes/app_client.py`。
4. 正式诈骗事件在判定达到非 S0 状态后写入 PostgreSQL，并由进程内实时 Broker 推送给 WebSocket 订阅者。WebSocket ticket 和订阅状态是内存态，后端重启后失效。证据：`backend/app/modules/fraud/service.py`、`backend/app/infrastructure/realtime_events.py`、`backend/app/api/v1/routes/realtime.py`。
5. 当前主后端测试从当前仓库执行结果为 **213 passed、3 skipped**。该结果只证明后端测试集通过，不替代外部摄像头、雷达与萤石真机验收。

### 2.4 摄像头跌倒与 Camera–Radar 引擎

1. 多模态引擎位于 `backend/app/modules/fall/multimodal_engine/`，FastAPI 入口 `main.py`；仓库脚本 `scripts/start-multimodal-engine.ps1` 明确把它启动在 8001。该服务不在 Compose 中。
2. 当前主后端若设置 `APP_FALL_RISK_BASE_URL=http://host.docker.internal:8001`，客厅跌倒概览读取引擎的 `/api/multimodal/camera-led-associated/latest`；未配置时才回退读取本地雷达 JSON。证据：`backend/app/modules/fall/service.py`、`backend/app/core/config.py`。
3. 当前实时视觉工作器通过 Windows 萤石 PC OpenSDK 解码视频，按 15 Hz 时基建立 45 个有效姿态帧窗口；人体检测/姿态使用 RTMDet + RTMPose3D，风险模型为六折 BioSTGCN 集成。入口由多模态引擎以 `python -m fall_inference.opensdk_stream_worker` 启动。证据：`backend/app/modules/fall/multimodal_engine/services/fall_live_monitor.py:345`、外部视觉工程 `fall_inference/opensdk_stream_worker.py`。
4. 摄像头链路输出 CameraEvidence；引擎轮询雷达服务得到 RadarEvidence，执行时间/空间关联和 Radar Eligibility Gate，再进入 Camera-led Evidence Fusion v2。v2 结果字段 `affects_app_result=true`、`shadow_only=false`，是 Android 客厅跌倒页面的实际结果源；它当前不直接创建正式告警事件。证据：`backend/app/modules/fall/multimodal_engine/services/camera_led_evidence_fusion_v2.py`、`schemas/multimodal.py:564`、`api/multimodal.py:32`。
5. v2 以摄像头分数为主，雷达用于证据模式、支持/冲突状态与可解释性，不重写摄像头概率，也不否决摄像头高风险。该行为属于代码的真实融合边界，不应表述为双模态概率加权。证据同上。
6. 摄像头不会随多模态服务启动自动采集。App 个人页“开始守护”调用主后端 guard-session，再由多模态引擎启动摄像头工作器、请求雷达运行并开启融合参与。证据：`backend/app/api/v1/routes/guard_session.py`、`multimodal_engine/services/guard_session.py`、Android `ProfileScreen.kt`。
7. 跌倒正式事件由主后端（不是引擎）产生：`FallAlertController`（`app/modules/fall/fall_alerts.py`）在 `/api/v1/fall-risk/overview` 每次轮询时观察客厅 camera-led 状态，进入 high/critical 的边沿写 `FALL_SUSPECTED` RiskEvent 并经 WS 推送；一次幕次一条、60 秒冷却、状态回落自动重新武装。引擎侧 `affects_alerts=false` 与 MySQL 事件关闭的表述不变。

### 2.5 IWR6843ISK 雷达链路

1. 雷达模块位于 `backend/app/modules/fall/radar_module/`；HTTP 服务入口是 `service/radar_api.py`，多模态引擎默认访问 `http://127.0.0.1:8010`。它不在 Compose 中，也没有仓库一键启动脚本。
2. 实时串口桥 `acquisition/ti_official_bridge.py` 依赖 TI Radar Toolbox 官方 `gui_parser.py`/`UARTParser`，使用 IWR6843ISK 的 CLI/Data 双串口，将 3D People Tracking 的点云、trackIndexes、trackData 转为 JSONL。默认示例为 COM5/COM6，但实际端口由 `TI_OFFICIAL_OUTPUT_COMMAND_JSON` 指定，不能照抄默认值。证据：`radar_module/acquisition/ti_official_bridge.py`、`radar_module/.env.example`。
3. 代码期待 TI Radar Toolbox 2.20.00.05 下的 `3D_People_Tracking/chirp_configs/ISK_6m_default.cfg`；当前仓库未包含 Toolbox、`gui_parser.py` 或该 cfg，需按 TI 许可单独准备。证据同上。
4. 当前 calibrated causal TCN 使用 `radar_module/checkpoints/experiments_v5/tcn_hard_negative/tcn_0p5_1p0_specificity_operating_point_v1.pt` 与 `radar_module/reports/domain_calibration_v1_full/calibrated_normalization_real_gaussian.json`，两者已在仓库。服务也能生成/使用测试 checkpoint，但测试模型不应进入比赛正式部署主流程。
5. 审计时雷达 API 可独立启动，`/health` 返回 `model_loaded=true`；未连接数据源时 `radar_connected=false`，`/api/radar/latest` 返回 503。这是正确的未采集状态，不代表已收到雷达数据。
6. 有效雷达证据必须满足 `score_valid=true`、数据质量通过、轨迹/点数/稳定性门限以及时间空间关联。`INSUFFICIENT_DATA`、WARMUP、时间间隙、点数不足会使 Eligibility Gate 拒绝雷达参与。

### 2.6 诈骗识别、ASR 与正式事件链路

1. 当前演示的真实入口是 **Android EZOpenSDK PCM → 主后端 `/devices/{id}/audio-pcm`**，不是服务端 HLS/RTMP 拉流。当前配置的 `APP_YS7_MEDIA_SOURCE=app_relay` 与此一致。
2. 后端把 PCM 切成 20 ms 帧，经 WebRTC VAD 形成语音段：启动阈值 200 ms、静音结束 700 ms、前后缓冲 300 ms、最长语音段 10 s、强制切段重叠 1 s。证据：`backend/app/modules/fraud/voice_activity.py`、`backend/app/workers/ys7_media_stream_worker.py`、`backend/app/core/config.py`。
3. 流式部分结果使用 Paraformer Streaming；语句结束后由 SenseVoiceSmall 生成最终文本。两者均为 FunASR/ModelScope 模型，按配置在运行时下载或从缓存加载，仓库未包含模型本体。证据：`backend/app/infrastructure/external/sensevoice/streaming.py`、`recognizer.py`、`backend/pyproject.toml`。
4. 文本和声学信号形成 Evidence，经轻量分类器、规则、状态机和可选 LLM 复核。轻量分类器不是外部 checkpoint，而是启动时使用 `fraud/data/train.jsonl` 训练的字符 TF-IDF + 校准 LogisticRegression；规则位于 `fraud/rules.json`。证据：`backend/app/modules/fraud/text_classifier.py`、`backend/app/modules/fraud/rules.json`、`backend/app/modules/fraud/fraud_state_machine.py`。
5. 正式事件阈值为 S2+：S1"待观察"多为环境噪声弱证据，只在会话内观察，不写 RiskEvent、不推送（`fraud/service.py` 两处 `state_index >= 2` 门控）。家属删除消息为软删除 `DELETE /api/v1/events/{event_id}`（RESOLVED + RETRACTED，保留审计动作），消息页含 `fall_suspected` 与 `fraud_suspected` 两类事件。
6. LLM 是 OpenAI-compatible 可选复核层；当前默认与本机配置均未开启，不是默认诈骗链路。语义模型、近期上下文和 preliminary 事件也默认关闭。

### 2.7 心理状态评估

1. 心理模块位于 `backend/app/modules/psychology/home_detection_pkg/`；批处理工作器入口 `service/psychology_worker_main.py`，不监听端口、不在 Compose 中。另有只读 FastAPI `service/api.py`，但主后端当前未使用它。
2. 当前主后端在启用心理功能且未配置外部 Base URL 时，直接读取 `home_out/latest/<sha256(subject_key)>.json`，通过 `/api/v1/psychology/overview` 返回 Care 页面。证据：`backend/app/modules/psychology/service.py`、`backend/app/api/v1/routes/psychology.py`。
3. 实际链路为：萤石视频 → OpenSDK HEVC 解码/临时 H.264 MP4 → OpenFace CSV → 多人轨迹启发式选择 → 视觉特征（关键点、凝视、头姿、AU）→ 两个 MCCL 模型 → XGBoost → PHQ-8 估计 → JSON snapshot。当前推理代码**没有音频融合**，也没有已注册身份的人脸识别；所谓“目标老人识别”实际是最长出现且运动较低轨迹的启发式选择。证据：`home_detection_pkg/service/psychology_worker_main.py`、`home_detection_pkg/scripts/extract_elderly.py`、`home_detection_pkg/scripts/mccl_home_inference.py`。
4. PHQ-8 风险映射为 `<10 no_risk`、`10–14 mild`、`15–19 moderate`、`>=20 severe`；结果被标记为辅助评估、非诊断。Care 页面优先显示当前结果，没有当前结果时显示 `latest_completed`。
5. MCCL 两个 checkpoint 与 XGBoost `pima.pickle.dat` 已在仓库；OpenFace Windows 包不在仓库。

### 2.8 认知障碍评估（新增功能，第二轮审计）

1. 位置与入口：新增独立包 `backend/app/modules/psychology/cognitive/`，文件为 `collector.py`、`worker.py`、`service.py`、`mapping.py`、`result_store.py`、`schemas.py`、`ports.py`。主后端在 `app/main.py:134` 构造 `CognitiveAudioCollector`、`app/main.py:145` 构造 `CognitiveOverviewService`，lifespan 内 `await cognitive_collector.start()/stop()`（`app/main.py:344`、`:348`）；MMSE 推理工作器入口是 `cognitive/worker.py` 的 `main()`（支持 `--once` 单任务模式），仓库无启动脚本、Compose 不启动，需以 `python -m app.modules.psychology.cognitive.worker` 方式从 `backend/` 目录手动运行。
2. 与心理模块关系：认知是 `app/modules/psychology/` 下的新增兄弟包，不共用 `home_detection_pkg` 的批处理工作器，不读写 `home_out` snapshot，OpenFace/MCCL/XGBoost 链路未改动。共用点只有三个：同一守护会话编排器 `GuardianSessionService`（`app/modules/guarding/service.py:57` start 时 attach、`:77` stop 时 detach）、同一个 Care 页面（Android `CareScreen.kt` 两张卡片并列）、同一个路由文件（`app/api/v1/routes/psychology.py:37` 新增 cognitive-overview）。
3. 输入：唯一模型输入是 Android 上传的 16 kHz/单声道/s16 PCM 侧信道。`POST /api/v1/devices/{id}/audio-pcm` 在推给诈骗 `AppPcmRelaySource` 之后同步调用 `cognitive_collector.push(...)`（`app/api/v1/routes/app_client.py:289`）；无视频、问卷或行为输入，也不依赖 SenseVoice/Paraformer（collector 自带 webrtcvad，`APP_COGNITIVE_VAD_MODE` 默认 2）。注意该路由被 `ys7_media_enabled` + `app_relay` 门控（`app_client.py:275`，否则 503），因此认知采集仍依赖 App 播放器 PCM 上传链路处于开启状态。采集参数：20 ms 帧 VAD 只保留有效语音帧（WAV 为通过帧的拼接，不是连续原音）；有效语音达到 target 120 s（下限 60 s，会话上限 30 min）即发布任务；guard stop 时不足 60 s 写 `insufficient_data`。
4. 处理流程：主后端进程内 asyncio 任务收集（队列 maxsize 8、满时丢最旧，`collector.py:223`）→ 达标后写 processing snapshot 并把 WAV + job manifest 原子发布到 `runtime/inbox`（先 WAV 后 manifest，`result_store.py:55`）→ worker 每 0.5 s 轮询，`os.replace` 移入 `processing`，librosa 以 16 kHz 载入 → wav2vec2 按 15 s 窗/10 s 步长滑窗回归（尾窗补零），logits 均值反标准化 `score = mean*7.1844 + 23.0280`（`worker.py:103`）→ 校验 0–30 → 写 completed snapshot，删除 WAV 与 manifest。失败码：`audio_missing`、`job_expired`（TTL 默认 1 h）、`inference_failed`、`score_out_of_range`、`insufficient_speech`。
5. 模型：`Wav2Vec2MmseRunner`（`worker.py:34`）用 HF `AutoModelForAudioClassification`（num_labels=1、problem_type="regression"）+ `AutoFeatureExtractor`，要求特征抽取器 sampling_rate=16000，否则启动报错。模型目录来自 `--model-dir` 或 `COGNITIVE_MODEL_DIR`（未配置则启动失败），帮助文本命名为 "wav2vec2_base_adress"；仓库不含该模型文件。`--device`/`COGNITIVE_DEVICE` 默认 cpu，请求 cuda 而 CUDA 不可用时直接报错。
6. 输出：worker 级 snapshot `CognitiveAssessmentSnapshot`（status、estimated_mmse_score、effective_speech_seconds、audio_window_count、window、failure_code 等，`schemas.py:17`）；App 契约 `CognitiveOverview`（`schemas.py:90`）字段为 `source_status/assessment_state`（processing/completed/failed/insufficient_data/unavailable）、`data_quality`（usable/limited/insufficient，完成结果 ≥120 s 为 usable）、`source_modality="voice_acoustic"`、`assessment_window`、`estimated_mmse_score`、`attention_level`（≥27 none、≥24 mild、≥18 moderate、否则 high，`mapping.py:113`）、`evidence_summary`、`guidance`、`updated_at`、`disclaimer`、`latest_completed`。completed 还要求有效语音 ≥60 s、window_count>0、分数有限且在 0–30（`mapping.py:94`）。
7. 存储：纯文件系统，无数据库表/迁移、无 RiskEvent、无 WebSocket 推送（`infrastructure/` 与 `alembic/` 无 cognitive 引用）。运行目录 `backend/app/modules/psychology/cognitive/runtime/`（`config.py:109` 默认），结构为 `latest/<sha256(subject_key)>.json`、`latest/completed/<sha256(subject_key)>.json`、`inbox/`、`processing/`；文件名哈希方式与心理模块一致，但 snapshot JSON 内含 subject_key 明文字段。
8. 主后端读取：`GET /api/v1/psychology/cognitive-overview`（bearer 鉴权、可选 elder_id，经 `AppClientService.resolve_elder` 取 `external_subject`，`app/api/v1/routes/psychology.py:37`），`CognitiveOverviewService` 读 latest、非 completed 时补 `latest_completed`。`APP_COGNITIVE_ENABLED=false` 时 service 持 None store，固定返回 unavailable overview（`service.py:22`）。
9. Android 已接入：`data/psychology/model/CognitiveModels.kt`、`PsychologyApi.kt:15` `@GET("api/v1/psychology/cognitive-overview")`、`PsychologyRepository.kt:30`、`CareViewModel.kt:43`、`CareScreen.kt:79` 认知卡片；DI 组装在 `SafeGuardApp.kt:34`。Care 页标题为“心理与认知评估”，认知卡片展示：当前状态、分析来源“语音声学特征”、completed 时“参考评估分数 x.x / 30（AI辅助MMSE估计）”、辅助关注程度、数据质量、最近评估时间；processing 时展示 `latest_completed` 上一轮结果；底部固定非诊断声明。
10. 不生成 RiskEvent：全链路不触碰 PostgreSQL，当前实际用途仅是 Care 页面的非诊断辅助观察展示；不触发告警、不进入 WebSocket 事件流。
11. 依赖与配置变化：collector 复用主后端已有直接依赖 `webrtcvad-wheels`（与诈骗 VAD 同包，`pyproject.toml:18`）；worker 运行时经 importlib 引 torch/transformers/librosa，三者都不是 pyproject 直接依赖（uv.lock 中 transformers 5.14.1、librosa 0.11.0 是 funasr/sensevoice extra 的传递依赖）；worker 代码避免 3.11+ API（`UTC_TZ` 注释）且留有 cpython-310 字节码，说明按 Python 3.10 兼容设计并曾在 3.10 环境运行。新增配置：`APP_COGNITIVE_ENABLED/RUNTIME_DIR/QUEUE_MAXSIZE/MIN_SPEECH_SECONDS/TARGET_SPEECH_SECONDS/MAX_SESSION_SECONDS/COOLDOWN_SECONDS/JOB_TTL_SECONDS/VAD_MODE`（`config.py:108`、`compose.yaml:94`、`.env.example:119`）与 worker 侧 `COGNITIVE_MODEL_DIR/COGNITIVE_DEVICE/COGNITIVE_RUNTIME_ROOT`（`worker.py:277`）。
12. 启动方式与既有结果文件：心理批处理工作器及其 `home_out` 结构未改动；Compose 仍是 postgres+backend 两个服务，backend 新增 APP_COGNITIVE_* 透传并把 `APP_COGNITIVE_RUNTIME_DIR` 固定为容器内路径（`compose.yaml:95`）；当前 `compose.yaml:122` 存在 `./backend/app:/app/backend/app` 本地开发绑定挂载，该容器内 runtime 目录因此实际落回宿主机仓库路径。根 `.env` 当前没有任何 `APP_COGNITIVE_*` 行，认知采集当前处于关闭状态（代码默认 false）。
13. 验证状态：仓库内单测覆盖 collector 5 项、worker 2 项（使用 FakeRunner，不含真实 wav2vec2 推理）、overview 映射 6 项、guard attach/detach（`test_guard_session.py:69`）、OpenAPI 契约路径与 bearer security（`test_app_client_api.py:36`、`:55`）；后端全套当前为 **236 passed、3 skipped**（此前审计 213+3）。Android 当前源码树 `assembleDebug` 通过（增量 up-to-date，产物含认知页面代码）。**未完成/未验证**：模型文件不在仓库且无获取说明文档；根 `.env` 未开启开关；`runtime/{inbox,latest,processing}` 均为空目录，无真实推理产物；无 worker 启动脚本；端到端（采集→推理→App 展示）真机证据不存在。`COGNITIVE_COOLDOWN_SECONDS`（默认 900 s）只被写入/清除、从未被读取（`collector.py:94`、`:163`、`:361`），不能表述为“15 分钟冷却已生效”；当前唯一的频控是“每个 subject 同时至多一个采集会话”（attach 冲突返回 False）。

### 2.9 萤石平台真实边界

1. 主后端集成萤石开放平台 token、直播地址和报警接口；Android 使用 EZOpenSDK 5.30.2 原生播放与 PCM 回调。
2. token 可由后端使用 AppKey/AppSecret 获取并缓存，也可由配置注入；遇到萤石 token 失效码会刷新重试。AppSecret 始终留在服务端。
3. 开放平台可返回 HLS、RTMP、FLV 地址；当前 Android 实时播放使用 EZOpenSDK 的 deviceSerial/channel，不经过后端转发视频。当前诈骗音频来源也是客户端 SDK PCM 回调。
4. 萤石平台负责设备接入、鉴权、直播/SDK 解码和可选平台报警信号；人体姿态、BioSTGCN、雷达 TCN、融合、诈骗识别、心理评估及 RiskEvent 均由本项目实现。

### 2.10 本次审计的可执行验证

1. `docker compose config --services` 从当前仓库解析为 `postgres`、`backend` 两项；审计时 Docker Desktop daemon 未运行，因此未把现成容器状态作为功能证据。
2. 主后端当前测试集执行结果为 236 passed、3 skipped（第二轮审计复核；第一轮为 213 passed、3 skipped，增量主要来自认知模块测试）。
3. Android `assembleDebug` 成功，APK 位于 `android/app/build/outputs/apk/debug/app-debug.apk`；第二轮审计增量复核通过，产物已包含认知页面代码。
4. Radar API 使用当前配置Python可在8010启动；无数据源时 `/health` 返回 `model_loaded=true, radar_connected=false`，`/api/radar/latest` 返回503。该服务随后已停止，未把空数据状态写成真机成功。
5. Camera/Radar/Psychology的历史 runtime 文件只作为产物结构与字段证据；完整真机演示仍须按正文第17节逐级重新验收。
6. 认知链路当前只有组件级单测、契约测试与 Android 构建证据；`cognitive/runtime` 为空目录、模型不在仓库、根 `.env` 未开启 `APP_COGNITIVE_ENABLED`，采集→推理→App 展示的端到端真机演示未验证。

## 3. 已实现但非默认运行

| 能力 | 当前状态 | 启用条件与证据 |
|---|---|---|
| Camera–Radar 多模态服务 | 需单独启动 | `scripts/start-multimodal-engine.ps1`；端口 8001；需独立 Python 3.10 环境、外部视觉工程和模型 |
| 实时摄像头跌倒 | 需 App 手动开始守护 | 摄像头工作器由 guard-session 启动；需萤石 PC OpenSDK、CUDA、RTMDet/RTMPose3D/BioSTGCN 文件 |
| 雷达 HTTP 服务 | 需单独启动 | `radar_module/service/radar_api.py`，端口 8010 |
| IWR6843ISK 实时 UART | 需硬件和手动启用 | `ti_official_bridge.py`；需 CLI/Data 串口、TI Toolbox、cfg |
| 心理批处理工作器 | 需单独启动 | `home_detection_pkg/service/psychology_worker_main.py`；需 OpenFace、模型、萤石凭据 |
| 心理只读 HTTP API | 已实现、当前未作为主源 | `home_detection_pkg/service/api.py`；端口由启动命令决定，仓库未固定 |
| 认知采集 Collector | 已实现、当前关闭 | `APP_COGNITIVE_ENABLED=true` 时随主后端 lifespan 启动（`main.py:344`）；PCM 入口复用 `/devices/{id}/audio-pcm`（需 `app_relay`），需 guard-session attach；根 `.env` 当前未开启 |
| 认知 MMSE 工作器 | 需单独启动 | `python -m app.modules.psychology.cognitive.worker`；需外部 wav2vec2 模型目录（`COGNITIVE_MODEL_DIR`）与 torch/transformers/librosa 环境 |
| 萤石服务端云媒体拉流 | 已实现、当前未用 | `APP_YS7_MEDIA_SOURCE` 改为云拉流并安装 FFmpeg/模型；当前是 `app_relay` |
| 萤石报警轮询/Webhook | 已实现、默认关闭 | `APP_YS7_SIGNAL_ENABLED`/alarm polling；会写 signal inbox、visual events 和 raw JSON |
| LLM 诈骗复核 | 已实现、默认关闭 | 配置 OpenAI-compatible Base URL/model/key 并启用开关 |
| MySQL 原型持久化 | 引擎可用、当前正式链路未用 | 需手动部署 MySQL 和执行 `multimodal_engine/sql/*.sql`；当前事件写入开关关闭 |

## 4. Shadow / 实验链路

1. 多模态固定权重基线结果 `/api/multimodal/latest` 与 `runtime/fusion_shadow.jsonl` 是 shadow 观测，不是 Android 客厅跌倒结果。
2. temporal associated、associated risk augmentation 仍作为 shadow/比较链路存在，不进入 App 正式结果。
3. Radar calibrated causal TCN 当前输出字段明确包含 `shadow=true`、`suppressed=true`、`formal=false`；其作用是提供可审计 RadarEvidence，不会单独写正式告警。
4. Camera-led Evidence Fusion v2 本身不是 shadow：它 `affects_app_result=true`；但当前 `affects_alerts=false`，因此只能表述为 App 风险状态源，不能表述为已自动生成 PostgreSQL 跌倒 RiskEvent。
5. 现有 `fusion_shadow.jsonl` 最新样本中相机/雷达证据缺失、alignment 为 `CAMERA_PERSON_MISSING`、v2 为 UNKNOWN；这些文件证明运行链路曾写产物，不证明当前有完整 MATCHED 真机样本。
6. Psychology App contract 固定返回 `operating_mode="shadow"`。Care 页面确实读取并展示该结果，但它是非诊断辅助观察，不写正式 RiskEvent。

## 5. 仅有代码但当前未接入正式应用链路

1. 多模态引擎自己的 MySQL `risk_events` 与主后端 PostgreSQL `risk_events` **是两套不同模型**，当前无桥接：
   - PostgreSQL：`backend/app/infrastructure/database/models.py` 的 `RiskEventModel`，UUID/JSONB，供 Android/REST/WebSocket 使用；
   - MySQL：`multimodal_engine/database/models/risk_event.py` 的 `RiskEvent`，BIGINT/evidence_json，供原型 dashboard 使用。
2. 多模态引擎 MySQL 事件写入开关当前关闭，且 Android 从不读取该库。因此不能宣称两套事件已统一。
3. 心理 HTTP 子服务、老版文件/上传式跌倒推理、研究期雷达 LSTM/PointNet/下降检测代码、历史融合比较实现均不在当前 App 主链路。
4. Android 的 Mock 实现存在，但 `useMockBackend` 在当前 buildTypes 中均固定为 false，不是可在运行时切换的比赛模式。
5. App 存在历史回放与“稍后提醒”的客户端入口/接口定义，但当前后端返回 501；不应列为已部署功能。
6. 未发现 App 内新增/绑定萤石设备的完整 UI；设备由后端种子数据或数据库/管理流程配置，App 只读取已绑定老人与设备。
7. 未发现心理部署推理中的音频融合或注册身份人脸识别证据；训练/研究目录中的音频代码不能作为当前部署链路描述。
8. 未发现 Compose 自动启动 MySQL、雷达、多模态引擎、心理工作器、萤石 PC SDK 工作器的配置。
9. `prototype/` 只有静态 HTML/CSS 设计稿，没有独立构建配置、服务入口或当前主后端静态挂载；它不属于运行系统。`tmp/psych_openface_spike.py` 是开发验证脚本，`samples/` 只有占位文件，均不进入部署链路。
10. `radar_module/service/radar_worker_main.py` 是 replay-only 文件状态写入器，需要 `--replay-file`，不提供HTTP；当前配置通过8001→8010链路使用 `radar_api.py`，不会自动启动该worker。

## 6. 现有运行产物审计

| 数据 | 位置 | 写入者 | 读取者/用途 | 审计结论 |
|---|---|---|---|---|
| PostgreSQL | Compose volume `postgres_data` | 主后端 | REST/WebSocket/Android | 正式业务数据源 |
| MySQL 原型库 | `elder_risk_prototype`（配置默认） | 多模态引擎可选 | 原型 dashboard | 未由 Compose 部署，当前事件写入关闭 |
| Fusion JSONL | `backend/app/modules/fall/multimodal_engine/runtime/fusion_shadow.jsonl` | shadow sampler | 技术观测/复盘 | 非 App 正式结果源 |
| Radar state JSON | `backend/app/modules/fall/radar_module/runtime_state/*_latest.json` | radar worker/API | LocalRadarSource/排障 | 现有 bathroom 文件 payload room 为 living_room，且 INSUFFICIENT_DATA，不可作为有效浴室证据 |
| Psychology latest | `backend/app/modules/psychology/home_detection_pkg/home_out/latest/*.json` | 心理 worker | 主后端本地源 | Care 实际读取源；文件名哈希但内容仍可能含 subject_key |
| Cognitive runtime | `backend/app/modules/psychology/cognitive/runtime/`（`inbox/`、`processing/`、`latest/`、`latest/completed/`） | 认知 collector（任务 WAV+manifest）与 MMSE worker（snapshot） | 主后端 cognitive-overview 读取 `latest/` | 审计时为空目录，无真实推理产物；snapshot 内含 subject_key 明文 |
| Psychology logs | `home_detection_pkg/logs/`、`home_out/` | 心理 worker | 排障/审计 | 可能含设备、时间窗、评分等敏感信息 |
| 萤石 raw signal | `backend/storage/ys7/raw/` | 可选 signal worker | 信号审计 | 仅开启萤石信号链路后产生 |

运行产物中所有 camera/radar processing latency 字段当前为 null，仓库没有正式端到端告警延迟记录。开发机真机运行日志（2112 条，仓库外）统计的模块处理延迟：Camera P50 109 ms / P95 188 ms；Radar P50 2.66 ms / P95 16.54 ms；0.53 s 是模型新结果更新周期（45 帧窗、步长 8、15 Hz），不是计算耗时。端到端"动作→App"时延仍以拍摄当天实测为准，提交材料引用模块延迟时应注明日志来源与统计口径。

## 7. 模型与大型外部资源事实

| 资源 | 程序期待位置/配置来源 | 仓库状态 | 缺失行为 | 提交包建议 |
|---|---|---|---|---|
| BioSTGCN 15 Hz 六折权重 | 外部视觉工程 `outputs/biostgcn_15hz/deployment15hz_45f_v1/folds/unified_foldXX/stage2_15hz_best.pt` | 不在本仓库 | 摄像头工作器启动或预测失败 | 按项目许可单独附带六折权重和 manifest |
| RTMPose3D | 外部视觉工程 `data/pose_models/rtmw3d-...pth` | 不在本仓库 | 无法产生 3D Pose | 另包附带或给出合规下载说明 |
| RTMDet | 外部视觉工程 `data/pose_models/rtmdet-...pth` | 不在本仓库 | 无人体框，Pose 链路不可用 | 同上 |
| 雷达 causal TCN | `radar_module/checkpoints/experiments_v5/tcn_hard_negative/tcn_0p5_1p0_specificity_operating_point_v1.pt` | 已在仓库 | 配置到不存在路径时模型加载失败/回退测试模型 | 已包含；比赛配置必须显式指向该文件 |
| 雷达 normalization stats | `radar_module/reports/domain_calibration_v1_full/calibrated_normalization_real_gaussian.json` | 已在仓库 | 无法进行真实域校准 | 已包含 |
| Camera–Radar calibration | `multimodal_engine/calibrations/living_room_grid9_shadow_v0.json` | 已在仓库 | 空间关联退化/无法匹配 | 已包含；换场地应重新标定 |
| MCCL model 1/2 | `home_detection_pkg/checkpoint/DAIC/current_model1`、`current_model2` | 已在仓库 | PHQ-8 推理失败 | 已包含 |
| XGBoost | `home_detection_pkg/checkpoint/DAIC/pima.pickle.dat` | 已在仓库 | PHQ-8 回归失败 | 已包含 |
| wav2vec2 MMSE 模型 | `COGNITIVE_MODEL_DIR`/`--model-dir` 指定的 HF 格式目录（feature extractor 必须 16000 Hz） | 不在仓库 | worker 启动即 FileNotFoundError；cognitive-overview 不会出现 completed | 按模型许可单独附带目录与获取说明 |
| OpenFace | `OPENFACE_EXE`/`PSYCH_OPENFACE_EXE` 或 `--openface-exe` 指定的 `FeatureExtraction.exe` | 不在仓库；代码未给固定绝对路径 | 无法生成面部特征，assessment 失败 | 不分发本体时提供官方获取与安装说明 |
| SenseVoiceSmall | FunASR/ModelScope 模型 ID或本地缓存 | 不在仓库 | 最终 ASR unavailable，诈骗链路降级 | 演示机预下载缓存；按模型许可分发 |
| Paraformer Streaming | FunASR/ModelScope 模型 ID或本地缓存 | 不在仓库 | 无流式 partial；最终 SenseVoice仍可独立工作 | 演示机预下载缓存 |
| 萤石 PC OpenSDK | `FALL_LIVE_OPENSDK_DIR` | 不在仓库 | 摄像头跌倒工作器不能打开设备 | 按萤石 SDK 许可单独安装 |
| TI Radar Toolbox/GUI parser/cfg | `TI_OFFICIAL_OUTPUT_COMMAND_JSON` 指定 | 不在仓库 | UART bridge 不能启动或不能配置雷达 | 单独安装 TI 官方包，不复制禁止分发文件 |
| LLM | 环境变量中的 OpenAI-compatible 配置 | 无模型文件，当前关闭 | 不影响默认规则/分类链路 | 比赛非必需；不得附带密钥 |

## 8. 关键环境与版本审计

| 项目 | 真实要求/状态 |
|---|---|
| 主后端 OS | Docker Linux（Debian Trixie 基础镜像）；主机可为 Windows/Linux |
| 摄像头/雷达/心理当前工程环境 | Windows，原因是 PC OpenSDK Win64、OpenFace `.exe`、COM 串口与现有 PowerShell 脚本 |
| 主后端 Python | 项目要求 >=3.12；Docker固定 Python 3.12 |
| 视觉/雷达/心理 Python | 当前可用外部环境均为 Python 3.10.9；仓库未提供统一环境锁 |
| 认知 Python | collector 运行于主后端容器 Python 3.12（webrtcvad 已在依赖内）；MMSE worker 代码按 3.10 兼容编写（避免 `datetime.UTC`）并留有 cpython-310 字节码，运行需 torch/transformers/librosa |
| JDK | 17 |
| Android | Gradle 9.6.1、AGP 9.3.1、Kotlin 2.4.10、SDK 37、minSdk 26 |
| PostgreSQL | 16-alpine |
| MySQL | 仓库未固定版本 |
| 主后端 PyTorch | SenseVoice extra 锁定 2.11 CPU 系列 |
| 视觉 CUDA/PyTorch | 审计机环境 torch 2.1.2+cu121；仓库没有对所有部署机固定 CUDA/驱动版本 |
| 雷达/心理 PyTorch | 审计机外部环境 torch 2.13.0+cu126；仓库未提供可复建锁文件 |
| FFmpeg | Docker apt 安装，版本未固定；Windows 必须可执行或配置路径 |
| OpenFace | 路径约定为 2.2.0 Windows 包；需单独安装 |
| MMPose | 当前视觉环境 mmpose 1.3.2、mmdet 3.3.0、mmengine 0.10.7、mmcv 2.2.0 |
| TI | IWR6843ISK + CLI/Data 双串口 + Radar Toolbox 2.20.00.05 目录约定 |

## 9. 历史设计稿需按本审计修订的表述

对 `docs/competition/app-architecture-design.md` 的可复用内容应进行以下事实修订：

1. 跌倒和心理页面/API 已有真实实现，不再是纯占位；但外部采集进程需要手动启动。
2. Android 会收到萤石 AppKey 和 AccessToken，只有 AppSecret 不下发。
3. 登录令牌存于 App 私有 SharedPreferences，但未做额外加密，不能写成“加密安全存储”。
4. 监控 Service 为 START_STICKY，但没有发现冷启动时读取持久化开关并自动恢复的代码；比赛操作仍需手动“开始守护”。
5. Compose 不是完整 AI 系统一键部署，只覆盖 PostgreSQL 和主后端。
6. 当前诈骗事件页只展示诈骗类型，不能表述为全类型风险统一列表。

## 10. 不应写入比赛材料的结论

1. 不应宣称 Compose 自动启动雷达、视觉、融合、心理或 MySQL。
2. 不应宣称 Camera–Radar v2 已自动写入 PostgreSQL 跌倒事件。
3. 不应把 shadow baseline、temporal associated 或 Radar formal=false 结果描述为正式告警。
4. 不应把心理链路写成已有人脸身份识别或音频融合。
5. 不应把引擎的 50 ms 轮询间隔写成雷达帧率，也不应把模型单次耗时写成端到端告警延迟。
6. 不应宣称现有 runtime 文件已经证明 MATCHED；必须以实时字段验证。
7. 不应写入任何真实 AppKey、AppSecret、AccessToken、验证码、设备序列号、数据库口令、LLM Key、手机号、姓名、音频文本或心理评分明细。

## 11. 《04_部署与技术文档》修改影响清单（认知障碍新增功能）

以下按《04_部署与技术文档》现行章节给出因新增认知障碍功能需要的修改，每条附代码依据。原则：只改与新代码事实冲突或缺失的内容。

### 11.1 必须修改的位置与依据

| # | 04 文档章节 | 应修改内容 | 修改依据 |
|---|---|---|---|
| 1 | §1 文档说明/一键部署边界 | "手动启动"清单加入认知 MMSE 工作器；说明认知 Collector 是主后端进程内组件（开关开启时随 backend lifespan 启动），不是独立微服务 | `main.py:134`、`:344`、`:348`；compose 无认知 worker 服务 |
| 2 | §2.1 系统总体架构分层图 | AI分析层加一行 "Cognition：Android PCM 侧信道 → webrtcvad 有效语音累积(≥60 s) → wav2vec2 滑窗回归 → MMSE 0–30"；后端服务层加 "Cognitive MMSE worker（手动，无端口）"；Android 层 Care 行改为"Care 心理与认知评估" | `cognitive/collector.py`、`cognitive/worker.py`、`CareScreen.kt:68` |
| 3 | §2 新增"认知真实数据流"小节 | 排在 2.4 心理数据流后：App PCM → `/devices/{id}/audio-pcm` → collector(VAD, ≥60 s/目标 120 s/上限 30 min) → `runtime/inbox` WAV+manifest → worker wav2vec2(15 s 窗/10 s 步长, `mean*7.1844+23.0280`) → `latest/<sha256>.json` → `/api/v1/psychology/cognitive-overview` → Care 页；注明不依赖 ASR 模型、不写 RiskEvent、开关关闭时 App 显示"服务暂不可用" | `app_client.py:289`、`collector.py:254-345`、`worker.py:79-108`、`psychology.py:37`、`mapping.py:83` |
| 4 | §3.1 环境表 | 加"认知 Python"行：collector 在主后端容器 Python 3.12（webrtcvad 已在依赖）；worker 按 3.10 兼容编写，运行需 torch/transformers/librosa，三者非 pyproject 直接依赖（uv.lock 中为 funasr 传递依赖） | `worker.py:49-52`、`pyproject.toml:7-30`、uv.lock |
| 5 | §3.2 已验证命令 | pytest 结果更新为 236 passed、3 skipped | 本轮审计实际执行 |
| 6 | §4 项目目录表 | 加两行：认知采集 Collector（`backend/app/modules/psychology/cognitive/`，随主后端，8000 内）；认知 MMSE 工作器（`worker.py`，无端口，写 runtime 文件，手动 `python -m app.modules.psychology.cognitive.worker`） | `main.py`、`worker.py:292` |
| 7 | §5 配置说明 | 5.1 补"认知 worker 读 COGNITIVE_* 环境变量与命令行参数，不读根 .env"；新增 APP_COGNITIVE_* 九项（ENABLED/RUNTIME_DIR/QUEUE_MAXSIZE/MIN_SPEECH_SECONDS/TARGET_SPEECH_SECONDS/MAX_SESSION_SECONDS/COOLDOWN_SECONDS/JOB_TTL_SECONDS/VAD_MODE）与 worker 侧 COGNITIVE_MODEL_DIR（必填）/COGNITIVE_DEVICE/COGNITIVE_RUNTIME_ROOT；5.8 关键开关表加 Cognitive Collector 行（默认 false、当前未开启） | `config.py:108-118`、`compose.yaml:94-102`、`.env.example:119-131`、`worker.py:277-286` |
| 8 | §6.3 模型资源清单 | 加 wav2vec2 MMSE 模型行：HF 目录、feature extractor 必须 16000 Hz、不在仓库、缺失时 worker 启动 FileNotFoundError 且 overview 恒 unavailable | `worker.py:34-77` |
| 9 | §7 服务启动 | 新增"启动认知 MMSE 工作器"小节（cd backend；设 COGNITIVE_MODEL_DIR；`python -m app.modules.psychology.cognitive.worker --model-dir ... --device cpu`；验证 runtime/latest 与 cognitive-overview）；7.4/7.9 注明 `APP_COGNITIVE_ENABLED=true` 时 collector 随 backend 启动 | `worker.py:292-307`、`main.py:344` |
| 10 | §13 心理模块部署 | 标题/内容扩展为"心理与认知模块部署"或新增 13.x 认知部署：采集条件（guard attach、VAD mode 2、≥60 s 有效语音、120 s 目标、30 min 上限）、attention_level 映射（≥27 none/≥24 mild/≥18 moderate/else high）、data_quality（≥120 s usable）、四种非完成状态语义；并说明 cooldown 当前未生效（只写不读），不得写成"15 分钟冷却" | `collector.py`、`mapping.py:107-120`、`collector.py:94/163/361` |
| 11 | §14 Android 客户端 | 14.2 第 9 步改为：Care 页（标题"心理与认知评估"）含抑郁评估与认知评估两张卡片；认知卡片 completed 时显示"参考评估分数 x.x/30（AI辅助MMSE估计）"、辅助关注程度、数据质量、评估时间，processing 时显示上一轮结果；无独立认知页面 | `CareScreen.kt:68`、`:79`、`:129-206` |
| 12 | §15 API 与服务通信 | 15.1 表加 `GET /api/v1/psychology/cognitive-overview`（返回 `CognitiveOverview` 字段；Android 实际调用=是，Care 页进入/重载时调用）；15.3 通信图加 "Cognitive MMSE worker ─inbox/latest 文件─> 主后端 CognitiveOverviewService" | `psychology.py:37-56`、`PsychologyApi.kt:15` |
| 13 | §16 数据组织 | 16.2 文件表加 Cognitive runtime 行（inbox/processing/latest/latest/completed；snapshot 含 subject_key 明文）；补"认知评估不写任何 RiskEvent、不进 WebSocket" | `result_store.py:40-67`、`schemas.py:17`；infrastructure/alembic 无 cognitive 引用 |
| 14 | §17 功能验证 | 17.1 验收顺序加认知步骤；新增 17.5 认知逐级验收表：App PCM 上传 → collector 计数/processing snapshot → inbox 任务 → worker completed snapshot（score/windows/speech_seconds） → cognitive-overview 映射 → Care 卡片 | 全链路代码；`tests/unit/test_cognitive_*.py` |
| 15 | §18 排查表 | 加认知行：worker 未启动或模型缺失 → 恒 unavailable；有效语音 <60 s → `insufficient_data`；模型输出越界 → failed(`score_out_of_range`)；任务过期 → failed(`job_expired`) | `worker.py:186-221`、`mapping.py:43-66`、`service.py:22` |
| 16 | §19 安全与隐私 | 补充：认知链路会在 runtime 落盘拼接语音 WAV（inbox/processing，完成或 TTL 清扫后删除）与含 subject_key、MMSE 分数的 snapshot；语音与认知分数按高敏感数据处理 | `result_store.py:55-67`、`collector.py:373-381`、`schemas.py:17-32` |
| 17 | §20 附录速查表 | 20.1 端口表加"无端口 认知 MMSE worker，手动启动"；20.2 节奏表加认知参数：20 ms 帧、队列 8 丢最旧、min 60 s/目标 120 s/上限 30 min、job TTL 1 h、worker 轮询 0.5 s、15 s 窗/10 s 步长 | `collector.py:22-97`、`worker.py:40-45`、`config.py:108-118` |
| 18 | §8 萤石平台算法边界、§21 最短部署路径 | §8.1 算法边界句加"wav2vec2 MMSE 认知估计由本项目完成"；§21 步骤 2 增加 APP_COGNITIVE_ENABLED=true 与 COGNITIVE_MODEL_DIR，步骤 7 附近加 worker 启动，步骤 11 Care 描述加认知卡片 | `worker.py`、`config.py:108`、`CareScreen.kt` |

### 11.2 修订时的措辞红线（防过度声明）

1. 认知结果是"AI辅助认知状态评估/日常关怀参考"，不是认知障碍诊断或正式量表施测；App 契约固定非诊断声明（`mapping.py:15`）。
2. 不得把 900 s cooldown 写成已生效的"15 分钟冷却"；当前唯一频控是每 subject 至多一个活动采集会话。
3. 不得把采集音频写成"连续录音"；WAV 是 VAD 通过帧的拼接。
4. completed 结果要求 ≥60 s 有效语音且分数在 0–30，否则呈现 insufficient_data/failed，不得伪造分数。
5. 不得宣称 Compose 启动认知 worker；根 `.env` 未开启 `APP_COGNITIVE_ENABLED` 时，Care 认知卡片显示"服务暂不可用"（unavailable），不得截图声称已出分。
6. 仓库 runtime 为空、模型不在仓库：在完成真机端到端验证前，不得把认知功能写成"已验证可用"。
