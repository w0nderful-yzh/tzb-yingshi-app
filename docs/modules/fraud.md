# 防诈模块

时效性、准确性、PARTIAL 两级预警和 RTX 4060 部署的分阶段实现路径见[防诈时效性与准确性优化实施路线图](../architecture/fraud-latency-accuracy-optimization.md)。该路线图属于实施进度跟踪，未完成阶段（含 `[GPU·待同事]`、`[⏳ 条件未满足·暂缓]` 项）不得写成当前能力。

## 当前状态

当前最终方案是“音频主判、视觉增强、分层决策、独立会话”：FFmpeg 连续解码萤石音轨，WebRTC VAD 按自然停顿切句；讲话期间 Paraformer Streaming 每 600 ms 更新部分转写，停顿后 SenseVoiceSmall 对完整语句生成最终转写和富标签。部分结果只负责 S1-S2 早期判断，默认不产生家属通知（开启 `APP_FRAUD_PRELIMINARY_ALERT_ENABLED` 后，连续强动作命中可先发 PRELIMINARY 待确认提醒）；最终结果才允许进入 S3-S5。规则、校准轻量分类器和 S0-S5 状态机同步完成本地判断，S2-S4 可异步进入 LLM 复核；LLM 不可用时不影响本地链路。

视觉事件只提供通话、访客、人数和独处等场景事实，不从画面猜测诈骗语义。单次人员出现不升级状态；持续访客证据只有与强语音动作共同出现时才可增强升级。

## 已实现

- `phone_call`、`people_count`、`person_detected` 视觉事件；
- 事件发生时间和服务端接收时间分离；
- 原始消息保存、消息 ID 去重和后台消费；
- 按设备查询统一视觉事件；
- 按设备和会话维护分阶段证据窗口（contact 300s / probing 180s / action 120s / control 120s / protective 300s / context 120s，带时间衰减），支持语音片段乱序重放；
- 按 `source_event_id` 幂等接收转写片段；
- 按设备、会话和 `chunk_id` 幂等接收短 WAV 音频块；
- SenseVoiceSmall 懒加载、串行推理和异常隔离，推理不占用 FastAPI 事件循环；
- SenseVoice 相对毫秒自动转换为音频块 `started_at` 对应的绝对时间；
- 保留 SenseVoice 的语言、情绪和音频事件富标签，不再只保留清洗后的文字；
- 使用 AppKey/AppSecret 自动获取和缓存萤石 accessToken；
- 获取 FLV、RTMP 或 HLS 直播地址，FFmpeg 解码为 16 kHz 单声道 PCM；
- 媒体断线指数退避重连，重连时重新申请直播地址；
- 有界实时音频队列，积压时丢弃最旧块而不是无限增加告警延迟；
- WebRTC VAD 自适应切句，保留句首句尾并对超长语音执行重叠切分；
- Paraformer Streaming 按 600 ms 增量识别，维护逐话轮缓存、热词和确定性热词纠错；
- `PARTIAL` 转写使用稳定事件 ID 原位更新，最多推动到 S2，不落风险告警、不触发 LLM；
- SenseVoice 在自然停顿后产生 `FINAL`，替换同一部分证据并负责 S3-S5 最终确认；
- 按绝对时间和文本相似度去除重叠音频产生的重复转写；
- 第一段有效语音或新 `phone_call` 创建独立会话，30 秒静音或 10 分钟上限后切换会话；
- 活动会话写入 PostgreSQL `fraud_sessions`，支持重启恢复，并在新会话开始时关闭旧会话；
- 将 `phone_call` 转换为通话语境，将单次人数/人员事件作为不单独升级的场景事实；
- 持续 3 秒以上且至少两次的人员事件可形成访客在场上下文，但只有与强语音动作组合时才能升级；
- 组合确定性规则与 sigmoid 校准的字符 TF-IDF 分类器提取语音证据；
- 输出 S0-S5 状态、风险等级、判断建议、转换原因和完整证据链；
- 使用 OpenAI-compatible `/chat/completions` 接口进行可选文本与图片 LLM 复核；
- 多模态复核复用萤石事件的 `image_url`，按证据窗口筛选、去重并限制最多图片数；
- LLM 在独立有界队列中运行，不阻塞萤石取流、SenseVoice 和本地状态机响应；
- LLM 输出使用严格 JSON，证据必须逐字引用转写原文，无法定位的引用会被丢弃；
- LLM 证据强度上限为 `medium`，不能单独推动 S4/S5；服务超时或失败自动降级；
- 非 S0 结果按设备和会话幂等写入 PostgreSQL `risk_events`；
- 萤石接收功能关闭时，FastAPI 其他接口正常启动；
- 启动预热分类器与 ASR 模型（`APP_FRAUD_CLASSIFIER_WARMUP_ENABLED` / `APP_SENSEVOICE_WARMUP_ENABLED` / `APP_STREAMING_ASR_WARMUP_ENABLED`），失败自动降级懒加载，媒体状态接口暴露 `models_ready` / `classifier_ready` / `warmup_error`；
- PARTIAL 两级预警（默认关闭）：连续两次同 kind 强动作（`credential_request` / `remote_control_instruction` / `money_instruction`，融合置信度≥0.90）写 PRELIMINARY 事件（REMINDER + OPEN），FINAL 按 `state_index` 确认或系统撤回，`partial_stability` 随会话持久化保证重启幂等；
- 媒体分析拆分为 streaming 与 final 两个有界队列与独立消费任务，`FraudAudioService` 按会话/FINAL 块/转写历史拆分锁，积压丢弃按任务类型记录原因；
- VAD 参数可配置（`APP_YS7_VAD_SPEECH_START_MS` / `APP_YS7_VAD_SILENCE_END_MS` / `APP_STREAMING_CHUNK_MS`，强制 20ms 整数倍）；
- 分阶段证据窗口与时间衰减（contact 300s / probing 180s / action 120s / control 120s / protective 300s / context 120s），原始与衰减后置信度都保留在证据链；
- 固定评测集与离线评测脚本（`evaluate_fraud_model.py`）、FALSE_ALARM 脱敏反馈导出（`export_fraud_feedback.py`）；语义检索与近期风险画像端口就绪，默认关闭。

## 决策链

```text
萤石音轨 → 16 kHz PCM → WebRTC VAD
  ├─ 每 600 ms → Paraformer PARTIAL → 规则 + 校准分类器 → 最多 S2
  └─ 自然停顿 → SenseVoice FINAL → 规则 + 校准分类器 → S0-S5 状态机
                                             ├─ 合并视觉场景事实
                                             ├─ 写 fraud_sessions / risk_events
                                             └─ S2-S4 异步 LLM 复核
```

四层各自职责固定：

1. 规则层识别验证码、转账、屏幕共享、保密要求等高确定性证据；
2. 校准分类器补充口语变体，只输出证据分数，不直接报警；
3. S0-S5 状态机按证据组合和先后顺序做唯一分级；
4. LLM 仅复核疑难上下文，引用必须能回指原文，且证据强度上限为 `medium`。

## 规划职责

- 管理语音转写时间窗口；
- 提取身份冒充、敏感信息、资金操作、远程控制、保密和紧迫性证据；
- 融合人员数量、打电话状态、老人独处和异常访客等视觉情境；
- 基于证据顺序和组合推断诈骗阶段；
- 生成统一风险事件、证据链和处置建议；
- 提供Mock接口、功能开关和模块测试。

## 输入边界

模块接收后端内部定义的语音和视觉事件，不直接解析 Android 请求、萤石供应商消息或数据库记录。

## 输出边界

模块内部结果由 Service 转换为 `FraudRiskEventWrite`，基础设施 Repository 再写入统一 `risk_events`，不得定义只供防诈页面使用的另一套持久化事件结构。

## 当前限制

- 防诈会话可从 PostgreSQL 恢复，但萤石统一视觉事件仍保存在进程内存，重启后不会恢复历史视觉上下文；
- 音频块只用于临时推理，不持久化；当前块级去重状态也会随进程重启丢失；
- 当前媒体 Worker 从配置启动一个设备；多设备动态增删尚未实现；
- 当前 LLM 接收最近转写、SenseVoice 标签、已有证据以及可选的萤石事件抓拍，不上传原始音频；
- 文本 LLM 只在本地状态达到配置阈值（默认 S2）且尚未达到 S5 时触发；
- 音频时间以首次解码块的服务端 UTC 时间为锚点，尚未解析摄像头 OSD 时间；
- 分类器分数经过 sigmoid 校准，但当前训练集规模仍小；对外 `confidence` 是参与转换的证据置信度最大值，不是整场诈骗概率；
- 人数事件当前只做可解释场景事实，不会单独升级状态；
- 正式萤石云信令 Topic、签名、解密和确认协议仍待官方资料。
- 尚未用标注的真实老人通话集给出准确率、召回率、误报率和端到端延迟结论；上线阈值必须以真实设备评估集确定。

## 双通道 ASR 运行方式

主后端仍使用 Python 3.12。FunASR 的间接依赖需要显式约束兼容版本，已封装在可选依赖中，普通 API/数据库开发不会被迫安装模型运行时：

```bash
cd backend
uv sync --extra sensevoice --dev
```

首次真实推理会下载 `paraformer-zh-streaming` 和 `iic/SenseVoiceSmall` 模型。两个识别器都是懒加载并支持启动预热（`APP_SENSEVOICE_WARMUP_ENABLED` / `APP_STREAMING_ASR_WARMUP_ENABLED`，预热失败自动降级懒加载）；开发测试通过注入假实现运行，不依赖网络和模型权重。

```env
APP_SENSEVOICE_ENABLED=true
APP_SENSEVOICE_MODEL=iic/SenseVoiceSmall
APP_SENSEVOICE_DEVICE=cpu
APP_STREAMING_ASR_ENABLED=true
APP_STREAMING_ASR_MODEL=paraformer-zh-streaming
APP_STREAMING_ASR_DEVICE=cpu
APP_STREAMING_ASR_HOTWORDS=验证码 安全账户 屏幕共享 远程控制 涉案资金 转账 汇款 取现
APP_STREAMING_ASR_HOTWORD_CORRECTIONS={}
```

## 多模态 LLM 复核

LLM 默认关闭。`APP_FRAUD_LLM_BASE_URL` 应包含兼容 API 的版本前缀，例如 `https://provider.example/v1`，代码会调用其 `/chat/completions`：

```env
APP_FRAUD_LLM_ENABLED=true
APP_FRAUD_LLM_BASE_URL=https://provider.example/v1
APP_FRAUD_LLM_API_KEY=replace-me
APP_FRAUD_LLM_MODEL=replace-me
APP_FRAUD_LLM_TIMEOUT_SECONDS=10
APP_FRAUD_LLM_ENABLE_THINKING=false
APP_FRAUD_LLM_QUEUE_MAXSIZE=32
APP_FRAUD_LLM_TRIGGER_STATE_INDEX=2
APP_FRAUD_LLM_MAX_TRANSCRIPT_CHARS=6000
APP_FRAUD_LLM_VISION_ENABLED=true
APP_FRAUD_LLM_MAX_IMAGES=4
```

调用链：

```text
SenseVoice 转写/标签 + 同一时间窗口内的萤石事件抓拍
  → 本地规则与 S0-S5 快照
  → S2-S4 请求立即入队，接口先返回本地快照
  → LLM 严格 JSON 复核
  → 原文引用校验并转换为 medium 证据
  → 重放状态机并更新同一 risk_events 记录
```

`qwen3.5-plus` 默认会进行深度思考，实时复核建议用 `APP_FRAUD_LLM_ENABLE_THINKING=false` 关闭，以降低首包延迟和输出费用。`APP_FRAUD_LLM_VISION_ENABLED` 默认关闭，确认模型渠道支持 `image_url` 后再开启。图片只用于核对场景，LLM 产出的诈骗证据仍必须逐字引用转写原文。`GET /api/v1/fraud/llm/status` 返回配置状态、视觉开关、Worker 状态、队列深度、成功/失败数量和不含密钥的最近错误。

## 非目标

- 不根据单个关键词直接报警；
- 不执行自动转账拦截或自动报警；
- 不进行诈骗者声纹身份识别；
- 不在模块中保存平台密钥；
- 不绕过统一事件中心直接向Android推送。
