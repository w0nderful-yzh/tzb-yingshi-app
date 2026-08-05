# 防诈模块

## 当前状态

已迁入原防诈工程中的规则、轻量文本分类器、语音证据提取、证据融合和 S0-S5 状态机，并接入当前 FastAPI 的萤石视觉事件与 PostgreSQL 风险事件表。后端既可接收短 WAV 音频块，也可使用萤石凭证获取标准直播地址，由 FFmpeg 连续解码音轨并自动切块进入 SenseVoice。S2-S4 会话可选进入异步 LLM 复核，支持将同一时间窗口内的萤石事件抓拍与转写一并发送；LLM 不可用时继续使用本地结果。

## 已实现

- `phone_call`、`people_count`、`person_detected` 视觉事件；
- 事件发生时间和服务端接收时间分离；
- 原始消息保存、消息 ID 去重和后台消费；
- 按设备查询统一视觉事件；
- 按设备和会话维护 120 秒有序证据窗口，支持语音片段乱序重放；
- 按 `source_event_id` 幂等接收转写片段；
- 按设备、会话和 `chunk_id` 幂等接收短 WAV 音频块；
- SenseVoiceSmall 懒加载、串行推理和异常隔离，推理不占用 FastAPI 事件循环；
- SenseVoice 相对毫秒自动转换为音频块 `started_at` 对应的绝对时间；
- 保留 SenseVoice 的语言、情绪和音频事件富标签，不再只保留清洗后的文字；
- 使用 AppKey/AppSecret 自动获取和缓存萤石 accessToken；
- 获取 FLV、RTMP 或 HLS 直播地址，FFmpeg 解码为 16 kHz 单声道 PCM；
- 媒体断线指数退避重连，重连时重新申请直播地址；
- 有界实时音频队列，积压时丢弃最旧块而不是无限增加告警延迟；
- 将 `phone_call` 转换为通话语境，将人数作为不单独升级的场景事实；
- 组合规则与字符 TF-IDF 轻量分类器提取语音证据；
- 输出 S0-S5 状态、风险等级、判断建议、转换原因和完整证据链；
- 使用 OpenAI-compatible `/chat/completions` 接口进行可选文本与图片 LLM 复核；
- 多模态复核复用萤石事件的 `image_url`，按证据窗口筛选、去重并限制最多图片数；
- LLM 在独立有界队列中运行，不阻塞萤石取流、SenseVoice 和本地状态机响应；
- LLM 输出使用严格 JSON，证据必须逐字引用转写原文，无法定位的引用会被丢弃；
- LLM 证据强度上限为 `medium`，不能单独推动 S4/S5；服务超时或失败自动降级；
- 非 S0 结果按设备和会话幂等写入 PostgreSQL `risk_events`；
- 萤石接收功能关闭时，FastAPI 其他接口正常启动。

## 下一步

1. 用真实萤石设备验证直播流是否包含音轨、协议权限和端到端延迟；
2. 明确萤石访客进入、停留等正式事件类型后补充视觉证据适配；
3. 若需要跳过 SenseVoice，再选用明确支持音频输入的模型并增加疑难片段的 PCM 环形缓冲；
4. 将单设备媒体配置扩展为数据库驱动的多设备 Worker；
5. 将活动会话从进程内存迁入数据库或 Redis，支持重启恢复与多实例；
6. 增加风险事件查询、人工处置和 WebSocket 通知。

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

- 活动会话与萤石统一视觉事件仍保存在进程内存，重启后不恢复；已生成的非 S0 风险快照会保存在 PostgreSQL；
- 音频块只用于临时推理，不持久化；当前块级去重状态也会随进程重启丢失；
- 当前媒体 Worker 从配置启动一个设备；多设备动态增删尚未实现；
- 当前 LLM 接收最近转写、SenseVoice 标签、已有证据以及可选的萤石事件抓拍，不上传原始音频；
- 文本 LLM 只在本地状态达到配置阈值（默认 S2）且尚未达到 S5 时触发；
- 音频时间以首次解码块的服务端 UTC 时间为锚点，尚未解析摄像头 OSD 时间；
- `confidence` 是参与转换的证据置信度最大值，不是经过校准的诈骗概率；
- 人数事件当前只做可解释场景事实，不会单独升级状态；
- 正式萤石云信令 Topic、签名、解密和确认协议仍待官方资料。

## SenseVoice 运行方式

主后端仍使用 Python 3.12。FunASR 的间接依赖需要显式约束兼容版本，已封装在可选依赖中，普通 API/数据库开发不会被迫安装模型运行时：

```bash
cd backend
uv sync --extra sensevoice --dev
```

首次真实推理会下载 `iic/SenseVoiceSmall` 模型。开发测试通过注入 `SpeechRecognizer` 假实现运行，不依赖网络和模型权重。

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
