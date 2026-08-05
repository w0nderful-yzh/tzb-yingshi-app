# API契约

App 端联调接口设计（老人端 / 家属端分角色）见 [app-client-api.md](app-client-api.md)。

## 当前状态

当前已实现健康检查、萤石模拟事件接收、接收器状态、萤石直播音轨状态、统一视觉事件查询、防诈转写分析、SenseVoice 音频块分析、文本 LLM 异步复核和活动会话风险查询。正式萤石消息协议仍需等待 Topic、签名/解密规则和完整消息样例后补充。

## 统一前缀

```text
/api/v1
```

## 规划接口

```text
POST   /api/v1/fall/analyze
GET    /api/v1/events
GET    /api/v1/events/{event_id}
PATCH  /api/v1/events/{event_id}/status
GET    /api/v1/devices
WS     /api/v1/ws/events
```

Android 第一版已接入的 App 客户端接口以 [app-client-api.md](app-client-api.md) 为准。联调前执行数据库迁移和 `uv run python -m app.scripts.seed_demo`，再启动 FastAPI。

## 已实现接口

### `GET /api/v1/health`

用于进程存活检查，不访问数据库或外部服务。

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "status": "ok",
    "service": "老年安全监测后端",
    "version": "0.1.0",
    "environment": "development"
  },
  "request_id": "req_xxx"
}
```

客户端可以传入合法的 `X-Request-ID`，服务端会复用并在响应头和响应体中返回；缺失或格式非法时由服务端生成。

### `POST /api/v1/integrations/ys7/events`

接收标准化的萤石模拟消息。请求必须携带 `X-YS7-Webhook-Token`，且服务端必须启用 `APP_YS7_SIGNAL_ENABLED`。接收回调只执行校验、原文保存、去重和入队，不执行状态机或模型推理。

```json
{
  "messageId": "msg-demo-001",
  "requestId": "request-demo-001",
  "eventId": "event-demo-001",
  "deviceId": "camera-01",
  "timestamp": "2026-08-04T12:00:00+08:00",
  "eventType": "phone_call",
  "confidence": 0.91,
  "peopleCount": 1,
  "boxes": [],
  "imageUrl": "https://example.invalid/evidence.jpg"
}
```

当前支持的 `eventType`：

```text
phone_call | people_count | person_detected
```

成功或重复消息均返回 HTTP 202：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "status": "accepted",
    "source_event_id": "event-demo-001",
    "raw_event_ref": "2026-08-04/xxx.json"
  },
  "request_id": "req_xxx"
}
```

`status=duplicate` 表示相同 `messageId`、`requestId` 或 `eventId` 已接收，不会再次生成视觉事件。队列满时返回 HTTP 503，上游应按其重试策略再次投递。

### `GET /api/v1/integrations/ys7/status`

返回接收功能开关、后台 Worker 状态和当前队列深度。

### `GET /api/v1/integrations/ys7/media/status`

返回直播音轨 Worker 的运行状态：

- `running`：后台任务是否存活；
- `connected`：当前是否已连接直播流；
- `session_id`：本次进程中的防诈直播会话；
- `queue_depth`：等待 SenseVoice 的音频块数量；
- `chunks_processed/chunks_dropped`：成功分析和为保持实时性而丢弃的块数；
- `reconnect_attempts/last_error`：当前重连次数和不含凭证的错误摘要。

### `GET /api/v1/fraud/visual-events`

查询后台适配后的统一视觉事件，支持 `device_id` 和 `limit` 参数。返回结果按 `occurred_at` 倒序排列，而不是按接收顺序排列。

### `POST /api/v1/fraud/analyze`

提交一条已完成的语音转写片段。`occurred_at` 和 `ended_at` 必须携带时区；同一设备、会话下重复的 `source_event_id` 返回 `status=duplicate`，不会重复加入证据链。

```json
{
  "session_id": "call-20260804-001",
  "source_event_id": "asr-segment-001",
  "device_id": "camera-01",
  "occurred_at": "2026-08-04T12:00:05+08:00",
  "ended_at": "2026-08-04T12:00:08+08:00",
  "text": "我是银行客服，把短信验证码告诉我",
  "elder_alone": true
}
```

响应中的 `risk` 包含 `state`、`risk_level`、`decision`、`transition_reason`、`next_stage_conditions`、`evidence_chain` 和 `state_history`。服务会按事件发生时间重放当前会话近 120 秒的语音证据，并合并同设备的萤石视觉事件。非 S0 结果在数据库启用时写入 `risk_events`。

### `POST /api/v1/fraud/audio/chunks`

以 `multipart/form-data` 上传短 WAV 音频块。服务先校验 WAV，再在线程中调用 SenseVoice，最后把转写片段交给与 `/fraud/analyze` 相同的业务链路。

| 字段 | 类型 | 说明 |
|---|---|---|
| `audio` | WAV 文件 | 单声道或双声道，8–48 kHz，不超过 15 秒和配置的字节上限 |
| `session_id` | string | 防诈会话 ID |
| `chunk_id` | string | 当前会话内稳定的音频块 ID，用于避免重复推理 |
| `device_id` | string | 外部摄像头设备 ID |
| `started_at` | datetime | 音频块第一个采样点的绝对时间，必须带时区 |
| `elder_alone` | boolean | 是否已明确老人独处 |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/fraud/audio/chunks \
  -F 'audio=@chunk.wav;type=audio/wav' \
  -F 'session_id=call-demo-001' \
  -F 'chunk_id=chunk-001' \
  -F 'device_id=camera-01' \
  -F 'started_at=2026-08-04T12:00:00+08:00' \
  -F 'elder_alone=true'
```

相同设备、会话和 `chunk_id` 重复提交时返回 `status=duplicate`，不会再次运行 SenseVoice。音频无有效语音时 `transcript_segments` 为空，`risk` 返回已有会话快照或 `null`。功能未启用或模型依赖缺失返回 HTTP 503，模型执行失败返回 HTTP 502。

### `GET /api/v1/fraud/sessions/{session_id}`

通过必填查询参数 `device_id` 获取当前进程内活动会话的最新风险快照。该查询会重新读取同设备视觉事件，因此后到达的萤石事件可以在下一次查询或分析时参与判断。会话不存在返回 HTTP 404。

### `GET /api/v1/fraud/llm/status`

返回文本 LLM 复核开关、配置完整性、后台 Worker、模型名、队列深度、成功/失败次数和最近错误。LLM 未配置、超时或调用失败不会影响本地规则和 S0-S5 状态机继续运行。

## 统一响应

```json
{
  "code": 0,
  "message": "success",
  "data": {},
  "request_id": "req_xxx"
}
```

当前公共错误码：

| code | HTTP 状态 | 含义 |
|---|---:|---|
| `10001` | 422 | 请求参数校验失败 |
| `10002` | 4xx | HTTP 请求错误 |
| `10003` | 500 | 未处理的服务端错误 |

萤石接收接口还可能返回：

| HTTP 状态 | 含义 |
|---:|---|
| 401 | Webhook 令牌无效 |
| 422 | 消息字段、时间、事件类型或检测框不合法 |
| 503 | 接收器未启用、令牌未配置或队列已满 |

SenseVoice 音频块接口还可能返回：

| HTTP 状态 | 含义 |
|---:|---|
| 415 | 上传内容不是 WAV 音频 |
| 422 | WAV、时间或块元数据不合法 |
| 502 | SenseVoice 已安装但本次推理失败 |
| 503 | 音频接收未启用或模型运行依赖未安装 |

## 变更流程

1. 先修改本目录中的契约文档；
2. 更新Pydantic Schema；
3. 更新后端契约测试；
4. 更新Android DTO；
5. 在PR中说明兼容性影响。
