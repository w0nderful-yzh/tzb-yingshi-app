# API契约

## 当前状态

当前已实现健康检查、萤石模拟事件接收、接收器状态、统一视觉事件查询、防诈转写分析和活动会话风险查询。正式萤石消息协议仍需等待 Topic、签名/解密规则和完整消息样例后补充。

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

### `GET /api/v1/fraud/sessions/{session_id}`

通过必填查询参数 `device_id` 获取当前进程内活动会话的最新风险快照。该查询会重新读取同设备视觉事件，因此后到达的萤石事件可以在下一次查询或分析时参与判断。会话不存在返回 HTTP 404。

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

## 变更流程

1. 先修改本目录中的契约文档；
2. 更新Pydantic Schema；
3. 更新后端契约测试；
4. 更新Android DTO；
5. 在PR中说明兼容性影响。
