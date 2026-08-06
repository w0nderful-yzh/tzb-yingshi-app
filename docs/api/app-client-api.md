# App 端联调接口设计（老人端 / 家属端）

> 状态：第一版 App 联调接口已落地。正式 Bearer 鉴权、可靠通知任务、联系人写接口和活动热力数据源仍按各节说明继续建设。
> 变更需遵循 [API 契约变更流程](README.md#变更流程)。当前 App 只消费防诈事件；跌倒与心理关怀按 [跨模块协作说明](../product/fraud-first-app.md) 保留接入边界。

第一阶段聚焦“身份与守护关系 → 统一事件 → SOS/处置 → 可靠通知 → 设备查看”的闭环。设备完整 CRUD、主动一键回呼和活动热力图列入第二阶段；数据库结构已升级到迁移头 `9b4e2f7a1c32`，接口实现状态仍以每节标记为准。

当前已实现并供 Android 关闭 Mock 后联调：

- `GET /users/me`、`GET /safety/status`；
- `POST /sos`；
- 事件列表、详情、老人确认和家属状态处置；
- 设备列表、萤石短时直播地址和原生 SDK 直播会话；
- 联系人只读列表、家属守护老人列表；
- 事件统计和 `/stats/activity` 保留后端兼容接口，当前防诈 App 不调用。

联调身份暂由 `X-Demo-Role: elder|family` 提供，仅在 `APP_DEMO_IDENTITY_ENABLED=true` 时生效；生产环境启用前必须替换为正式 Bearer Token。

## 1. 角色与端能力划分

| 能力 | 老人端 | 家属端 |
|---|:---:|---|
| 查看防诈守护状态 | ✅ | ✅（远程查看被守护老人） |
| 一键紧急求助（SOS） | ✅ | — |
| 防诈确认（停止操作 / 需要家属帮助） | ✅ | — |
| 防诈处置（知晓、解决、误报） | — | ✅ |
| 从风险详情查看实时画面 | ✅ | ✅ |
| 设备绑定与管理 | 只读查看 | 第二阶段 |
| 紧急联系人 | 只读查看 | 只读查看 |
| 防诈能力接入状态 | 只读查看 | 只读查看 |
| 跌倒/心理关怀 | 占位 | 占位 |
| 家属绑定（扫码） | 出示绑定码 | 扫码绑定 |

鉴权按角色区分：`role=elder` 的令牌只能访问本人数据；`role=family` 的令牌只能访问已绑定老人的数据。API 的 `family` 对应数据库角色 `GUARDIAN`。服务端必须逐请求校验 `elder_id ∈ 当前用户 ACTIVE 绑定范围`，不依赖客户端传入。

## 2. 通用约定

- 统一前缀：`/api/v1`
- 统一响应、错误码、`X-Request-ID`：沿用 [API 契约](README.md#统一响应)。
- 鉴权【规划】：请求头 `Authorization: Bearer <token>`。登录/换发接口本期由后端另行约定，联调阶段可用 Mock 固定令牌。
- 所有时间字段为 ISO 8601 且必须带时区。
- 列表接口统一分页参数：`cursor`（可选）+ `limit`（默认 20，最大 100），响应含 `next_cursor`。事件列表固定按 `(occurred_at DESC, event_id DESC)` 排序，`cursor` 是同时包含这两个值的不透明字符串。
- 会触发事件、状态变化或外部通知的接口必须携带 `Idempotency-Key`；相同用户、接口和 Key 重试时返回首次结果，不重复执行副作用。
- 更新事件或设置时携带当前 `version`。版本不匹配返回 HTTP `409`，客户端重新读取后再提交。

第一阶段业务错误码：

| code | HTTP 状态 | 含义 |
|---|---:|---|
| `21001` | 403 | 当前家属未绑定目标老人 |
| `21002` | 404 | 事件不存在或当前用户不可见 |
| `21003` | 409 | 事件状态流转不合法 |
| `21004` | 409 | 数据版本冲突 |
| `21005` | 409 | 同一幂等 Key 被用于不同请求体 |
| `21006` | 422 | 绑定码错误、过期或已使用 |

### 告警分级与状态机映射

| App 分级 | 含义 | 对应后端状态 | 推送方式 |
|---|---|---|---|
| `reminder`（提醒） | 久坐、离床、设备异常等 | 防诈 S1-S2 / 跌倒低置信 | 仅 App 内消息 |
| `warning`（警告） | 疑似诈骗、陌生人、长时间无活动 | 防诈 S3-S4 / 跌倒中置信 | App 推送 |
| `emergency`（紧急） | 疑似跌倒、SOS 主动求助 | 防诈 S5 / 跌倒高置信持续 | App 推送 + 短信 + 自动外呼升级 |

`level` 是 App 告警等级；算法原始的 `risk_level=low|medium|high|critical` 仅在详情的 `analysis` 中展示，两者不能混用。

事件状态流转：`open → acknowledged（已知晓）→ resolved（已处理）| false_alarm（误报）`。数据库使用对应的大写值。`emergency` 超过 60 秒无 `acknowledged` 时，由持久化通知任务自动升级外呼；服务重启不能丢失任务。

## 3. 老人端接口

### 3.1 获取当前用户信息【规划】

```text
GET /api/v1/users/me
```

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "user_id": "u-elder-001",
    "role": "elder",
    "name": "王秀兰",
    "bound_family_count": 2,
    "font_size": "extra_large",
    "voice_assist_enabled": true
  },
  "request_id": "req_xxx"
}
```

### 3.2 首页安全状态总览【规划】

聚合接口，避免老人端多请求拼装。

```text
GET /api/v1/safety/status
```

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "overall": "safe",
    "overall_label": "一切正常",
    "active_event_count": 0,
    "highest_active_level": null,
    "devices_online": 3,
    "devices_total": 3,
    "checked_at": "2026-08-04T16:45:02+08:00",
    "today": {
      "event_count": 0
    }
  },
  "request_id": "req_xxx"
}
```

`overall` 枚举：`safe | attention | danger`，与最高活动告警级别对应。

### 3.3 一键紧急求助（SOS）【规划】

```text
POST /api/v1/sos
Idempotency-Key: <由客户端为本次长按生成的 UUID>
```

```json
{
  "trigger": "long_press",
  "client_request_id": "019fcb12-31d4-7472-9f25-45a340248b5a",
  "location": { "lat": 30.2741, "lng": 120.1551 },
  "occurred_at": "2026-08-04T16:45:02+08:00"
}
```

响应 `data`：`{ "event_id": "evt_xxx", "status": "dispatched", "notified_contacts": 2 }`。服务端生成 `emergency` 级事件并立即走通知家属流程。相同 `Idempotency-Key/client_request_id` 重试返回同一 `event_id` 和 `status=duplicate`；不能仅按 60 秒时间窗口合并，以免吞掉第二次真实求助。

### 3.4 告警确认（我没事 / 我需要帮助）【规划】

该接口保留给未来的设备端或其他老人触点，当前家属端 App 不调用。

```text
POST /api/v1/events/{event_id}/confirm
Idempotency-Key: <UUID>
```

```json
{ "action": "im_ok", "version": 1 }
```

`action` 枚举：`im_ok`（事件置为 `resolved`）、`need_help`（事件置为 `acknowledged`，但仍保持未解决，并立即创建外呼紧急联系人的升级任务）。老人端告警全屏页 60 秒倒计时无操作由服务端自动升级，无需客户端轮询。版本冲突返回 `409`。

### 3.5 我的防诈事件列表【已实现】

```text
GET /api/v1/events?level=&status=&limit=20&cursor=
```

老人端不传 `elder_id`（固定为本人）。响应条目：

```json
{
  "event_id": "evt_001",
  "type": "fraud_suspected",
  "level": "warning",
  "title": "疑似诈骗风险",
  "summary": "出现身份冒充和验证码索取证据",
  "device_id": "camera-01",
  "occurred_at": "2026-08-04T15:02:11+08:00",
  "status": "open",
  "version": 1,
  "evidence_image_url": null,
  "evidence_frames": [
    { "captured_at": "2026-08-04T15:02:00+08:00", "image_url": "https://cdn.example/frame-00.jpg" },
    { "captured_at": "2026-08-04T15:02:11+08:00", "image_url": "https://cdn.example/frame-11.jpg" }
  ],
  "location": "客厅",
  "fraud_scene": "telecom",
  "fraud_state": "S4_ACTION_INDUCEMENT",
  "fraud_state_index": 4,
  "fraud_state_label": "敏感操作诱导",
  "fraud_decision": "block"
}
```

`type` 枚举（初版）：`fall_suspected | fraud_suspected | stranger | inactivity | sos | device_offline | night_leave_bed | sedentary`。
当前 App 只展示 `fraud_suspected`。防诈事件的 `fraud_scene` 为 `telecom | home_visit | unknown`；其他事件返回 `null`。
`fraud_state*` 与 `fraud_decision` 是列表页使用的预测摘要，避免首页为每条风险再次请求详情。首页不按 `fraud_scene` 拆分入口，具体场景只在预警消息和详情中展示。
`evidence_frames` 保持抓拍地址与采集时间戳一一对应，按时间升序返回；历史数据只有单张图时，服务端会用 `occurred_at` 兼容生成一个画面条目。

`next_cursor=null` 表示已经到末页。客户端不得解析或自行拼接 `cursor`。

### 3.6 设备列表（只读）【规划】

```text
GET /api/v1/devices
```

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "devices": [
      {
        "device_id": "camera-01",
        "name": "客厅摄像头",
        "room": "living_room",
        "online": true,
        "signal": "good",
        "last_seen_at": "2026-08-04T16:44:58+08:00"
      }
    ]
  },
  "request_id": "req_xxx"
}
```

### 3.7 实时画面取流地址【已实现】

```text
GET /api/v1/devices/{device_id}/live-url
```

返回短时有效的萤石播放地址（协议由 `APP_YS7_LIVE_PROTOCOL` 决定，AppSecret 不下发客户端）：

```json
{ "code": 0, "message": "success", "data": { "url": "https://...flv", "protocol": "flv", "expires_in": 300 }, "request_id": "req_xxx" }
```

标准 FLV/HLS 地址适用于 H.264 设备。设备输出 H.265 时，萤石标准地址会返回“视频编码类型非 H264”的提示画面，Android App 应改用下方原生 SDK 会话接口。

### 3.8 原生 SDK 直播会话【已实现】

```text
GET /api/v1/devices/{device_id}/live-sdk-session
```

用于 Android 端通过 `EZOpenSDK` 播放设备原始 H.264/H.265 码流：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "app_key": "<ys7-app-key>",
    "access_token": "<short-lived-access-token>",
    "device_serial": "camera-01",
    "channel_no": 1,
    "expires_in": 300
  },
  "request_id": "req_xxx"
}
```

- 响应携带 `Cache-Control: no-store`，客户端只在内存中持有本次会话，不写日志和持久化存储；
- `AppSecret` 只保存在后端，绝不下发客户端；
- 当前 Demo 使用萤石 AccessToken 授权。生产环境应启用正式用户鉴权，并切换为设备/通道级小权限 Token；
- SDK 内部调试日志必须关闭，因为其请求日志可能包含 AccessToken。

### 3.9 事件时间点历史回放【TODO，当前返回 501】

```text
GET /api/v1/devices/{device_id}/history-playback?elder_id=&at=&duration_seconds=30
```

App 已声明并接入入口。后端完成设备归属校验后明确返回 `501 Not Implemented`；待萤石历史回放能力接入后，返回短时有效的 `url`、`protocol`、`start_at` 和 `expires_in`，不得用直播地址冒充历史回放。

### 3.10 守护设置查询【规划】

```text
GET /api/v1/settings
```

老人端只读；修改走家属端 4.9。

## 4. 家属端接口

### 4.1 扫码绑定老人【规划】

老人端先生成 5 分钟有效的绑定码。响应中的明文码只返回一次，数据库只保存哈希：

```text
POST /api/v1/family/bind-codes
Idempotency-Key: <UUID>
```

```json
{ "code": 0, "message": "success", "data": { "bind_code": "8F3K2Q", "expires_at": "2026-08-04T16:50:02+08:00" }, "request_id": "req_xxx" }
```

家属扫码后消费绑定码：

```text
POST /api/v1/family/bind
Idempotency-Key: <UUID>
```

```json
{ "bind_code": "8F3K2Q", "relation": "son", "display_name": "张伟" }
```

响应 `data`：`{ "binding_id": "bind_xxx", "elder_id": "u-elder-001", "elder_name": "王秀兰", "bound": true }`。合法短期绑定码视为老人当次授权，绑定立即进入 `active`。

第一阶段只允许老人端撤销绑定：`DELETE /api/v1/family/bindings/{binding_id}`。家属端不能绕过老人直接解除守护关系。解绑后关系置为 `revoked`，保留历史授权记录。

### 4.2 我守护的老人列表【规划】

```text
GET /api/v1/family/elders
```

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "elders": [
      {
        "elder_id": "u-elder-001",
        "name": "王秀兰",
        "relation": "son",
        "overall": "danger",
        "last_active_at": "2026-08-04T16:33:00+08:00",
        "pending_event_count": 1
      }
    ]
  },
  "request_id": "req_xxx"
}
```

### 4.3 被守护老人的事件列表【规划】

```text
GET /api/v1/events?elder_id=u-elder-001&level=&status=&type=&from=&to=&limit=20&cursor=
```

与 3.5 同构，家属端必须显式传 `elder_id`，服务端校验绑定关系，未绑定返回 `403`。事件详情按 ID 访问时，对不存在和无权查看统一返回 `404`，避免泄露其他家庭的事件是否存在。

### 4.4 事件详情【已实现】

```text
GET /api/v1/events/{event_id}
```

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "event_id": "evt_002",
    "type": "fraud_suspected",
    "level": "warning",
    "status": "open",
    "version": 1,
    "device_id": "camera-01",
    "occurred_at": "2026-08-04T11:20:05+08:00",
    "evidence_image_url": null,
    "evidence_frames": [
      { "captured_at": "2026-08-04T11:19:50+08:00", "image_url": "https://cdn.example/frame-01.jpg" },
      { "captured_at": "2026-08-04T11:20:05+08:00", "image_url": "https://cdn.example/frame-02.jpg" }
    ],
    "location": "客厅",
    "analysis": {
      "confidence": 0.87,
      "reasons": [
        { "key": "identity_claim", "label": "身份冒充", "value": "自称银行客服" },
        { "key": "credential_request", "label": "敏感信息", "value": "索要短信验证码" }
      ],
      "disclaimer": "AI 辅助判断，不替代专业结论"
    },
    "notifications": [],
    "escalation": { "auto_call_at": null, "status": "pending" },
    "fraud": {
      "scene": "telecom",
      "state": "S4_ACTION_INDUCEMENT",
      "state_index": 4,
      "state_label": "敏感操作诱导",
      "decision": "block",
      "transition_reason": "出现验证码索取和身份冒充证据"
    }
  },
  "request_id": "req_xxx"
}
```

防诈类事件的 `analysis.reasons` 复用防诈状态机的 `evidence_chain` 生成；`fraud` 提供场景、S0-S5 阶段、决策与迁移原因。非防诈事件的 `fraud` 为 `null`。

### 4.5 事件处置【已实现】

```text
PATCH /api/v1/events/{event_id}/status
Idempotency-Key: <UUID>
```

```json
{ "status": "acknowledged", "note": "已电话确认母亲无恙", "version": 1 }
```

`status` 枚举：`acknowledged | resolved | false_alarm`。不允许从终态返回 `open`；非法流转或版本冲突返回 `409`。`false_alarm` 样本回流用于降低误报（写入事件备注，不直接参与模型训练）。每次处置都追加 `event_actions` 审计记录。

### 4.6 设备语音介入提醒【TODO，当前返回 501】

```text
POST /api/v1/events/{event_id}/intervention-reminder
Idempotency-Key: <UUID>
```

```json
{ "channel": "device_voice", "message": "请暂停当前操作，家人正在联系您核实情况。" }
```

App 详情页已保留操作入口。后端先校验事件查看权限，再明确返回 `501 Not Implemented`；待接入萤石设备语音或真实外呼后，必须记录送达状态和动作审计，不能仅返回固定成功文案。

### 4.7 一键回呼老人【第二阶段】

```text
POST /api/v1/events/{event_id}/call
```

服务端通过萤石设备双向语音或电话外呼发起，响应 `data`：`{ "call_status": "ringing" }`。也可不带事件独立呼叫：`POST /api/v1/family/elders/{elder_id}/call`。

### 4.8 紧急联系人管理【规划】

```text
GET /api/v1/contacts?elder_id=
POST /api/v1/contacts
PATCH /api/v1/contacts/{contact_id}
DELETE /api/v1/contacts/{contact_id}
POST /api/v1/contacts/{contact_id}/confirm
```

```json
{
  "contacts": [
    { "contact_id": "contact_001", "order": 1, "name": "张伟", "relation": "son", "phone_masked": "138****6688", "channels": ["push", "sms", "call"], "status": "active" },
    { "contact_id": "contact_002", "order": 2, "name": "张莉", "relation": "daughter", "phone_masked": "139****2233", "channels": ["push", "sms"], "status": "active" }
  ]
}
```

`POST/PATCH/DELETE` 均需 `Idempotency-Key`。新增或修改联系人时请求体提交完整 `phone`，服务端加密落库，只额外保存末四位；API 响应和日志仅输出 `phone_masked`。变更先写入待确认内容，原有 `active` 联系人在审核期间继续用于告警；老人端调用确认接口 `{ "approved": true }` 后才应用修改或删除。按 `order` 顺序升级呼叫。

### 4.9 设备管理【第二阶段】

```text
GET   /api/v1/devices?elder_id=
POST  /api/v1/devices                 # 绑定萤石设备：{ "elder_id", "device_serial", "channel_no", "name", "room" }
PATCH /api/v1/devices/{device_id}     # 改名、调整房间、启停监测
DELETE /api/v1/devices/{device_id}    # 解绑，需二次确认
```

设备状态细节（取流 Worker、队列深度）复用【已实现】`GET /api/v1/integrations/ys7/media/status` 诊断，不向 App 暴露凭证字段。

### 4.10 守护设置管理【规划】

```text
GET /api/v1/settings?elder_id=
PUT /api/v1/settings?elder_id=
Idempotency-Key: <UUID>
```

```json
{
  "fraud_monitor_enabled": true,
  "fall_detect_enabled": true,
  "night_leave_bed_enabled": true,
  "voice_broadcast_enabled": true,
  "fall_sensitivity": "medium",
  "inactivity_threshold_hours": 6,
  "version": 1
}
```

`fall_sensitivity` 枚举：`low | medium | high`，映射到跌倒模块的持续时间与置信度阈值组合，具体映射表由跌倒模块设计补充。`PUT` 必须提交当前 `version`，成功后返回递增的新版本。

### 4.11 数据看板统计【规划】

```text
GET /api/v1/stats/events?elder_id=&days=30
```

`/stats/events` 按周/天返回分级别计数：`{ "buckets": [{ "period": "2026-W31", "reminder": 4, "warning": 1, "emergency": 0 }] }`。
活动热力图没有稳定的持久化数据源，`/stats/activity` 移至第二阶段；情绪趋势统计随心理关怀模块一并暂缓。

### 4.12 推送端点注册【规划】

```text
POST   /api/v1/users/me/push-endpoints
DELETE /api/v1/users/me/push-endpoints/{install_id}
```

```json
{
  "install_id": "android-install-uuid",
  "platform": "android",
  "provider": "fcm",
  "push_token": "provider-issued-token"
}
```

`POST` 按当前用户和 `install_id` 幂等覆盖。推送 Token 加密落库并保存不可逆指纹，不在查询接口和日志中返回。用户退出登录或卸载时调用 `DELETE` 使端点失效。

## 5. 实时通知（WebSocket）【规划】

```text
POST /api/v1/ws/tickets
WS /api/v1/ws/events?ticket=<60 秒有效的一次性票据>
```

Bearer Token 不放在 WebSocket 查询参数中，避免被代理访问日志记录。客户端先用正常鉴权请求换取一次性票据，再建立连接。

连接后按角色推送。老人端仅收本户事件；家属端收全部绑定老人的事件：

```json
{
  "msg_type": "risk_event",
  "data": {
    "event_id": "evt_001",
    "elder_id": "u-elder-001",
    "type": "fall_suspected",
    "level": "emergency",
    "title": "疑似跌倒",
    "occurred_at": "2026-08-04T15:02:11+08:00"
  }
}
```

其他 `msg_type`：`device_status`（上下线）、`escalation`（外呼升级进度）、`pong`（心跳应答，客户端 30 秒发一次 `ping`）。断线重连由客户端指数退避，重连后用 `GET /events?status=open` 补齐。

`emergency` 级事件同时触发服务端短信/外呼通道，不依赖 App 在线。

## 6. 复用已实现的后端能力

以下接口已存在（见 [API 契约](README.md)），App 联调时作为数据源或联调工具，不重复建设：

| 接口 | 用途 |
|---|---|
| `GET /api/v1/health` | 联调环境连通性检查 |
| `POST /api/v1/integrations/ys7/events` | 联调期注入模拟视觉事件（跌倒/人数/打电话） |
| `GET /api/v1/fraud/visual-events` | 统一视觉事件查询，事件中心数据源 |
| `POST /api/v1/fraud/analyze` | 联调期注入模拟通话转写，触发防诈告警 |
| `POST /api/v1/fraud/audio/chunks` | 真实音频链路验证 |
| `GET /api/v1/fraud/sessions/{session_id}` | 防诈会话风险快照查询 |
| `GET /api/v1/integrations/ys7/media/status` | 取流 Worker 诊断 |

## 7. 暂缓清单（本期不做）

- 家属主动一键回呼老人；紧急事件的服务端自动外呼仍属于第一阶段；
- 家属端新增、修改和解绑萤石设备；第一阶段只提供已绑定设备查询、短时直播地址和原生 SDK 直播会话；
- 活动热力图 `GET /stats/activity` 及其小时级活动汇总任务；
- 心理关怀模块全部接口：情绪打卡（`POST /moods`）、情绪趋势（`GET /moods/trend`）、AI 陪伴对话（`POST /companion/chat`）；
- 部署与运维相关内容（docker 化、CI 发布）；
- 登录注册与令牌换发的正式方案（联调期使用 Mock 令牌）。
