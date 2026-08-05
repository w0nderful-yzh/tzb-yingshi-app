# PostgreSQL 数据库设计

## 当前状态

已接入 PostgreSQL 16、SQLAlchemy 2.x、psycopg 3 和 Alembic。当前数据库已升级到 App 第一阶段迁移头 `9b4e2f7a1c32`。

当前已完成数据库结构和连接生命周期，防诈 Service 已通过 Repository 幂等写入 `risk_events`。家属绑定、联系人、推送端点、守护设置和告警投递已完成表结构，业务 API 尚待实现。萤石 Inbox、统一视觉事件和活动会话仍使用磁盘/内存实现，尚未写入对应数据库表。

## 实体关系

```mermaid
erDiagram
    USERS ||--o{ DEVICES : "elder_user_id"
    USERS ||--o{ FAMILY_BINDINGS : "guardian_user_id"
    USERS ||--o{ FAMILY_BINDINGS : "elder_user_id"
    USERS ||--o{ BINDING_CODES : "elder_user_id"
    USERS ||--o{ USER_PUSH_ENDPOINTS : "user_id"
    USERS ||--o{ IDEMPOTENCY_RECORDS : "user_id"
    USERS ||--o| ELDER_SAFETY_SETTINGS : "elder_user_id"
    USERS ||--o{ EMERGENCY_CONTACTS : "elder_user_id"
    USERS ||--o{ EVENT_ACTIONS : "actor_user_id"
    DEVICES ||--o{ YS7_SIGNAL_INBOX : "device_id"
    DEVICES ||--o{ VISUAL_EVENTS : "device_id"
    DEVICES ||--o{ MODEL_RUNS : "device_id"
    DEVICES ||--o{ RISK_EVENTS : "device_id"
    USERS ||--o{ RISK_EVENTS : "elder_user_id"
    YS7_SIGNAL_INBOX ||--o| VISUAL_EVENTS : "raw_signal_id"
    MODEL_RUNS ||--o{ RISK_EVENTS : "model_run_id"
    RISK_EVENTS ||--o{ EVENT_ACTIONS : "risk_event_id"
    RISK_EVENTS ||--o{ EVENT_DELIVERIES : "risk_event_id"
    EMERGENCY_CONTACTS ||--o{ EVENT_DELIVERIES : "contact_id"
```

所有业务主键使用 UUID。`external_device_id` 同时保留在事件表中，即使设备尚未注册或后来解绑，原始事件仍然可以按供应商设备 ID 查询；`device_id` 是可空的内部外键。

## 表职责

| 表 | 所有者 | 用途 | 关键约束 |
|---|---|---|---|
| `users` | 公共 | 老人、家属和管理员基础身份 | `external_subject` 唯一；不存明文密码和完整手机号 |
| `family_bindings` | App 守护关系 | 家属可见老人范围与绑定确认状态 | `(guardian_user_id, elder_user_id)` 唯一；授权只认 `ACTIVE` |
| `binding_codes` | App 守护关系 | 5 分钟绑定码的哈希、过期和消费状态 | 只存 `code_hash`，不存明文绑定码 |
| `user_push_endpoints` | 通知 | App 安装实例与推送地址 | Token 加密保存，指纹唯一；一个用户可有多个安装实例 |
| `idempotency_records` | App 基础设施 | 跨重试保存请求哈希和首次响应 | `(user_id, scope, idempotency_key)` 唯一；过期后定期清理 |
| `elder_safety_settings` | App 守护设置 | 老人维度的监测开关和阈值 | `elder_user_id` 唯一；`version` 用于并发更新 |
| `emergency_contacts` | 通知 | 紧急联系人、升级顺序、通道及待确认变更 | 完整手机号加密；`pending_payload` 在老人确认前不覆盖生效数据 |
| `devices` | 公共 | 摄像头及其他安全设备 | `external_device_id` 唯一；解绑用户时保留设备 |
| `ys7_signal_inbox` | 萤石接入 | 原始消息和可靠消费状态 | `dedup_key` 唯一；状态为 Pending/Processing/Processed/Failed |
| `visual_events` | 防诈/防摔共享 | 供应商消息转换后的视觉事实 | `(source, source_event_id)` 唯一；一条原始消息最多生成一条视觉事件 |
| `model_runs` | 公共 | 一次防诈或防摔算法运行记录 | 保留模型名称、版本、输入引用和运行状态 |
| `risk_events` | 统一事件中心 | 对 App 输出的防诈、防摔、SOS 和系统安全事件 | `(source, source_event_id)` 唯一；算法风险等级和 App 告警等级分列 |
| `event_actions` | 统一事件中心 | 家属确认、误报、处理和联系老人等审计记录 | 只追加；风险事件和操作记录禁止级联删除 |
| `event_deliveries` | 通知 | 推送、短信、自动外呼任务及投递结果 | `dedup_key` 唯一；按 `next_attempt_at` 可靠重试 |

## 时间与乱序

当前防诈 Repository 以“外部设备 ID + 会话 ID”的哈希作为稳定 `source_event_id`，并使用 `source=FRAUD_ENGINE` 形成复合幂等键。同一会话从 S1 升级到 S5 时更新同一风险事件，同时保留已由人工修改的 `status`。

统一事件状态使用 `OPEN → ACKNOWLEDGED → RESOLVED | FALSE_ALARM`。`risk_level` 保留算法的 `LOW/MEDIUM/HIGH/CRITICAL`，`alert_level` 保存 App 实际使用的 `REMINDER/WARNING/EMERGENCY`。SOS、设备离线等非模型事件允许置信度和模型字段为空。

只有能够解析出 `elder_user_id` 的事件才允许出现在 App 事件中心。未注册设备产生的原始数据先保留在 Inbox/视觉事实层，待设备归属明确后再生成 App 可见风险事件。

- `occurred_at`：摄像头、云端算法或业务模块认定的事件发生时间；
- `received_at`：FastAPI 收到消息或结果的时间；
- `created_at`：数据库写入业务记录的时间。

状态机按 `occurred_at` 重放证据，`received_at` 用于监控延迟和排查乱序。相关事件表均使用 `TIMESTAMP WITH TIME ZONE`。

## JSONB 边界

以下字段允许 JSONB：

- `ys7_signal_inbox.raw_payload`：不可变的供应商原始消息；
- `visual_events.boxes`：不同视觉算法的检测框；
- `risk_events.evidence`：防诈和防摔不同的证据详情；
- `model_runs.input_refs/output_summary`：运行输入引用与轻量结果摘要；
- `devices.settings`：低频设备配置。
- `users.preferences`：字号、语音辅助等无需过滤的客户端偏好；
- `emergency_contacts.channels`：联系人启用的推送、短信和外呼通道；
- `emergency_contacts.pending_payload`：家属提交、等待老人确认的联系人变更；
- `event_actions.metadata`：处置动作的少量扩展审计信息。
- `idempotency_records.response_body`：重试时需要原样返回的首次响应体。

事件 ID、状态、等级、时间、置信度、设备和模型版本必须使用独立列，禁止只放进 JSONB，否则无法稳定去重、过滤和建立索引。

## 索引策略

- Inbox：`(processing_status, received_at)` 支持 Worker 拉取待处理消息；
- Inbox/视觉事件：`(external_device_id, occurred_at)` 支持按设备重放；
- 风险事件：`(external_device_id, occurred_at)`、`(status, occurred_at)`、`(elder_user_id, status, occurred_at, id)` 和 `(elder_user_id, event_type, occurred_at)`；
- 处置记录：`(risk_event_id, created_at)`；
- 告警投递：`(status, next_attempt_at)` 和 `(risk_event_id, created_at)`；
- 家属绑定：家属和老人分别按 `(user_id, status)` 查询；
- 幂等记录：`expires_at` 支持定期清理过期响应；
- 模型运行：`(device_id, started_at)`。

目前不为 JSONB 建立 GIN 索引。比赛阶段查询以结构化字段为主，等出现明确的证据内容检索需求后再增加，避免无依据的写入和存储开销。

## 删除与审计

- 家属解绑通过 `family_bindings.status=REVOKED` 保留授权历史，不删除用户；
- 老人和家属使用 `users.is_active` 软失效，已产生的安全事件和通知记录不物理删除；
- 设备删除后，事件保留外部设备 ID，内部设备外键置空；
- 原始消息被视觉事件引用时禁止删除；
- 风险事件存在处置记录时禁止删除；
- 联系人或用户存在告警投递记录时禁止删除，改用状态失效；
- 正式业务应使用状态失效或数据保留任务，而不是直接物理删除审计链。

## 迁移命令

```bash
cd backend
uv run alembic upgrade head
uv run alembic current
uv run alembic check
```

任何后续表结构修改必须创建新的 Alembic revision，不得修改已经合入共享分支的历史迁移。
