# 跌倒风险 App Adapter

## 正式决策路径

- `camera_led_radar_evidence`：客厅使用 Camera 主风险判断，关联后的 TI Tracking 雷达运动证据只做增强；风险分数取 `camera_score`。
- `camera_only`：客厅 Camera 可用但雷达证据暂不可用或未关联时，继续使用 Camera 正式判断。
- `radar_only`：卫生间及无摄像头卧室使用 Radar TCN 短时风险；风险分数取真实字段 `pre_fall_score`，但 UI 不得称为“跌倒概率”。
- `unavailable`：该房间正式路径所需的主传感器不可用。

Radar-only 是无摄像头空间的独立能力，不是客厅 C 路的 fallback。当前算法工程的 TCN 响应仍明确标记为 `shadow_only=true`、`alert_suppressed=true`，Adapter 会读取并闭锁为 `unavailable`，直到上游使用同一契约发布正式结果。

## 代码边界

```text
backend/app/modules/fall/
├── schemas.py          # 稳定 App-facing contract
├── source_schemas.py   # 算法服务输入边界，忽略未知调试字段
├── ports.py            # FallRiskSource 接口
├── mapping.py          # 正式状态到业务字段和中文摘要的映射
├── service.py          # 房间能力路由与 overview 聚合
├── radar_module/       # IWR6843 B0 calibrated TCN 独立算法进程
└── multimodal_engine/  # Camera、Alignment、Eligibility、Fusion v2 独立进程

backend/app/infrastructure/external/fall_risk/
└── client.py           # 独立算法服务 HTTP client

android/.../data/fall/
├── model/FallRiskModels.kt
├── network/FallRiskApi.kt
└── repository/FallRiskRepository.kt
```

## App API

`GET /api/v1/fall-risk/overview?elder_id=...` 使用现有 Bearer 身份和老人绑定关系校验，返回房间级：

- `decision_path`
- `risk_level` 与风险指数 `risk_score`（不得在 UI 中称为跌倒概率）
- `prediction_state`、`fall_event_status`
- `camera_status`、`radar_status`
- `association_status`、`joint_assessment`
- `evidence_summary`、`updated_at`

App contract 不包含实验名、checkpoint、raw track ID、sync delta、association confidence、shadow-only 或大量 reason codes。

## 上游接口配置

启用 `APP_FALL_RISK_ENABLED=true` 后配置：

- 客厅 Camera-led 路：`APP_FALL_RISK_BASE_URL` 指向仓库内 `backend/app/modules/fall/multimodal_engine` 独立算法服务，请求 `GET /api/multimodal/camera-led-associated/latest`。该端点以 `camera_led_evidence_fusion_v2` 作为 App 实时风险结果（`realtime_active=true`、`shadow_only=false`、`affects_app_result=true`），仍保持 `affects_alerts=false`。Track-first / Risk-late-binding 继续仅在独立 Shadow Pipeline 中运行。
- 卫生间/卧室 Radar-only：`APP_FALL_RISK_RADAR_ROOM_URLS` 以 JSON 对象配置每个房间的 Radar 服务基址，各自请求 `GET /api/radar/latest`。Radar 服务实例的响应内 `room` 必须和配置键一致。

真实算法端点都没有 `elder_id` 参数，Adapter 不会把 App 用户标识转发给算法服务。路径可以通过环境变量调整；上游不可用、房间不匹配或载荷不合法时只把对应房间映射为 `unavailable`，不阻断首页其他能力。

## 真实字段边界

- 客厅主风险：`camera_led_evidence_fusion_v2.camera_led_score`、`camera_led_state`；该分数由契约保证与有效 BioSTGCN Camera score 相同。
- 关联证据：`fusion_mode`、`association_state`、`radar_motion_evidence_strength` 与 `radar_eligible`；只转换成用户可理解的 `decision_path`、`joint_assessment` 与摘要。
- 传感器状态：两路各自的 `available` 与 `quality_level`。
- 事件：`fall_event.fall_event_status`。
- Radar-only：优先 `calibrated_tcn_prediction`，其次 `tcn_prediction` 或 `tcn_baseline`；读取 `pre_fall_score`、正式状态、`score_valid`、`data_quality`、`shadow_only` 与 `alert_suppressed`。

checkpoint、阈值、raw track/sync/association 字段和 reason codes 只留在算法内部或调试日志，不进入 App-facing contract。

## 一键守护会话生命周期

Camera 直播预览不是守护会话的一部分，进入直播页即可独立取流。正式 UI 只提供一个“开始/停止守护”入口，用同一个会话统一控制 Camera 跌倒分析、诈骗 PCM 转发、心理周期观察和 Radar Evidence 参与状态：

```text
App 开始守护
  -> POST /api/v1/guard-session/start（幂等）
  -> 多模态引擎 /api/guard-session/start（幂等）
  -> Camera 分析启用
  -> Radar API ensure_running（系统级单例）
  -> Radar Evidence 绑定当前 session

App 停止守护
  -> Camera 分析停止、诈骗 PCM 转发停止、心理周期观察停止
  -> Radar Evidence 解除当前 session 绑定
  -> Radar Worker 保持运行
```

生命周期接口：

- `POST /api/v1/guard-session/start`：需要登录、老人权限和 `Idempotency-Key`；重复请求返回同一活动会话。
- `POST /api/v1/guard-session/stop`：需要登录和 `Idempotency-Key`；重复停止安全返回 `STOPPED`。
- `GET /api/v1/guard-session/status`：返回 Camera 分析、诈骗监听、心理观察、Radar Worker、Radar 会话参与和 Fusion 的独立状态。
- Radar 服务的 `POST /api/radar/ensure-running` 与 `GET /api/radar/status` 只管理/查询系统级单例，不属于最终 App UI。

Radar 缺失、关联失败或质量不足不会阻止会话启动；状态中保留原因码并让 Fusion 安全降级为 Camera-only。该生命周期没有修改 BioSTGCN、B0 calibrated TCN、Fusion v2、阈值或 50 ms Gate。
