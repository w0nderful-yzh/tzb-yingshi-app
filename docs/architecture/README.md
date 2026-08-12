# 架构文档

## 当前状态

当前已落地 FastAPI 公共入口、配置、请求 ID、统一响应、异常处理、版本化路由、萤石告警主动轮询与模拟事件适配、Android 前台服务连续音轨、SenseVoice/Paraformer、防诈状态机、PostgreSQL 风险事件写入和单进程 WebSocket 实时通知。跌倒业务、多设备动态取流、跨进程消息总线和正式萤石云推送鉴权/解密仍未实现。

## 已实现后端入口

```text
app/main.py
   ├── core/config.py           环境配置
   ├── core/request_id.py       请求追踪
   ├── core/exceptions.py       统一错误响应
   └── api/v1/router.py
          └── routes/health.py  存活检查
```

跌倒业务后续放在 `app/modules/fall/`，再由 `app/api/v1/routes/` 中的路由调用；算法逻辑不得直接写进路由。

## 萤石事件接收第一阶段

```text
萤石告警列表主动轮询 ──→ Ys7AlarmMapper
                               ↓
HTTP模拟推送 ────────────────→ 统一消息结构
                               ↓
Ys7SignalListener
   ├── EventDeduplicator
   ├── RawSignalStore
   └── Ys7EventQueue（有界、非阻塞入队）
              ↓
       Ys7EventWorker
              ↓
       Ys7EventAdapter
              ↓
       VisualEventStore
              ├──────────────→ GET /api/v1/fraud/visual-events
              ↓
POST /api/v1/fraud/analyze ← 带绝对时间的语音转写
POST /api/v1/fraud/audio/chunks
              ↓ 后台线程 SenseVoiceSmall
       绝对时间转写片段
              ↓
       FraudSessionService
              ↓
规则 + 轻量分类器 + S0-S5 状态机
              ↓
    RiskEventRepository → PostgreSQL risk_events（事务提交）
                              ↓
                       RealtimeEventBroker
                              ↓
                    一次性票据 WebSocket → Android 通知
```

`occurred_at` 是设备或云端产生事件的时间，`received_at` 是后端接收时间。视觉事件查询按前者排序，为后续按设备会话重放乱序证据做准备。

当前使用内存队列、内存视觉事件仓库、内存活动防诈会话和磁盘原始消息；状态机生成的非 S0 风险快照已持久化到 PostgreSQL。原始消息可以用于排查和演示，但进程重启后尚不会自动扫描未消费文件；后续应将接收链路接入已有 Inbox/Visual Event 表并实现启动恢复。

主动轮询使用 AppKey/AppSecret 或 accessToken 请求萤石告警列表，不要求后端具有公网入口。比赛配置每 5 秒轮询并回看 120 秒，设备事件发现延迟平均约 2.5 秒、最坏约 5 秒，使用 `alarmId` 幂等处理重叠结果。5 秒间隔若持续 24 小时会超过萤石个人版 1 万次/天额度，因此长期运行应改为 10 秒或切换正式消息回调。当前只把明确的人体、人脸、人数和通话类告警转换成统一视觉事件，未知类型保留在轮询状态中等待按真实样例适配。

HTTP 共享令牌只是正式推送签名协议到位前的开发保护措施。拿到萤石正式消息样例后，只替换 `external/ys7/` 中的鉴权、解密和解析代码，不改变统一视觉事件和防诈业务层。

SenseVoice 通过 `SpeechRecognizer` 端口与业务层隔离，FunASR 只存在于 `infrastructure/external/sensevoice/`。模型懒加载且推理串行化；API 使用工作线程调用同步模型，模型缺失或失败不会阻止健康检查、萤石视觉接收和已有转写 API。

运行设备属于语音识别基础设施配置，不改变防诈业务层和 API 契约。当前默认 Docker 构建固定使用 CPU-only Torch/Torchaudio，以便普通电脑开箱运行；Windows + NVIDIA RTX 4060 后续可通过 Docker Desktop WSL 2、独立 CUDA 镜像和 Compose GPU override 加速 SenseVoice/Paraformer 推理。GPU 方案在完成依赖锁定、容器内 CUDA 检测和端到端性能验证前只属于可选部署规划，默认 CPU 镜像必须继续保留为兼容和故障回退路径。

## 萤石直播音轨与持续守护

```text
家属显式开启持续守护
      ↓
Android Ys7MonitorService（前台服务、静音、断线重连）
      ↓ EZOpenSDK 解码 16 kHz mono PCM
POST /devices/{device_id}/audio-pcm（1 秒传输批次）
      ↓
AppPcmRelaySource（恢复为 20 ms 连续帧）
      ↓
WebRTC VAD → Paraformer PARTIAL → SenseVoice FINAL → S0-S5
```

前台服务不依赖 Activity 或 Composable，用户返回桌面或锁屏后继续监听；用户从系统设置强制停止 App 后无法继续。Token、直播地址和设备验证码不得写入日志。当前仅处理音轨；视频诈骗证据仍优先来自萤石云算法事件，避免重复持续运行本地视觉模型。

`RealtimeEventBroker` 当前是有界内存广播，符合 Compose 单 Uvicorn Worker 部署。多 Worker 或多实例部署前必须替换为跨进程消息总线。

## 数据库

PostgreSQL 表结构、关系、索引、JSONB 边界和删除策略见 [PostgreSQL 数据库设计](database-schema.md)。数据库结构必须通过 Alembic 迁移维护，应用启动时不调用 `create_all`。

## 当前主链路

```text
Android Ys7MonitorService
   ↓ PCM / REST
FastAPI API层
   ↓
业务Service层
   ├── 防诈Detector
   ├── 跌倒Detector
   └── 萤石适配器
   ↓
统一风险事件中心
   ↓ 先提交
PostgreSQL
   ↓ 后广播
RealtimeEventBroker → 一次性票据 WebSocket → Android 系统通知
```

## 架构原则

- FastAPI 是唯一后端入口；
- 算法模块不直接依赖 Android；
- Android 不直接访问数据库和算法脚本；
- 萤石原始消息先转换成项目内部事件；
- 防诈和跌倒模块使用统一风险事件外层结构；
- 外部服务通过 `infrastructure/external/` 适配，不泄漏供应商类型到业务层；
- 一个模块关闭或失败时，其他模块仍可使用。

## 设计决策记录

新增重大架构决策时，在本目录增加独立Markdown文件，至少记录背景、决策、替代方案、影响和回滚方式。
