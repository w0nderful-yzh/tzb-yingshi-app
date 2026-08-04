# 架构文档

## 当前状态

当前已落地 FastAPI 公共入口、配置、请求 ID、统一响应、异常处理、版本化路由、萤石模拟事件适配、防诈 Service/状态机和 PostgreSQL 风险事件写入。跌倒业务、统一事件查询与处置、WebSocket 和正式萤石平台鉴权/解密仍未实现。

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
HTTP模拟推送
   ↓ 共享令牌和消息结构校验
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
              ↓
       FraudSessionService
              ↓
规则 + 轻量分类器 + S0-S5 状态机
              ↓
    RiskEventRepository → PostgreSQL risk_events
```

`occurred_at` 是设备或云端产生事件的时间，`received_at` 是后端接收时间。视觉事件查询按前者排序，为后续按设备会话重放乱序证据做准备。

当前使用内存队列、内存视觉事件仓库、内存活动防诈会话和磁盘原始消息；状态机生成的非 S0 风险快照已持久化到 PostgreSQL。原始消息可以用于排查和演示，但进程重启后尚不会自动扫描未消费文件；后续应将接收链路接入已有 Inbox/Visual Event 表并实现启动恢复。

当前 HTTP 共享令牌只是正式签名协议到位前的开发保护措施。拿到萤石正式消息样例后，只替换 `external/ys7/` 中的鉴权、解密和解析代码，不改变统一视觉事件和防诈业务层。

## 数据库

PostgreSQL 表结构、关系、索引、JSONB 边界和删除策略见 [PostgreSQL 数据库设计](database-schema.md)。数据库结构必须通过 Alembic 迁移维护，应用启动时不调用 `create_all`。

## 规划结构

```text
Android
   ↓ REST / WebSocket
FastAPI API层
   ↓
业务Service层
   ├── 防诈Detector
   ├── 跌倒Detector
   └── 萤石适配器
   ↓
统一风险事件中心
   ├── PostgreSQL
   └── WebSocket通知
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
