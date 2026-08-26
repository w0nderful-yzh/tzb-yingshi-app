# ORM模型层

- `MonitoringSession` 对应 `monitoring_sessions`；
- `RiskEvent` 对应 `risk_events`。

字段、约束和索引与 `backend/sql/002_create_tables.sql` 保持一致。数据库结构仍由SQL文件初始化，不调用 `Base.metadata.create_all()` 自动建表。
