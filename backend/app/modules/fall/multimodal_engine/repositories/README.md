# Repository层

- `MonitoringRepository`：负责 `monitoring_sessions` 的基础CRUD；
- `RiskEventRepository`：负责 `risk_events` 的基础CRUD。

Repository只执行SQLAlchemy查询、写入和`flush`，不提交事务，也不处理模拟规则、风险分级或外部平台逻辑。
