# Service层

- `MonitoringService`：创建监测会话并查询当前运行会话；
- `RiskEventService`：保存统一风险事件、检查重复`event_id`并更新处理状态。
- `SimulationService`：读取固定场景、绑定当前会话并调用`RiskEventService`。

Service负责业务编排和事务边界。`SimulationService`不访问Repository；模拟、未来算法和萤石事件统一通过`RiskEventService`保存。
