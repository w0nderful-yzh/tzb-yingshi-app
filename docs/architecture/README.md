# 架构文档

## 当前状态

当前只有目录和文档，没有可运行架构。本文件记录后续实现必须遵循的模块边界。

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
