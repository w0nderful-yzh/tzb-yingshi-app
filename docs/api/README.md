# API契约

## 当前状态

当前没有已实现接口。以下内容是后续联调的规划契约，正式实现时必须补充请求、响应、错误码和示例。

## 统一前缀

```text
/api/v1
```

## 规划接口

```text
POST   /api/v1/fraud/analyze
POST   /api/v1/fall/analyze
GET    /api/v1/events
GET    /api/v1/events/{event_id}
PATCH  /api/v1/events/{event_id}/status
GET    /api/v1/devices
WS     /api/v1/ws/events
```

## 统一响应

```json
{
  "code": 0,
  "message": "success",
  "data": {},
  "request_id": "req_xxx"
}
```

## 变更流程

1. 先修改本目录中的契约文档；
2. 更新Pydantic Schema；
3. 更新后端契约测试；
4. 更新Android DTO；
5. 在PR中说明兼容性影响。
