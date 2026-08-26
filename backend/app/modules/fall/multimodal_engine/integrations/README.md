# 外部集成层

本目录用于隔离外部平台SDK、HTTP API和协议适配，避免外部平台逻辑进入风险事件业务核心。

当前只实现萤石服务器侧认证、设备列表查询和播放配置获取：

```text
integrations/
└── ezviz/
    ├── client.py
    ├── auth.py
    └── schemas.py
```

未来必须遵守以下边界：

1. AppKey、AppSecret、AccessToken和设备验证码仅由FastAPI后端管理；
2. Vue不直接调用萤石API；
3. 萤石设备事件如需形成风险事件，必须转换为统一RiskEvent后进入RiskEventService；
4. 萤石客户端不直接写MySQL；
5. 播放配置接口只返回后续播放器所需配置，不执行视频播放或拉流；
6. 当前不包含播放器、视频解码、Webhook或算法推理。
