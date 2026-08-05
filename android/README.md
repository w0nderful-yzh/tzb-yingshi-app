# SafeGuard Android App（守护安）

老年居家安全 App Demo：老人端 + 家属端双角色。界面还原 `prototype/pages/` 原型，接口对齐 `docs/api/app-client-api.md`。

## 技术栈

- Kotlin 2.4 + Jetpack Compose（Material 3）+ MVVM + StateFlow
- Retrofit 2 + OkHttp + kotlinx.serialization
- 萤石 Android EZOpenSDK 5.30.2（原生 H.264/H.265 实时预览）
- Navigation Compose；无 Hilt（`ServiceLocator` 手动装配，Demo 阶段降低接入成本）
- AGP 9（内置 Kotlin，不再应用 `org.jetbrains.kotlin.android` 插件）

## 分层结构

```text
app/src/main/java/com/tzb/safeguard/
├── SafeGuardApp.kt          # ServiceLocator（DI）+ Session（当前角色）
├── MainActivity.kt
├── data/
│   ├── model/Models.kt      # 契约模型，字段与接口文档 snake_case 一致
│   ├── network/
│   │   ├── ApiService.kt        # Retrofit 接口，路径对应接口文档
│   │   ├── MockInterceptor.kt   # 规划接口的本地 Mock（契约级演示数据）
│   │   └── NetworkModule.kt     # OkHttp/Retrofit 装配
│   └── repository/SafeRepository.kt  # 统一 ApiResponse -> Result<T>，异常兜底
└── ui/
    ├── theme/Theme.kt       # 适老化配色与字号（对照 prototype/assets/style.css）
    ├── components/          # 状态容器、告警卡片、大按钮、底导航、Canvas 图表
    ├── navigation/NavGraph.kt
    └── screens/             # role / home / monitor / alerts / alertdetail / care / profile / family
```

## 构建运行

```bash
# local.properties 中配置 sdk.dir（本机已生成）
gradle :app:assembleDebug
# 产物：app/build/outputs/apk/debug/app-debug.apk
```

- `MOCK_MODE`（`app/build.gradle.kts`）：debug 和 release 均为 `false`，请求直接发送到真实后端；需要离线演示时再临时开启。
- `API_BASE_URL`：默认 `http://127.0.0.1:8000/`，模拟器/USB 真机先执行 `adb reverse tcp:8000 tcp:8000`；无线真机可用 `./gradlew :app:assembleDebug -PAPI_BASE_URL=http://局域网IP:8000/` 覆盖。
- 联调期每个请求携带 `X-Demo-Role`，由身份选择页动态切换 `elder` / `family`；写请求自动携带 `Idempotency-Key`。正式登录完成后替换为 Bearer Token。

后端首次联调先执行：

```bash
cd ../backend
uv run alembic upgrade head
uv run python -m app.scripts.seed_demo
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# 另开终端，为当前模拟器或 USB 真机转发端口
adb reverse tcp:8000 tcp:8000
```

## 页面与接口对应

| 页面 | 路由 | 主要接口 |
|---|---|---|
| 身份选择 | role | —（联调期手动入口） |
| 首页总览（老人） | home | `GET /users/me`、`GET /safety/status`、`GET /devices`、`GET /events`、`POST /sos` |
| 实时监控 | monitor | `GET /devices`、`GET /devices/{id}/live-sdk-session`，EZOpenSDK 播放设备原始 H.264/H.265 直播 |
| 消息中心 | alerts | `GET /events`（级别/状态筛选） |
| 告警详情 | alert_detail/{id} | `GET /events/{id}`、`POST /events/{id}/confirm`、`PATCH /events/{id}/status`、`POST /events/{id}/call` |
| 心理关怀 | care | 接口暂缓，本地演示数据 |
| 个人/设备 | profile | `GET /users/me`、`GET /contacts`、`GET /devices` |
| 家属看板 | family | `GET /family/elders`、`GET /events?status=open`、`GET /stats/events`、`GET /stats/activity` |

## 待办（联调后端时）

1. 鉴权：在 `NetworkModule` 追加 `Authorization` 拦截器；
2. 实时推送：接 `WS /api/v1/ws/events`，告警到达时刷新列表并弹系统通知；
3. 心理关怀：接口落地后接入 `care` 页仓库层。
