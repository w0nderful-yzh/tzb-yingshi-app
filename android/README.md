# SafeGuard Android App（守护安）

面向家属的风险预测与提前介入 App。当前落地防诈 S0-S5 趋势预测、实时复核和处置审计；跌倒风险预测和心理健康趋势保留诚实占位。

## 技术栈

- Kotlin 2.4 + Jetpack Compose（Material 3）+ MVVM + StateFlow
- Retrofit 2 + OkHttp + kotlinx.serialization
- 萤石 Android EZOpenSDK 5.30.2（原生 H.264/H.265 实时预览与 16 kHz PCM 转发）
- Navigation Compose；无 Hilt（`ServiceLocator` 手动装配，Demo 阶段降低接入成本）
- AGP 9（内置 Kotlin，不再应用 `org.jetbrains.kotlin.android` 插件）

## 分层结构

```text
app/src/main/java/com/tzb/safeguard/
├── SafeGuardApp.kt          # ServiceLocator（DI）+ 家属守护上下文
├── MainActivity.kt
├── data/
│   ├── auth/AuthStore.kt    # Bearer Token 本地登录态
│   ├── media/CameraAudioRelay.kt # SDK PCM 的有界、非阻塞网络转发
│   ├── media/Ys7MonitorService.kt # 页面关闭后持续取流、重连和通知
│   ├── realtime/AlertWebSocketClient.kt # 一次性票据 WebSocket
│   ├── model/Models.kt      # 契约模型，字段与接口文档 snake_case 一致
│   ├── network/
│   │   ├── ApiService.kt        # Retrofit 接口，路径对应接口文档
│   │   ├── MockInterceptor.kt   # 规划接口的本地 Mock（契约级演示数据）
│   │   └── NetworkModule.kt     # OkHttp/Retrofit 装配
│   └── repository/SafeRepository.kt  # 统一 ApiResponse -> Result<T>，异常兜底
└── ui/
    ├── theme/Theme.kt       # 家属端白底极简视觉
    ├── components/          # 状态容器、防诈事件卡片、大按钮、底导航
    ├── navigation/NavGraph.kt
    └── screens/             # login / home / monitor / alerts / alertdetail / care / profile
```

## 构建运行

```bash
# local.properties 中配置 sdk.dir（本机已生成）
gradle :app:assembleDebug
# 产物：app/build/outputs/apk/debug/app-debug.apk
```

- `MOCK_MODE`（`app/build.gradle.kts`）：debug 和 release 均为 `false`，请求直接发送到真实后端；需要离线演示时再临时开启。
- `API_BASE_URL`：默认 `http://127.0.0.1:8000/`。模拟器每次启动后需执行 `adb reverse tcp:8000 tcp:8000`；无线真机可用 `./gradlew :app:assembleDebug -PAPI_BASE_URL=http://局域网IP:8000/` 覆盖。
- 登录成功后 Token 保存在 App 私有 SharedPreferences；OkHttp 自动携带 `Authorization: Bearer`。接口返回 401 时清除本地登录态并回到登录页。

后端首次联调先执行：

```bash
cd ../backend
uv run alembic upgrade head
uv run python -m app.scripts.seed_demo
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# 模拟器每次重启后都要重新建立转发
adb reverse tcp:8000 tcp:8000
```

Docker 默认演示家属账号为 `guardian / guardian123`，可在根目录 `.env` 中通过 `APP_DEMO_GUARDIAN_LOGIN` 和 `APP_DEMO_GUARDIAN_PASSWORD` 修改。页面展示姓名来自后端用户表，不再写死在 App 中。

当后端设置 `APP_YS7_MEDIA_SOURCE=app_relay` 时，用户在“我的”页开启持续守护，`Ys7MonitorService` 会保持 SDK 静音解码，并把 16 kHz 单声道 PCM 按 1 秒网络批次发送到后端。直播页面只负责可视预览，不再承担采集。返回桌面、切换页面或锁屏不会停止服务；系统设置中的“强制停止”会终止全部 App 能力。

服务同时通过 `POST /ws/tickets` 获取 60 秒一次性票据并建立 `WS /ws/events`。风险事件到达后直接创建高优先级系统通知，点击进入事件详情；断线后重新申请票据并以 1–30 秒指数退避重连。

## 页面与接口对应

| 页面 | 路由 | 主要接口 |
|---|---|---|
| 登录 | login | `POST /auth/login`，令牌失效后自动返回 |
| 家人看护首页 | home | `GET /family/elders`、`GET /devices`、`GET /events`、`GET /devices/{id}/live-sdk-session`；首页按需播放真实直播 |
| 现场复核（二级页） | monitor | `GET /devices`、`GET /devices/{id}/live-sdk-session`，EZOpenSDK 播放设备原始 H.264/H.265 直播 |
| 消息 | alerts | `GET /events`，只展示 `fraud_suspected`，卡片显示事件时间戳画面 |
| 预警详情 | alert_detail/{id} | `GET /events/{id}`、`PATCH /events/{id}/status`；主画面与关联画面来自 `evidence_frames` |
| 后续模块占位 | care | 不请求接口，不展示模拟结果 |
| 个人/设备 | profile | `GET /users/me`、`GET /contacts`、`GET /devices`；显式启停持续守护 |

## 待办（联调后端时）

1. 认证增强：增加注册、改密、找回密码和刷新令牌；正式发布时将 Token 存储升级为 Android Keystore 保护；
2. 离线必达：当前 WebSocket 依赖持续守护服务在线；生产版补充国内厂商推送或 FCM；
3. 入户场景：接入萤石正式访客/停留事件字段，补足跨时间证据；
4. 跌倒与心理关怀：由对应团队按 `docs/product/fraud-first-app.md` 的统一事件边界接入。
5. 历史回放与设备语音提醒：App 已接 `GET /devices/{id}/history-playback`、`POST /events/{id}/intervention-reminder`，后端当前以 `501/TODO` 明确占位。
