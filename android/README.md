# SafeGuard Android App（守护安）

面向家属的风险预测与提前介入 App。当前落地防诈 S0-S5 趋势预测、实时复核和处置审计；跌倒风险预测和心理健康趋势保留诚实占位。

## 技术栈

- Kotlin 2.4 + Jetpack Compose（Material 3）+ MVVM + StateFlow
- Retrofit 2 + OkHttp + kotlinx.serialization
- 萤石 Android EZOpenSDK 5.30.2（原生 H.264/H.265 实时预览）
- Navigation Compose；无 Hilt（`ServiceLocator` 手动装配，Demo 阶段降低接入成本）
- AGP 9（内置 Kotlin，不再应用 `org.jetbrains.kotlin.android` 插件）

## 分层结构

```text
app/src/main/java/com/tzb/safeguard/
├── SafeGuardApp.kt          # ServiceLocator（DI）+ 家属守护上下文
├── MainActivity.kt
├── data/
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
    └── screens/             # home / monitor / alerts / alertdetail / care / profile
```

## 构建运行

```bash
# local.properties 中配置 sdk.dir（本机已生成）
gradle :app:assembleDebug
# 产物：app/build/outputs/apk/debug/app-debug.apk
```

- `MOCK_MODE`（`app/build.gradle.kts`）：debug 和 release 均为 `false`，请求直接发送到真实后端；需要离线演示时再临时开启。
- `API_BASE_URL`：默认 `http://127.0.0.1:8000/`。模拟器每次启动后需执行 `adb reverse tcp:8000 tcp:8000`；无线真机可用 `./gradlew :app:assembleDebug -PAPI_BASE_URL=http://局域网IP:8000/` 覆盖。
- 联调期请求固定携带 `X-Demo-Role: family`，写请求自动携带 `Idempotency-Key`。正式登录完成后替换为 Bearer Token。

后端首次联调先执行：

```bash
cd ../backend
uv run alembic upgrade head
uv run python -m app.scripts.seed_demo
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# 模拟器每次重启后都要重新建立转发
adb reverse tcp:8000 tcp:8000
```

## 页面与接口对应

| 页面 | 路由 | 主要接口 |
|---|---|---|
| 家人看护首页 | home | `GET /family/elders`、`GET /devices`、`GET /events`、`GET /devices/{id}/live-sdk-session`；首页直接播放真实直播 |
| 现场复核（二级页） | monitor | `GET /devices`、`GET /devices/{id}/live-sdk-session`，EZOpenSDK 播放设备原始 H.264/H.265 直播 |
| 消息 | alerts | `GET /events`，只展示 `fraud_suspected`，卡片显示事件时间戳画面 |
| 预警详情 | alert_detail/{id} | `GET /events/{id}`、`PATCH /events/{id}/status`；主画面与关联画面来自 `evidence_frames` |
| 后续模块占位 | care | 不请求接口，不展示模拟结果 |
| 个人/设备 | profile | `GET /users/me`、`GET /contacts`、`GET /devices` |

## 待办（联调后端时）

1. 鉴权：在 `NetworkModule` 追加 `Authorization` 拦截器；
2. 实时推送：接 `WS /api/v1/ws/events`，风险阶段变化时刷新列表并弹系统通知；
3. 入户场景：接入萤石正式访客/停留事件字段，补足跨时间证据；
4. 跌倒与心理关怀：由对应团队按 `docs/product/fraud-first-app.md` 的统一事件边界接入。
5. 历史回放与设备语音提醒：App 已接 `GET /devices/{id}/history-playback`、`POST /events/{id}/intervention-reminder`，后端当前以 `501/TODO` 明确占位。
