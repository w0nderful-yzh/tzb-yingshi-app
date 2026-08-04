# 合作开发约束

## 1. 适用范围

本约束适用于本仓库中的 Android、FastAPI、防诈、跌倒、统一事件和萤石平台接入开发。当前团队按两人协作设计，目标是保证模块可以并行开发、独立降级并通过统一协议联调。

主分支为 `master`。当前仓库只有空骨架，文中接口和结构均为后续实现必须遵守的契约，不代表功能已经完成。

## 2. 核心原则

1. FastAPI 是唯一后端入口，Android 不直接调用算法脚本或数据库。
2. 防诈和跌倒模块必须输出统一风险事件。
3. 模块算法失败不得导致整个后端无法启动。
4. 未完成能力必须提供 Mock 或功能开关后再进入联调。
5. 共享接口、数据库和公共模块的修改必须由另一人审查。
6. 不提交密钥、模型权重、真实老人数据和真实音视频。
7. README、API契约和实现必须保持一致。

## 3. 模块边界

### 3.1 防诈负责人

主要负责：

```text
backend/app/modules/fraud/
docs/modules/fraud.md
Android防诈功能包（页面、ViewModel和Repository）
```

职责包括语音文本窗口、风险证据提取、风险阶段判断、事件证据生成、防诈接口和对应 Android 页面。

### 3.2 跌倒负责人

主要负责：

```text
backend/app/modules/fall/
docs/modules/fall.md
Android跌倒功能包（页面、ViewModel和Repository）
```

职责包括人体或姿态结果处理、时序判断、跌倒置信度、事件证据生成、跌倒接口和对应 Android 页面。

### 3.3 共享区域

以下内容属于共享区域：

```text
backend/app/api/
backend/app/common/
backend/app/core/
backend/app/infrastructure/
backend/app/modules/events/
backend/app/modules/devices/
Android公共模型、网络、存储和设计系统
Android全局导航与共享页面
docs/api/
docs/architecture/
```

修改共享区域必须说明影响模块、补充契约测试、同步文档并由另一人审查。引入不兼容变更的人负责完成防诈、跌倒和 Android 三侧适配。

## 4. 后端分层

### API层

- 接收请求和校验参数；
- 调用 Service；
- 返回统一响应；
- 不写算法，不直接访问数据库。

### Service层

- 编排业务流程；
- 调用 Detector 和 Repository；
- 执行去重、状态判断和统一事件转换；
- 不包含具体模型加载代码。

### Detector层

- 负责模型生命周期、特征处理、规则和推理；
- 只返回模块内部检测结果；
- 不保存数据库，不推送 Android，不决定业务事件状态。

### Repository层

- 只负责持久化；
- 不包含算法、外部HTTP调用或风险等级判断。

## 5. 统一风险事件

两个算法模块必须转换为同一外层结构：

```python
class RiskEventCreate(BaseModel):
    source_event_id: str
    device_id: str
    event_type: EventType
    risk_level: RiskLevel
    confidence: float
    summary: str
    occurred_at: datetime
    evidence: dict[str, Any]
    model_name: str
    model_version: str
```

统一枚举：

```text
EventType: FRAUD_SUSPECTED | FALL_SUSPECTED
RiskLevel: LOW | MEDIUM | HIGH | CRITICAL
EventStatus: PENDING | CONFIRMED | FALSE_ALARM | RESOLVED
```

禁止各模块自行增加另一套风险等级拼写。模块可以定义不同 `evidence` 内容，但不得改变统一外层字段。

## 6. API契约

- 统一前缀：`/api/v1`；
- 所有时间使用带时区的 ISO 8601；
- 所有请求携带或生成 `request_id`；
- 错误不得把原始堆栈返回 Android；
- 公共字段变更必须先修改 `docs/api/`，再修改后端Schema、契约测试和Android DTO。

统一响应：

```json
{
  "code": 0,
  "message": "success",
  "data": {},
  "request_id": "req_xxx"
}
```

## 7. 数据库约束

- 使用 PostgreSQL、SQLAlchemy 2.x 和 Alembic；
- 所有结构变更必须通过迁移；
- 禁止手工修改共享数据库结构；
- 迁移合入 `master` 后不得重写历史；
- `source_event_id` 建立唯一约束，防止重复事件；
- 后合并迁移的人负责解决迁移头分叉。

基础实体规划为 `users`、`devices`、`risk_events`、`event_actions` 和 `model_runs`。引入实体前必须在架构文档中说明用途和所有权。

## 8. 事件幂等与实时消息

- 防诈和跌倒事件都必须提供稳定的 `source_event_id`；
- 重复事件不重复创建，可以更新证据或置信度；
- 云信令和 WebSocket 消息必须保留事件发生时间与接收时间；
- 消费端必须处理重复、迟到、乱序、断线和重连；
- 云信令监听收到消息后先入队，不在监听回调中执行耗时推理。

## 9. 配置与密钥

只提交不含真实值的 `.env.example`。以下内容禁止进入Git：

- `.env`；
- 数据库密码和JWT密钥；
- 萤石 AppSecret、accessToken 和设备验证码；
- 模型服务Key；
- 私钥、证书和签名文件；
- 真实老人数据、音视频和数据库备份；
- 模型权重和大型数据集。

Android 中不得保存服务端密钥。日志不得包含完整Token、验证码、银行卡号、身份证号、完整通话文本或未脱敏音视频地址。

## 10. Git协作

禁止直接推送 `master`。分支命名：

```text
feat/fraud-xxx
feat/fall-xxx
feat/android-xxx
fix/fraud-xxx
fix/fall-xxx
refactor/shared-xxx
docs/xxx
test/xxx
chore/xxx
```

提交格式：

```text
<type>(<scope>): <description>
```

一个提交只完成一个明确目标。PR必须包含：

- 本次修改；
- 影响模块；
- 接口或数据库变更；
- 测试方式和结果；
- 截图或请求示例；
- 已知问题。

PR只有在CI通过、另一人审查、文档同步、核心测试通过且不包含敏感文件时才能合并。推荐 Squash Merge。

## 11. 编码约束

### Python

- 使用 `uv` 管理依赖；
- 使用 `ruff`、`mypy` 和 `pytest`；
- 对外函数必须有类型标注；
- Pydantic Schema 与数据库Model分离；
- 禁止裸 `except:`；
- 禁止在模块导入时执行耗时推理；
- 模型在应用或Worker生命周期中受控加载；
- 路径使用 `pathlib.Path`。

### Android

- 使用 ViewModel 和 StateFlow 管理页面状态；
- Composable 不直接调用 Retrofit；
- Repository 负责数据来源；
- DTO、Domain Model 和 UI State 分离；
- 后端地址不得硬编码在Kotlin文件中；
- 网络操作必须有加载、成功和失败状态；
- 风险等级必须同时使用文字和图标表达，不能只依赖颜色。

## 12. 测试要求

每个模块至少包含单元测试、异常测试和统一事件契约测试。共享接口变更必须同时验证防诈、跌倒和 Android DTO。

后续提交前的目标检查命令：

```text
uv run ruff check .
uv run ruff format --check .
uv run mypy app
uv run pytest
./gradlew lint
./gradlew testDebugUnitTest
./gradlew assembleDebug
```

当前仓库尚无可运行工程，因此这些命令暂不可执行。首次初始化工程的人负责补全实际命令和CI。

## 13. Definition of Done

一个功能只有同时满足以下条件才算完成：

- 功能和异常处理完成；
- 单元测试和必要的契约测试完成；
- API、数据库和Android适配同步；
- 文档与README更新；
- 本地运行和CI通过；
- 不包含密钥、模型和敏感数据；
- 另一人完成审查；
- 合入后 `master` 仍可运行。

## 14. 文档维护

- 修改启动方式、环境变量或目录结构时必须更新根README；
- 修改公共接口时必须更新 `docs/api/`；
- 修改模块职责或数据流时必须更新 `docs/modules/` 或 `docs/architecture/`；
- 规划能力与已实现能力必须明确分开；
- 不允许只在聊天消息中宣布不兼容变更。
