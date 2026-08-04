# 老年安全监测项目空骨架设计

日期：2026-08-04

## 1. 目标

在 `/Users/yzh666/workspace/tzb-yingshi-app` 建立两人协作开发使用的单仓库空骨架，并连接 GitHub 仓库 `w0nderful-yzh/tzb-yingshi-app`。

本次只交付目录、占位文件和文档，不交付任何业务实现或可运行工程。

## 2. 范围

本次创建：

- Android、FastAPI、文档、样例、脚本、Docker 和 GitHub 配置的约定目录；
- 用于保留空目录的 `.gitkeep`；
- 根目录 `README.md`；
- 根目录 `.gitignore`；
- 合作开发约束文档；
- 防诈、跌倒、API、架构和测试文档入口。

本次不创建：

- Kotlin、Python 或 SQL 源代码；
- Gradle、`pyproject.toml`、Alembic、Docker Compose 或 CI 工作流配置；
- 数据库、模型、数据集、音视频样本和密钥；
- 对现有防诈项目的复制、迁移或引用；
- 任何已实现功能声明。

## 3. 目录设计

```text
tzb-yingshi-app/
├── android/
├── backend/
│   ├── app/
│   │   ├── api/v1/
│   │   ├── common/
│   │   ├── core/
│   │   ├── infrastructure/
│   │   │   ├── database/
│   │   │   ├── external/ys7/
│   │   │   ├── storage/
│   │   │   └── websocket/
│   │   ├── modules/
│   │   │   ├── devices/
│   │   │   ├── events/
│   │   │   ├── fall/
│   │   │   ├── fraud/
│   │   │   ├── notifications/
│   │   │   └── users/
│   │   └── workers/
│   ├── alembic/
│   ├── models/
│   ├── storage/
│   └── tests/
│       ├── contract/
│       ├── fixtures/
│       └── integration/
├── docker/
├── docs/
│   ├── api/
│   ├── architecture/
│   ├── modules/
│   ├── superpowers/specs/
│   └── testing/
├── samples/
│   ├── audio/
│   └── video/
├── scripts/
├── .github/workflows/
├── .gitignore
└── README.md
```

Android 内部源码目录暂不展开。正式初始化 Compose 工程时由 Android 构建工具生成，避免当前骨架制造一套无法构建的伪工程。

## 4. 文档设计

`README.md` 说明赛题背景、规划能力、架构边界、目录用途、未来启动位置、环境变量原则和当前状态。所有能力均标为“规划”或“未实现”。

`docs/DEVELOPMENT_CONSTRAINTS.md` 约束模块所有权、统一风险事件、API 规范、Git 分支与 PR、密钥管理、测试和 Definition of Done。

其余文档入口只定义后续内容应该写在哪里，不提前设计实现细节。

## 5. Git 设计

- 主分支使用 `master`；
- 远端名称使用 `origin`；
- 远端地址为 `https://github.com/w0nderful-yzh/tzb-yingshi-app.git`；
- 本次不向远端推送；
- 后续功能通过分支和 Pull Request 合入 `master`。

## 6. 验收标准

- 目标目录是独立 Git 仓库，当前分支为 `master`；
- `origin` 指向指定 GitHub 仓库；
- 目录与协作规范一致；
- 除 `.gitkeep`、Markdown 和 `.gitignore` 外没有新增项目文件；
- README 不把规划能力写成已实现能力；
- 仓库中不存在密钥、模型、数据集或媒体文件。
