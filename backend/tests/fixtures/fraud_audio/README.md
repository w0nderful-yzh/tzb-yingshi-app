# 防诈回放测试样本说明

本目录用于存放防诈决策管线的离线回放样本。**不提交真实老人通话、原始音频、
模型权重、密钥或未脱敏数据。**

## 两类样本

### 1. 文本 manifest（可提交 Git）

纯文本的 JSONL 清单，用于驱动决策管线基准（规则 + 校准分类器 + S0-S5 状态机），
无需 ASR 模型即可运行。参考 `demo_manifest.jsonl`。

每行一个 JSON 对象：

```json
{
  "id": "sample-001",
  "turns": [{"text": "我是银行客服"}, {"text": "把短信验证码告诉我"}],
  "expected_labels": ["identity_claim", "credential_request"],
  "expected_state": "S4_ACTION_INDUCEMENT",
  "scenario": "fake_customer_service",
  "split_group": "conversation-001",
  "elder_alone": true
}
```

字段说明：

- `turns`：多轮话术数组；单条样本可用 `text` 字段代替。
- `expected_labels`：期望检出的证据标签（11 类之一）。
- `expected_state`：期望的 S0-S5 最终状态。
- `scenario`：场景标签，用于分类统计。
- `split_group`：同一原始对话及其改写必须使用相同值，禁止跨训练集/评测集，防止数据泄漏。
- `elder_alone`：是否老人独处（影响 S5 触发）。

运行：

```bash
python -m app.scripts.benchmark_fraud_pipeline \
    --manifest tests/fixtures/fraud_audio/demo_manifest.jsonl
```

### 2. 脱敏 WAV（不提交 Git）

用于端到端回放（VAD 切句 + SenseVoice FINAL + Paraformer PARTIAL）的本地 WAV 文件
清单。文件本体**不进入 Git**，仅在本地受控目录保存，使用前须完成授权与脱敏。

端到端回放工具将在此目录读取 WAV 清单，输出每条样本的：期望证据标签与状态、
音频时长、VAD 切段结果、PARTIAL 首次有效文本时间、FINAL 推理时间、最终状态与
证据链、CPU/GPU、模型版本与 Git commit。

## 隐私约束

- 电话号码、身份证号、银行卡号、验证码等敏感实体必须脱敏或替换为占位符。
- 真实老人数据只允许进入受控私有评测目录（`backend/evaluation/private/`，不提交 Git）。
- 任何对外报告不得包含可定位到个人的原始内容。
