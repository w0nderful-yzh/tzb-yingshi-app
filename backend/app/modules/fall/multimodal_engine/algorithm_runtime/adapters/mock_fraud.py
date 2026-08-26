from datetime import datetime, timezone
from typing import Any

from app.modules.fall.multimodal_engine.algorithm_runtime.adapter import AlgorithmAdapter
from app.modules.fall.multimodal_engine.algorithm_runtime.contracts import AlgorithmFinding
from app.modules.fall.multimodal_engine.schemas.risk_event import EvidenceItem, RiskLevel, RiskModule


class MockFraudAdapter(AlgorithmAdapter[Any]):
    """固定诈骗风险假结果；不运行ASR、音频处理或真实规则模型。"""

    module = RiskModule.FRAUD
    model_version = "mock-fraud-adapter-1.0"

    def consume(self, input_data: Any = None) -> AlgorithmFinding:
        self._ensure_running()
        return AlgorithmFinding(
            module=self.module,
            event_type="SUSPICIOUS_SPEECH",
            occurred_at=datetime.now(timezone.utc),
            risk_score=0.89,
            risk_level=RiskLevel.HIGH,
            summary="Mock文本同时命中验证码索取与资金操作风险规则",
            evidence=[
                EvidenceItem(
                    code="mock_verification_code_request",
                    label="模拟命中索取验证码规则",
                    value=True,
                ),
                EvidenceItem(
                    code="mock_transfer_request",
                    label="模拟命中资金操作规则",
                    value=True,
                ),
            ],
            recommended_action="提示老人暂停操作并联系家属核实",
            model_version=self.model_version,
        )
