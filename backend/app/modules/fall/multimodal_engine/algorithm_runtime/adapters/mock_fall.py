from datetime import datetime, timezone
from typing import Any

from app.modules.fall.multimodal_engine.algorithm_runtime.adapter import AlgorithmAdapter
from app.modules.fall.multimodal_engine.algorithm_runtime.contracts import AlgorithmFinding
from app.modules.fall.multimodal_engine.schemas.risk_event import EvidenceItem, RiskLevel, RiskModule


class MockFallAdapter(AlgorithmAdapter[Any]):
    """Phase 7-B固定假结果；不读取视频、不加载模型。"""

    module = RiskModule.FALL
    model_version = "mock-fall-adapter-1.0"

    def consume(self, input_data: Any = None) -> AlgorithmFinding:
        self._ensure_running()
        return AlgorithmFinding(
            module=self.module,
            event_type="PRE_FALL_RISK",
            occurred_at=datetime.now(timezone.utc),
            risk_score=0.91,
            risk_level=RiskLevel.HIGH,
            summary="Mock适配器检测到连续失衡与快速下坠风险",
            evidence=[
                EvidenceItem(
                    code="mock_fall_probability",
                    label="模拟跌倒风险概率",
                    value=0.91,
                ),
                EvidenceItem(
                    code="mock_continuous_frames",
                    label="模拟连续风险帧数",
                    value=12,
                    unit="frame",
                ),
            ],
            recommended_action="进行语音提醒并关注后续状态",
            model_version=self.model_version,
        )
