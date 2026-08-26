from datetime import datetime, timezone
from typing import Any

from app.modules.fall.multimodal_engine.algorithm_runtime.adapter import AlgorithmAdapter
from app.modules.fall.multimodal_engine.algorithm_runtime.contracts import AlgorithmFinding
from app.modules.fall.multimodal_engine.schemas.risk_event import EvidenceItem, RiskLevel, RiskModule


class MockMentalAdapter(AlgorithmAdapter[Any]):
    """固定状态趋势假结果；不执行疾病诊断或真实行为分析。"""

    module = RiskModule.MENTAL_STATE
    model_version = "mock-mental-adapter-1.0"

    def consume(self, input_data: Any = None) -> AlgorithmFinding:
        self._ensure_running()
        return AlgorithmFinding(
            module=self.module,
            event_type="ACTIVITY_DECLINE",
            occurred_at=datetime.now(timezone.utc),
            risk_score=0.63,
            risk_level=RiskLevel.MEDIUM,
            summary="Mock行为窗口显示活动量下降并伴有长时间静止",
            evidence=[
                EvidenceItem(
                    code="mock_activity_decline",
                    label="模拟活动量下降比例",
                    value=28,
                    unit="%",
                ),
                EvidenceItem(
                    code="mock_inactivity_duration",
                    label="模拟连续静止时长",
                    value=95,
                    unit="min",
                ),
            ],
            recommended_action="建议关注近期活动与生活节律变化",
            model_version=self.model_version,
        )
