from app.modules.fall.multimodal_engine.algorithm_runtime.adapter import AlgorithmAdapter
from app.modules.fall.multimodal_engine.algorithm_runtime.contracts import AlgorithmFinding
from app.modules.fall.multimodal_engine.data_sources import UnifiedDataPacket
from app.modules.fall.multimodal_engine.schemas.risk_event import EvidenceItem, RiskLevel, RiskModule


class MockRadarRiskAdapter(AlgorithmAdapter[UnifiedDataPacket]):
    """验证RADAR Packet接入风险事件链路，不执行真实雷达推理。"""

    module = RiskModule.FALL
    model_version = "mock-radar-risk-adapter-1.0"

    def consume(self, input_data: UnifiedDataPacket) -> AlgorithmFinding:
        self._ensure_running()
        self._validate_input(input_data)
        return AlgorithmFinding(
            module=self.module,
            event_type="PRE_FALL_RISK",
            occurred_at=input_data.timestamp,
            risk_score=0.72,
            risk_level=RiskLevel.MEDIUM,
            summary="Radar Mock适配器收到模拟雷达数据并生成风险结果",
            evidence=[
                EvidenceItem(
                    code="mock_packet_id",
                    label="模拟输入数据包",
                    value=input_data.packet_id,
                ),
                EvidenceItem(
                    code="mock_source_id",
                    label="模拟数据来源",
                    value=input_data.source_id,
                ),
                EvidenceItem(
                    code="mock_radar_distance",
                    label="模拟目标距离",
                    value=input_data.data.get("distance_m"),
                    unit="m",
                ),
            ],
            recommended_action="仅用于验证第二数据源与风险事件链路",
            model_version=self.model_version,
        )

    def _validate_input(self, input_data: UnifiedDataPacket) -> None:
        if input_data.modality != "RADAR":
            raise ValueError("MockRadarRiskAdapter only accepts RADAR packets")
        if input_data.session_id != self.context.session_id:
            raise ValueError("packet session_id does not match adapter context")
        if input_data.device_id != self.context.device_id:
            raise ValueError("packet device_id does not match adapter context")
