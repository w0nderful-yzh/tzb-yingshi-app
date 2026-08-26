from datetime import datetime, timezone
from uuid import uuid4

from app.modules.fall.multimodal_engine.data_sources.adapter import DataSourceAdapter
from app.modules.fall.multimodal_engine.data_sources.contracts import UnifiedDataPacket


class DummyRadarAdapter(DataSourceAdapter):
    """模拟第二平台雷达数据的Phase 7-C1架构验证数据源。"""

    def __init__(
        self,
        *,
        source_id: str = "dummy-radar-platform",
        device_id: str = "radar-001",
        distance_m: float = 1.35,
        vertical_velocity_mps: float = -0.42,
        height_m: float = 0.81,
    ) -> None:
        super().__init__()
        self.source_id = source_id
        self.device_id = device_id
        self.distance_m = distance_m
        self.vertical_velocity_mps = vertical_velocity_mps
        self.height_m = height_m

    def read(self) -> UnifiedDataPacket:
        self._ensure_running()
        return UnifiedDataPacket(
            packet_id=f"dummy-radar-{uuid4().hex}",
            session_id=self.session_id,
            source_id=self.source_id,
            device_id=self.device_id,
            modality="RADAR",
            timestamp=datetime.now(timezone.utc),
            data={
                "distance_m": self.distance_m,
                "vertical_velocity_mps": self.vertical_velocity_mps,
                "height_m": self.height_m,
            },
        )
