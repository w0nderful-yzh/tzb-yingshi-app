from datetime import datetime, timezone
from uuid import uuid4

from app.modules.fall.multimodal_engine.data_sources.adapter import DataSourceAdapter
from app.modules.fall.multimodal_engine.data_sources.contracts import UnifiedDataPacket


class MockCameraAdapter(DataSourceAdapter):
    """产生摄像头引用数据的Phase 7-C1测试数据源。"""

    def __init__(
        self,
        *,
        source_id: str = "mock-camera",
        device_id: str = "camera-001",
        frame_ref: str = "mock://camera/frame-001",
    ) -> None:
        super().__init__()
        self.source_id = source_id
        self.device_id = device_id
        self.frame_ref = frame_ref

    def read(self) -> UnifiedDataPacket:
        self._ensure_running()
        return UnifiedDataPacket(
            packet_id=f"mock-camera-{uuid4().hex}",
            session_id=self.session_id,
            source_id=self.source_id,
            device_id=self.device_id,
            modality="VIDEO",
            timestamp=datetime.now(timezone.utc),
            data={"frame_ref": self.frame_ref},
        )
