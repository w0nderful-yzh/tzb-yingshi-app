from abc import ABC, abstractmethod

from app.modules.fall.multimodal_engine.data_sources.contracts import UnifiedDataPacket


class DataSourceAdapter(ABC):
    """比赛原型中不同数据源共用的最小同步读取边界。"""

    def __init__(self) -> None:
        self._session_id: str | None = None

    @property
    def is_running(self) -> bool:
        return self._session_id is not None

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            raise RuntimeError("data source adapter has not been started")
        return self._session_id

    def start(self, session_id: str) -> None:
        normalized = session_id.strip()
        if not normalized:
            raise ValueError("session_id must not be blank")
        self._session_id = normalized

    @abstractmethod
    def read(self) -> UnifiedDataPacket | None:
        """读取并标准化一个数据包；当前没有数据时允许返回None。"""

    def stop(self) -> None:
        self._session_id = None

    def _ensure_running(self) -> None:
        if not self.is_running:
            raise RuntimeError("data source adapter is not running")
