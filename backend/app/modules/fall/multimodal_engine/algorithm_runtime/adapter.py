from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar

from app.modules.fall.multimodal_engine.algorithm_runtime.contracts import AdapterContext, AlgorithmFinding
from app.modules.fall.multimodal_engine.schemas.risk_event import RiskModule


InputT = TypeVar("InputT")


class AdapterState(str, Enum):
    CREATED = "CREATED"
    LOADED = "LOADED"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class AlgorithmAdapter(ABC, Generic[InputT]):
    """真实成员算法与系统之间的最小适配器边界。"""

    module: RiskModule
    model_version: str

    def __init__(self) -> None:
        self._state = AdapterState.CREATED
        self._context: AdapterContext | None = None

    @property
    def state(self) -> AdapterState:
        return self._state

    @property
    def context(self) -> AdapterContext:
        if self._context is None:
            raise RuntimeError("adapter has not been started")
        return self._context

    def load(self) -> None:
        """未来真实Adapter可重写此方法加载权重。"""

        self._state = AdapterState.LOADED

    def start(self, context: AdapterContext) -> None:
        if self._state not in {AdapterState.LOADED, AdapterState.STOPPED}:
            raise RuntimeError(f"adapter cannot start from state {self._state.value}")
        self._context = context
        self._state = AdapterState.RUNNING

    @abstractmethod
    def consume(self, input_data: InputT) -> AlgorithmFinding | None:
        """消费一次模块输入；没有形成风险结论时返回None。"""

    def flush(self) -> list[AlgorithmFinding]:
        """未来时序Adapter可重写此方法处理尚未完成的窗口。"""

        return []

    def stop(self) -> None:
        self._context = None
        self._state = AdapterState.STOPPED

    def health(self) -> AdapterState:
        return self._state

    def _ensure_running(self) -> None:
        if self._state is not AdapterState.RUNNING:
            raise RuntimeError("adapter is not running")
