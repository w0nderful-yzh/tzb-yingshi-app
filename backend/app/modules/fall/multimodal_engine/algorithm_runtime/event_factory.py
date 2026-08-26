from collections.abc import Callable
from uuid import uuid4

from app.modules.fall.multimodal_engine.algorithm_runtime.contracts import AdapterContext, AlgorithmFinding
from app.modules.fall.multimodal_engine.schemas.risk_event import EventSource, RiskEventInput


EventIdFactory = Callable[[AlgorithmFinding], str]


def _default_event_id(finding: AlgorithmFinding) -> str:
    return f"alg-{finding.module.value.lower()}-{uuid4().hex}"


class RiskEventFactory:
    """为算法Finding补齐系统身份字段并生成既有RiskEvent契约。"""

    def __init__(self, event_id_factory: EventIdFactory | None = None) -> None:
        self._event_id_factory = event_id_factory or _default_event_id

    def create(
        self,
        finding: AlgorithmFinding,
        context: AdapterContext,
    ) -> RiskEventInput:
        return RiskEventInput(
            schema_version="1.0",
            event_id=self._event_id_factory(finding),
            session_id=context.session_id,
            device_id=context.device_id,
            module=finding.module,
            event_type=finding.event_type,
            occurred_at=finding.occurred_at,
            risk_score=finding.risk_score,
            risk_level=finding.risk_level,
            summary=finding.summary,
            evidence=finding.evidence,
            recommended_action=finding.recommended_action,
            snapshot_path=finding.snapshot_path,
            clip_path=finding.clip_path,
            model_version=finding.model_version,
            source=EventSource.ALGORITHM,
        )
