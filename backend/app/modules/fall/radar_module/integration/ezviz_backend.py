from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import httpx

from app.algorithm_runtime.adapter import AlgorithmAdapter
from app.algorithm_runtime.contracts import AlgorithmFinding
from app.data_sources.adapter import DataSourceAdapter
from app.data_sources.contracts import UnifiedDataPacket
from app.schemas.risk_event import EvidenceItem, RiskLevel, RiskModule


DEMO_DISCLAIMER = "当前为DEMO风险推理框架结果，不能代表真实跌倒预测能力"
VALID_SOURCE_MODES = {"REAL", "REPLAY"}
VALID_MODEL_MODES = {"TEST_CHECKPOINT", "TRAINED_CHECKPOINT"}


class RadarServiceDataSourceAdapter(DataSourceAdapter):
    """从独立Radar FastAPI读取当前结果并转换为既有UnifiedDataPacket。

    服务离线、雷达未连接或时间戳未更新时返回None，绝不复用旧结果。
    """

    def __init__(
        self,
        base_url: str,
        *,
        source_id: str = "iwr6843isk-radar-service",
        timeout_seconds: float = 2.0,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__()
        self.source_id = source_id
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )
        self._last_timestamp: datetime | None = None
        self._online = False

    @property
    def online(self) -> bool:
        return self._online

    def read(self) -> UnifiedDataPacket | None:
        self._ensure_running()
        try:
            health_response = self._client.get("/health")
            health_response.raise_for_status()
            health = health_response.json()
            if not health.get("radar_connected"):
                self._online = False
                return None

            response = self._client.get("/api/radar/latest")
            response.raise_for_status()
            payload = response.json()
            timestamp = datetime.fromisoformat(
                str(payload["timestamp"]).replace("Z", "+00:00")
            )
            if self._last_timestamp is not None and timestamp <= self._last_timestamp:
                return None
            self._last_timestamp = timestamp
            self._online = True
            return UnifiedDataPacket(
                packet_id=f"radar-{uuid4().hex}",
                session_id=self.session_id,
                source_id=self.source_id,
                device_id=str(payload["device_id"]),
                modality="RADAR",
                timestamp=timestamp,
                data={
                    "room": payload["room"],
                    "source_mode": payload["source_mode"],
                    "human_state": payload["human_state"],
                    "risk_score": payload["risk_score"],
                    "model_mode": payload["model_mode"],
                    "disclaimer": payload.get("disclaimer"),
                    "event_triggered": bool(payload.get("event_triggered", False)),
                },
            )
        except (
            httpx.HTTPError,
            KeyError,
            TypeError,
            ValueError,
        ):
            self._online = False
            return None

    def stop(self) -> None:
        super().stop()
        self._last_timestamp = None
        self._online = False

    def close(self) -> None:
        self.stop()
        if self._owns_client:
            self._client.close()


class RadarRiskAdapter(AlgorithmAdapter[UnifiedDataPacket]):
    """将Radar结果路由为既有AlgorithmFinding。

    默认禁止REAL+TRAINED生成正式PRE_FALL_RISK。完成P2验证后，必须显式
    设置``allow_formal_predictions=True``才会启用正式事件。
    """

    module = RiskModule.FALL
    model_version = "radar-risk-http-adapter-1.0"

    def __init__(self, *, allow_formal_predictions: bool = False) -> None:
        super().__init__()
        self.allow_formal_predictions = allow_formal_predictions

    def consume(
        self,
        input_data: UnifiedDataPacket,
    ) -> AlgorithmFinding | None:
        self._ensure_running()
        self._validate_input(input_data)
        data = input_data.data
        if data["human_state"] != "FALL_RISK":
            return None
        if not bool(data.get("event_triggered")):
            return None

        source_mode = str(data["source_mode"])
        model_mode = str(data["model_mode"])
        event_type, summary_prefix = self._route_event(
            source_mode=source_mode,
            model_mode=model_mode,
        )
        if event_type is None:
            return None

        risk_score = float(data["risk_score"])
        disclaimer = data.get("disclaimer")
        evidence = [
            EvidenceItem(
                code="radar_room",
                label="监测房间",
                value=str(data["room"]),
            ),
            EvidenceItem(
                code="source_mode",
                label="雷达数据源模式",
                value=source_mode,
            ),
            EvidenceItem(
                code="model_mode",
                label="雷达模型模式",
                value=model_mode,
            ),
        ]
        if model_mode == "TEST_CHECKPOINT":
            evidence.extend(
                [
                    EvidenceItem(
                        code="demo",
                        label="DEMO事件",
                        value=True,
                    ),
                    EvidenceItem(
                        code="disclaimer",
                        label="能力声明",
                        value=str(disclaimer or DEMO_DISCLAIMER),
                    ),
                ]
            )

        return AlgorithmFinding(
            module=self.module,
            event_type=event_type,
            occurred_at=input_data.timestamp,
            risk_score=risk_score,
            risk_level=(
                RiskLevel.HIGH if risk_score >= 0.85 else RiskLevel.MEDIUM
            ),
            summary=(
                f"{summary_prefix}毫米波风险推理结果："
                f"{data['room']}风险分数{risk_score:.3f}"
            ),
            evidence=evidence,
            recommended_action=(
                "仅验证系统链路，不触发正式跌倒处置"
                if model_mode == "TEST_CHECKPOINT"
                else "离线回放结果，仅用于模型测试"
                if source_mode == "REPLAY"
                else "请按正式预警流程核实老人状态"
            ),
            model_version=self.model_version,
        )

    def _route_event(
        self,
        *,
        source_mode: str,
        model_mode: str,
    ) -> tuple[str | None, str]:
        if model_mode == "TEST_CHECKPOINT":
            return "RADAR_DEMO_RISK", "[DEMO] "
        if source_mode == "REPLAY":
            return "RADAR_REPLAY_RISK", "[REPLAY] "
        if self.allow_formal_predictions:
            return "PRE_FALL_RISK", ""
        return None, ""

    def _validate_input(self, input_data: UnifiedDataPacket) -> None:
        if input_data.modality != "RADAR":
            raise ValueError("RadarRiskAdapter only accepts RADAR packets")
        if input_data.session_id != self.context.session_id:
            raise ValueError("packet session_id does not match adapter context")
        if input_data.device_id != self.context.device_id:
            raise ValueError("packet device_id does not match adapter context")
        required = {
            "room",
            "source_mode",
            "human_state",
            "risk_score",
            "model_mode",
        }
        missing = sorted(required.difference(input_data.data))
        if missing:
            raise ValueError(f"radar packet is missing fields: {missing}")
        if input_data.data["source_mode"] not in VALID_SOURCE_MODES:
            raise ValueError("invalid source_mode")
        if input_data.data["model_mode"] not in VALID_MODEL_MODES:
            raise ValueError("invalid model_mode")
        if input_data.data["model_mode"] == "TEST_CHECKPOINT":
            disclaimer = input_data.data.get("disclaimer")
            if disclaimer != DEMO_DISCLAIMER:
                raise ValueError("TEST_CHECKPOINT must carry the DEMO disclaimer")
