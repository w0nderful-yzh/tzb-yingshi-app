from app.modules.fall.multimodal_engine.algorithm_runtime.adapter import AlgorithmAdapter
from app.modules.fall.multimodal_engine.algorithm_runtime.contracts import AlgorithmFinding
from app.modules.fall.multimodal_engine.data_sources import UnifiedDataPacket
from app.modules.fall.multimodal_engine.schemas.risk_event import EvidenceItem, RiskLevel, RiskModule


VALID_SOURCE_MODES = {"REAL", "REPLAY"}
VALID_MODEL_MODES = {"TEST_CHECKPOINT", "TRAINED_CHECKPOINT"}


class RadarRiskAdapter(AlgorithmAdapter[UnifiedDataPacket]):
    """把Radar Packet路由成既有AlgorithmFinding。

    第一阶段默认禁止REAL+TRAINED生成正式PRE_FALL_RISK；P2验证完成后才
    能显式启用``allow_formal_predictions``。
    """

    module = RiskModule.FALL
    model_version = "radar-risk-adapter-1.2"

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
        if not bool(data.get("event_triggered", False)):
            return None

        source_mode = str(data["source_mode"])
        model_mode = str(data["model_mode"])
        event_kind = str(data.get("event_kind", "PREDICTION"))
        event_type, summary_prefix = self._route_event(
            source_mode=source_mode,
            model_mode=model_mode,
            event_kind=event_kind,
        )
        if event_type is None:
            return None

        risk_score = float(data["risk_score"])
        room_label = {
            "living_room": "客厅",
            "bedroom": "卧室",
            "bathroom": "卫生间",
        }.get(str(data["room"]), str(data["room"]))
        trigger_reasons = [str(item) for item in data.get("trigger_reasons", [])]
        reason_label = "+".join(
            {
                "ACTION_RISK": "动作风险",
                "PREFALL_PREDICTION": "跌倒预测",
            }.get(item, item)
            for item in trigger_reasons
        ) or "雷达风险"
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
        research = data.get("research")
        if event_kind == "UNIFIED_FALL_RISK" and isinstance(research, dict):
            evidence.extend(
                [
                    EvidenceItem(
                        code="trigger_reasons",
                        label="触发依据",
                        value=reason_label,
                    ),
                    EvidenceItem(
                        code="prefall_prediction_score",
                        label="跌倒预测分数",
                        value=float(research.get("pre_fall_score", 0.0)),
                    ),
                    EvidenceItem(
                        code="action_risk_score",
                        label="当前动作风险分数",
                        value=float(research.get("fall_risk_score", risk_score)),
                    ),
                    EvidenceItem(
                        code="room_risk_score_5s",
                        label="近5秒风险峰值",
                        value=float(
                            research.get("fall_risk_score_5s", risk_score)
                        ),
                    ),
                    EvidenceItem(
                        code="data_quality",
                        label="雷达数据质量",
                        value=str(research.get("data_quality", "UNKNOWN")),
                    ),
                ]
            )
            components = research.get("rule_components")
            if isinstance(components, dict):
                diagnostic_labels = {
                    "height_drop_m_0_6s": ("0.6秒人体中位高度下降", "m"),
                    "median_z_slope_mps": ("人体中部垂直速度", "m/s"),
                    "lower_z_slope_mps": ("人体下部垂直速度", "m/s"),
                    "upper_z_slope_mps": ("人体上部垂直速度", "m/s"),
                    "median_point_count_0_6s": ("0.6秒中位点数", "count"),
                    "median_core_height_m_0_6s": ("0.6秒中位人体跨度", "m"),
                    "coherent_body_descent_gate": ("人体整体一致下降", None),
                    "body_structure_gate": ("人体结构证据", None),
                }
                for code, (label, unit) in diagnostic_labels.items():
                    value = components.get(code)
                    if isinstance(value, (int, float)):
                        evidence.append(
                            EvidenceItem(
                                code=code,
                                label=label,
                                value=float(value),
                                unit=unit,
                            )
                        )

        return AlgorithmFinding(
            module=self.module,
            event_type=event_type,
            occurred_at=input_data.timestamp,
            risk_score=risk_score,
            risk_level=(
                RiskLevel.HIGH if risk_score >= 0.60 else RiskLevel.MEDIUM
            ),
            summary=(
                f"{summary_prefix}毫米波综合跌倒风险："
                f"{room_label}触发时综合风险分数{risk_score:.3f}，"
                f"触发依据：{reason_label}"
            ),
            evidence=evidence,
            recommended_action=(
                "请核实当前人员状态，并在处置记录中填写核实结果"
                if event_kind == "UNIFIED_FALL_RISK" and source_mode == "REAL"
                else "离线回放结果，仅用于模型验证"
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
        event_kind: str,
    ) -> tuple[str | None, str]:
        if event_kind == "UNIFIED_FALL_RISK":
            if source_mode == "REPLAY":
                return "RADAR_REPLAY_RISK", "[回放] "
            return "RADAR_FALL_RISK", ""
        if model_mode == "TEST_CHECKPOINT":
            return None, ""
        if source_mode == "REPLAY":
            return "RADAR_REPLAY_RISK", "[回放] "
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
