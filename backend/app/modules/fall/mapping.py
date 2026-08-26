"""Map real algorithm response fields to the stable App-facing contract."""

from app.modules.fall.schemas import (
    AssociationStatus,
    CameraAlgorithmStatus,
    CameraMonitoringStatus,
    CameraStreamStatus,
    DecisionPath,
    FallEventStatus,
    JointAssessment,
    PredictionState,
    RiskLevel,
    RoomFallRisk,
    SensorStatus,
)
from app.modules.fall.source_schemas import (
    CameraLedEvidenceFusionMode,
    CameraLedSourceSnapshot,
    CameraMonitoringSourceStatus,
    QualityLevel,
    RadarCalibratedTcnPredictionSource,
    RadarGateState,
    RadarOnlySourceSnapshot,
    RadarTcnRiskState,
)


def map_camera_monitoring_status(
    snapshot: CameraMonitoringSourceStatus,
) -> CameraMonitoringStatus:
    if not snapshot.enabled or snapshot.state == "DISABLED":
        stream = CameraStreamStatus.UNAVAILABLE
        algorithm = CameraAlgorithmStatus.UNAVAILABLE
    elif snapshot.state == "STOPPED":
        stream = CameraStreamStatus.STOPPED
        algorithm = CameraAlgorithmStatus.STOPPED
    elif snapshot.state == "ERROR":
        stream = CameraStreamStatus.ERROR
        algorithm = CameraAlgorithmStatus.ERROR
    elif snapshot.state in {"STARTING", "LOADING_MODELS"}:
        stream = CameraStreamStatus.CONNECTING
        algorithm = CameraAlgorithmStatus.STARTING
    elif snapshot.state == "CONNECTING":
        stream = CameraStreamStatus.RECONNECTING
        algorithm = CameraAlgorithmStatus.WAITING_DATA
    elif snapshot.input_state == "WAITING":
        stream = CameraStreamStatus.STREAMING
        algorithm = CameraAlgorithmStatus.WAITING_DATA
    else:
        stream = CameraStreamStatus.STREAMING
        algorithm = CameraAlgorithmStatus.RUNNING
    return CameraMonitoringStatus(
        camera_stream_status=stream,
        camera_algorithm_status=algorithm,
        detail=snapshot.error or snapshot.input_message,
        updated_at=snapshot.checked_at,
    )


def map_camera_led_snapshot(
    snapshot: CameraLedSourceSnapshot,
    *,
    room_id: str,
    room_name: str,
) -> RoomFallRisk:
    camera_status = _sensor_status(snapshot.camera.available, snapshot.camera.quality_level)
    radar_status = _sensor_status(snapshot.radar.available, snapshot.radar.quality_level)
    fusion_v2 = snapshot.camera_led_evidence_fusion_v2
    association_raw = fusion_v2.association_state
    association = _association_status(association_raw)

    if camera_status not in {SensorStatus.AVAILABLE, SensorStatus.DEGRADED}:
        return unavailable_room(
            room_id=room_id,
            room_name=room_name,
            camera_status=camera_status,
            radar_status=radar_status,
        )

    uses_associated_evidence = fusion_v2.fusion_mode in {
        "RADAR_SUPPORTED",
        "CAMERA_RADAR_CONSISTENT",
        "RADAR_CONFLICT",
    }
    decision_path = (
        DecisionPath.CAMERA_LED_RADAR_EVIDENCE
        if uses_associated_evidence
        else DecisionPath.CAMERA_ONLY
    )
    risk_level = _camera_led_risk_level(fusion_v2.camera_led_state)
    prediction = _camera_prediction_state(fusion_v2.camera_led_state)
    event_status = _camera_event_status(snapshot.fall_event.fall_event_status)
    joint = _camera_led_joint_assessment(
        fusion_v2.fusion_mode,
        fusion_v2.camera_led_state,
    )
    return RoomFallRisk(
        room_id=room_id,
        room_name=room_name,
        decision_path=decision_path,
        risk_level=risk_level,
        # Fusion v2 keeps the BioSTGCN score unchanged; Fixed remains baseline-only.
        risk_score=fusion_v2.camera_led_score,
        prediction_state=prediction,
        fall_event_status=event_status,
        camera_status=camera_status,
        radar_status=radar_status,
        association_status=association,
        joint_assessment=joint,
        evidence_summary=_camera_summary(joint, risk_level),
        updated_at=snapshot.timestamp,
    )


def map_radar_only_snapshot(
    snapshot: RadarOnlySourceSnapshot,
    *,
    room_name: str,
) -> RoomFallRisk:
    radar_status = _sensor_status(snapshot.score_valid, snapshot.data_quality)
    if radar_status not in {SensorStatus.AVAILABLE, SensorStatus.DEGRADED}:
        return unavailable_room(
            room_id=snapshot.room,
            room_name=room_name,
            camera_status=SensorStatus.NOT_APPLICABLE,
            radar_status=radar_status,
        )

    # The current Radar FastAPI explicitly marks TCN outputs as shadow/suppressed.
    # Read those real fields and fail closed until upstream promotes a formal result.
    if snapshot.shadow_only or snapshot.alert_suppressed:
        return RoomFallRisk(
            room_id=snapshot.room,
            room_name=room_name,
            decision_path=DecisionPath.UNAVAILABLE,
            risk_level=RiskLevel.UNKNOWN,
            risk_score=None,
            prediction_state=PredictionState.UNAVAILABLE,
            fall_event_status=FallEventStatus.NONE,
            camera_status=SensorStatus.NOT_APPLICABLE,
            radar_status=radar_status,
            association_status=AssociationStatus.NOT_REQUIRED,
            joint_assessment=JointAssessment.UNAVAILABLE,
            evidence_summary="当前房间雷达风险判断暂不可用，请稍后查看",
            updated_at=snapshot.timestamp,
        )

    state, event_triggered = _radar_state_and_event(snapshot)
    risk_level = _radar_risk_level(state)
    prediction = _radar_prediction_state(state)
    event_status = _radar_event_status(state, event_triggered)
    return RoomFallRisk(
        room_id=snapshot.room,
        room_name=room_name,
        decision_path=DecisionPath.RADAR_ONLY,
        risk_level=risk_level,
        # This is a short-term risk score, never labelled as a fall probability in the App.
        risk_score=snapshot.pre_fall_score,
        prediction_state=prediction,
        fall_event_status=event_status,
        camera_status=SensorStatus.NOT_APPLICABLE,
        radar_status=radar_status,
        association_status=AssociationStatus.NOT_REQUIRED,
        joint_assessment=JointAssessment.RADAR_ONLY,
        evidence_summary=_radar_summary(risk_level),
        updated_at=snapshot.timestamp,
    )


def unavailable_room(
    *,
    room_id: str,
    room_name: str,
    camera_status: SensorStatus,
    radar_status: SensorStatus,
) -> RoomFallRisk:
    return RoomFallRisk(
        room_id=room_id,
        room_name=room_name,
        decision_path=DecisionPath.UNAVAILABLE,
        risk_level=RiskLevel.UNKNOWN,
        risk_score=None,
        prediction_state=PredictionState.UNAVAILABLE,
        fall_event_status=FallEventStatus.NONE,
        camera_status=camera_status,
        radar_status=radar_status,
        association_status=AssociationStatus.UNAVAILABLE,
        joint_assessment=JointAssessment.UNAVAILABLE,
        evidence_summary="当前房间的跌倒风险监测暂不可用",
        updated_at=None,
    )


def _sensor_status(available: bool, quality: QualityLevel) -> SensorStatus:
    if not available:
        return SensorStatus.UNAVAILABLE
    if quality == "GOOD":
        return SensorStatus.AVAILABLE
    return SensorStatus.DEGRADED


def _association_status(value: str) -> AssociationStatus:
    if value == "MATCHED":
        return AssociationStatus.ASSOCIATED
    return AssociationStatus.NOT_ASSOCIATED


def _camera_led_risk_level(value: str) -> RiskLevel:
    return {
        "NORMAL": RiskLevel.NORMAL,
        "WATCH": RiskLevel.MEDIUM,
        "HIGH": RiskLevel.HIGH,
        "IMMINENT": RiskLevel.CRITICAL,
        "UNKNOWN": RiskLevel.UNKNOWN,
    }.get(value, RiskLevel.UNKNOWN)


def _camera_prediction_state(value: str) -> PredictionState:
    return {
        "LOW": PredictionState.STABLE,
        "NORMAL": PredictionState.STABLE,
        "MEDIUM": PredictionState.ELEVATED_RISK,
        "WATCH": PredictionState.ELEVATED_RISK,
        "HIGH": PredictionState.SHORT_TERM_WARNING,
        "IMMINENT": PredictionState.SHORT_TERM_WARNING,
        "UNKNOWN": PredictionState.UNKNOWN,
    }.get(value, PredictionState.UNKNOWN)


def _camera_event_status(value: str) -> FallEventStatus:
    return {
        "NO_EVENT": FallEventStatus.NONE,
        "SUSPECTED": FallEventStatus.SUSPECTED,
        "CONFIRMED": FallEventStatus.CONFIRMED,
        "UNKNOWN": FallEventStatus.NONE,
    }[value]


def _camera_led_joint_assessment(
    mode: CameraLedEvidenceFusionMode,
    state: str,
) -> JointAssessment:
    if mode == "CAMERA_RADAR_CONSISTENT":
        return (
            JointAssessment.CORROBORATED_HIGH
            if state in {"HIGH", "IMMINENT"}
            else JointAssessment.RADAR_SUPPORTS_CAMERA
        )
    if mode == "RADAR_SUPPORTED":
        return JointAssessment.RADAR_SUPPORTS_CAMERA
    if mode == "RADAR_CONFLICT":
        return JointAssessment.MODALITY_CONFLICT
    if mode == "CAMERA_ONLY":
        return JointAssessment.CAMERA_ONLY
    return JointAssessment.UNAVAILABLE


def _radar_state_and_event(
    snapshot: RadarOnlySourceSnapshot,
) -> tuple[RadarTcnRiskState | RadarGateState, bool]:
    if isinstance(snapshot, RadarCalibratedTcnPredictionSource):
        return snapshot.gate_state, snapshot.formal_alert
    return snapshot.risk_state, snapshot.event_triggered


def _radar_risk_level(value: RadarTcnRiskState | RadarGateState) -> RiskLevel:
    return {
        "NORMAL": RiskLevel.NORMAL,
        "WATCH": RiskLevel.MEDIUM,
        "IMMINENT": RiskLevel.HIGH,
        "SUPPRESSED_RECOVERY": RiskLevel.MEDIUM,
        "CONFIRMED": RiskLevel.CRITICAL,
        "UNKNOWN": RiskLevel.UNKNOWN,
    }[value]


def _radar_prediction_state(value: RadarTcnRiskState | RadarGateState) -> PredictionState:
    return {
        "NORMAL": PredictionState.STABLE,
        "WATCH": PredictionState.ELEVATED_RISK,
        "IMMINENT": PredictionState.SHORT_TERM_WARNING,
        "SUPPRESSED_RECOVERY": PredictionState.ELEVATED_RISK,
        "CONFIRMED": PredictionState.FALL_DETECTED,
        "UNKNOWN": PredictionState.UNKNOWN,
    }[value]


def _radar_event_status(
    value: RadarTcnRiskState | RadarGateState,
    event_triggered: bool,
) -> FallEventStatus:
    if value == "CONFIRMED":
        return FallEventStatus.CONFIRMED
    if event_triggered:
        return FallEventStatus.SUSPECTED
    if value == "IMMINENT":
        return FallEventStatus.PREDICTED
    return FallEventStatus.NONE


def _camera_summary(joint: JointAssessment, risk_level: RiskLevel) -> str:
    summaries = {
        JointAssessment.CORROBORATED_HIGH: "多传感器检测到一致高风险",
        JointAssessment.RADAR_SUPPORTS_CAMERA: "雷达运动证据支持视觉风险判断",
        JointAssessment.MODALITY_CONFLICT: "当前多传感器证据不一致，以视觉主判断持续监测",
        JointAssessment.NOT_ASSOCIATED: "当前无法进行联合判断，继续使用视觉风险监测",
        JointAssessment.CAMERA_ONLY: "视觉风险监测正常，雷达联合证据暂不可用",
    }
    return summaries.get(joint, f"视觉主路径持续监测，当前风险等级为{risk_level.value}")


def _radar_summary(risk_level: RiskLevel) -> str:
    if risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        return "雷达检测到短时跌倒高风险，请及时关注"
    return "当前使用雷达单模态进行短时跌倒风险监测"
