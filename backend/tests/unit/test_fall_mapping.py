from app.modules.fall.mapping import map_camera_led_snapshot, map_radar_only_snapshot
from app.modules.fall.schemas import (
    AssociationStatus,
    DecisionPath,
    JointAssessment,
    PredictionState,
    RiskLevel,
    SensorStatus,
)
from app.modules.fall.source_schemas import (
    CameraLedSourceSnapshot,
    RadarCalibratedTcnPredictionSource,
    RadarTcnPredictionSource,
)


def camera_payload(*, evidence_state: str, association_state: str = "MATCHED") -> dict:
    return {
        "camera": {
            "camera_score": 0.72,
            "camera_risk_state": "HIGH",
            "quality_level": "GOOD",
            "timestamp": "2026-08-15T10:00:00+08:00",
            "available": True,
            "checkpoint": "debug-only",
        },
        "radar": {
            "radar_score": 0.41,
            "radar_risk_state": "WATCH",
            "quality_level": "GOOD",
            "timestamp": "2026-08-15T10:00:00+08:00",
            "available": True,
            "room": "living_room",
        },
        "alignment": {
            "association_state": association_state,
            "eligible_for_temporal_association": association_state == "MATCHED",
            "radar_track_id": 7,
            "sync_delta_ms": 38,
        },
        "associated_risk_augmentation": {
            "associated_short_term_fall_score": 0.72,
            "associated_risk_state": "HIGH",
            "associated_evidence_state": evidence_state,
            "base_camera_score": 0.72,
            "base_camera_state": "HIGH",
            "radar_motion_evidence_strength": "STRONG",
            "association_state": association_state,
            "shadow_only": True,
            "affects_alerts": False,
            "camera_score_unchanged": True,
            "uses_radar_tcn_score": False,
            "reason_codes": ["DEBUG_ONLY"],
        },
        "fall_event": {
            "fall_event_status": "SUSPECTED",
            "summary": "上游事件摘要",
            "reason_codes": ["DEBUG_ONLY"],
        },
        "timestamp": "2026-08-15T10:00:01+08:00",
        "fusion": {"fusion_score": 0.99, "method": "fixed_weighted"},
        "temporal_associated_fusion": {"fusion_score": 0.98},
    }


def test_camera_led_mapping_uses_camera_score_and_associated_radar_evidence() -> None:
    snapshot = CameraLedSourceSnapshot.model_validate(
        camera_payload(evidence_state="CORROBORATED_HIGH")
    )

    result = map_camera_led_snapshot(
        snapshot,
        room_id="living_room",
        room_name="客厅",
    )

    assert result.decision_path is DecisionPath.CAMERA_LED_RADAR_EVIDENCE
    assert result.risk_score == 0.72
    assert result.risk_level is RiskLevel.HIGH
    assert result.prediction_state is PredictionState.SHORT_TERM_WARNING
    assert result.association_status is AssociationStatus.ASSOCIATED
    assert result.joint_assessment is JointAssessment.CORROBORATED_HIGH
    assert "track" not in result.model_dump()
    assert "fusion" not in result.model_dump()


def test_living_room_degrades_to_camera_only_for_real_not_associated_state() -> None:
    snapshot = CameraLedSourceSnapshot.model_validate(
        camera_payload(
            evidence_state="CAMERA_ONLY_HIGH",
            association_state="RADAR_TRACK_MISSING",
        )
    )

    result = map_camera_led_snapshot(
        snapshot,
        room_id="living_room",
        room_name="客厅",
    )

    assert result.decision_path is DecisionPath.CAMERA_ONLY
    assert result.joint_assessment is JointAssessment.CAMERA_ONLY
    assert result.risk_score == 0.72


def test_current_shadow_radar_tcn_is_read_but_not_promoted_to_app_risk() -> None:
    snapshot = RadarCalibratedTcnPredictionSource.model_validate(
        {
            "schema_version": "radar_calibrated_tcn_live_v1",
            "timestamp": "2026-08-15T10:00:00+08:00",
            "device_id": "radar-bathroom",
            "room": "bathroom",
            "pre_fall_score": 0.64,
            "score_valid": True,
            "tcn_risk_state": "WATCH",
            "gate_state": "WATCH",
            "formal_alert": False,
            "data_quality": "GOOD",
            "shadow_only": True,
            "alert_suppressed": True,
            "checkpoint_sha256": "debug-only",
        }
    )

    result = map_radar_only_snapshot(snapshot, room_name="卫生间")

    assert result.decision_path is DecisionPath.UNAVAILABLE
    assert result.risk_score is None
    assert result.camera_status is SensorStatus.NOT_APPLICABLE
    assert result.evidence_summary == "当前房间雷达风险判断暂不可用，请稍后查看"
    assert "TCN" not in result.evidence_summary
    assert "影子" not in result.evidence_summary


def test_formal_radar_tcn_contract_maps_to_independent_radar_only_path() -> None:
    snapshot = RadarTcnPredictionSource.model_validate(
        {
            "schema_version": "radar_tcn_live_v1",
            "timestamp": "2026-08-15T10:00:00+08:00",
            "device_id": "radar-bedroom",
            "room": "bedroom",
            "risk_state": "IMMINENT",
            "pre_fall_score": 0.64,
            "score_valid": True,
            "event_triggered": False,
            "data_quality": "GOOD",
            "shadow_only": False,
            "alert_suppressed": False,
        }
    )

    result = map_radar_only_snapshot(snapshot, room_name="卧室")

    assert result.decision_path is DecisionPath.RADAR_ONLY
    assert result.prediction_state is PredictionState.SHORT_TERM_WARNING
    assert result.risk_score == 0.64
    assert result.association_status is AssociationStatus.NOT_REQUIRED
    assert result.joint_assessment is JointAssessment.RADAR_ONLY
