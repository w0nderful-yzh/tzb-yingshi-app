import pytest

from app.modules.fall.schemas import DecisionPath, RiskLevel
from app.modules.fall.service import FallRiskService
from app.modules.fall.source_schemas import (
    CameraLedSourceSnapshot,
    RadarOnlySourceSnapshot,
    RadarTcnPredictionSource,
)


class FakeFallRiskSource:
    def __init__(self) -> None:
        self.camera_rooms: list[str] = []
        self.radar_rooms: list[str] = []

    async def get_camera_led_risk(
        self,
        *,
        elder_id: str,
        room_id: str,
    ) -> CameraLedSourceSnapshot:
        assert elder_id == "elder-001"
        self.camera_rooms.append(room_id)
        return CameraLedSourceSnapshot.model_validate(
            {
                "camera": {
                    "camera_score": 0.1,
                    "camera_risk_state": "LOW",
                    "quality_level": "GOOD",
                    "timestamp": "2026-08-15T10:00:00+08:00",
                    "available": True,
                },
                "radar": {
                    "radar_score": 0.08,
                    "radar_risk_state": "NORMAL",
                    "quality_level": "GOOD",
                    "timestamp": "2026-08-15T10:00:00+08:00",
                    "available": True,
                    "room": room_id,
                },
                "alignment": {
                    "association_state": "MATCHED",
                    "eligible_for_temporal_association": True,
                },
                "associated_risk_augmentation": {
                    "associated_short_term_fall_score": 0.1,
                    "associated_risk_state": "NORMAL",
                    "associated_evidence_state": "NORMAL_CORROBORATED",
                    "base_camera_score": 0.1,
                    "base_camera_state": "LOW",
                    "radar_motion_evidence_strength": "NONE",
                    "association_state": "MATCHED",
                },
                "fall_event": {
                    "fall_event_status": "NO_EVENT",
                    "summary": "未检测到跌倒事件",
                },
                "timestamp": "2026-08-15T10:00:01+08:00",
            }
        )

    async def get_radar_only_risk(
        self,
        *,
        elder_id: str,
        room_id: str,
    ) -> RadarOnlySourceSnapshot:
        assert elder_id == "elder-001"
        self.radar_rooms.append(room_id)
        return RadarTcnPredictionSource.model_validate(
            {
                "schema_version": "radar_tcn_live_v1",
                "timestamp": "2026-08-15T10:00:00+08:00",
                "device_id": f"radar-{room_id}",
                "room": room_id,
                "risk_state": "WATCH" if room_id == "bathroom" else "NORMAL",
                "pre_fall_score": 0.55 if room_id == "bathroom" else 0.08,
                "score_valid": True,
                "data_quality": "GOOD",
                "shadow_only": False,
                "alert_suppressed": False,
            }
        )


@pytest.mark.asyncio
async def test_service_routes_rooms_by_formal_capability() -> None:
    source = FakeFallRiskSource()
    result = await FallRiskService(source).get_overview(elder_id="elder-001")

    assert source.camera_rooms == ["living_room"]
    assert source.radar_rooms == ["bathroom", "bedroom"]
    assert [room.decision_path for room in result.rooms] == [
        DecisionPath.CAMERA_LED_RADAR_EVIDENCE,
        DecisionPath.RADAR_ONLY,
        DecisionPath.RADAR_ONLY,
    ]
    assert result.overall_risk_level is RiskLevel.MEDIUM


@pytest.mark.asyncio
async def test_service_returns_explicit_unavailable_contract_without_source() -> None:
    result = await FallRiskService(None).get_overview(elder_id="elder-001")

    assert all(room.decision_path is DecisionPath.UNAVAILABLE for room in result.rooms)
    assert result.rooms[1].camera_status.value == "not_applicable"
    assert result.overall_risk_level is RiskLevel.UNKNOWN
