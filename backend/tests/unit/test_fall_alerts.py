from datetime import UTC, datetime, timedelta

import pytest

from app.modules.fall.fall_alerts import FallAlertController
from app.modules.fall.ports import FallRiskEventWrite
from app.modules.fall.schemas import (
    AssociationStatus,
    CameraAlgorithmStatus,
    CameraMonitoringStatus,
    CameraStreamStatus,
    DecisionPath,
    FallEventStatus,
    FallRiskOverview,
    JointAssessment,
    PredictionState,
    RoomFallRisk,
    SensorStatus,
)


class FakeSink:
    def __init__(self) -> None:
        self.events: list[FallRiskEventWrite] = []

    async def upsert_fall_event(self, event: FallRiskEventWrite) -> None:
        self.events.append(event)


def _room(level: str) -> RoomFallRisk:
    return RoomFallRisk(
        room_id="living_room",
        room_name="客厅",
        decision_path=DecisionPath.CAMERA_LED_RADAR_EVIDENCE,
        risk_level=level,  # type: ignore[arg-type]
        risk_score=0.9 if level in {"high", "critical"} else 0.1,
        prediction_state=PredictionState.STABLE,
        fall_event_status=FallEventStatus.NONE,
        camera_status=SensorStatus.AVAILABLE,
        radar_status=SensorStatus.AVAILABLE,
        association_status=AssociationStatus.ASSOCIATED,
        joint_assessment=JointAssessment.CAMERA_LED,
        evidence_summary="ok",
    )


def _overview(level: str) -> FallRiskOverview:
    return FallRiskOverview(
        overall_risk_level=level,  # type: ignore[arg-type]
        rooms=[_room(level)],
        camera_monitoring=CameraMonitoringStatus(
            camera_stream_status=CameraStreamStatus.STREAMING,
            camera_algorithm_status=CameraAlgorithmStatus.RUNNING,
            detail="",
            updated_at=datetime.now(UTC),
        ),
        generated_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_high_risk_emits_single_event_per_episode() -> None:
    sink = FakeSink()
    controller = FallAlertController(sink, cooldown_seconds=60.0)
    await controller.observe(elder_user_id="elder-1", overview=_overview("high"))
    await controller.observe(elder_user_id="elder-1", overview=_overview("high"))
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.risk_level == "HIGH"
    assert event.elder_user_id == "elder-1"
    assert event.evidence["state_label"] == "跌倒高风险"


@pytest.mark.asyncio
async def test_recovery_rearms_and_cooldown_absorbs_flapping() -> None:
    sink = FakeSink()
    controller = FallAlertController(sink, cooldown_seconds=60.0)
    await controller.observe(elder_user_id="elder-1", overview=_overview("high"))
    # 回落解除幕次，但冷却期内再次进入高风险不应产生第二个事件。
    await controller.observe(elder_user_id="elder-1", overview=_overview("normal"))
    await controller.observe(elder_user_id="elder-1", overview=_overview("high"))
    assert len(sink.events) == 1
    # 冷却期过后再次进入高风险 → 新幕次、新事件。
    controller._last_emitted_at["elder-1"] -= timedelta(seconds=61)
    await controller.observe(elder_user_id="elder-1", overview=_overview("normal"))
    await controller.observe(elder_user_id="elder-1", overview=_overview("critical"))
    assert len(sink.events) == 2
    assert sink.events[1].risk_level == "CRITICAL"
    assert sink.events[0].source_event_id != sink.events[1].source_event_id


@pytest.mark.asyncio
async def test_low_risk_states_never_emit() -> None:
    sink = FakeSink()
    controller = FallAlertController(sink)
    for level in ("normal", "low", "medium", "unknown"):
        await controller.observe(elder_user_id="elder-1", overview=_overview(level))
    assert sink.events == []
