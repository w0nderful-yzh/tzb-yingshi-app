import httpx
import pytest

from app.infrastructure.external.fall_risk.client import HttpFallRiskSource
from app.modules.fall.source_schemas import RadarTcnPredictionSource


@pytest.mark.asyncio
async def test_client_uses_real_multimodal_and_per_room_radar_endpoints() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "camera.test":
            return httpx.Response(
                200,
                json={
                    "camera": {
                        "camera_score": 0.72,
                        "camera_risk_state": "HIGH",
                        "quality_level": "GOOD",
                        "timestamp": "2026-08-15T10:00:00+08:00",
                        "available": True,
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
                        "association_state": "MATCHED",
                        "eligible_for_temporal_association": True,
                    },
                    "associated_risk_augmentation": {
                        "associated_short_term_fall_score": 0.72,
                        "associated_risk_state": "HIGH",
                        "associated_evidence_state": "CORROBORATED_HIGH",
                        "base_camera_score": 0.72,
                        "base_camera_state": "HIGH",
                        "radar_motion_evidence_strength": "STRONG",
                        "association_state": "MATCHED",
                    },
                    "fall_event": {
                        "fall_event_status": "SUSPECTED",
                        "summary": "疑似跌倒事件",
                    },
                    "timestamp": "2026-08-15T10:00:01+08:00",
                    "fusion": {"fusion_score": 0.99},
                },
            )
        return httpx.Response(
            200,
            json={
                "tcn_prediction": {
                    "schema_version": "radar_tcn_live_v1",
                    "timestamp": "2026-08-15T10:00:00+08:00",
                    "device_id": "radar-bathroom",
                    "room": "bathroom",
                    "risk_state": "WATCH",
                    "pre_fall_score": 0.55,
                    "score_valid": True,
                    "event_triggered": False,
                    "data_quality": "GOOD",
                    "shadow_only": True,
                    "alert_suppressed": True,
                    "checkpoint_sha256": "debug-only",
                }
            },
        )

    source = HttpFallRiskSource(
        camera_base_url="https://camera.test",
        radar_room_base_urls={"bathroom": "https://bathroom.test"},
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        camera = await source.get_camera_led_risk(
            elder_id="elder-not-forwarded",
            room_id="living_room",
        )
        radar = await source.get_radar_only_risk(
            elder_id="elder-not-forwarded",
            room_id="bathroom",
        )
    finally:
        await source.close()

    assert camera.camera.camera_score == 0.72
    assert isinstance(radar, RadarTcnPredictionSource)
    assert radar.pre_fall_score == 0.55
    assert requests[0].url.path == "/api/multimodal/camera-led-associated/latest"
    assert not requests[0].url.params
    assert requests[1].url.path == "/api/radar/latest"
    assert not requests[1].url.params
    assert all("elder_id" not in request.url.params for request in requests)
