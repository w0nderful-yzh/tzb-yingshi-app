from fastapi import APIRouter, HTTPException, Query, Request

from app.modules.fall.multimodal_engine.schemas.multimodal import (
    CameraLedAssociatedAlignmentProjection,
    CameraLedAssociatedCameraProjection,
    CameraLedAssociatedFallEventProjection,
    CameraLedAssociatedLatestResponse,
    CameraLedAssociatedRadarProjection,
    CameraLedAssociatedRiskProjection,
    MultimodalLatestResponse,
    OfflineReplayLatestResponse,
)


router = APIRouter(prefix="/api/multimodal", tags=["multimodal"])


@router.get("/latest", response_model=MultimodalLatestResponse)
def get_multimodal_latest(
    request: Request,
    method: str = Query(
        default="fixed_weighted",
        pattern="^(fixed_weighted|quality_weighted|radar_quality_adaptive|mlp)$",
    ),
) -> MultimodalLatestResponse:
    try:
        return request.app.state.multimodal_fusion_service.get_latest(method=method)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/camera-led-associated/latest",
    response_model=CameraLedAssociatedLatestResponse,
)
def get_camera_led_associated_latest(
    request: Request,
) -> CameraLedAssociatedLatestResponse:
    """Expose the active Camera-led Fusion v2 result to the App adapter."""

    try:
        latest = request.app.state.multimodal_fusion_service.get_latest()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    augmentation = latest.associated_risk_augmentation
    return CameraLedAssociatedLatestResponse(
        camera=CameraLedAssociatedCameraProjection(
            camera_score=latest.camera.camera_score,
            camera_risk_state=latest.camera.camera_risk_state,
            quality_level=latest.camera.quality_level,
            timestamp=latest.camera.timestamp,
            available=latest.camera.available,
        ),
        radar=CameraLedAssociatedRadarProjection(
            radar_score=latest.radar.radar_score,
            radar_risk_state=latest.radar.radar_risk_state,
            quality_level=latest.radar.quality_level,
            timestamp=latest.radar.timestamp,
            available=latest.radar.available,
            room=latest.radar.room,
        ),
        alignment=CameraLedAssociatedAlignmentProjection(
            association_state=latest.alignment.association_state,
            eligible_for_temporal_association=(
                latest.alignment.eligible_for_temporal_association
            ),
        ),
        associated_risk_augmentation=(
            CameraLedAssociatedRiskProjection(
                associated_short_term_fall_score=(
                    augmentation.associated_short_term_fall_score
                ),
                associated_risk_state=augmentation.associated_risk_state,
                associated_evidence_state=augmentation.associated_evidence_state,
                base_camera_score=augmentation.base_camera_score,
                base_camera_state=augmentation.base_camera_state,
                radar_motion_evidence_strength=(
                    augmentation.radar_motion_evidence_strength
                ),
                association_state=augmentation.association_state,
                shadow_only=augmentation.shadow_only,
                affects_alerts=augmentation.affects_alerts,
                camera_score_unchanged=augmentation.camera_score_unchanged,
                uses_radar_tcn_score=augmentation.uses_radar_tcn_score,
            )
            if augmentation is not None
            else None
        ),
        camera_led_evidence_fusion_v2=latest.camera_led_evidence_fusion_v2,
        fall_event=CameraLedAssociatedFallEventProjection(
            fall_event_status=latest.fall_event.fall_event_status,
            summary=latest.fall_event.summary,
        ),
        timestamp=latest.timestamp,
    )


@router.get("/replay/latest", response_model=OfflineReplayLatestResponse)
def get_multimodal_replay_latest(
    request: Request,
    cursor: int = Query(default=0, ge=0),
    method: str = Query(
        default="fixed_weighted",
        pattern="^(fixed_weighted|quality_weighted|radar_quality_adaptive)$",
    ),
) -> OfflineReplayLatestResponse:
    try:
        return request.app.state.offline_evidence_replay_service.get(
            cursor,
            method=method,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
