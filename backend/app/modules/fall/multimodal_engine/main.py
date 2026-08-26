from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.modules.fall.multimodal_engine.api.dashboard import router as dashboard_router
from app.modules.fall.multimodal_engine.api.ezviz import router as ezviz_router
from app.modules.fall.multimodal_engine.api.fall_inference import router as fall_inference_router
from app.modules.fall.multimodal_engine.api.fall_live import router as fall_live_router
from app.modules.fall.multimodal_engine.api.guard_session import router as guard_session_router
from app.modules.fall.multimodal_engine.api.events import router as events_router
from app.modules.fall.multimodal_engine.api.health import router as health_router
from app.modules.fall.multimodal_engine.api.monitoring import router as monitoring_router
from app.modules.fall.multimodal_engine.api.multimodal import router as multimodal_router
from app.modules.fall.multimodal_engine.api.radar import router as radar_router
from app.modules.fall.multimodal_engine.api.simulation import router as simulation_router
from app.modules.fall.multimodal_engine.core.config import get_settings
from app.modules.fall.multimodal_engine.data_sources.adapters.radar_service import RadarServiceDataSourceAdapter
from app.modules.fall.multimodal_engine.services.radar_integration import RadarIntegrationService
from app.modules.fall.multimodal_engine.services.fall_live_monitor import FallLiveMonitorService
from app.modules.fall.multimodal_engine.services.multimodal_fusion import MultimodalFusionService
from app.modules.fall.multimodal_engine.services.fusion_runtime import (
    FusionShadowLogger,
    FusionShadowSampler,
    FusionStateConfig,
)
from app.modules.fall.multimodal_engine.services.fusion_event_bridge import FusionRiskEventBridge
from app.modules.fall.multimodal_engine.services.offline_evidence_replay import OfflineEvidenceReplayService
from app.modules.fall.multimodal_engine.services.temporal_associated_fusion import TemporalAssociationConfig
from app.modules.fall.multimodal_engine.services.camera_radar_alignment import (
    CameraRadarAlignmentAdapter,
    RadarTrackEvidenceBuffer,
)
from app.modules.fall.multimodal_engine.services.alignment_aware_risk_augmentation import AssociatedEvidenceConfig
from app.modules.fall.multimodal_engine.services.radar_eligibility import RadarEligibilityConfig
from app.modules.fall.multimodal_engine.services.guard_session import MultimodalGuardSessionService


def create_app() -> FastAPI:
    settings = get_settings()

    radar_source = RadarServiceDataSourceAdapter(
        settings.radar_service_url,
        timeout_seconds=settings.radar_request_timeout_seconds,
    )
    radar_track_buffer = RadarTrackEvidenceBuffer()
    radar_integration = RadarIntegrationService(
        radar_source,
        poll_interval_seconds=settings.radar_poll_interval_seconds,
        radar_risk_events_enabled=settings.radar_risk_events_enabled,
        allow_formal_predictions=settings.radar_formal_predictions_enabled,
        radar_track_buffer=radar_track_buffer,
        session_enabled=False,
    )
    fall_live_monitor = FallLiveMonitorService(settings)
    guard_session = MultimodalGuardSessionService(
        fall_live_monitor,
        radar_integration,
        radar_source,
    )
    fusion_shadow_logger = FusionShadowLogger(
        settings.fusion_shadow_log_path,
        enabled=settings.fusion_shadow_log_enabled,
        max_bytes=settings.fusion_shadow_log_max_mb * 1024 * 1024,
        backup_count=settings.fusion_shadow_log_backup_count,
    )
    fusion_event_bridge = FusionRiskEventBridge(
        enabled=settings.fusion_risk_events_enabled,
        cooldown_seconds=settings.fall_live_event_cooldown_seconds,
        auto_create_session=settings.fall_live_auto_create_session,
    )
    alignment_adapter = CameraRadarAlignmentAdapter(
        settings.fusion_alignment_calibration_path,
        enabled=settings.fusion_alignment_shadow_enabled,
        realtime_active=True,
        radar_track_buffer=radar_track_buffer,
    )
    fusion_state_config = FusionStateConfig(
        ema_alpha=settings.fusion_ema_alpha,
        watch_enter=settings.fusion_medium_threshold,
        watch_exit=settings.fusion_watch_exit_threshold,
        high_enter=settings.fusion_high_threshold,
        high_exit=settings.fusion_high_exit_threshold,
        imminent_enter=settings.fusion_imminent_threshold,
        watch_confirmation_windows=settings.fusion_watch_confirmation_windows,
        high_confirmation_windows=settings.fusion_high_confirmation_windows,
        normal_confirmation_windows=settings.fusion_normal_confirmation_windows,
        conflict_score_gap=settings.fusion_conflict_score_gap,
        minimum_modality_quality=settings.fusion_minimum_modality_quality,
    )
    multimodal_fusion = MultimodalFusionService(
        fall_live_monitor.get_status,
        radar_integration.get_status,
        camera_weight=settings.fusion_camera_weight,
        radar_weight=settings.fusion_radar_weight,
        sync_tolerance_seconds=settings.fusion_sync_tolerance_seconds,
        medium_threshold=settings.fusion_medium_threshold,
        high_threshold=settings.fusion_high_threshold,
        state_config=fusion_state_config,
        shadow_logger=fusion_shadow_logger,
        response_callback=fusion_event_bridge.handle,
        temporal_association_config=TemporalAssociationConfig(
            window_seconds=settings.fusion_temporal_window_seconds,
            confirmation_windows=settings.fusion_temporal_confirmation_windows,
            minimum_quality=settings.fusion_minimum_modality_quality,
        ),
        alignment_adapter=alignment_adapter,
        associated_evidence_config=AssociatedEvidenceConfig(
            window_seconds=settings.fusion_associated_window_seconds,
            minimum_track_samples=settings.fusion_associated_minimum_track_samples,
            minimum_point_count=settings.fusion_associated_minimum_point_count,
            minimum_track_stability=(
                settings.fusion_associated_minimum_track_stability
            ),
            weak_vertical_velocity_mps=(
                settings.fusion_associated_weak_vertical_velocity_mps
            ),
            strong_vertical_velocity_mps=(
                settings.fusion_associated_strong_vertical_velocity_mps
            ),
            weak_height_drop_m=settings.fusion_associated_weak_height_drop_m,
            strong_height_drop_m=settings.fusion_associated_strong_height_drop_m,
        ),
        radar_eligibility_config=RadarEligibilityConfig(
            enabled=settings.fusion_radar_eligibility_enabled,
            history_window_seconds=(
                settings.fusion_radar_eligibility_history_seconds
            ),
            minimum_track_samples=(
                settings.fusion_radar_eligibility_minimum_track_samples
            ),
            minimum_point_count=(
                settings.fusion_radar_eligibility_minimum_point_count
            ),
            reference_point_count=(
                settings.fusion_radar_eligibility_reference_point_count
            ),
            minimum_track_stability=(
                settings.fusion_radar_eligibility_minimum_track_stability
            ),
            minimum_radar_quality=(
                settings.fusion_radar_eligibility_minimum_quality
            ),
            maximum_velocity_jump_mps=(
                settings.fusion_radar_eligibility_maximum_velocity_jump_mps
            ),
            height_consistency_tolerance_m=(
                settings.fusion_radar_eligibility_height_tolerance_m
            ),
        ),
    )
    offline_evidence_replay = OfflineEvidenceReplayService(
        settings.fusion_offline_replay_preview_path,
        multimodal_fusion,
        state_config=fusion_state_config,
    )
    fusion_shadow_sampler = FusionShadowSampler(
        lambda: multimodal_fusion.get_latest(method=settings.fusion_default_method),
        enabled=(
            settings.fusion_shadow_log_enabled
            and settings.fusion_shadow_sampler_enabled
        ),
        interval_seconds=settings.fusion_shadow_sample_interval_seconds,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.radar_integration_enabled:
            radar_integration.start()
        fusion_shadow_sampler.start()
        try:
            yield
        finally:
            fusion_shadow_sampler.stop()
            radar_integration.stop()
            fall_live_monitor.stop()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.state.radar_integration_service = radar_integration
    app.state.fall_live_monitor_service = fall_live_monitor
    app.state.guard_session_service = guard_session
    app.state.multimodal_fusion_service = multimodal_fusion
    app.state.fusion_event_bridge = fusion_event_bridge
    app.state.fusion_shadow_sampler = fusion_shadow_sampler
    app.state.offline_evidence_replay_service = offline_evidence_replay
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Content-Type"],
    )
    app.include_router(health_router)
    app.include_router(monitoring_router)
    app.include_router(events_router)
    app.include_router(dashboard_router)
    app.include_router(radar_router)
    app.include_router(multimodal_router)
    app.include_router(simulation_router)
    app.include_router(ezviz_router)
    app.include_router(fall_inference_router)
    app.include_router(fall_live_router)
    app.include_router(guard_session_router)
    return app


app = create_app()
