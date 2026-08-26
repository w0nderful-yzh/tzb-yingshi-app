from __future__ import annotations

import threading
from datetime import datetime, timezone
from time import monotonic
from uuid import uuid4

from app.modules.fall.multimodal_engine.data_sources.adapters.radar_service import (
    RadarServiceDataSourceAdapter,
)
from app.modules.fall.multimodal_engine.schemas.fall_live import FallLiveState
from app.modules.fall.multimodal_engine.schemas.guard_session import (
    GuardCapabilityState,
    GuardCapabilityStatus,
    GuardSessionState,
    MultimodalGuardSessionStatus,
)
from app.modules.fall.multimodal_engine.services.fall_live_monitor import (
    FallLiveMonitorService,
)
from app.modules.fall.multimodal_engine.services.radar_integration import (
    RadarIntegrationService,
)


class MultimodalGuardSessionService:
    """Idempotent Camera/Radar analysis-session lifecycle.

    The Radar process is system-owned. Starting a guard session only asks the
    Radar API to ensure that singleton is available and then enables its
    evidence for this session. Stopping never calls the Radar stop endpoint.
    """

    def __init__(
        self,
        fall_live_monitor: FallLiveMonitorService,
        radar_integration: RadarIntegrationService,
        radar_source: RadarServiceDataSourceAdapter,
        radar_retry_interval_seconds: float = 2.0,
    ) -> None:
        self._fall_live_monitor = fall_live_monitor
        self._radar_integration = radar_integration
        self._radar_source = radar_source
        self._lock = threading.RLock()
        self._active = False
        self._session_id: str | None = None
        self._started_at: datetime | None = None
        self._radar_ensure_error: str | None = None
        self._radar_retry_interval_seconds = max(0.0, radar_retry_interval_seconds)
        self._last_radar_ensure_attempt = 0.0

    def start(self, session_id: str | None = None) -> MultimodalGuardSessionStatus:
        with self._lock:
            if self._active:
                return self._status_locked()
            normalized = (session_id or f"guard-{uuid4().hex}").strip()
            if not normalized:
                raise ValueError("session_id must not be blank")
            self._active = True
            self._session_id = normalized
            self._started_at = datetime.now(timezone.utc)
            self._radar_ensure_error = None
            self._try_ensure_radar_locked(force=True)
            self._radar_integration.enable_for_session(normalized)
            self._fall_live_monitor.start()
            return self._status_locked()

    def stop(self) -> MultimodalGuardSessionStatus:
        with self._lock:
            if not self._active:
                return self._status_locked()
            self._fall_live_monitor.stop()
            self._radar_integration.disable_for_session()
            self._active = False
            self._session_id = None
            self._started_at = None
            self._radar_ensure_error = None
            return self._status_locked()

    def get_status(self) -> MultimodalGuardSessionStatus:
        with self._lock:
            if self._active and not self._radar_source.online:
                self._try_ensure_radar_locked(force=False)
            return self._status_locked()

    def _try_ensure_radar_locked(self, *, force: bool) -> None:
        now = monotonic()
        if (
            not force
            and now - self._last_radar_ensure_attempt
            < self._radar_retry_interval_seconds
        ):
            return
        self._last_radar_ensure_attempt = now
        try:
            self._radar_source.ensure_running()
            self._radar_ensure_error = None
        except (RuntimeError, ValueError) as exc:
            # Radar is optional for session startup. Status polling retries the
            # idempotent ensure call; Fusion remains Camera-only meanwhile.
            self._radar_ensure_error = f"{type(exc).__name__}: {exc}"

    def _status_locked(self) -> MultimodalGuardSessionStatus:
        camera = self._fall_live_monitor.get_status()
        radar = self._radar_integration.get_status()
        reasons: list[str] = []

        if not self._active:
            stopped = GuardCapabilityStatus(
                state=GuardCapabilityState.STOPPED,
                enabled_for_session=False,
                detail="当前未启用风险分析会话",
            )
            return MultimodalGuardSessionStatus(
                active=False,
                state=GuardSessionState.STOPPED,
                camera_analysis=stopped,
                radar_worker=GuardCapabilityStatus(
                    state=(
                        GuardCapabilityState.RUNNING
                        if self._radar_source.online
                        else GuardCapabilityState.UNAVAILABLE
                    ),
                    enabled_for_session=False,
                    detail="Radar Worker保持系统级运行，不随会话停止",
                ),
                radar_participation=stopped,
                fusion=stopped,
            )

        camera_state = self._camera_capability_state(camera.state)
        if camera_state in {GuardCapabilityState.DEGRADED, GuardCapabilityState.UNAVAILABLE}:
            reasons.append("CAMERA_ANALYSIS_UNAVAILABLE")

        radar_worker_running = self._radar_source.online or radar.online
        if self._radar_ensure_error is not None:
            reasons.append("RADAR_ENSURE_RUNNING_FAILED")
        if not radar_worker_running:
            reasons.append("RADAR_UNAVAILABLE_CAMERA_ONLY")

        radar_participating = self._radar_integration.session_enabled and radar.online
        overall_state = (
            GuardSessionState.ACTIVE
            if camera_state is GuardCapabilityState.RUNNING and radar_participating
            else GuardSessionState.STARTING
            if camera_state is GuardCapabilityState.STARTING
            else GuardSessionState.DEGRADED
        )
        return MultimodalGuardSessionStatus(
            session_id=self._session_id,
            active=True,
            state=overall_state,
            camera_analysis=GuardCapabilityStatus(
                state=camera_state,
                enabled_for_session=True,
                detail=camera.input_message or camera.error or "Camera分析已启用",
            ),
            radar_worker=GuardCapabilityStatus(
                state=(
                    GuardCapabilityState.RUNNING
                    if radar_worker_running
                    else GuardCapabilityState.UNAVAILABLE
                ),
                enabled_for_session=False,
                detail=(
                    "Radar Worker系统级单例运行中"
                    if radar_worker_running
                    else self._radar_ensure_error or "Radar Worker暂不可用"
                ),
            ),
            radar_participation=GuardCapabilityStatus(
                state=(
                    GuardCapabilityState.RUNNING
                    if radar_participating
                    else GuardCapabilityState.DEGRADED
                ),
                enabled_for_session=True,
                detail=(
                    "Radar Evidence已绑定当前会话"
                    if radar_participating
                    else "Radar不可用，当前安全降级为Camera-only"
                ),
            ),
            fusion=GuardCapabilityStatus(
                state=(
                    GuardCapabilityState.RUNNING
                    if camera_state is GuardCapabilityState.RUNNING
                    else GuardCapabilityState.DEGRADED
                ),
                enabled_for_session=True,
                detail=(
                    "Camera-led Fusion v2联合判断"
                    if radar_participating
                    else "Camera-led Fusion v2处于Camera-only降级"
                ),
            ),
            reason_codes=reasons,
            started_at=self._started_at,
        )

    @staticmethod
    def _camera_capability_state(state: FallLiveState) -> GuardCapabilityState:
        if state in {FallLiveState.RUNNING}:
            return GuardCapabilityState.RUNNING
        if state in {
            FallLiveState.STARTING,
            FallLiveState.LOADING_MODELS,
            FallLiveState.CONNECTING,
        }:
            return GuardCapabilityState.STARTING
        if state in {FallLiveState.ERROR}:
            return GuardCapabilityState.DEGRADED
        if state in {FallLiveState.DISABLED}:
            return GuardCapabilityState.UNAVAILABLE
        return GuardCapabilityState.STOPPED
