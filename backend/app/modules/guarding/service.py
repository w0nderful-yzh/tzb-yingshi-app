from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from app.modules.fall.ports import FallRiskSourceError, GuardSessionControl
from app.modules.fall.source_schemas import GuardCapabilitySourceStatus, GuardSessionSourceStatus
from app.modules.guarding.psychology_observation import PsychologyObservationController
from app.modules.guarding.schemas import CapabilityStatus, GuardianSessionStatus, LifecycleState


class GuardianSessionService:
    def __init__(
        self,
        *,
        fall_control: GuardSessionControl | None,
        psychology: PsychologyObservationController,
        fraud_monitoring_enabled: bool,
    ) -> None:
        self._fall_control = fall_control
        self._psychology = psychology
        self._fraud_monitoring_enabled = fraud_monitoring_enabled
        self._lock = asyncio.Lock()
        self._active = False
        self._session_id: str | None = None
        self._subject_key: str | None = None
        self._started_at: datetime | None = None
        self._fall_status: GuardSessionSourceStatus | None = None

    async def start(self, *, subject_key: str) -> GuardianSessionStatus:
        async with self._lock:
            if self._active:
                return self._build_status()
            self._active = True
            self._session_id = f"guard-{uuid4().hex}"
            self._subject_key = subject_key
            self._started_at = datetime.now(UTC)
            self._psychology.start(subject_key)
            if self._fall_control is not None:
                try:
                    self._fall_status = await self._fall_control.start_guard_session(
                        self._session_id
                    )
                    # If the upstream process survived an App-backend restart,
                    # adopt its already-active idempotent session identity.
                    if self._fall_status.session_id:
                        self._session_id = self._fall_status.session_id
                except (FallRiskSourceError, ValueError):
                    self._fall_status = None
            return self._build_status()

    async def stop(self) -> GuardianSessionStatus:
        async with self._lock:
            if not self._active:
                return self._build_status()
            if self._fall_control is not None:
                try:
                    self._fall_status = await self._fall_control.stop_guard_session()
                except (FallRiskSourceError, ValueError):
                    self._fall_status = None
            await self._psychology.stop()
            self._active = False
            self._session_id = None
            self._subject_key = None
            self._started_at = None
            return self._build_status()

    async def get_status(self) -> GuardianSessionStatus:
        async with self._lock:
            if self._active and self._fall_control is not None:
                try:
                    self._fall_status = await self._fall_control.get_guard_session_status()
                except (FallRiskSourceError, ValueError):
                    self._fall_status = None
            return self._build_status()

    async def shutdown(self) -> None:
        await self.stop()

    def _build_status(self) -> GuardianSessionStatus:
        now = datetime.now(UTC)
        if not self._active:
            stopped = CapabilityStatus(
                state=LifecycleState.STOPPED,
                enabled=False,
                detail="未启用",
            )
            return GuardianSessionStatus(
                active=False,
                state=LifecycleState.STOPPED,
                camera_analysis=stopped,
                fraud_monitoring=stopped,
                psychology_observation=stopped,
                radar_worker=self._radar_worker_status(enabled=False),
                radar_participation=stopped,
                fusion=stopped,
                updated_at=now,
            )

        fall = self._fall_status
        reasons: list[str] = []
        if fall is None:
            reasons.append("MULTIMODAL_SERVICE_UNAVAILABLE")
        elif not fall.radar_participation.enabled_for_session or (
            fall.radar_participation.state != "RUNNING"
        ):
            reasons.append("RADAR_UNAVAILABLE_CAMERA_ONLY")

        camera = self._map_fall_capability(
            fall.camera_analysis if fall is not None else None,
            unavailable_detail="Camera跌倒预测服务不可用",
        )
        radar_participation = self._map_fall_capability(
            fall.radar_participation if fall is not None else None,
            unavailable_detail="Radar未参与，Fusion降级Camera-only",
        )
        fusion = self._map_fall_capability(
            fall.fusion if fall is not None else None,
            unavailable_detail="Fusion服务不可用",
        )
        psychology = CapabilityStatus(
            state=(
                LifecycleState.RUNNING if self._psychology.active else LifecycleState.UNAVAILABLE
            ),
            enabled=self._psychology.active,
            detail=(
                "心理观察已启用，按周期读取有限时窗评估"
                if self._psychology.active
                else "心理观察不可用"
            ),
        )
        fraud = CapabilityStatus(
            state=(
                LifecycleState.RUNNING
                if self._fraud_monitoring_enabled
                else LifecycleState.UNAVAILABLE
            ),
            enabled=self._fraud_monitoring_enabled,
            detail=(
                "诈骗语音/视觉监听已启用"
                if self._fraud_monitoring_enabled
                else "诈骗实时监听未配置"
            ),
        )
        state = (
            LifecycleState.RUNNING
            if camera.state is LifecycleState.RUNNING
            and fraud.state is LifecycleState.RUNNING
            and radar_participation.state is LifecycleState.RUNNING
            else LifecycleState.STARTING
            if camera.state is LifecycleState.STARTING
            else LifecycleState.DEGRADED
        )
        return GuardianSessionStatus(
            session_id=self._session_id,
            active=True,
            state=state,
            camera_analysis=camera,
            fraud_monitoring=fraud,
            psychology_observation=psychology,
            radar_worker=self._radar_worker_status(enabled=True),
            radar_participation=radar_participation,
            fusion=fusion,
            reason_codes=reasons,
            started_at=self._started_at,
            updated_at=now,
        )

    def _radar_worker_status(self, *, enabled: bool) -> CapabilityStatus:
        fall = self._fall_status
        if fall is not None:
            return self._map_fall_capability(
                fall.radar_worker,
                unavailable_detail="Radar Worker暂不可用",
                enabled=enabled,
            )
        return CapabilityStatus(
            state=LifecycleState.UNAVAILABLE,
            enabled=False,
            detail="Radar Worker状态不可用；不会阻止其他模块",
        )

    @staticmethod
    def _map_fall_capability(
        source: GuardCapabilitySourceStatus | None,
        *,
        unavailable_detail: str,
        enabled: bool | None = None,
    ) -> CapabilityStatus:
        if source is None:
            return CapabilityStatus(
                state=LifecycleState.UNAVAILABLE,
                enabled=False,
                detail=unavailable_detail,
            )
        mapping = {
            "STOPPED": LifecycleState.STOPPED,
            "STARTING": LifecycleState.STARTING,
            "RUNNING": LifecycleState.RUNNING,
            "DEGRADED": LifecycleState.DEGRADED,
            "UNAVAILABLE": LifecycleState.UNAVAILABLE,
        }
        return CapabilityStatus(
            state=mapping[source.state],
            enabled=source.enabled_for_session if enabled is None else enabled,
            detail=source.detail,
        )
