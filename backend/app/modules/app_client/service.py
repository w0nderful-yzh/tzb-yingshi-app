from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, cast
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.infrastructure.database.models import (
    DeviceModel,
    EmergencyContactModel,
    EventActionModel,
    FamilyBindingModel,
    RiskEventModel,
    UserModel,
)
from app.infrastructure.external.ys7.api_client import (
    Ys7ApiError,
    Ys7LiveAddressProvider,
    Ys7SdkCredentialProvider,
)
from app.modules.app_client.schemas import (
    ActivityData,
    AnalysisData,
    ContactItem,
    ContactsData,
    DeviceItem,
    DeviceListData,
    ElderItem,
    EldersData,
    EmptyData,
    EscalationData,
    EventDetailData,
    EventListData,
    EventsStatsData,
    EvidenceFrameData,
    FraudContextData,
    LiveSdkSessionData,
    LiveUrlData,
    ReasonItem,
    RiskEventItem,
    SafetyStatus,
    SosRequest,
    SosResult,
    StatsBucket,
    TodayStats,
    UserInfo,
)

DemoRole = Literal["elder", "family"]

_EVENT_TYPES = {
    "FRAUD_SUSPECTED": "fraud_suspected",
    "FALL_SUSPECTED": "fall_suspected",
    "STRANGER": "stranger",
    "INACTIVITY": "inactivity",
    "SOS": "sos",
    "DEVICE_OFFLINE": "device_offline",
    "NIGHT_LEAVE_BED": "night_leave_bed",
    "SEDENTARY": "sedentary",
}
_EVENT_TITLES = {
    "FRAUD_SUSPECTED": "疑似诈骗风险",
    "FALL_SUSPECTED": "疑似跌倒",
    "STRANGER": "陌生人到访",
    "INACTIVITY": "长时间无活动",
    "SOS": "紧急求助",
    "DEVICE_OFFLINE": "设备离线",
    "NIGHT_LEAVE_BED": "夜间离床",
    "SEDENTARY": "久坐提醒",
}
_OVERALL_BY_LEVEL = {
    None: "safe",
    "REMINDER": "attention",
    "WARNING": "attention",
    "EMERGENCY": "danger",
}
_LEVEL_RANK = {"REMINDER": 1, "WARNING": 2, "EMERGENCY": 3}


@dataclass(frozen=True, slots=True)
class AppIdentity:
    user: UserModel
    role: DemoRole


def _external_user_id(user: UserModel) -> str:
    return user.external_subject or str(user.id)


def _event_id(event: RiskEventModel) -> str:
    return f"evt_{event.id}"


def _parse_event_id(value: str) -> UUID:
    raw = value.removeprefix("evt_")
    try:
        return UUID(raw)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="event not found") from exc


def _evidence_image_url(evidence: dict[str, object]) -> str | None:
    direct = evidence.get("evidence_image_url") or evidence.get("image_url")
    if isinstance(direct, str):
        return direct
    chain = evidence.get("evidence_chain")
    if isinstance(chain, list):
        for item in reversed(chain):
            if isinstance(item, dict):
                image_url = item.get("image_url")
                if isinstance(image_url, str):
                    return image_url
    return None


def _parse_evidence_time(value: object, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        seconds = float(value) / 1000 if value > 10_000_000_000 else float(value)
        return datetime.fromtimestamp(seconds, UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        except ValueError:
            return fallback
    return fallback


def _evidence_frames(
    evidence: dict[str, object],
    occurred_at: datetime,
) -> list[EvidenceFrameData]:
    frames: list[EvidenceFrameData] = []
    sources = evidence.get("evidence_frames")
    candidates = sources if isinstance(sources, list) else []
    chain = evidence.get("evidence_chain")
    if isinstance(chain, list):
        candidates = [*candidates, *chain]

    seen: set[tuple[str, datetime]] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        image_url = item.get("image_url") or item.get("evidence_image_url")
        if not isinstance(image_url, str) or not image_url:
            continue
        raw_time = item.get("captured_at") or item.get("timestamp")
        end_ms = item.get("end_ms")
        captured_at = (
            occurred_at + timedelta(milliseconds=float(end_ms))
            if raw_time is None and isinstance(end_ms, (int, float))
            else _parse_evidence_time(raw_time, occurred_at)
        )
        key = (image_url, captured_at)
        if key not in seen:
            seen.add(key)
            frames.append(EvidenceFrameData(captured_at=captured_at, image_url=image_url))

    direct = _evidence_image_url(evidence)
    if direct and not any(frame.image_url == direct for frame in frames):
        frames.append(EvidenceFrameData(captured_at=occurred_at, image_url=direct))
    return sorted(frames, key=lambda frame: frame.captured_at)[-8:]


def _event_location(evidence: dict[str, object]) -> str:
    value = evidence.get("location") or evidence.get("room")
    return value if isinstance(value, str) else ""


def _fraud_scene(evidence: dict[str, object]) -> Literal["telecom", "home_visit", "unknown"]:
    chain = evidence.get("evidence_chain")
    if not isinstance(chain, list):
        return "unknown"

    items = [item for item in chain if isinstance(item, dict)]
    kinds = {str(item.get("kind", "")) for item in items}
    if "visitor_presence" in kinds:
        return "home_visit"
    if "phone_call_active" in kinds:
        return "telecom"

    people_counts = [
        item.get("people_count")
        for item in items
        if item.get("kind") == "people_count_context"
    ]
    if any(isinstance(count, int) and count >= 2 for count in people_counts):
        return "home_visit"
    if any(item.get("source") == "speech" for item in items):
        return "telecom"
    return "unknown"


def _fraud_context(event: RiskEventModel) -> FraudContextData | None:
    if event.event_type != "FRAUD_SUSPECTED":
        return None
    evidence = event.evidence or {}
    raw_index = evidence.get("state_index", 0)
    state_index = raw_index if isinstance(raw_index, int) else 0
    return FraudContextData(
        scene=_fraud_scene(evidence),
        state=str(evidence.get("state", "S0_NORMAL")),
        state_index=max(0, min(5, state_index)),
        state_label=str(evidence.get("state_label", "风险分析")),
        decision=str(evidence.get("decision", "observe")),
        transition_reason=str(evidence.get("transition_reason", event.summary)),
    )


def _highest_level(events: list[RiskEventModel]) -> str | None:
    return max(
        (event.alert_level for event in events),
        key=lambda level: _LEVEL_RANK[level],
        default=None,
    )


def _device_signal(device: DeviceModel) -> Literal["good", "weak", "offline"]:
    if device.status in {"OFFLINE", "UNKNOWN"}:
        return "offline"
    configured = str(device.settings.get("signal", "good"))
    return cast(
        Literal["good", "weak", "offline"],
        configured if configured in {"good", "weak", "offline"} else "good",
    )


class AppClientService:
    def __init__(
        self,
        *,
        session: AsyncSession,
        settings: Settings,
        live_address_provider: Ys7LiveAddressProvider,
        sdk_credential_provider: Ys7SdkCredentialProvider,
    ) -> None:
        self._session = session
        self._settings = settings
        self._live_address_provider = live_address_provider
        self._sdk_credential_provider = sdk_credential_provider

    async def resolve_identity(self, role: DemoRole) -> AppIdentity:
        if not self._settings.demo_identity_enabled or self._settings.environment == "production":
            raise HTTPException(status_code=401, detail="authentication is not configured")
        subject = (
            self._settings.demo_elder_subject
            if role == "elder"
            else self._settings.demo_guardian_subject
        )
        expected_role = "ELDER" if role == "elder" else "GUARDIAN"
        user = await self._session.scalar(
            select(UserModel).where(
                UserModel.external_subject == subject,
                UserModel.role == expected_role,
                UserModel.is_active.is_(True),
            )
        )
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="demo identity is not initialized; run the demo seed command",
            )
        return AppIdentity(user=user, role=role)

    async def resolve_elder(
        self,
        identity: AppIdentity,
        elder_id: str | None,
    ) -> UserModel:
        if identity.role == "elder":
            if elder_id is not None and elder_id != _external_user_id(identity.user):
                raise HTTPException(status_code=403, detail="elder access denied")
            return identity.user

        statement = (
            select(UserModel)
            .join(
                FamilyBindingModel,
                FamilyBindingModel.elder_user_id == UserModel.id,
            )
            .where(
                FamilyBindingModel.guardian_user_id == identity.user.id,
                FamilyBindingModel.status == "ACTIVE",
                UserModel.is_active.is_(True),
            )
            .order_by(FamilyBindingModel.created_at)
        )
        if elder_id is not None:
            statement = statement.where(UserModel.external_subject == elder_id)
        elder = await self._session.scalar(statement.limit(1))
        if elder is None:
            raise HTTPException(status_code=403, detail="elder access denied")
        return elder

    async def get_me(self, identity: AppIdentity) -> UserInfo:
        preferences = identity.user.preferences or {}
        bound_count = 0
        if identity.role == "elder":
            bound_count = int(
                await self._session.scalar(
                    select(func.count(FamilyBindingModel.id)).where(
                        FamilyBindingModel.elder_user_id == identity.user.id,
                        FamilyBindingModel.status == "ACTIVE",
                    )
                )
                or 0
            )
        return UserInfo(
            user_id=_external_user_id(identity.user),
            role=identity.role,
            name=identity.user.display_name,
            bound_family_count=bound_count,
            font_size=str(preferences.get("font_size", "extra_large")),
            voice_assist_enabled=bool(preferences.get("voice_assist_enabled", True)),
        )

    async def get_safety_status(self, elder: UserModel) -> SafetyStatus:
        active_events = list(
            (
                await self._session.scalars(
                    select(RiskEventModel).where(
                        RiskEventModel.elder_user_id == elder.id,
                        RiskEventModel.status.in_(("OPEN", "ACKNOWLEDGED")),
                    )
                )
            ).all()
        )
        devices = list(
            (
                await self._session.scalars(
                    select(DeviceModel).where(
                        DeviceModel.elder_user_id == elder.id,
                        DeviceModel.monitoring_enabled.is_(True),
                    )
                )
            ).all()
        )
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        today_events = int(
            await self._session.scalar(
                select(func.count(RiskEventModel.id)).where(
                    RiskEventModel.elder_user_id == elder.id,
                    RiskEventModel.occurred_at >= today_start,
                )
            )
            or 0
        )
        highest = _highest_level(active_events)
        overall = cast(Literal["safe", "attention", "danger"], _OVERALL_BY_LEVEL[highest])
        label = {
            "safe": "一切正常",
            "attention": f"有 {len(active_events)} 条安全提醒待查看",
            "danger": f"有 {len(active_events)} 条紧急告警待确认",
        }[overall]
        return SafetyStatus(
            overall=overall,
            overall_label=label,
            active_event_count=len(active_events),
            highest_active_level=highest.lower() if highest else None,
            devices_online=sum(device.status == "ONLINE" for device in devices),
            devices_total=len(devices),
            checked_at=datetime.now(UTC),
            today=TodayStats(event_count=today_events),
        )

    async def list_events(
        self,
        elder: UserModel,
        *,
        level: str | None,
        status: str | None,
        limit: int,
    ) -> EventListData:
        statement = select(RiskEventModel).where(RiskEventModel.elder_user_id == elder.id)
        if level is not None:
            statement = statement.where(RiskEventModel.alert_level == level.upper())
        if status is not None:
            statement = statement.where(RiskEventModel.status == status.upper())
        events = list(
            (
                await self._session.scalars(
                    statement.order_by(
                        RiskEventModel.occurred_at.desc(),
                        RiskEventModel.id.desc(),
                    ).limit(limit)
                )
            ).all()
        )
        return EventListData(events=[self._event_item(event) for event in events])

    async def get_event(self, identity: AppIdentity, event_id: str) -> EventDetailData:
        event = await self._authorized_event(identity, event_id)
        evidence = event.evidence or {}
        reasons: list[ReasonItem] = []
        chain = evidence.get("evidence_chain")
        if isinstance(chain, list):
            for item in chain[-8:]:
                if not isinstance(item, dict):
                    continue
                reasons.append(
                    ReasonItem(
                        key=str(item.get("kind", "evidence")),
                        label=str(item.get("reason", "风险证据")),
                        value=str(item.get("text", "")),
                    )
                )
        if not reasons:
            reasons.append(
                ReasonItem(
                    key="summary",
                    label="风险摘要",
                    value=event.summary,
                )
            )
        return EventDetailData(
            event_id=_event_id(event),
            type=_EVENT_TYPES[event.event_type],
            level=event.alert_level.lower(),
            status=event.status.lower(),
            version=event.version,
            device_id=event.external_device_id or "",
            occurred_at=event.occurred_at,
            evidence_image_url=_evidence_image_url(evidence),
            evidence_frames=_evidence_frames(evidence, event.occurred_at),
            location=_event_location(evidence),
            analysis=AnalysisData(
                confidence=float(event.confidence or 0.0),
                reasons=reasons,
                disclaimer="AI 辅助判断，不替代医疗、公安或金融机构的专业结论。",
            ),
            notifications=[],
            escalation=EscalationData(),
            fraud=_fraud_context(event),
        )

    async def create_sos(
        self,
        elder: UserModel,
        payload: SosRequest,
        idempotency_key: str,
    ) -> SosResult:
        source_event_id = f"app-sos:{idempotency_key}"
        existing = await self._session.scalar(
            select(RiskEventModel).where(
                RiskEventModel.source == "APP_SOS",
                RiskEventModel.source_event_id == source_event_id,
            )
        )
        if existing is not None:
            return SosResult(
                event_id=_event_id(existing),
                status="duplicate",
                notified_contacts=0,
            )
        event = RiskEventModel(
            source="APP_SOS",
            source_event_id=source_event_id,
            elder_user_id=elder.id,
            event_type="SOS",
            risk_level="CRITICAL",
            alert_level="EMERGENCY",
            status="OPEN",
            confidence=1.0,
            summary="老人通过 App 发起紧急求助。",
            occurred_at=payload.occurred_at,
            received_at=datetime.now(UTC),
            evidence={"trigger": payload.trigger},
            model_name="app_sos",
            model_version="1.0",
        )
        self._session.add(event)
        await self._session.commit()
        return SosResult(event_id=_event_id(event), status="recorded", notified_contacts=0)

    async def confirm_event(
        self,
        identity: AppIdentity,
        event_id: str,
        *,
        action: Literal["im_ok", "need_help"],
        version: int | None,
        idempotency_key: str,
    ) -> EmptyData:
        event = await self._authorized_event(identity, event_id)
        target_status = "RESOLVED" if action == "im_ok" else "ACKNOWLEDGED"
        action_type = "CONFIRM" if action == "im_ok" else "NEED_HELP"
        await self._apply_event_action(
            event,
            identity=identity,
            target_status=target_status,
            action_type=action_type,
            note=None,
            version=version,
            idempotency_key=idempotency_key,
        )
        return EmptyData()

    async def patch_event_status(
        self,
        identity: AppIdentity,
        event_id: str,
        *,
        status: Literal["acknowledged", "resolved", "false_alarm"],
        note: str,
        version: int | None,
        idempotency_key: str,
    ) -> EmptyData:
        event = await self._authorized_event(identity, event_id)
        target_status = status.upper()
        action_type = {
            "ACKNOWLEDGED": "ACKNOWLEDGE",
            "RESOLVED": "RESOLVE",
            "FALSE_ALARM": "FALSE_ALARM",
        }[target_status]
        await self._apply_event_action(
            event,
            identity=identity,
            target_status=target_status,
            action_type=action_type,
            note=note or None,
            version=version,
            idempotency_key=idempotency_key,
        )
        return EmptyData()

    async def list_devices(self, elder: UserModel) -> DeviceListData:
        devices = list(
            (
                await self._session.scalars(
                    select(DeviceModel)
                    .where(
                        DeviceModel.elder_user_id == elder.id,
                        DeviceModel.monitoring_enabled.is_(True),
                    )
                    .order_by(DeviceModel.created_at)
                )
            ).all()
        )
        return DeviceListData(
            devices=[
                DeviceItem(
                    device_id=device.external_device_id,
                    name=device.name,
                    room=device.room or "",
                    online=device.status == "ONLINE",
                    signal=_device_signal(device),
                    last_seen_at=device.last_seen_at,
                )
                for device in devices
            ]
        )

    async def get_live_url(self, elder: UserModel, device_id: str) -> LiveUrlData:
        device = await self._session.scalar(
            select(DeviceModel).where(
                DeviceModel.elder_user_id == elder.id,
                DeviceModel.external_device_id == device_id,
                DeviceModel.monitoring_enabled.is_(True),
            )
        )
        if device is None:
            raise HTTPException(status_code=404, detail="device not found")
        protocol = self._settings.ys7_live_protocol
        url = await self._live_address_provider.get_live_address(
            device_serial=device.external_device_id,
            channel_no=device.channel_no,
            protocol=protocol,
            quality=self._settings.ys7_live_quality,
        )
        device.status = "ONLINE"
        device.last_seen_at = datetime.now(UTC)
        await self._session.commit()
        return LiveUrlData(url=url, protocol=protocol, expires_in=300)

    async def get_live_sdk_session(
        self,
        elder: UserModel,
        device_id: str,
    ) -> LiveSdkSessionData:
        device = await self._session.scalar(
            select(DeviceModel).where(
                DeviceModel.elder_user_id == elder.id,
                DeviceModel.external_device_id == device_id,
                DeviceModel.monitoring_enabled.is_(True),
            )
        )
        if device is None:
            raise HTTPException(status_code=404, detail="device not found")
        if self._settings.ys7_app_key is None:
            raise Ys7ApiError("YS7 AppKey is not configured")

        access_token = await self._sdk_credential_provider.get_access_token()
        device.status = "ONLINE"
        device.last_seen_at = datetime.now(UTC)
        await self._session.commit()
        return LiveSdkSessionData(
            app_key=self._settings.ys7_app_key.get_secret_value(),
            access_token=access_token,
            device_serial=device.external_device_id,
            channel_no=device.channel_no,
            expires_in=300,
        )

    async def list_contacts(self, elder: UserModel) -> ContactsData:
        contacts = list(
            (
                await self._session.scalars(
                    select(EmergencyContactModel)
                    .where(
                        EmergencyContactModel.elder_user_id == elder.id,
                        EmergencyContactModel.status == "ACTIVE",
                    )
                    .order_by(EmergencyContactModel.priority_order)
                )
            ).all()
        )
        return ContactsData(
            contacts=[
                ContactItem(
                    order=contact.priority_order,
                    name=contact.name,
                    relation=contact.relation,
                    phone=f"****{contact.phone_last4}",
                    channels=[str(channel).lower() for channel in contact.channels],
                )
                for contact in contacts
            ]
        )

    async def list_elders(self, identity: AppIdentity) -> EldersData:
        if identity.role != "family":
            raise HTTPException(status_code=403, detail="family role required")
        rows = (
            await self._session.execute(
                select(UserModel, FamilyBindingModel)
                .join(
                    FamilyBindingModel,
                    FamilyBindingModel.elder_user_id == UserModel.id,
                )
                .where(
                    FamilyBindingModel.guardian_user_id == identity.user.id,
                    FamilyBindingModel.status == "ACTIVE",
                    UserModel.is_active.is_(True),
                )
                .order_by(FamilyBindingModel.created_at)
            )
        ).all()
        elders: list[ElderItem] = []
        for elder, binding in rows:
            active_events = list(
                (
                    await self._session.scalars(
                        select(RiskEventModel).where(
                            RiskEventModel.elder_user_id == elder.id,
                            RiskEventModel.status.in_(("OPEN", "ACKNOWLEDGED")),
                        )
                    )
                ).all()
            )
            last_active_at = await self._session.scalar(
                select(func.max(DeviceModel.last_seen_at)).where(
                    DeviceModel.elder_user_id == elder.id
                )
            )
            highest = _highest_level(active_events)
            overall = cast(
                Literal["safe", "attention", "danger"],
                _OVERALL_BY_LEVEL[highest],
            )
            elders.append(
                ElderItem(
                    elder_id=_external_user_id(elder),
                    name=elder.display_name,
                    relation=binding.relation,
                    overall=overall,
                    last_active_at=last_active_at,
                    pending_event_count=len(active_events),
                )
            )
        return EldersData(elders=elders)

    async def get_event_stats(self, elder: UserModel, days: int) -> EventsStatsData:
        since = datetime.now(UTC) - timedelta(days=days)
        events = list(
            (
                await self._session.scalars(
                    select(RiskEventModel).where(
                        RiskEventModel.elder_user_id == elder.id,
                        RiskEventModel.occurred_at >= since,
                    )
                )
            ).all()
        )
        buckets: dict[str, dict[str, int]] = {}
        for event in events:
            year, week, _ = event.occurred_at.isocalendar()
            period = f"{year}-W{week:02d}"
            counts = buckets.setdefault(
                period,
                {"reminder": 0, "warning": 0, "emergency": 0},
            )
            counts[event.alert_level.lower()] += 1
        return EventsStatsData(
            buckets=[StatsBucket(period=period, **buckets[period]) for period in sorted(buckets)]
        )

    @staticmethod
    def get_activity_stats() -> ActivityData:
        return ActivityData(hours=[])

    def _event_item(self, event: RiskEventModel) -> RiskEventItem:
        evidence = event.evidence or {}
        fraud = _fraud_context(event)
        return RiskEventItem(
            event_id=_event_id(event),
            type=_EVENT_TYPES[event.event_type],
            level=event.alert_level.lower(),
            title=_EVENT_TITLES[event.event_type],
            summary=event.summary,
            device_id=event.external_device_id or "",
            occurred_at=event.occurred_at,
            status=event.status.lower(),
            version=event.version,
            evidence_image_url=_evidence_image_url(evidence),
            evidence_frames=_evidence_frames(evidence, event.occurred_at),
            location=_event_location(evidence),
            fraud_scene=fraud.scene if fraud else None,
            fraud_state=fraud.state if fraud else None,
            fraud_state_index=fraud.state_index if fraud else None,
            fraud_state_label=fraud.state_label if fraud else None,
            fraud_decision=fraud.decision if fraud else None,
        )

    async def _authorized_event(
        self,
        identity: AppIdentity,
        event_id: str,
    ) -> RiskEventModel:
        event = await self._session.get(RiskEventModel, _parse_event_id(event_id))
        if event is None or event.elder_user_id is None:
            raise HTTPException(status_code=404, detail="event not found")
        if identity.role == "elder":
            authorized = event.elder_user_id == identity.user.id
        else:
            authorized = (
                await self._session.scalar(
                    select(FamilyBindingModel.id).where(
                        FamilyBindingModel.guardian_user_id == identity.user.id,
                        FamilyBindingModel.elder_user_id == event.elder_user_id,
                        FamilyBindingModel.status == "ACTIVE",
                    )
                )
                is not None
            )
        if not authorized:
            raise HTTPException(status_code=404, detail="event not found")
        return event

    async def _apply_event_action(
        self,
        event: RiskEventModel,
        *,
        identity: AppIdentity,
        target_status: str,
        action_type: str,
        note: str | None,
        version: int | None,
        idempotency_key: str,
    ) -> None:
        existing = await self._session.scalar(
            select(EventActionModel).where(EventActionModel.idempotency_key == idempotency_key)
        )
        if existing is not None:
            if existing.risk_event_id != event.id:
                raise HTTPException(status_code=409, detail="idempotency key conflict")
            return
        if version is not None and version != event.version:
            raise HTTPException(status_code=409, detail="event version conflict")
        if event.status in {"RESOLVED", "FALSE_ALARM"} and event.status != target_status:
            raise HTTPException(status_code=409, detail="event status transition is not allowed")
        previous_status = event.status
        event.status = target_status
        event.version += 1
        self._session.add(
            EventActionModel(
                risk_event_id=event.id,
                actor_user_id=identity.user.id,
                action_type=action_type,
                previous_status=previous_status,
                new_status=target_status,
                note=note,
                idempotency_key=idempotency_key,
                action_metadata={},
            )
        )
        await self._session.commit()
