from datetime import datetime
from uuid import uuid4

import httpx
from pydantic import ValidationError

from app.modules.fall.multimodal_engine.data_sources.adapter import DataSourceAdapter
from app.modules.fall.multimodal_engine.data_sources.contracts import UnifiedDataPacket
from app.modules.fall.multimodal_engine.schemas.radar import (
    RadarAlignmentEvidencePayload,
    RadarCalibratedTcnPredictionPayload,
    RadarDescentPredictionPayload,
    RadarEvidencePayload,
    RadarFallRiskAssessmentPayload,
    RadarHealthPayload,
    RadarLatestPayload,
    RadarPointNetLatestPayload,
    RadarPointNetPredictionPayload,
    RadarTcnLatestPayload,
    RadarTcnPredictionPayload,
)


class RadarServiceDataSourceAdapter(DataSourceAdapter):
    """轮询独立Radar FastAPI并输出统一RADAR Packet。

    只消费雷达服务已经计算好的结构化风险结果。服务不可达、状态不一致或
    结果时间戳未更新时返回None，不复用旧数据。
    """

    def __init__(
        self,
        service_url: str,
        *,
        source_id: str = "iwr6843isk-radar-service",
        timeout_seconds: float = 2.0,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__()
        self.service_url = service_url.rstrip("/")
        self.source_id = source_id
        self._timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client or self._create_client()
        self._last_timestamp: datetime | None = None
        self._latest_payload: (
            RadarLatestPayload
            | RadarTcnPredictionPayload
            | RadarPointNetPredictionPayload
            | RadarCalibratedTcnPredictionPayload
            | None
        ) = None
        self._online = False
        self._last_error: str | None = None
        self._latest_health: RadarHealthPayload | None = None
        self._latest_tcn_baseline: RadarTcnPredictionPayload | None = None
        self._latest_descent: RadarDescentPredictionPayload | None = None
        self._latest_risk_assessment: RadarFallRiskAssessmentPayload | None = None
        self._latest_alignment_evidence: list[RadarAlignmentEvidencePayload] = []

    @property
    def online(self) -> bool:
        return self._online

    @property
    def latest_payload(
        self,
    ) -> (
        RadarLatestPayload
        | RadarTcnPredictionPayload
        | RadarPointNetPredictionPayload
        | RadarCalibratedTcnPredictionPayload
        | None
    ):
        return self._latest_payload

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def latest_health(self) -> RadarHealthPayload | None:
        return self._latest_health

    @property
    def latest_tcn_baseline(self) -> RadarTcnPredictionPayload | None:
        return self._latest_tcn_baseline

    @property
    def latest_descent(self) -> RadarDescentPredictionPayload | None:
        return self._latest_descent

    @property
    def latest_risk_assessment(self) -> RadarFallRiskAssessmentPayload | None:
        return self._latest_risk_assessment

    @property
    def latest_alignment_evidence(self) -> list[RadarAlignmentEvidencePayload]:
        return [item.model_copy(deep=True) for item in self._latest_alignment_evidence]

    def start(self, session_id: str) -> None:
        # FastAPI TestClient can enter the same application lifespan repeatedly.
        # Recreate only the internally owned client after a previous shutdown.
        if self._owns_client and self._client.is_closed:
            self._client = self._create_client()
        super().start(session_id)

    def read(self) -> UnifiedDataPacket | None:
        self._ensure_running()
        try:
            health_response = self._client.get("/health")
            health_response.raise_for_status()
            health = RadarHealthPayload.model_validate(health_response.json())
            self._latest_health = health
            if (
                health.status != "ok"
                or not health.radar_connected
                or not health.model_loaded
            ):
                self._mark_offline("radar service reports no active data source")
                return None

            latest_response = self._client.get("/api/radar/latest")
            latest_response.raise_for_status()
            latest_json = latest_response.json()
            alignment_evidence: list[RadarAlignmentEvidencePayload] = []
            for item in latest_json.get("alignment_evidence") or []:
                try:
                    alignment_evidence.append(
                        RadarAlignmentEvidencePayload.model_validate(item)
                    )
                except (ValidationError, TypeError, ValueError):
                    continue
            # Alignment is an optional shadow sidecar. Keep the authoritative
            # Radar inference envelopes strict by validating them without the
            # sidecar instead of widening any model/checkpoint contract.
            inference_json = {
                key: value
                for key, value in latest_json.items()
                if key != "alignment_evidence"
            }
            tcn_baseline = None
            calibrated_tcn = None
            descent = None
            risk_assessment = None
            if "calibrated_tcn_prediction" in latest_json:
                calibrated_tcn = RadarCalibratedTcnPredictionPayload.model_validate(
                    latest_json["calibrated_tcn_prediction"]
                )
                tcn_baseline = (
                    RadarTcnPredictionPayload.model_validate(
                        latest_json["tcn_baseline"]
                    )
                    if latest_json.get("tcn_baseline")
                    else None
                )
                latest = calibrated_tcn
            elif "pointnet_prediction" in latest_json:
                pointnet_envelope = RadarPointNetLatestPayload.model_validate(
                    inference_json
                )
                latest = pointnet_envelope.pointnet_prediction
                tcn_baseline = pointnet_envelope.tcn_baseline
            elif "tcn_prediction" in latest_json:
                latest = RadarTcnLatestPayload.model_validate(
                    inference_json
                ).tcn_prediction
            else:
                latest = RadarLatestPayload.model_validate(inference_json)
            if latest_json.get("descent_prediction"):
                try:
                    descent = RadarDescentPredictionPayload.model_validate(
                        latest_json["descent_prediction"]
                    )
                except ValidationError:
                    descent = None
            if latest_json.get("fall_risk_assessment"):
                try:
                    risk_assessment = RadarFallRiskAssessmentPayload.model_validate(
                        latest_json["fall_risk_assessment"]
                    )
                except ValidationError:
                    risk_assessment = None
            if health.source_mode != latest.source_mode:
                self._mark_offline("radar health/latest source_mode mismatch")
                return None
            if health.model_mode != latest.model_mode:
                self._mark_offline("radar health/latest model_mode mismatch")
                return None

            self._online = True
            self._last_error = None
            self._latest_payload = latest
            self._latest_tcn_baseline = tcn_baseline
            self._latest_descent = descent
            self._latest_risk_assessment = risk_assessment
            self._latest_alignment_evidence = alignment_evidence
            if (
                self._last_timestamp is not None
                and latest.timestamp <= self._last_timestamp
            ):
                return None
            self._last_timestamp = latest.timestamp
            if isinstance(
                latest,
                (
                    RadarTcnPredictionPayload,
                    RadarPointNetPredictionPayload,
                    RadarCalibratedTcnPredictionPayload,
                ),
            ):
                evidence_source = (
                    tcn_baseline
                    if isinstance(latest, RadarPointNetPredictionPayload)
                    and tcn_baseline is not None
                    else latest
                )
                evidence = RadarEvidencePayload(
                    radar_score=(
                        evidence_source.pre_fall_score
                        if evidence_source.score_valid
                        else None
                    ),
                    risk_state=(
                        "UNKNOWN"
                        if not evidence_source.score_valid
                        else latest.gate_state
                        if isinstance(latest, RadarCalibratedTcnPredictionPayload)
                        else evidence_source.risk_state
                    ),
                    timestamp=evidence_source.timestamp,
                    room=evidence_source.room,
                    device_id=evidence_source.device_id,
                    quality=evidence_source.data_quality,
                    model_version=evidence_source.model_version,
                )
                data = {
                    "room": latest.room,
                    "source_mode": latest.source_mode,
                    "model_mode": latest.model_mode,
                    "disclaimer": latest.disclaimer,
                    "event_triggered": (
                        latest.event_triggered
                        if hasattr(latest, "event_triggered")
                        else False
                    ),
                    "radar_evidence": evidence.model_dump(mode="json"),
                }
                if isinstance(latest, RadarPointNetPredictionPayload):
                    data["pointnet_prediction"] = latest.model_dump(mode="json")
                    data["tcn_baseline"] = (
                        tcn_baseline.model_dump(mode="json")
                        if tcn_baseline is not None
                        else None
                    )
                elif isinstance(latest, RadarCalibratedTcnPredictionPayload):
                    data["calibrated_tcn_prediction"] = latest.model_dump(mode="json")
                    data["tcn_baseline"] = (
                        tcn_baseline.model_dump(mode="json")
                        if tcn_baseline is not None
                        else None
                    )
                else:
                    data["tcn_prediction"] = latest.model_dump(mode="json")
                debug: dict[str, object] = {
                    "affects_risk_state": False,
                    "affects_alerts": False,
                }
                if descent is not None:
                    debug["descent_prediction"] = descent.model_dump(mode="json")
                if risk_assessment is not None:
                    debug["fall_risk_assessment"] = risk_assessment.model_dump(mode="json")
                if descent is not None or risk_assessment is not None:
                    data["radar_debug"] = debug
            else:
                data = {
                    "room": latest.room,
                    "source_mode": latest.source_mode,
                    "human_state": latest.human_state,
                    "risk_score": latest.risk_score,
                    "model_mode": latest.model_mode,
                    "disclaimer": latest.disclaimer,
                    "event_triggered": latest.event_triggered,
                    "research": (
                        latest.research.model_dump(mode="json")
                        if latest.research is not None
                        else None
                    ),
                }
            return UnifiedDataPacket(
                packet_id=f"radar-{uuid4().hex}",
                session_id=self.session_id,
                source_id=self.source_id,
                device_id=latest.device_id,
                modality="RADAR",
                timestamp=latest.timestamp,
                data=data,
            )
        except (
            httpx.HTTPError,
            ValidationError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            self._mark_offline(f"{type(exc).__name__}: {exc}")
            return None

    def stop(self) -> None:
        super().stop()
        self._last_timestamp = None
        self._latest_tcn_baseline = None
        self._mark_offline(None)

    def close(self) -> None:
        self.stop()
        if self._owns_client:
            self._client.close()

    def _mark_offline(self, error: str | None) -> None:
        self._online = False
        self._latest_payload = None
        self._latest_health = None
        self._latest_descent = None
        self._latest_risk_assessment = None
        self._latest_alignment_evidence = []
        self._last_error = error

    def _create_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.service_url,
            timeout=self._timeout_seconds,
        )
