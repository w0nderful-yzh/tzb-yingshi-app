import unittest
from datetime import datetime, timezone

import httpx

from app.modules.fall.multimodal_engine.algorithm_runtime.adapters.radar_risk import RadarRiskAdapter
from app.modules.fall.multimodal_engine.algorithm_runtime.contracts import AdapterContext
from app.modules.fall.multimodal_engine.data_sources.adapters.radar_service import RadarServiceDataSourceAdapter
from app.modules.fall.multimodal_engine.data_sources.contracts import UnifiedDataPacket
from app.modules.fall.multimodal_engine.services.camera_radar_alignment import RadarTrackEvidenceBuffer
from app.modules.fall.multimodal_engine.services.radar_integration import RadarIntegrationService


TEST_DISCLAIMER = "基础风险模型输出，不作为已验证的跌倒预测结论"


class RadarServiceDataSourceAdapterTest(unittest.TestCase):
    def test_owned_http_client_is_recreated_after_lifespan_restart(self) -> None:
        adapter = RadarServiceDataSourceAdapter("http://127.0.0.1:9")
        adapter.start("first-lifespan")
        first_client = adapter._client
        adapter.close()
        self.assertTrue(first_client.is_closed)

        adapter.start("second-lifespan")
        self.assertIsNot(adapter._client, first_client)
        self.assertFalse(adapter._client.is_closed)
        adapter.close()

    def test_valid_radar_service_result_becomes_unified_packet(self) -> None:
        client = self._client()
        adapter = RadarServiceDataSourceAdapter(
            "http://radar.test",
            client=client,
        )
        adapter.start("radar-session")
        packet = adapter.read()

        self.assertTrue(adapter.online)
        self.assertIsInstance(packet, UnifiedDataPacket)
        self.assertEqual(packet.modality, "RADAR")
        self.assertEqual(packet.device_id, "iwr6843isk-01")
        self.assertEqual(packet.data["room"], "living_room")
        self.assertEqual(packet.data["source_mode"], "REPLAY")
        self.assertEqual(packet.data["model_mode"], "TEST_CHECKPOINT")
        self.assertEqual(packet.data["disclaimer"], TEST_DISCLAIMER)
        self.assertEqual(packet.data["research"]["prediction_state"], "WATCH")
        self.assertTrue(packet.data["research"]["alert_suppressed"])
        adapter.close()
        client.close()

    def test_tcn_shadow_result_is_forwarded_without_legacy_rule_score(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json=self._tcn_health())
            if request.url.path == "/api/radar/latest":
                return httpx.Response(200, json=self._tcn_latest())
            return httpx.Response(404)

        client = httpx.Client(
            base_url="http://radar.test",
            transport=httpx.MockTransport(handler),
        )
        source = RadarServiceDataSourceAdapter(
            "http://radar.test",
            client=client,
        )
        source.start("radar-background")
        packet = source.read()

        self.assertTrue(source.online)
        self.assertIsNotNone(packet)
        self.assertEqual(packet.device_id, "iwr6843_revd_bathroom")
        self.assertEqual(packet.data["room"], "bathroom")
        self.assertNotIn("risk_score", packet.data)
        self.assertNotIn("research", packet.data)
        self.assertAlmostEqual(
            packet.data["radar_evidence"]["radar_score"], 0.287
        )
        self.assertEqual(
            packet.data["tcn_prediction"]["risk_state"],
            "IMMINENT",
        )
        self.assertTrue(packet.data["tcn_prediction"]["shadow_only"])

        buffer = RadarTrackEvidenceBuffer(
            clock=lambda: datetime.fromisoformat(
                "2026-08-09T15:27:17.500+08:00"
            )
        )
        service = RadarIntegrationService(source, radar_track_buffer=buffer)
        service.process_once()
        status = service.get_status()
        self.assertTrue(status.online)
        self.assertEqual(status.model_mode, "RESEARCH_WEAK_SUPERVISION")
        self.assertIsNone(status.human_state)
        self.assertIsNone(status.risk_score)
        self.assertEqual(status.tcn_prediction.risk_state, "IMMINENT")
        self.assertAlmostEqual(status.radar_evidence.radar_score, 0.287)
        self.assertEqual(status.radar_evidence.quality, "GOOD")
        self.assertEqual(status.sensor_metrics.frame_rate_hz, 10.0)
        self.assertEqual(status.sensor_metrics.point_count, 17)
        self.assertEqual(len(status.alignment_evidence), 1)
        self.assertEqual(status.alignment_evidence[0].track_id, 7)
        self.assertEqual(status.alignment_evidence[0].frame_number, 314)
        self.assertEqual(status.alignment_evidence[0].radar_config_name, "ISK_6m_67ms_ab.cfg")
        self.assertEqual(
            buffer.frame_count(room="bathroom", device_id="iwr6843_revd_bathroom"),
            1,
        )
        self.assertIsNone(service._prepare_event_packet(packet))

        service._set_offline("unit-test disconnect")
        self.assertEqual(
            buffer.frame_count(room="bathroom", device_id="iwr6843_revd_bathroom"),
            0,
        )

        source.close()
        client.close()

    def test_calibrated_tcn_gate_is_final_evidence_and_other_branches_are_debug(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json=self._tcn_health())
            if request.url.path == "/api/radar/latest":
                return httpx.Response(200, json=self._calibrated_tcn_latest())
            return httpx.Response(404)

        client = httpx.Client(
            base_url="http://radar.test",
            transport=httpx.MockTransport(handler),
        )
        source = RadarServiceDataSourceAdapter("http://radar.test", client=client)
        source.start("radar-background")
        packet = source.read()

        self.assertIsNotNone(packet)
        self.assertNotIn("risk_score", packet.data)
        self.assertNotIn("descent_prediction", packet.data)
        self.assertNotIn("fall_risk_assessment", packet.data)
        self.assertEqual(packet.data["radar_evidence"]["risk_state"], "IMMINENT")
        self.assertAlmostEqual(packet.data["radar_evidence"]["radar_score"], 0.52)
        self.assertFalse(packet.data["radar_debug"]["affects_risk_state"])
        self.assertFalse(packet.data["radar_debug"]["affects_alerts"])

        service = RadarIntegrationService(source)
        service.process_once()
        status = service.get_status()
        self.assertIsNone(status.risk_score)
        self.assertEqual(status.radar_evidence.risk_state, "IMMINENT")
        self.assertAlmostEqual(status.radar_evidence.radar_score, 0.52)
        self.assertFalse(status.radar_debug.affects_risk_state)
        self.assertFalse(status.radar_debug.affects_alerts)
        self.assertAlmostEqual(
            status.radar_debug.fall_risk_assessment.risk_score, 0.88
        )
        self.assertIsNone(service._prepare_event_packet(packet))

        source.close()
        client.close()

    def test_pointnet_branch_remains_debug_while_tcn_is_final_evidence(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                health = self._tcn_health()
                health["feature_version"] = "radar_point_sequence_v2"
                return httpx.Response(200, json=health)
            if request.url.path == "/api/radar/latest":
                tcn = self._tcn_latest()["tcn_prediction"]
                pointnet = {
                    **tcn,
                    "schema_version": "radar_pointnet_live_v1",
                    "pre_fall_score": 0.412,
                    "risk_state": "WATCH",
                    "consecutive_high_windows": 1,
                    "event_triggered": False,
                    "event_id": None,
                    "observed_frame_count": 20,
                    "point_count": 17,
                    "snr_available_fraction": 1.0,
                    "model_version": "pointnet_gru_prefall_formal_v1",
                    "model_variant": "P2_DGUHA_JOINT",
                    "architecture": "pointnet_gru",
                    "feature_version": "radar_point_sequence_v2",
                    "threshold": 0.43,
                    "disclaimer": "PointNet-GRU雷达证据，不触发正式告警",
                }
                pointnet.pop("longest_unresolved_gap_seconds")
                return httpx.Response(200, json={
                    "pointnet_prediction": pointnet,
                    "tcn_baseline": tcn,
                })
            return httpx.Response(404)

        client = httpx.Client(
            base_url="http://radar.test", transport=httpx.MockTransport(handler)
        )
        source = RadarServiceDataSourceAdapter("http://radar.test", client=client)
        source.start("radar-background")
        packet = source.read()

        self.assertIsNotNone(packet)
        self.assertNotIn("risk_score", packet.data)
        self.assertAlmostEqual(packet.data["radar_evidence"]["radar_score"], 0.287)
        self.assertEqual(packet.data["radar_evidence"]["risk_state"], "IMMINENT")
        self.assertEqual(
            packet.data["radar_evidence"]["model_version"],
            "radar_temporal_experiment_v3",
        )
        self.assertEqual(packet.data["pointnet_prediction"]["architecture"], "pointnet_gru")
        self.assertEqual(packet.data["tcn_baseline"]["architecture"], "causal_tcn")

        service = RadarIntegrationService(source)
        service.process_once()
        status = service.get_status()
        self.assertEqual(status.pointnet_prediction.risk_state, "WATCH")
        self.assertEqual(status.tcn_baseline.architecture, "causal_tcn")
        self.assertIsNone(status.tcn_prediction)
        self.assertIsNone(service._prepare_event_packet(packet))

        source.close()
        client.close()

    def test_unreachable_service_marks_offline_and_clears_old_payload(self) -> None:
        calls = {"health": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                calls["health"] += 1
                if calls["health"] > 1:
                    return httpx.Response(503)
                return httpx.Response(200, json=self._health())
            return httpx.Response(200, json=self._latest())

        client = httpx.Client(
            base_url="http://radar.test",
            transport=httpx.MockTransport(handler),
        )
        adapter = RadarServiceDataSourceAdapter(
            "http://radar.test",
            client=client,
        )
        adapter.start("radar-session")
        self.assertIsNotNone(adapter.read())
        self.assertIsNone(adapter.read())
        self.assertFalse(adapter.online)
        self.assertIsNone(adapter.latest_payload)
        adapter.close()
        client.close()

    def test_integration_service_exposes_latest_status_without_event_database(self) -> None:
        client = self._client(event_triggered=False)
        source = RadarServiceDataSourceAdapter(
            "http://radar.test",
            client=client,
        )
        source.start("radar-background")
        service = RadarIntegrationService(source)
        service.process_once()
        status = service.get_status()

        self.assertTrue(status.online)
        self.assertEqual(status.room, "living_room")
        self.assertEqual(status.source_mode, "REPLAY")
        self.assertEqual(status.model_mode, "TEST_CHECKPOINT")
        self.assertEqual(status.human_state, "FALL_RISK")
        self.assertAlmostEqual(status.risk_score, 0.68)
        self.assertEqual(status.research.prediction_state, "WATCH")
        self.assertEqual(status.research.prediction_horizon_seconds, (1.0, 2.0))
        source.close()
        client.close()

    def test_unified_risk_creates_one_packet_per_risk_episode(self) -> None:
        client = self._client(event_triggered=True)
        source = RadarServiceDataSourceAdapter(
            "http://radar.test",
            client=client,
        )
        source.start("radar-background")
        packet = source.read()
        self.assertIsNotNone(packet)
        service = RadarIntegrationService(
            source,
        )

        action_packet = service._prepare_event_packet(packet)
        self.assertIsNotNone(action_packet)
        self.assertEqual(action_packet.data["event_kind"], "UNIFIED_FALL_RISK")
        self.assertEqual(action_packet.data["trigger_reasons"], ["ACTION_RISK"])
        self.assertAlmostEqual(action_packet.data["risk_score"], 0.68)
        self.assertIsNone(service._prepare_event_packet(packet))

        low_packet = packet.model_copy(
            update={
                "data": {
                    **packet.data,
                    "event_triggered": False,
                    "research": {
                        **packet.data["research"],
                        "prediction_state": "NORMAL",
                        "pre_fall_score": 0.10,
                        "fall_risk_level": "LOW",
                        "fall_risk_score": 0.10,
                        "fall_risk_score_5s": 0.10,
                        "action_risk_event_triggered": False,
                    },
                }
            }
        )
        self.assertIsNone(service._prepare_event_packet(low_packet))
        self.assertIsNotNone(service._prepare_event_packet(packet))

        prediction_service = RadarIntegrationService(source)
        prediction_packet = packet.model_copy(
            update={
                "data": {
                    **packet.data,
                    "research": {
                        **packet.data["research"],
                        "prediction_state": "IMMINENT",
                        "pre_fall_score": 0.72,
                        "fall_risk_score": 0.10,
                        "fall_risk_score_5s": 0.10,
                        "fall_risk_level": "LOW",
                    },
                }
            }
        )
        prediction_event = prediction_service._prepare_event_packet(
            prediction_packet
        )
        self.assertIsNotNone(prediction_event)
        self.assertEqual(
            prediction_event.data["trigger_reasons"],
            ["PREFALL_PREDICTION"],
        )
        self.assertAlmostEqual(prediction_event.data["risk_score"], 0.72)

        source.close()
        client.close()

    def _client(self, *, event_triggered: bool = True) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json=self._health())
            if request.url.path == "/api/radar/latest":
                return httpx.Response(
                    200,
                    json=self._latest(event_triggered=event_triggered),
                )
            return httpx.Response(404)

        return httpx.Client(
            base_url="http://radar.test",
            transport=httpx.MockTransport(handler),
        )

    @staticmethod
    def _health() -> dict:
        return {
            "status": "ok",
            "radar_connected": True,
            "model_loaded": True,
            "source_mode": "REPLAY",
            "model_mode": "TEST_CHECKPOINT",
            "feature_version": "radar_features_v1",
        }

    @staticmethod
    def _latest(*, event_triggered: bool = True) -> dict:
        return {
            "room": "living_room",
            "device_id": "iwr6843isk-01",
            "timestamp": "2026-07-24T12:30:03.200+08:00",
            "source_mode": "REPLAY",
            "human_state": "FALL_RISK",
            "risk_score": 0.7310586,
            "model_mode": "TEST_CHECKPOINT",
            "disclaimer": TEST_DISCLAIMER,
            "event_triggered": event_triggered,
            "research": {
                "timestamp": "2026-07-24T12:30:03.200+08:00",
                "prediction_state": "WATCH",
                "pre_fall_score": 0.42,
                "fall_risk_score": 0.68,
                "fall_risk_score_5s": 0.74,
                "fall_risk_level": "HIGH",
                "action_risk_event_triggered": True,
                "data_quality": "GOOD",
                "threshold": 0.872,
                "prediction_horizon_seconds": [1.0, 2.0],
                "positive_anchor": "near_floor_level_reached",
                "model_mode": "RESEARCH_WEAK_SUPERVISION",
                "shadow_only": True,
                "alert_suppressed": True,
                "disclaimer": "雷达研究影子输出，不触发告警",
            },
        }

    @staticmethod
    def _tcn_health() -> dict:
        return {
            "status": "ok",
            "radar_connected": True,
            "model_loaded": True,
            "source_mode": "REAL",
            "model_mode": "RESEARCH_WEAK_SUPERVISION",
            "feature_version": "radar_features_v2",
            "frame_rate_hz": 10.0,
            "point_count": 17,
        }

    @staticmethod
    def _tcn_latest() -> dict:
        return {
            "alignment_evidence": [
                {
                    "frame_number": 314,
                    "source_timestamp": "2026-08-09T15:27:17.394+08:00",
                    "track_id": 7,
                    "x": 0.42,
                    "y": 2.35,
                    "z": 1.10,
                    "vx": 0.10,
                    "vy": -0.20,
                    "vz": -0.08,
                    "point_count": 17,
                    "radar_score": 0.287,
                    "radar_quality": 0.9,
                    "radar_state": "IMMINENT",
                    "radar_config_name": "ISK_6m_67ms_ab.cfg",
                    "target_confidence": 0.86,
                    "shadow_only": True,
                }
            ],
            "tcn_prediction": {
                "schema_version": "radar_tcn_live_v1",
                "timestamp": "2026-08-09T15:27:17.394+08:00",
                "emitted_at": "2026-08-09T15:27:17.410+08:00",
                "device_id": "iwr6843_revd_bathroom",
                "room": "bathroom",
                "source_mode": "REAL",
                "risk_state": "IMMINENT",
                "pre_fall_score": 0.287,
                "score_valid": True,
                "consecutive_high_windows": 3,
                "event_triggered": True,
                "event_id": "radar-prefall-test",
                "unknown_reason": None,
                "data_quality": "GOOD",
                "missing_frame_ratio": 0.0,
                "longest_unresolved_gap_seconds": 0.0,
                "model_version": "radar_temporal_experiment_v3",
                "model_mode": "RESEARCH_WEAK_SUPERVISION",
                "architecture": "causal_tcn",
                "checkpoint_sha256": "0" * 64,
                "feature_version": "radar_features_v2",
                "threshold": 0.35,
                "threshold_policy": "validation_specificity_priority",
                "prediction_horizon_seconds": [0.5, 1.0],
                "positive_anchor": "descent_onset",
                "shadow_only": True,
                "alert_suppressed": True,
                "disclaimer": "TCN影子验证，不触发正式告警",
            }
        }

    @classmethod
    def _calibrated_tcn_latest(cls) -> dict:
        baseline = cls._tcn_latest()["tcn_prediction"]
        calibrated = {
            **baseline,
            "schema_version": "radar_calibrated_tcn_live_v1",
            "pre_fall_score": 0.52,
            "tcn_risk_state": "WATCH",
            "gate_state": "IMMINENT",
            "formal_alert": False,
            "suppressed_reason": None,
            "recovery_window_active": False,
            "recovery_count": 0,
            "consecutive_high_windows": 3,
            "threshold_crossed_at": "2026-08-09T15:27:17.100+08:00",
            "confirmed_at": "2026-08-09T15:27:17.394+08:00",
            "confirmation_latency_seconds": 0.294,
            "centroid_z": 0.91,
            "vertical_velocity": -0.82,
            "height_delta_0_6s": -0.21,
            "feature_point_count": 42.0,
        }
        calibrated.pop("risk_state")
        calibrated.pop("event_triggered")
        calibrated.pop("event_id")
        calibrated.pop("missing_frame_ratio")
        calibrated.pop("longest_unresolved_gap_seconds")
        return {
            "calibrated_tcn_prediction": calibrated,
            "tcn_baseline": baseline,
            "descent_prediction": {
                "schema_version": "radar_descent_live_v1",
                "timestamp": baseline["timestamp"],
                "emitted_at": baseline["emitted_at"],
                "device_id": baseline["device_id"],
                "room": baseline["room"],
                "source_mode": baseline["source_mode"],
                "descent_score": 1.0,
                "score_valid": True,
                "risk_state": "WATCH",
                "consecutive_high_windows": 1,
                "event_triggered": False,
                "event_id": None,
                "unknown_reason": None,
                "data_quality": "DEGRADED",
                "model_version": "radar_descent_detection_v1",
                "model_mode": "RESEARCH_DESCENT_DETECTION_V1",
                "architecture": "causal_tcn",
                "checkpoint_sha256": "1" * 64,
                "feature_version": "radar_features_v2",
                "threshold": 0.05,
                "prediction_horizon_seconds": [0.0, 1.5],
                "positive_anchor": "descent_interval",
                "shadow_only": True,
                "alert_suppressed": True,
                "disclaimer": "debug only",
            },
            "fall_risk_assessment": {
                "schema_version": "radar_risk_assessment_live_v1",
                "timestamp": baseline["timestamp"],
                "device_id": baseline["device_id"],
                "room": baseline["room"],
                "risk_level": "HIGH",
                "risk_score": 0.88,
                "sway_risk": 1.0,
                "mobility_risk": 0.6,
                "descent_risk": 0.8,
                "assessment_window_seconds": 60.0,
                "valid_window_count": 18,
                "observed_duration_seconds": 4.0,
                "unknown_reason": None,
                "shadow_only": True,
                "alert_suppressed": True,
                "disclaimer": "debug only",
            },
        }


class RadarRiskAdapterTest(unittest.TestCase):
    def test_real_unified_risk_creates_one_fall_risk_record(self) -> None:
        packet = self._packet(
            source_mode="REAL",
            model_mode="TEST_CHECKPOINT",
            disclaimer=TEST_DISCLAIMER,
            event_kind="UNIFIED_FALL_RISK",
            trigger_reasons=["ACTION_RISK", "PREFALL_PREDICTION"],
        )
        finding = self._consume(RadarRiskAdapter(), packet)
        self.assertEqual(finding.event_type, "RADAR_FALL_RISK")
        self.assertEqual(finding.risk_level.value, "HIGH")
        self.assertIn("毫米波综合跌倒风险", finding.summary)
        self.assertIn("客厅触发时综合风险分数", finding.summary)
        self.assertIn("动作风险+跌倒预测", finding.summary)
        self.assertNotEqual(finding.event_type, "PRE_FALL_RISK")
        evidence = {item.code: item.value for item in finding.evidence}
        self.assertEqual(evidence["source_mode"], "REAL")
        self.assertEqual(evidence["model_mode"], "TEST_CHECKPOINT")
        self.assertAlmostEqual(evidence["prefall_prediction_score"], 0.42)
        self.assertAlmostEqual(evidence["action_risk_score"], 0.68)
        self.assertAlmostEqual(evidence["room_risk_score_5s"], 0.74)

    def test_test_checkpoint_is_not_routed_as_prediction(self) -> None:
        packet = self._packet(
            source_mode="REAL",
            model_mode="TEST_CHECKPOINT",
            disclaimer=TEST_DISCLAIMER,
        )
        self.assertIsNone(self._consume(RadarRiskAdapter(), packet))

    def test_trained_replay_is_not_formal_event(self) -> None:
        packet = self._packet(
            source_mode="REPLAY",
            model_mode="TRAINED_CHECKPOINT",
            disclaimer=None,
        )
        finding = self._consume(RadarRiskAdapter(), packet)
        self.assertEqual(finding.event_type, "RADAR_REPLAY_RISK")

    def test_trained_real_requires_explicit_p2_switch(self) -> None:
        packet = self._packet(
            source_mode="REAL",
            model_mode="TRAINED_CHECKPOINT",
            disclaimer=None,
        )
        self.assertIsNone(self._consume(RadarRiskAdapter(), packet))
        finding = self._consume(
            RadarRiskAdapter(allow_formal_predictions=True),
            packet,
        )
        self.assertEqual(finding.event_type, "PRE_FALL_RISK")

    @staticmethod
    def _consume(
        adapter: RadarRiskAdapter,
        packet: UnifiedDataPacket,
    ):
        adapter.load()
        adapter.start(
            AdapterContext(
                session_id=packet.session_id,
                device_id=packet.device_id,
            )
        )
        try:
            return adapter.consume(packet)
        finally:
            adapter.stop()

    @staticmethod
    def _packet(
        *,
        source_mode: str,
        model_mode: str,
        disclaimer: str | None,
        event_kind: str = "PREDICTION",
        trigger_reasons: list[str] | None = None,
    ) -> UnifiedDataPacket:
        return UnifiedDataPacket(
            packet_id="radar-packet-001",
            session_id="radar-session",
            source_id="radar-service",
            device_id="iwr6843isk-01",
            modality="RADAR",
            timestamp=datetime.now(timezone.utc),
            data={
                "room": "living_room",
                "source_mode": source_mode,
                "human_state": "FALL_RISK",
                "risk_score": 0.7310586,
                "model_mode": model_mode,
                "disclaimer": disclaimer,
                "event_triggered": True,
                "event_kind": event_kind,
                "trigger_reasons": trigger_reasons or [],
                "research": {
                    "prediction_state": "WATCH",
                    "pre_fall_score": 0.42,
                    "fall_risk_score": 0.68,
                    "fall_risk_score_5s": 0.74,
                    "fall_risk_level": "HIGH",
                    "data_quality": "GOOD",
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
