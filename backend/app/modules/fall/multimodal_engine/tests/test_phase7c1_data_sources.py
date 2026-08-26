import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from app.modules.fall.multimodal_engine.data_sources import DataSourceAdapter, UnifiedDataPacket
from app.modules.fall.multimodal_engine.data_sources.adapters import DummyRadarAdapter, MockCameraAdapter


class Phase7C1UnifiedDataPacketTest(unittest.TestCase):
    def test_valid_packet_preserves_minimum_contract(self) -> None:
        timestamp = datetime.now(timezone.utc)
        packet = UnifiedDataPacket(
            packet_id="packet-001",
            session_id="session-001",
            source_id="source-001",
            device_id="device-001",
            modality="VIDEO",
            timestamp=timestamp,
            data={"frame_ref": "mock://camera/frame-001"},
        )

        self.assertEqual(packet.packet_id, "packet-001")
        self.assertEqual(packet.timestamp, timestamp)
        self.assertEqual(packet.modality, "VIDEO")
        self.assertEqual(packet.data["frame_ref"], "mock://camera/frame-001")

    def test_packet_rejects_timestamp_without_timezone(self) -> None:
        with self.assertRaises(ValidationError):
            UnifiedDataPacket(
                packet_id="packet-001",
                session_id="session-001",
                source_id="source-001",
                device_id="device-001",
                modality="VIDEO",
                timestamp=datetime(2026, 7, 22, 12, 0, 0),
                data={},
            )

    def test_packet_rejects_blank_identifier_invalid_modality_and_extra_field(self) -> None:
        valid_payload = {
            "packet_id": "packet-001",
            "session_id": "session-001",
            "source_id": "source-001",
            "device_id": "device-001",
            "modality": "VIDEO",
            "timestamp": datetime.now(timezone.utc),
            "data": {},
        }

        for field in ("packet_id", "session_id", "source_id", "device_id"):
            with self.subTest(blank_field=field), self.assertRaises(ValidationError):
                UnifiedDataPacket.model_validate({**valid_payload, field: "   "})

        with self.assertRaises(ValidationError):
            UnifiedDataPacket.model_validate({**valid_payload, "modality": "video"})

        with self.assertRaises(ValidationError):
            UnifiedDataPacket.model_validate({**valid_payload, "unexpected": True})

    def test_packet_rejects_non_mapping_data(self) -> None:
        with self.assertRaises(ValidationError):
            UnifiedDataPacket.model_validate(
                {
                    "packet_id": "packet-001",
                    "session_id": "session-001",
                    "source_id": "source-001",
                    "device_id": "device-001",
                    "modality": "VIDEO",
                    "timestamp": datetime.now(timezone.utc),
                    "data": ["not", "a", "mapping"],
                }
            )


class Phase7C1DataSourceAdapterTest(unittest.TestCase):
    def test_mock_camera_uses_common_adapter_and_packet_contract(self) -> None:
        adapter = MockCameraAdapter()
        self.assertIsInstance(adapter, DataSourceAdapter)
        self.assertFalse(adapter.is_running)

        with self.assertRaises(RuntimeError):
            adapter.read()

        adapter.start("session-camera")
        packet = adapter.read()

        self.assertTrue(adapter.is_running)
        self.assertIsInstance(packet, UnifiedDataPacket)
        self.assertEqual(packet.session_id, "session-camera")
        self.assertEqual(packet.source_id, "mock-camera")
        self.assertEqual(packet.device_id, "camera-001")
        self.assertEqual(packet.modality, "VIDEO")
        self.assertEqual(packet.data["frame_ref"], "mock://camera/frame-001")

        adapter.stop()
        self.assertFalse(adapter.is_running)
        with self.assertRaises(RuntimeError):
            adapter.read()

    def test_dummy_radar_uses_same_adapter_and_packet_contract(self) -> None:
        adapter = DummyRadarAdapter(
            source_id="second-platform",
            device_id="radar-test-01",
            distance_m=2.1,
            vertical_velocity_mps=-0.5,
            height_m=0.75,
        )
        self.assertIsInstance(adapter, DataSourceAdapter)

        adapter.start("session-radar")
        packet = adapter.read()

        self.assertIsInstance(packet, UnifiedDataPacket)
        self.assertEqual(packet.session_id, "session-radar")
        self.assertEqual(packet.source_id, "second-platform")
        self.assertEqual(packet.device_id, "radar-test-01")
        self.assertEqual(packet.modality, "RADAR")
        self.assertEqual(
            packet.data,
            {
                "distance_m": 2.1,
                "vertical_velocity_mps": -0.5,
                "height_m": 0.75,
            },
        )

        adapter.stop()

    def test_start_rejects_blank_session_id(self) -> None:
        adapter = MockCameraAdapter()
        with self.assertRaises(ValueError):
            adapter.start("   ")


if __name__ == "__main__":
    unittest.main()
