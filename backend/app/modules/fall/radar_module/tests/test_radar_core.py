from __future__ import annotations

import ast
import inspect
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from radar_module.acquisition.ti_reader import (
    JsonlReplayAdapter,
    TiOfficialOutputAdapter,
    TiRadarReader,
)
from radar_module.acquisition.ti_official_bridge import _decoded_mapping, run_bridge
from radar_module.contracts import (
    DEMO_DISCLAIMER,
    FeatureVector,
    HumanState,
    RadarFrame,
    RadarPoint,
    Room,
    SourceMode,
)
from radar_module.inference.risk_prediction import RadarRiskPredictor
from radar_module.model import radar_lstm
from radar_module.model.radar_lstm import RadarLSTM
from radar_module.preprocess.feature_extraction import RadarFeatureExtractor
from radar_module.preprocess.pointcloud_processing import map_official_points
from radar_module.preprocess.window_generation import FeatureWindowGenerator


class RadarCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.timestamp = datetime(2026, 7, 24, 12, 30, tzinfo=timezone.utc)
        self.extractor = RadarFeatureExtractor()

    def test_official_callback_and_replay_use_same_radar_frame_contract(self) -> None:
        payload = {
            "timestamp": self.timestamp.isoformat(),
            "source_monotonic_ns": 123_000_000,
            "ti_frame_number": 42,
            "radar_config_name": "ISK_6m_55ms_ab.cfg",
            "points": [
                {
                    "posX": 1,
                    "posY": 2,
                    "posZ": 1.5,
                    "doppler": -0.2,
                    "track_id": 7,
                }
            ],
            "targets": [
                {
                    "track_id": 7,
                    "x": 1.0,
                    "y": 2.0,
                    "z": 1.5,
                    "velocity_x": 0.1,
                    "velocity_y": 0.2,
                    "velocity_z": -0.3,
                    "confidence": 0.9,
                }
            ],
        }
        official = TiOfficialOutputAdapter(decoded_callback=lambda: payload)
        reader = TiRadarReader(official, "radar-01", Room.LIVING_ROOM)
        reader.start()
        real_frame = reader.read()
        reader.stop()

        self.assertIsInstance(real_frame, RadarFrame)
        self.assertEqual(real_frame.source_mode, SourceMode.REAL)
        self.assertEqual(real_frame.points[0].velocity, -0.2)
        self.assertEqual(real_frame.frame_number, 42)
        self.assertEqual(real_frame.radar_config_name, "ISK_6m_55ms_ab.cfg")
        self.assertEqual(real_frame.targets[0].track_id, 7)
        self.assertAlmostEqual(real_frame.targets[0].velocity_z, -0.3)

        demo_file = Path(__file__).resolve().parents[1] / "data/replay/demo_session.jsonl"
        replay = JsonlReplayAdapter(demo_file, speed=100)
        replay_reader = TiRadarReader(replay, "radar-01", Room.LIVING_ROOM)
        replay_reader.start()
        replay_frame = replay_reader.read()
        replay_reader.stop()

        self.assertIsInstance(replay_frame, RadarFrame)
        self.assertEqual(replay_frame.source_mode, SourceMode.REPLAY)
        self.assertEqual(type(real_frame), type(replay_frame))

    def test_official_process_exit_preserves_stderr_detail(self) -> None:
        official = TiOfficialOutputAdapter(
            command=(
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('bridge detail\\n'); sys.exit(7)",
            )
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "bridge detail"):
                official.start()
        finally:
            official.stop()

    def test_official_process_receives_graceful_stop_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            marker = Path(temporary_directory) / "stop_request.txt"
            script = (
                "import json, pathlib, sys; "
                "print(json.dumps({'points': []}), flush=True); "
                "line = sys.stdin.readline().strip(); "
                "pathlib.Path(sys.argv[1]).write_text(line, encoding='utf-8')"
            )
            official = TiOfficialOutputAdapter(
                command=(sys.executable, "-c", script, str(marker))
            )
            official.start()
            try:
                for _ in range(20):
                    if official.read_decoded() is not None:
                        break
                    time.sleep(0.01)
            finally:
                official.stop()

            self.assertEqual(marker.read_text(encoding="utf-8"), "STOP")

    def test_official_process_start_waits_for_first_decoded_frame(self) -> None:
        script = (
            "import json, sys, time; "
            "time.sleep(0.15); "
            "print(json.dumps({'points': []}), flush=True); "
            "sys.stdin.readline()"
        )
        official = TiOfficialOutputAdapter(
            command=(sys.executable, "-c", script),
            startup_timeout_seconds=2.0,
        )
        started_at = time.monotonic()
        official.start()
        elapsed = time.monotonic() - started_at
        try:
            payload = official.read_decoded()
        finally:
            official.stop()

        self.assertGreaterEqual(elapsed, 0.1)
        self.assertEqual(payload, {"points": []})

    def test_point_mapping_filters_invalid_incomplete_and_out_of_range_points(self) -> None:
        points = map_official_points(
            [
                {
                    "x": 1,
                    "y": 2,
                    "z": 1,
                    "velocity": 0.5,
                    "snr": 12.0,
                    "target_id": 3,
                },
                {"x": float("nan"), "y": 2, "z": 1, "velocity": 0.5},
                {"x": 1, "y": 2, "z": 1},
                [9, 0, 0, 0],
                [0.1, 0.2, 0.3, -0.1],
                np.asarray([0.2, 0.3, 0.4, -0.2], dtype=np.float32),
            ],
            max_distance_m=8,
        )
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0].snr, 12.0)
        self.assertEqual(points[0].track_id, 3)
        self.assertAlmostEqual(points[1].velocity, -0.1)
        self.assertAlmostEqual(points[2].velocity, -0.2)

    def test_ti_official_bridge_maps_decoded_point_cloud(self) -> None:
        decoded = _decoded_mapping(
            {
                "frameNum": 42,
                "error": 0,
                "pointCloud": np.asarray(
                    [
                        [1.0, 2.0, 1.5, -0.25, 8.0, 1.0, 4.0],
                        [float("nan"), 2.0, 1.0, 0.1, 8.0, 1.0, 255.0],
                    ],
                    dtype=np.float64,
                ),
            },
            radar_config_name="ISK_6m_55ms_ab.cfg",
        )
        self.assertEqual(decoded["ti_frame_number"], 42)
        self.assertEqual(decoded["ti_parser_error"], 0)
        self.assertEqual(len(decoded["points"]), 1)
        self.assertEqual(decoded["points"][0]["velocity"], -0.25)
        self.assertEqual(decoded["points"][0]["snr"], 8.0)
        self.assertEqual(decoded["points"][0]["track_id"], 4)
        self.assertEqual(decoded["radar_config_name"], "ISK_6m_55ms_ab.cfg")

    def test_ti_official_bridge_uses_target_index_tlv_and_rejects_special_ids(self) -> None:
        decoded = _decoded_mapping(
            {
                "frameNum": 43,
                "pointCloud": np.asarray(
                    [
                        [1.0, 2.0, 1.5, -0.25, 8.0, 1.0, 4.0],
                        [1.1, 2.1, 1.4, -0.20, 7.0, 1.0, 5.0],
                    ],
                    dtype=np.float64,
                ),
                "trackIndexes": np.asarray([7, 253], dtype=np.uint8),
                "trackData": np.asarray(
                    [[7, 1.0, 2.0, 1.5, 0.1, 0.2, -0.3, 0, 0, 0, 0, 0.9]],
                    dtype=np.float64,
                ),
            }
        )

        self.assertEqual(decoded["points"][0]["track_id"], 7)
        self.assertNotIn("track_id", decoded["points"][1])
        self.assertEqual(decoded["track_indexes"], [7, 253])
        self.assertEqual(decoded["targets"][0]["track_id"], 7)
        self.assertAlmostEqual(decoded["targets"][0]["velocity_z"], -0.3)
        self.assertAlmostEqual(decoded["targets"][0]["confidence"], 0.9)

    def test_ti_bridge_reuses_existing_config_with_sensor_start_zero(self) -> None:
        fake_parser = _FakeOfficialParser()

        class FakeParserType:
            def __new__(cls, *_args: object, **_kwargs: object) -> object:
                return fake_parser

        with tempfile.TemporaryDirectory() as temporary_directory:
            config = Path(temporary_directory) / "unused.cfg"
            output = Path(temporary_directory) / "radar_frames.jsonl"
            config.write_text("sensorStop\nsensorStart\n", encoding="utf-8")
            with patch(
                "radar_module.acquisition.ti_official_bridge._load_uart_parser",
                return_value=FakeParserType,
            ):
                run_bridge(
                    cli_port="COM5",
                    data_port="COM6",
                    config_path=config,
                    common_dir=Path(temporary_directory),
                    max_frames=1,
                    reuse_existing_config=True,
                    output_jsonl=output,
                )
            self.assertEqual(
                fake_parser.cliCom.commands,
                ["sensorStop", "sensorStart 0", "sensorStop"],
            )
            stored = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(stored["radar_config_name"], "unused.cfg")

    def test_v1_feature_order_and_values_are_fixed(self) -> None:
        feature = self.extractor.extract(self._frame())
        self.assertEqual(feature.version, "radar_features_v1")
        self.assertEqual(feature.names, RadarFeatureExtractor.feature_names)
        self.assertEqual(feature.values.shape, (8,))
        self.assertAlmostEqual(float(feature.values[0]), 0.5)
        self.assertAlmostEqual(float(feature.values[3]), 1.0)
        self.assertAlmostEqual(float(feature.values[7]), 2.0)

    def test_window_accepts_only_feature_vectors_and_resets_on_stream_change(self) -> None:
        generator = FeatureWindowGenerator(
            window_size=3,
            feature_version=self.extractor.feature_version,
            feature_names=self.extractor.feature_names,
        )
        feature = self.extractor.extract(self._frame())
        self.assertIsNone(generator.consume(feature))
        self.assertIsNone(generator.consume(feature))
        window = generator.consume(feature)
        self.assertEqual(window.shape, (3, 8))

        changed_room = FeatureVector(
            timestamp=feature.timestamp,
            device_id=feature.device_id,
            room=Room.BEDROOM,
            source_mode=feature.source_mode,
            human_present=True,
            version=feature.version,
            names=feature.names,
            values=feature.values,
        )
        self.assertIsNone(generator.consume(changed_room))
        self.assertEqual(generator.current_size, 1)
        with self.assertRaises(TypeError):
            generator.consume(self._frame())  # type: ignore[arg-type]

    def test_lstm_returns_logits_and_never_references_radar_frame(self) -> None:
        model = RadarLSTM(input_size=8, hidden_size=64)
        logits = model(torch.zeros((2, 30, 8), dtype=torch.float32))
        self.assertEqual(tuple(logits.shape), (2,))
        syntax_tree = ast.parse(inspect.getsource(radar_lstm))
        imported_names = {
            alias.name
            for node in ast.walk(syntax_tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("RadarFrame", imported_names)

    def test_checkpoint_contract_rejects_feature_incompatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test.pt"
            RadarLSTM.create_test_checkpoint(
                checkpoint_path,
                feature_names=self.extractor.feature_names,
            )
            with self.assertRaisesRegex(ValueError, "feature_version"):
                RadarLSTM.load_checkpoint(
                    checkpoint_path,
                    expected_feature_version="radar_features_v2",
                    expected_feature_names=self.extractor.feature_names,
                    expected_window_size=30,
                    expected_input_size=8,
                )

    def test_test_checkpoint_is_demo_and_triggers_only_after_three_high_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test.pt"
            RadarLSTM.create_test_checkpoint(
                checkpoint_path,
                feature_names=self.extractor.feature_names,
            )
            loaded = RadarLSTM.load_checkpoint(
                checkpoint_path,
                expected_feature_version=self.extractor.feature_version,
                expected_feature_names=self.extractor.feature_names,
                expected_window_size=30,
                expected_input_size=8,
            )
            predictor = RadarRiskPredictor(loaded)

            results = []
            for index in range(32):
                frame = self._frame(
                    timestamp=self.timestamp + timedelta(milliseconds=100 * index)
                )
                results.append(predictor.consume(self.extractor.extract(frame)))

            self.assertEqual(results[28].risk_score, 0)
            self.assertEqual(results[29].human_state, HumanState.FALL_RISK)
            self.assertFalse(results[29].event_triggered)
            self.assertFalse(results[30].event_triggered)
            self.assertTrue(results[31].event_triggered)
            self.assertAlmostEqual(results[31].risk_score, 0.7310586, places=5)
            self.assertEqual(results[31].disclaimer, DEMO_DISCLAIMER)

            no_person = RadarFrame(
                timestamp=self.timestamp + timedelta(seconds=4),
                device_id="radar-01",
                room=Room.LIVING_ROOM,
                source_mode=SourceMode.REPLAY,
                points=(),
            )
            result = predictor.consume(self.extractor.extract(no_person))
            self.assertEqual(result.human_state, HumanState.NO_PERSON)
            self.assertEqual(result.risk_score, 0)

    def _frame(self, *, timestamp: datetime | None = None) -> RadarFrame:
        return RadarFrame(
            timestamp=timestamp or self.timestamp,
            device_id="radar-01",
            room=Room.LIVING_ROOM,
            source_mode=SourceMode.REPLAY,
            points=(
                RadarPoint(x=0.0, y=2.0, z=0.5, velocity=-0.2),
                RadarPoint(x=1.0, y=2.0, z=1.5, velocity=0.4),
            ),
        )


class _FakeSerialPort:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self._response = bytearray()
        self.is_open = True

    @property
    def in_waiting(self) -> int:
        return len(self._response)

    def reset_input_buffer(self) -> None:
        self._response.clear()

    def write(self, payload: bytes) -> int:
        command = payload.decode("ascii").strip()
        self.commands.append(command)
        self._response.extend(
            f"{command}\r\nDone\r\nmmwDemo:/>".encode("ascii")
        )
        return len(payload)

    def read(self, size: int) -> bytes:
        chunk = bytes(self._response[:size])
        del self._response[:size]
        return chunk

    def close(self) -> None:
        self.is_open = False


class _FakeOfficialParser:
    def __init__(self) -> None:
        self.cliCom = _FakeSerialPort()
        self.dataCom = _FakeSerialPort()

    def connectComPorts(self, _cli_port: str, _data_port: str) -> None:
        return None

    def readAndParseUartDoubleCOMPort(self) -> dict[str, object]:
        return {
            "frameNum": 1,
            "error": 0,
            "pointCloud": np.empty((0, 7), dtype=np.float64),
        }


if __name__ == "__main__":
    unittest.main()
