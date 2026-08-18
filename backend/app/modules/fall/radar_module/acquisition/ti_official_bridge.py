from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


LOGGER = logging.getLogger("ti-official-bridge")


def _load_uart_parser(common_dir: Path) -> type[Any]:
    """Load TI's parser from the installed Radar Toolbox at runtime.

    The application does not copy or reimplement TI UART/TLV parsing.  The
    Toolbox directory is added to ``sys.path`` only inside this bridge process
    because the official visualizer uses local absolute imports.
    """

    resolved = common_dir.resolve()
    parser_file = resolved / "gui_parser.py"
    if not parser_file.is_file():
        raise FileNotFoundError(f"TI official parser not found: {parser_file}")
    sys.path.insert(0, str(resolved))
    module = importlib.import_module("gui_parser")
    return module.UARTParser


def _decoded_mapping(
    output: dict[str, Any],
    *,
    radar_config_name: str | None = None,
) -> dict[str, Any]:
    """Convert an already decoded TI frame to the radar-module JSON contract."""

    raw_cloud = np.asarray(output.get("pointCloud", ()), dtype=np.float64)
    raw_track_indexes = np.asarray(output.get("trackIndexes", ())).reshape(-1)
    points: list[dict[str, float | int]] = []
    if raw_cloud.ndim == 2 and raw_cloud.shape[1] >= 4:
        for point_index, row in enumerate(raw_cloud):
            x, y, z, velocity = (float(value) for value in row[:4])
            if np.isfinite((x, y, z, velocity)).all():
                point: dict[str, float | int] = {
                    "x": x,
                    "y": y,
                    "z": z,
                    "velocity": velocity,
                }
                if raw_cloud.shape[1] >= 5 and np.isfinite(row[4]):
                    point["snr"] = float(row[4])
                track_id: int | None = None
                if (
                    point_index < raw_track_indexes.size
                    and np.isfinite(raw_track_indexes[point_index])
                ):
                    track_id = int(raw_track_indexes[point_index])
                elif raw_cloud.shape[1] >= 7 and np.isfinite(row[6]):
                    # Compatibility with adapters that already merged the
                    # official target-index TLV into pointCloud column 6.
                    track_id = int(row[6])
                if track_id is not None:
                    # TI reserves 253/254/255 for weak/unassociated/noise
                    # points.  Only 0..252 are real target identifiers.
                    if 0 <= track_id < 253:
                        point["track_id"] = track_id
                points.append(point)
    raw_targets = np.asarray(output.get("trackData", ()), dtype=np.float64)
    targets: list[dict[str, float | int]] = []
    if raw_targets.ndim == 2 and raw_targets.shape[1] >= 4:
        for row in raw_targets:
            if not np.isfinite(row[:4]).all():
                continue
            target: dict[str, float | int] = {
                "track_id": int(row[0]),
                "x": float(row[1]),
                "y": float(row[2]),
                "z": float(row[3]),
            }
            # Target List TLV 状态向量布局（GTRACK_targetDesc.S）：
            # [x,y,z,vx,vy,vz,(ax,ay,az)]，3DA 模式含 acc
            if raw_targets.shape[1] >= 7 and np.isfinite(row[4:7]).all():
                target.update(
                    velocity_x=float(row[4]),
                    velocity_y=float(row[5]),
                    velocity_z=float(row[6]),
                )
            if raw_targets.shape[1] >= 10 and np.isfinite(row[7:10]).all():
                target.update(
                    accel_x=float(row[7]),
                    accel_y=float(row[8]),
                    accel_z=float(row[9]),
                )
            # row[10] = G（gain），row[11] = confidenceLevel
            if raw_targets.shape[1] >= 12 and np.isfinite(row[11]):
                target["confidence"] = float(row[11])
            targets.append(target)
    received_at = datetime.now(timezone.utc).isoformat()
    captured_monotonic = time.monotonic()
    return {
        "timestamp": received_at,
        "source_timestamp": received_at,
        "received_at": received_at,
        "captured_monotonic": captured_monotonic,
        "source_monotonic_ns": int(round(captured_monotonic * 1_000_000_000)),
        "clock_domain": "SYSTEM_MONOTONIC_SAME_HOST",
        "points": points,
        "track_indexes": [
            int(value) for value in raw_track_indexes if np.isfinite(value)
        ],
        "targets": targets,
        "ti_frame_number": int(output.get("frameNum", 0)),
        "ti_parser_error": int(output.get("error", 0)),
        "radar_config_name": radar_config_name,
    }


def run_bridge(
    *,
    cli_port: str,
    data_port: str,
    config_path: Path,
    common_dir: Path,
    max_frames: int | None = None,
    stop_event: threading.Event | None = None,
    reuse_existing_config: bool = False,
    output_jsonl: Path | None = None,
) -> None:
    uart_parser_type = _load_uart_parser(common_dir)
    parser = uart_parser_type("DoubleCOMPort")
    parser.connectComPorts(cli_port, data_port)
    sensor_started = False
    output_handle = None
    try:
        if output_jsonl is not None:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            # Refuse accidental overwrite/mixing of A/B sessions.
            output_handle = output_jsonl.open("x", encoding="utf-8")
        if reuse_existing_config:
            startup_commands: Sequence[str] = ("sensorStop", "sensorStart 0")
        else:
            startup_commands = config_path.read_text(encoding="utf-8").splitlines(True)
        _send_official_config(parser.cliCom, startup_commands)
        sensor_started = True

        emitted_frames = 0
        while stop_event is None or not stop_event.is_set():
            output = parser.readAndParseUartDoubleCOMPort()
            if not output:
                continue
            line = json.dumps(
                _decoded_mapping(
                    output,
                    radar_config_name=config_path.name,
                ),
                ensure_ascii=False,
            )
            print(line, flush=True)
            if output_handle is not None:
                output_handle.write(line + "\n")
                output_handle.flush()
            emitted_frames += 1
            if max_frames is not None and emitted_frames >= max_frames:
                return
    finally:
        if output_handle is not None:
            output_handle.close()
        if sensor_started:
            _send_official_config(parser.cliCom, ("sensorStop",))
        for port_name in ("cliCom", "dataCom"):
            port = getattr(parser, port_name, None)
            if port is not None and getattr(port, "is_open", False):
                port.close()


def _send_official_config(
    cli_port: Any,
    config_lines: Sequence[str],
    *,
    command_timeout_seconds: float = 3.0,
    sensor_start_timeout_seconds: float = 10.0,
) -> None:
    """Send TI's official cfg while validating each CLI acknowledgement.

    Radar Toolbox ``UARTParser.sendCfg`` reads exactly two newline records per
    command.  The IWR6843ISK firmware also emits a prompt, so records can become
    offset when the process attaches to an already running sensor.  This loader
    sends the same official text commands but waits for the complete prompt and
    verifies ``Done``.  It does not decode UART data or implement TLV parsing.
    """

    cli_port.reset_input_buffer()
    commands = [
        line.strip()
        for line in config_lines
        if line.strip() and not line.lstrip().startswith("%")
    ]
    for command in commands:
        command_name = command.split(maxsplit=1)[0]
        timeout_seconds = (
            sensor_start_timeout_seconds
            if command_name == "sensorStart"
            else command_timeout_seconds
        )
        cli_port.write((command + "\n").encode("ascii"))
        deadline = time.monotonic() + timeout_seconds
        response = bytearray()
        while time.monotonic() < deadline:
            waiting = int(getattr(cli_port, "in_waiting", 0))
            chunk = cli_port.read(waiting or 1)
            if chunk:
                response.extend(chunk)
                if b"mmwDemo:/>" in response:
                    break
        text_response = response.decode("utf-8", errors="replace")
        accepted_stop = command == "sensorStop" and "Ignored:" in text_response
        if "Done" not in text_response and not accepted_stop:
            normalized = " ".join(text_response.split())
            raise RuntimeError(
                f"TI cfg command did not complete: {command!r}; "
                f"response={normalized!r}"
            )


def _default_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    toolbox_root = project_root / "radar_toolbox_2_20_00_05"
    common_dir = (
        toolbox_root
        / "tools"
        / "visualizers"
        / "Applications_Visualizer"
        / "common"
    )
    config_path = (
        toolbox_root
        / "source"
        / "ti"
        / "examples"
        / "People_Tracking"
        / "3D_People_Tracking"
        / "chirp_configs"
        / "ISK_6m_default.cfg"
    )
    return common_dir, config_path


def _watch_stop_requests(stop_event: threading.Event) -> None:
    """Translate the adapter's process-control line into a graceful stop."""

    for line in sys.stdin:
        if line.strip() == "STOP":
            stop_event.set()
            return


def main(argv: Sequence[str] | None = None) -> int:
    default_common, default_config = _default_paths()
    argument_parser = argparse.ArgumentParser(
        description="Bridge TI Radar Toolbox decoded People Tracking frames to JSONL"
    )
    argument_parser.add_argument("--cli-port", default="COM5")
    argument_parser.add_argument("--data-port", default="COM6")
    argument_parser.add_argument("--config", type=Path, default=default_config)
    argument_parser.add_argument(
        "--official-common-dir",
        type=Path,
        default=default_common,
    )
    argument_parser.add_argument(
        "--max-frames",
        type=int,
        help="stop after N decoded frames (diagnostics only)",
    )
    argument_parser.add_argument(
        "--reuse-existing-config",
        action="store_true",
        help=(
            "restart a previously configured People Tracking sensor with "
            "'sensorStart 0' instead of resending the full cfg "
            "(diagnostics only; restart reliability is firmware-dependent)"
        ),
    )
    argument_parser.add_argument(
        "--output-jsonl",
        type=Path,
        help=(
            "optional raw decoded-frame log; the path must not already exist "
            "so A/B sessions cannot be mixed accidentally"
        ),
    )
    args = argument_parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        argument_parser.error("--max-frames must be greater than zero")

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    stop_event = threading.Event()
    threading.Thread(
        target=_watch_stop_requests,
        args=(stop_event,),
        name="ti-official-bridge-control",
        daemon=True,
    ).start()
    try:
        run_bridge(
            cli_port=args.cli_port,
            data_port=args.data_port,
            config_path=args.config.resolve(),
            common_dir=args.official_common_dir.resolve(),
            max_frames=args.max_frames,
            stop_event=stop_event,
            reuse_existing_config=args.reuse_existing_config,
            output_jsonl=(
                args.output_jsonl.resolve() if args.output_jsonl is not None else None
            ),
        )
    except KeyboardInterrupt:
        return 0
    except BaseException:
        LOGGER.exception("TI official output bridge stopped")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
