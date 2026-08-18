"""纯雷达真机 pre-fall 特征可观测性采集工具（pilot 版）。

Motivation
----------
DGUHA 点云过于稀疏(3-8点/帧)无法支撑姿态/前兆代理特征。本工具在真机
IWR6843ISK 上按受控协议录制点云会话，用于验证哪些 pre-fall / instability
特征在密点云上稳定可观测。

协议（每个动作一个 session，每个 repeat 独立保存）：
- 动作前静止约 still_seconds(默认 2s) → 执行动作 → 动作后静止约 still_seconds
- 每类动作重复 repeats 次(默认 5)
- session 名即粗标签；每个 repeat 单独落盘

每个 repeat 独立保存（这是本轮关键要求）：
- <output_root>/<action_name>/repeat_<NN>/frames.jsonl
- <output_root>/<action_name>/repeat_<NN>/meta.json
- meta.json 记录 repeat_id / action_name / pre_start / action_start /
  action_end / post_end（单调时钟 + UTC 时间戳）

仅采集点云(复用 ti_official_bridge 官方链路)，不实例化任何 TCN/PointNet
模型、不修改 checkpoint、不改 UART/TLV/固件/实时链路。

交互模式(默认):
- 每次重复前操作者就位静止，按 Enter 触发本次重复(still_pre 开始)
- 动作开始用蜂鸣提示；动作阶段操作者按 Enter 结束
- 动作后自动静置 still_post

固定时长模式(--non-interactive):
- 动作阶段使用固定 --action-seconds，全程无按键

Pilot 模式(--pilot):
- 限定 4 类动作：standing / fast_sitting / forward_instability_recovery /
  controlled_forward_fall

Version: radar_real_prefall_capture_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from radar_module.acquisition.ti_reader import (
    JsonlReplayAdapter,
    RadarSourceAdapter,
    TiOfficialOutputAdapter,
    TiRadarReader,
)
from radar_module.analysis.sensor_to_world_audit_v1 import to_world
from radar_module.contracts import Room

DEFAULT_STILL_SECONDS = 2.0
DEFAULT_REPEATS = 5

# 安装参数（从 .env 读取；默认按当前真实测试 1m / 5°）
ENV_SENSOR_HEIGHT_M = "RADAR_SENSOR_HEIGHT_M"
ENV_ELEV_TILT_DEG = "RADAR_ELEV_TILT_DEG"
ENV_AZI_TILT_DEG = "RADAR_AZI_TILT_DEG"
DEFAULT_SENSOR_HEIGHT_M = 1.0
DEFAULT_ELEV_TILT_DEG = 5.0
DEFAULT_AZI_TILT_DEG = 0.0
# 配置文件里声明的安装参数（用于审计一致性；ISK_6m_default.cfg）
DEFAULT_CFG_SENSOR_HEIGHT_M = 2.0
DEFAULT_CFG_ELEV_TILT_DEG = 15.0
DEFAULT_CFG_AZI_TILT_DEG = 0.0

# 动作目录：session 名即粗标签
ACTIONS = [
    "standing",
    "walking",
    "bending",
    "squatting",
    "sitting",
    "fast_sitting",
    "forward_lean_recovery",
    "forward_instability_recovery",
    "lateral_instability_recovery",
    "controlled_forward_fall",
]

# pilot 轮次限定的 4 类动作
PILOT_ACTIONS = [
    "standing",
    "fast_sitting",
    "forward_instability_recovery",
    "controlled_forward_fall",
]

# 每个动作的建议时长(s)，供固定时长模式参考
SUGGESTED_ACTION_SECONDS = {
    "standing": 5.0,
    "walking": 5.0,
    "bending": 4.0,
    "squatting": 5.0,
    "sitting": 5.0,
    "fast_sitting": 3.0,
    "forward_lean_recovery": 4.0,
    "forward_instability_recovery": 5.0,
    "lateral_instability_recovery": 5.0,
    "controlled_forward_fall": 6.0,
}

# 时间标记名称
MARK_PRE_START = "pre_start"
MARK_ACTION_START = "action_start"
MARK_ACTION_END = "action_end"
MARK_POST_END = "post_end"


@dataclass(frozen=True, slots=True)
class TimeMark:
    name: str
    monotonic: float
    utc_iso: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "monotonic": self.monotonic,
            "utc_iso": self.utc_iso,
        }


@dataclass(slots=True)
class RepeatMetaV1:
    schema_version: str
    repeat_id: str
    action_name: str
    device_id: str
    room: str
    source_mode: str
    session_id: str
    still_seconds: float
    action_seconds: float | None
    interactive: bool
    captured_at: str
    marks: list[dict[str, object]] = field(default_factory=list)
    frame_count: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class _DecodedObservingAdapter:
    """透传 decoded 元数据供 raw_point_count / ti_frame_number 等字段使用。

    仅观察、不修改 payload，复用 RadarSourceAdapter 协议。
    """

    def __init__(self, delegate: RadarSourceAdapter) -> None:
        self.delegate = delegate
        self.last_decoded: Mapping[str, Any] | None = None

    @property
    def source_mode(self):
        return self.delegate.source_mode

    def start(self) -> None:
        self.last_decoded = None
        self.delegate.start()

    def read_decoded(self) -> Mapping[str, Any] | None:
        payload = self.delegate.read_decoded()
        self.last_decoded = payload
        return payload

    def stop(self) -> None:
        self.delegate.stop()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_install_params() -> dict[str, float]:
    """从环境读取安装参数（sensor height / tilt）。未设置则用默认 1m/5°。"""
    def _float_env(name: str, default: float) -> float:
        raw = os.getenv(name, "")
        if raw == "":
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    return {
        "sensor_height_m": _float_env(ENV_SENSOR_HEIGHT_M, DEFAULT_SENSOR_HEIGHT_M),
        "elev_tilt_deg": _float_env(ENV_ELEV_TILT_DEG, DEFAULT_ELEV_TILT_DEG),
        "azi_tilt_deg": _float_env(ENV_AZI_TILT_DEG, DEFAULT_AZI_TILT_DEG),
    }


def _sensor_to_world(points: Sequence[Mapping[str, Any]], params: dict[str, float]) -> list[dict[str, object]]:
    """sensor-frame 点云 → world-frame（保留原字段，只改 x/y/z）。"""
    if not points:
        return []
    xs, ys, zs = to_world(
        list(points),
        sensor_height_m=params["sensor_height_m"],
        elev_tilt_deg=params["elev_tilt_deg"],
        azi_tilt_deg=params["azi_tilt_deg"],
    )
    world: list[dict[str, object]] = []
    for p, (nx, ny, nz) in zip(points, zip(xs, ys, zs)):
        world.append({
            "x": float(nx),
            "y": float(ny),
            "z": float(nz),
            "velocity": p.get("velocity", 0.0),
            **({"snr": p.get("snr")} if p.get("snr") is not None else {}),
            **({"track_id": p.get("track_id")} if p.get("track_id") is not None else {}),
        })
    return world


def _prompt(text: str) -> None:
    print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {text}", flush=True)
    print("\a", end="", flush=True)


def _now_mark(name: str) -> TimeMark:
    return TimeMark(name=name, monotonic=time.monotonic(),
                    utc_iso=datetime.now(timezone.utc).isoformat())


def _target_to_dict(target: Any) -> dict[str, object]:
    """RadarTarget → dict（完整透传 TI Target List TLV 字段）。"""
    out: dict[str, object] = {
        "track_id": target.track_id,
        "pos_x": target.x,
        "pos_y": target.y,
        "pos_z": target.z,
    }
    if target.velocity_x is not None:
        out["vel_x"] = target.velocity_x
    if target.velocity_y is not None:
        out["vel_y"] = target.velocity_y
    if target.velocity_z is not None:
        out["vel_z"] = target.velocity_z
    if target.accel_x is not None:
        out["acc_x"] = target.accel_x
    if target.accel_y is not None:
        out["acc_y"] = target.accel_y
    if target.accel_z is not None:
        out["acc_z"] = target.accel_z
    if target.confidence is not None:
        out["confidence"] = target.confidence
    return out


def _record_frame(
    frame_timestamp_iso: str,
    points_sensor: Sequence[dict[str, object]],
    points_world: Sequence[dict[str, object]],
    targets: Sequence[Any],
    decoded: Mapping[str, Any],
    *,
    repeat_id: str,
    phase: str,
    action_name: str,
    monotonic: float,
    repeat_start_monotonic: float,
) -> dict[str, object]:
    return {
        "timestamp": frame_timestamp_iso,
        "action_name": action_name,
        "repeat_id": repeat_id,
        "phase": phase,
        "monotonic": monotonic,
        "monotonic_since_repeat_start": monotonic - repeat_start_monotonic,
        "points_sensor": [dict(p) for p in points_sensor],
        "points_world": [dict(p) for p in points_world],
        "targets": [_target_to_dict(t) for t in targets],
        "accepted_point_count": len(points_sensor),
        "raw_point_count": len(decoded.get("points", ())),
        "ti_frame_number": decoded.get("ti_frame_number"),
        "ti_parser_error": decoded.get("ti_parser_error"),
    }


def run_capture(
    source_adapter: RadarSourceAdapter,
    *,
    output_directory: str | Path,
    session_id: str,
    action_name: str,
    device_id: str = "iwr6843isk-01",
    room: Room | str = Room.BATHROOM,
    repeats: int = DEFAULT_REPEATS,
    still_seconds: float = DEFAULT_STILL_SECONDS,
    action_seconds: float | None = None,
    interactive_actions: bool = True,
    max_action_seconds: float = 30.0,
    install_params: dict[str, float] | None = None,
) -> list[RepeatMetaV1]:
    """录制单个动作 session，每个 repeat 独立保存。

    interactive_actions=True: 每次重复按 Enter 触发；动作阶段按 Enter 结束。
    interactive_actions=False: 动作阶段使用固定 --action-seconds。
    install_params: sensor height/tilt（world-frame 转换用）。缺省从环境读。
    """
    if action_name not in ACTIONS:
        raise ValueError(f"unknown action_name: {action_name}")
    if action_seconds is None and not interactive_actions:
        raise ValueError("--action-seconds is required when non-interactive")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if install_params is None:
        install_params = _read_install_params()

    output_dir = Path(output_directory).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 透传 decoded 元数据供 raw_point_count / ti_frame_number 等字段
    diagnostic_adapter = _DecodedObservingAdapter(source_adapter)
    reader = TiRadarReader(
        diagnostic_adapter,
        device_id=device_id,
        room=Room(room),
    )
    repeat_metas: list[RepeatMetaV1] = []

    try:
        reader.start()
        # 先清空管线中的历史帧，确保静置阶段从干净起点开始
        drain_started = time.monotonic()
        while time.monotonic() - drain_started < 1.0:
            reader.read()

        for repeat_index in range(1, repeats + 1):
            repeat_id = f"{action_name}_r{repeat_index:02d}"
            records: list[dict[str, object]] = []
            marks: list[TimeMark] = []
            discarded_old_frames = 0

            _prompt(
                f"重复 {repeat_index}/{repeats} | 动作 [{action_name}]"
            )
            if interactive_actions:
                print("  请就位并保持静止，就绪后按 Enter 开始", flush=True)
                input()  # 阻塞等待，可靠；期间帧在队列积压，稍后统一丢弃
                # 清空等待期间积压的旧帧，确保 pre_start 从干净起点开始
                while _drain_discard(reader, diagnostic_adapter):
                    pass
            else:
                print(
                    f"  自动开始（非交互）——{still_seconds:.0f}s 静置后执行动作",
                    flush=True,
                )
                time.sleep(0.5)

            pre_start = _now_mark(MARK_PRE_START)
            marks.append(pre_start)
            repeat_start_monotonic = pre_start.monotonic

            # 静置阶段（动作前）
            _prompt(f"保持静止 {still_seconds:.0f}s…")
            while time.monotonic() - pre_start.monotonic < still_seconds:
                _drain_frame(
                    reader, diagnostic_adapter, records,
                    action_name, repeat_id, "still_pre",
                    repeat_start_monotonic, install_params,
                )

            # 动作阶段
            action_start = _now_mark(MARK_ACTION_START)
            marks.append(action_start)
            if interactive_actions:
                _prompt("动作开始！完成后按 Enter")
                _capture_until_enter(
                    reader, diagnostic_adapter, records,
                    action_name, repeat_id, "action",
                    repeat_start_monotonic, install_params,
                    max_action_seconds=max_action_seconds,
                )
            else:
                _prompt(f"动作开始！保持 {float(action_seconds):.0f}s")
                while (
                    time.monotonic() - action_start.monotonic < float(action_seconds)
                ):
                    _drain_frame(
                        reader, diagnostic_adapter, records,
                        action_name, repeat_id, "action",
                        repeat_start_monotonic, install_params,
                    )
            action_end = _now_mark(MARK_ACTION_END)
            marks.append(action_end)

            # 动作后静置
            _prompt(f"动作结束，保持静止 {still_seconds:.0f}s…")
            while time.monotonic() - action_end.monotonic < still_seconds:
                _drain_frame(
                    reader, diagnostic_adapter, records,
                    action_name, repeat_id, "still_post",
                    repeat_start_monotonic, install_params,
                )
            post_end = _now_mark(MARK_POST_END)
            marks.append(post_end)

            repeat_dir = output_dir / f"repeat_{repeat_index:02d}"
            repeat_dir.mkdir(parents=True, exist_ok=True)
            _write_jsonl(repeat_dir / "frames.jsonl", records)
            meta = RepeatMetaV1(
                schema_version="radar_real_prefall_repeat_v1",
                repeat_id=repeat_id,
                action_name=action_name,
                device_id=device_id,
                room=Room(room).value,
                source_mode=source_adapter.source_mode.value,
                session_id=session_id,
                still_seconds=still_seconds,
                action_seconds=action_seconds,
                interactive=interactive_actions,
                captured_at=datetime.now(timezone.utc).isoformat(),
                marks=[m.to_dict() for m in marks],
                frame_count=len(records),
            )
            _write_json(repeat_dir / "meta.json", meta.to_dict())
            repeat_metas.append(meta)
            print(
                f"  {repeat_id} 完成: {len(records)} 帧 -> "
                f"{repeat_dir.relative_to(output_dir)}",
                flush=True,
            )
    except BaseException as exc:
        # 保留已采集的 repeat 数据，写出错误标记
        (output_dir / "capture_error.json").write_text(
            json.dumps(
                {
                    "schema_version": "radar_real_prefall_capture_error_v1",
                    "session_id": session_id,
                    "action_name": action_name,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed_repeats": [m.to_dict() for m in repeat_metas],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    finally:
        reader.stop()

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "radar_real_prefall_session_v1",
                "session_id": session_id,
                "action_name": action_name,
                "device_id": device_id,
                "room": Room(room).value,
                "source_mode": source_adapter.source_mode.value,
                "repeats": len(repeat_metas),
                "still_seconds": still_seconds,
                "action_seconds": action_seconds,
                "interactive_actions": interactive_actions,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "repeat_ids": [m.repeat_id for m in repeat_metas],
                "total_frames": sum(m.frame_count for m in repeat_metas),
                "install_params_applied": install_params,
                "cfg_sensor_position_declared": {
                    "sensor_height_m": DEFAULT_CFG_SENSOR_HEIGHT_M,
                    "elev_tilt_deg": DEFAULT_CFG_ELEV_TILT_DEG,
                    "azi_tilt_deg": DEFAULT_CFG_AZI_TILT_DEG,
                    "note": "ISK_6m_default.cfg sensorPosition 2 0 15",
                },
                "coordinate_note": (
                    "points_sensor = TI 原始 sensor-frame；points_world = "
                    "eulerRot(elev,az) + z += sensorHeight（按 install_params）"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"\n完成: {output_dir}（{len(repeat_metas)} repeats, "
        f"{sum(m.frame_count for m in repeat_metas)} 帧）",
        flush=True,
    )
    return repeat_metas


def _drain_discard(
    reader: TiRadarReader,
    diagnostic_adapter: _DecodedObservingAdapter,
) -> bool:
    """读出一帧但不保留（丢弃）。返回是否实际丢弃了一帧。"""
    frame = reader.read()
    if frame is None:
        return False
    return True


def _capture_until_enter(
    reader: TiRadarReader,
    diagnostic_adapter: _DecodedObservingAdapter,
    records: list[dict[str, object]],
    action_name: str,
    repeat_id: str,
    phase: str,
    repeat_start_monotonic: float,
    install_params: dict[str, float],
    *,
    max_action_seconds: float,
) -> None:
    """交互式动作阶段：后台线程持续采集，主线程等待 Enter 结束。

    相比在等待期间轮询 _key_pressed()，这里用 input() 阻塞主线程（在
    PowerShell / 非 console stdin 下可靠），后台线程负责读帧，避免
    input() 阻塞期间漏采。
    """
    import threading

    stop_event = threading.Event()
    started_monotonic = time.monotonic()
    capture_error: list[BaseException] = []

    def _worker() -> None:
        try:
            while not stop_event.is_set():
                if (
                    time.monotonic() - started_monotonic
                    > max_action_seconds
                ):
                    break
                _drain_frame(
                    reader, diagnostic_adapter, records,
                    action_name, repeat_id, phase,
                    repeat_start_monotonic, install_params,
                )
        except BaseException as exc:  # pragma: no cover - defensive
            capture_error.append(exc)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    try:
        input()  # 阻塞等待操作者按 Enter 结束动作
    finally:
        stop_event.set()
        worker.join(timeout=2.0)
    if capture_error:
        raise capture_error[0]


def _drain_frame(
    reader: TiRadarReader,
    diagnostic_adapter: _DecodedObservingAdapter,
    records: list[dict[str, object]],
    action_name: str,
    repeat_id: str,
    phase: str,
    repeat_start_monotonic: float,
    install_params: dict[str, float],
) -> None:
    frame = reader.read()
    if frame is None:
        return
    decoded = diagnostic_adapter.last_decoded or {}
    points_sensor = tuple(
        {
            "x": p.x,
            "y": p.y,
            "z": p.z,
            "velocity": p.velocity,
            **({"snr": p.snr} if p.snr is not None else {}),
        }
        for p in frame.points
    )
    points_world = _sensor_to_world(points_sensor, install_params)
    records.append(
        _record_frame(
            frame.timestamp.isoformat(),
            points_sensor,
            points_world,
            frame.targets,
            decoded,
            repeat_id=repeat_id,
            phase=phase,
            action_name=action_name,
            monotonic=time.monotonic(),
            repeat_start_monotonic=repeat_start_monotonic,
        )
    )


def _key_pressed() -> bool:
    """非阻塞检测 Enter；Windows 下用 msvcrt，Linux/测试用 select。"""
    try:
        if sys.platform == "win32":
            import msvcrt

            if msvcrt.kbhit():
                msvcrt.getwch()
                return True
            return False
        import select

        if sys.stdin in select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.readline()
            return True
        return False
    except Exception:
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record pure-radar pre-fall observability sessions "
        "(IWR6843ISK real capture, no model inference)."
    )
    parser.add_argument("--action-name", choices=ACTIONS, required=True)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--still-seconds", type=float, default=DEFAULT_STILL_SECONDS)
    parser.add_argument("--action-seconds", type=float, default=None)
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="use fixed --action-seconds instead of pressing Enter for each action",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="restrict action to the 4-class pilot set "
        "(standing/fast_sitting/forward_instability_recovery/controlled_forward_fall)",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--source", choices=("real", "replay"), default="real")
    parser.add_argument("--replay")
    parser.add_argument("--room", choices=tuple(r.value for r in Room), default="bathroom")
    parser.add_argument("--device-id", default="iwr6843isk-01")
    parser.add_argument("--output-root", type=Path, default=Path("reports/real_prefall_capture_v1"))
    parser.add_argument("--max-action-seconds", type=float, default=30.0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.pilot and args.action_name not in PILOT_ACTIONS:
        raise SystemExit(
            f"--pilot restricts action to {PILOT_ACTIONS}; got {args.action_name}"
        )
    if args.source == "real":
        _load_env_file(Path(args.env_file).resolve())
        command_json = os.getenv("TI_OFFICIAL_OUTPUT_COMMAND_JSON", "")
        if not command_json:
            raise SystemExit("TI_OFFICIAL_OUTPUT_COMMAND_JSON is missing")
        command = json.loads(command_json)
        if not isinstance(command, list) or not all(
            isinstance(item, str) and item for item in command
        ):
            raise SystemExit("TI_OFFICIAL_OUTPUT_COMMAND_JSON must be a string array")
        cwd = os.getenv("TI_OFFICIAL_OUTPUT_CWD", "").strip() or None
        source: RadarSourceAdapter = TiOfficialOutputAdapter(
            command=command,
            cwd=cwd,
        )
    else:
        if not args.replay:
            raise SystemExit("--replay is required for replay source")
        source = JsonlReplayAdapter(Path(args.replay))

    session_id = args.session_id or (
        f"{args.action_name}_p01_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    )
    run_capture(
        source,
        output_directory=args.output_root / args.action_name,
        session_id=session_id,
        action_name=args.action_name,
        device_id=args.device_id,
        room=args.room,
        repeats=args.repeats,
        still_seconds=args.still_seconds,
        action_seconds=args.action_seconds,
        interactive_actions=not args.non_interactive,
        max_action_seconds=args.max_action_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
