from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

from radar_module.contracts import RadarFrame, RadarTarget, Room, SourceMode
from radar_module.preprocess.pointcloud_processing import map_official_points


@runtime_checkable
class RadarSourceAdapter(Protocol):
    """已解码TI输出与雷达业务模块之间的边界。

    Adapter只能提供官方parser/demo已经解码的数据，不负责UART、TLV或
    任何毫米波底层协议解析。
    """

    source_mode: SourceMode

    def start(self) -> None: ...

    def read_decoded(self) -> Mapping[str, Any] | None: ...

    def stop(self) -> None: ...


class TiRadarReader:
    """通过组合RadarSourceAdapter，将已解码输出统一为RadarFrame。"""

    def __init__(
        self,
        source_adapter: RadarSourceAdapter,
        device_id: str,
        room: Room | str,
        *,
        max_distance_m: float | None = 8.0,
    ) -> None:
        if not isinstance(source_adapter, RadarSourceAdapter):
            raise TypeError("source_adapter must implement RadarSourceAdapter")
        if not device_id.strip():
            raise ValueError("device_id must not be blank")
        self.source_adapter = source_adapter
        self.device_id = device_id.strip()
        self.room = Room(room)
        self.max_distance_m = max_distance_m
        self._started = False

    @property
    def source_mode(self) -> SourceMode:
        return self.source_adapter.source_mode

    @property
    def is_running(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            return
        self.source_adapter.start()
        self._started = True

    def read(self) -> RadarFrame | None:
        if not self._started:
            raise RuntimeError("TiRadarReader has not been started")
        decoded = self.source_adapter.read_decoded()
        if decoded is None:
            return None

        timestamp = _parse_timestamp(decoded.get("timestamp"))
        device_id = str(decoded.get("device_id") or self.device_id).strip()
        room = Room(decoded.get("room") or self.room.value)
        raw_points = decoded.get("points", ())
        points = map_official_points(
            raw_points,
            max_distance_m=self.max_distance_m,
        )
        targets = _parse_targets(decoded.get("targets", ()))
        return RadarFrame(
            timestamp=timestamp,
            device_id=device_id,
            room=room,
            source_mode=self.source_mode,
            points=points,
            frame_number=_optional_int(
                decoded.get("ti_frame_number", decoded.get("frame_number"))
            ),
            source_timestamp=_optional_text(
                decoded.get("source_timestamp") or decoded.get("timestamp")
            ),
            source_monotonic_ns=_optional_int(decoded.get("source_monotonic_ns")),
            received_at=_optional_text(decoded.get("received_at")),
            targets=targets,
            radar_config_name=_optional_text(decoded.get("radar_config_name")),
        )

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self.source_adapter.stop()
        finally:
            self._started = False


DecodedCallback = Callable[[], Mapping[str, Any] | None]


def _parse_targets(value: Any) -> tuple[RadarTarget, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    targets: list[RadarTarget] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        try:
            target = RadarTarget(
                track_id=int(item["track_id"]),
                x=float(item["x"]),
                y=float(item["y"]),
                z=float(item["z"]),
                velocity_x=_optional_float(item.get("velocity_x")),
                velocity_y=_optional_float(item.get("velocity_y")),
                velocity_z=_optional_float(item.get("velocity_z")),
                accel_x=_optional_float(item.get("accel_x") or item.get("acc_x")),
                accel_y=_optional_float(item.get("accel_y") or item.get("acc_y")),
                accel_z=_optional_float(item.get("accel_z") or item.get("acc_z")),
                confidence=_optional_float(item.get("confidence")),
            )
        except (KeyError, TypeError, ValueError):
            continue
        targets.append(target)
    return tuple(targets)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


class TiOfficialOutputAdapter:
    """适配TI官方parser/demo的已解码实时输出。

    支持两种接法：
    1. ``decoded_callback``：由官方Demo的回调/导出层返回Mapping；
    2. ``command``：启动官方或桥接进程，并从stdout逐行读取JSON对象。

    本类不导入TI parser，也不解释UART/TLV字节。
    """

    source_mode = SourceMode.REAL

    def __init__(
        self,
        *,
        decoded_callback: DecodedCallback | None = None,
        command: Sequence[str] | None = None,
        cwd: str | Path | None = None,
        startup_timeout_seconds: float = 10.0,
    ) -> None:
        if (decoded_callback is None) == (command is None):
            raise ValueError(
                "configure exactly one official-output source: "
                "decoded_callback or command"
            )
        if command is not None and not command:
            raise ValueError("command must not be empty")
        self._callback = decoded_callback
        self._command = tuple(command) if command is not None else None
        self._cwd = Path(cwd).resolve() if cwd is not None else None
        self._startup_timeout_seconds = startup_timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._prefetched_output: Mapping[str, Any] | None = None
        self._output_queue: queue.Queue[Mapping[str, Any] | BaseException | None] = (
            queue.Queue(maxsize=256)
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        if self._callback is not None:
            self._started = True
            return

        assert self._command is not None
        self._prefetched_output = None
        self._process = subprocess.Popen(
            list(self._command),
            cwd=str(self._cwd) if self._cwd is not None else None,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
        )
        if self._process.stdout is None:
            self._process.terminate()
            raise RuntimeError("official output process has no stdout stream")
        self._reader_thread = threading.Thread(
            target=self._read_process_output,
            args=(self._process.stdout,),
            name="ti-official-output-reader",
            daemon=True,
        )
        self._reader_thread.start()
        self._started = True

        deadline = time.monotonic() + self._startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                stderr = (
                    self._process.stderr.read().strip()
                    if self._process.stderr is not None
                    else ""
                )
                self.stop()
                raise RuntimeError(
                    f"official output process exited during startup: {stderr}"
                )
            remaining = max(0.0, deadline - time.monotonic())
            try:
                item = self._output_queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue
            if isinstance(item, BaseException):
                self.stop()
                raise RuntimeError("invalid official decoded output") from item
            if item is None:
                if self._process.poll() is None:
                    try:
                        self._process.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        pass
                error = self._stopped_process_error()
                self.stop()
                raise error
            self._prefetched_output = item
            return
        self.stop()
        raise RuntimeError(
            "official output process did not emit a decoded frame before "
            f"the {self._startup_timeout_seconds:.1f}s startup timeout"
        )

    def read_decoded(self) -> Mapping[str, Any] | None:
        if not self._started:
            raise RuntimeError("TiOfficialOutputAdapter has not been started")
        if self._callback is not None:
            payload = self._callback()
            return _validate_decoded_mapping(payload)
        if self._prefetched_output is not None:
            payload = self._prefetched_output
            self._prefetched_output = None
            return payload

        try:
            item = self._output_queue.get(timeout=0.25)
        except queue.Empty:
            if self._process is not None and self._process.poll() is not None:
                raise self._stopped_process_error()
            return None
        if isinstance(item, BaseException):
            raise RuntimeError("invalid official decoded output") from item
        if item is None:
            if self._process is not None:
                if self._process.poll() is None:
                    try:
                        self._process.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        pass
                if self._process.poll() is not None:
                    raise self._stopped_process_error()
            raise RuntimeError("official output process closed its stdout stream")
        return item

    def _stopped_process_error(self) -> RuntimeError:
        stderr = ""
        process = self._process
        if (
            process is not None
            and process.poll() is not None
            and process.stderr is not None
        ):
            stderr = process.stderr.read().strip()
        detail = f": {stderr}" if stderr else ""
        return RuntimeError(f"official output process has stopped{detail}")

    def stop(self) -> None:
        self._started = False
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            try:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.write("STOP\n")
                    process.stdin.flush()
                process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=3)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1)
            self._reader_thread = None
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self._prefetched_output = None
        _clear_queue(self._output_queue)

    def _read_process_output(self, output: TextIO) -> None:
        try:
            for line in output:
                normalized = line.strip()
                if not normalized:
                    continue
                payload = json.loads(normalized)
                mapping = _validate_decoded_mapping(payload)
                if mapping is not None:
                    self._output_queue.put(mapping)
        except BaseException as exc:
            self._output_queue.put(exc)
        finally:
            self._output_queue.put(None)


class ReconnectingTiOfficialOutputAdapter:
    """Keep one logical REAL source alive while its bridge reconnects.

    The Radar API owns exactly one inference worker.  When the TI bridge or
    serial connection drops, this adapter recreates only the bridge inside
    that worker; it never starts a second Radar inference process.
    """

    source_mode = SourceMode.REAL

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: str | Path | None = None,
        reconnect_seconds: float = 2.0,
    ) -> None:
        if reconnect_seconds <= 0:
            raise ValueError("reconnect_seconds must be positive")
        self._command = tuple(command)
        self._cwd = cwd
        self._reconnect_seconds = reconnect_seconds
        self._delegate: TiOfficialOutputAdapter | None = None
        self._started = False
        self._stop_event = threading.Event()
        self.last_error: str | None = None

    def start(self) -> None:
        if self._started:
            return
        self._stop_event.clear()
        self._started = True
        try:
            self._connect()
        except BaseException:
            self._started = False
            self._stop_delegate()
            raise

    def read_decoded(self) -> Mapping[str, Any] | None:
        if not self._started:
            raise RuntimeError("reconnecting TI adapter has not been started")
        while self._started and not self._stop_event.is_set():
            delegate = self._delegate
            if delegate is None:
                try:
                    self._connect()
                except (OSError, RuntimeError, ValueError) as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self._stop_event.wait(self._reconnect_seconds)
                    continue
                delegate = self._delegate
            assert delegate is not None
            try:
                payload = delegate.read_decoded()
                self.last_error = None
                return payload
            except (OSError, RuntimeError, ValueError) as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._stop_delegate()
                self._stop_event.wait(self._reconnect_seconds)
        return None

    def stop(self) -> None:
        self._started = False
        self._stop_event.set()
        self._stop_delegate()

    def _connect(self) -> None:
        delegate = TiOfficialOutputAdapter(
            command=self._command,
            cwd=self._cwd,
        )
        delegate.start()
        self._delegate = delegate
        self.last_error = None

    def _stop_delegate(self) -> None:
        delegate = self._delegate
        self._delegate = None
        if delegate is not None:
            delegate.stop()


class JsonlReplayAdapter:
    """以标准化RadarFrame JSONL作为离线备用数据源。"""

    source_mode = SourceMode.REPLAY

    def __init__(
        self,
        file_path: str | Path,
        *,
        speed: float = 1.0,
        loop: bool = False,
    ) -> None:
        if speed <= 0:
            raise ValueError("speed must be greater than zero")
        self.file_path = Path(file_path).resolve()
        self.speed = float(speed)
        self.loop = bool(loop)
        self._records: list[Mapping[str, Any]] = []
        self._index = 0
        self._started = False
        self._wall_started_at = 0.0
        self._first_record_time: datetime | None = None
        self._stop_event = threading.Event()

    @property
    def finished(self) -> bool:
        return self._started and not self.loop and self._index >= len(self._records)

    def start(self) -> None:
        if self._started:
            return
        if not self.file_path.is_file():
            raise FileNotFoundError(f"replay file does not exist: {self.file_path}")
        records: list[Mapping[str, Any]] = []
        with self.file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                normalized = line.strip()
                if not normalized:
                    continue
                try:
                    payload = json.loads(normalized)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at line {line_number}: {exc.msg}"
                    ) from exc
                mapping = _validate_decoded_mapping(payload)
                if mapping is None:
                    raise ValueError(f"empty replay record at line {line_number}")
                records.append(mapping)
        if not records:
            raise ValueError("replay file contains no records")
        self._records = records
        self._index = 0
        self._first_record_time = _parse_timestamp(records[0].get("timestamp"))
        self._wall_started_at = time.monotonic()
        self._stop_event.clear()
        self._started = True

    def read_decoded(self) -> Mapping[str, Any] | None:
        if not self._started:
            raise RuntimeError("JsonlReplayAdapter has not been started")
        if self._index >= len(self._records):
            if not self.loop:
                return None
            self._index = 0
            self._first_record_time = _parse_timestamp(
                self._records[0].get("timestamp")
            )
            self._wall_started_at = time.monotonic()

        record = self._records[self._index]
        assert self._first_record_time is not None
        record_time = _parse_timestamp(record.get("timestamp"))
        intended_delay = (
            record_time - self._first_record_time
        ).total_seconds() / self.speed
        elapsed = time.monotonic() - self._wall_started_at
        remaining = intended_delay - elapsed
        if remaining > 0 and self._stop_event.wait(remaining):
            return None
        if not self._started:
            return None
        self._index += 1
        return record

    def stop(self) -> None:
        self._stop_event.set()
        self._records = []
        self._index = 0
        self._first_record_time = None
        self._started = False


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif value is None:
        parsed = datetime.now(timezone.utc)
    else:
        raise ValueError(f"unsupported timestamp type: {type(value).__name__}")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("decoded timestamp must include a timezone offset")
    return parsed


def _validate_decoded_mapping(
    payload: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError("decoded output must be a Mapping")
    points = payload.get("points", ())
    if (
        not isinstance(points, Iterable)
        or isinstance(points, (str, bytes, bytearray, Mapping))
    ):
        raise TypeError("decoded output field 'points' must be a sequence")
    return payload


def _clear_queue(output_queue: queue.Queue[Any]) -> None:
    while True:
        try:
            output_queue.get_nowait()
        except queue.Empty:
            return
