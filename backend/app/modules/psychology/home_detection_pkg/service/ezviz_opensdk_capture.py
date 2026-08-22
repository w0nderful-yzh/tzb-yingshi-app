# -*- coding: utf-8 -*-
"""EZVIZ Windows OpenSDK → 临时视频文件录制器（流式心理 worker 用）。

背景：这台 C6c 设备经 `v2/live/address/get` 只返回带 #EXT-X-ENDLIST 的单片段
占位 HLS（错误码 9053），标准 openlive 流（仅支持 H.264）不可用；而 OpenSDK
回调能拿到压缩的 HEVC/H.264 分块（实测约 13fps）。本模块把这些分块经系统
ffmpeg 转码为临时 H.264 mp4，供 OpenFace 批处理读取（OpenFace 自带 OpenCV
能解 HEVC 与 H.264）。视频不长期保存：文件由调用方负责删除。

自包含实现（仅 stdlib + 系统 ffmpeg + OpenNetStream.dll），不依赖 fall 工程。
OpenSDK 函数绑定与参数参考 `摔倒预测模块/.../ezviz_opensdk_capture.py`。

用法（worker 内，每窗口一次）：
    with EzvizOpenSdkRecorder(
        sdk_root=..., app_key=..., access_token=..., device_serial=...,
        verify_code="", stream_type=2, out_dir=work_dir,
    ) as recorder:
        video = recorder.record(seconds=138.0)   # -> work_dir/xxx.mp4
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from queue import Full, Queue
from threading import Thread

AUTH_URLS = (b"https://openauth.ys7.com", b"https://open.ys7.com")
SDK_STALL_TIMEOUT_SECONDS = 10.0
OUTPUT_FPS = 15.0  # 输出恒定帧率（实际 ~13fps，CFR 归一化时间戳便于 OpenFace 读）


def _bind(lib, name: str, argtypes: list[object], restype=ctypes.c_int):
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


class EzvizOpenSdkRecorder:
    """启动 OpenSDK 会话并把压缩流转码写入临时 mp4，record() 结束时关闭会话。"""

    def __init__(
        self,
        *,
        sdk_root: str | Path,
        app_key: str,
        access_token: str,
        device_serial: str,
        verify_code: str = "",
        stream_type: int = 2,
        ffmpeg: str | None = None,
        out_dir: str | Path,
    ) -> None:
        sdk_root = Path(sdk_root).resolve()
        library_path = sdk_root / "lib" / "win64" / "OpenNetStream.dll"
        if not library_path.is_file():
            raise FileNotFoundError(f"OpenNetStream.dll not found below {sdk_root}")
        self.app_key = app_key.strip()
        self.access_token = access_token.strip()
        self.device_serial = device_serial.strip()
        # 未加密设备（isEncrypt=0）的验证码字段可传占位值；SDK 仅在有视频加密时校验。
        self.verify_code = (verify_code or "000000").strip()
        self.stream_type = int(stream_type)
        if self.stream_type not in (1, 2):
            raise ValueError("stream_type must be 1 (main) or 2 (sub)")
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg")
        if not self.ffmpeg:
            raise RuntimeError("ffmpeg is not available on PATH")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._library_dir = sdk_root / "lib" / "win64"
        self._library_path = library_path
        self._chunks: Queue = Queue(maxsize=512)
        self._process: subprocess.Popen | None = None
        self._writer: Thread | None = None
        self._dll_dir_handle = None
        self._initialized = False
        self._started = False
        self._session: bytes | None = None
        self._last_data_monotonic = 0.0
        self._first_data_monotonic = 0.0
        self._callback_chunks = 0
        self._callback_bytes = 0
        self.dropped_chunks = 0
        self.stderr_tail: list[str] = []
        self._trace_label = f"{self.out_dir.parent.name}/{self.out_dir.name}/{uuid.uuid4().hex[:8]}"

    def _trace(self, event: str, **fields: object) -> None:
        """Emit low-volume lifecycle diagnostics without changing SDK behavior."""
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        suffix = f" {details}" if details else ""
        print(
            f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] "
            f"OpenSDK[{self._trace_label}] {event}{suffix}",
            flush=True,
        )

    # ---------- 录制主体 ----------

    def record(self, seconds: float) -> Path:
        out_path = self.out_dir / f"sdk-{uuid.uuid4().hex}.mp4"
        deadline = time.monotonic() + seconds
        self._trace("record.begin", seconds=f"{seconds:.1f}", output=out_path.name)
        try:
            self._start_ffmpeg(out_path)
            self._start_sdk()
            while time.monotonic() < deadline:
                if self._process is None or self._process.poll() is not None:
                    returncode = self._process.returncode if self._process else None
                    self._trace("record.ffmpeg_exited", returncode=returncode)
                    raise RuntimeError("ffmpeg exited early")
                # 流停顿保护：一段时间没有数据则提前结束，保留已录内容
                idle_seconds = time.monotonic() - self._last_data_monotonic
                if idle_seconds > SDK_STALL_TIMEOUT_SECONDS:
                    self.stderr_tail = self.stderr_tail + ["sdk stream stalled"]
                    self._trace("record.stream_stalled", idle_seconds=f"{idle_seconds:.1f}")
                    break
                time.sleep(0.5)
        finally:
            self._trace("record.close.begin")
            self.close()
            self._trace("record.close.end")
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise RuntimeError(f"OpenSDK recording produced empty file: {out_path}")
        self._trace("record.end", output_bytes=out_path.stat().st_size)
        return out_path

    # ---------- SDK ----------

    def _start_sdk(self) -> None:
        library_dir = self._library_dir
        self._trace("sdk.load.begin", dll=self._library_path.name)
        if hasattr(os, "add_dll_directory"):
            self._dll_dir_handle = os.add_dll_directory(str(library_dir))
        library = ctypes.WinDLL(str(self._library_path))
        self._trace("sdk.load.end")

        message_cb = ctypes.WINFUNCTYPE(
            None, ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_char_p, ctypes.c_void_p
        )
        data_cb = ctypes.WINFUNCTYPE(
            None, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p
        )

        init = _bind(library, "OpenSDK_InitLib", [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool])
        fini = _bind(library, "OpenSDK_FiniLib", [])
        set_config = _bind(library, "OpenSDK_SetConfigInfo", [ctypes.c_int, ctypes.c_int], None)
        set_token = _bind(library, "OpenSDK_SetAccessToken", [ctypes.c_char_p])
        alloc = _bind(
            library,
            "OpenSDK_AllocSessionEx",
            [message_cb, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)],
        )
        free_data = _bind(library, "OpenSDK_Data_Free", [ctypes.c_void_p])
        free_session = _bind(library, "OpenSDK_FreeSession", [ctypes.c_char_p])
        set_callback = _bind(library, "OpenSDK_SetDataCallBack", [ctypes.c_char_p, data_cb, ctypes.c_void_p])
        set_session_config = _bind(library, "OpenSDK_SetSessionConfig", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int], None)
        start_play = _bind(
            library,
            "OpenSDK_StartPlayWithStreamType",
            [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int],
        )
        stop_play = _bind(library, "OpenSDK_StopRealPlayEx", [ctypes.c_char_p])
        last_error = _bind(library, "OpenSDK_GetLastErrorCode", [])
        self._library = library
        self._fini = fini
        self._stop_play = stop_play
        self._set_callback = set_callback
        self._free_session = free_session
        self._data_callback_type = data_cb

        @message_cb
        def on_message(_session, _message_type, error_code, _message, _user) -> None:
            if int(error_code):
                self.stderr_tail = (self.stderr_tail + [f"OpenSDK callback error {int(error_code)}"])[-20:]

        @data_cb
        def on_data(data_type, pointer, length, _user, _session) -> None:
            if int(data_type) not in (1, 2) or not pointer or int(length) <= 0:
                return
            now = time.monotonic()
            self._last_data_monotonic = now
            self._callback_chunks += 1
            self._callback_bytes += int(length)
            if self._first_data_monotonic == 0.0:
                self._first_data_monotonic = now
                self._trace("callback.first_packet", data_type=int(data_type), bytes=int(length))
            try:
                self._chunks.put_nowait(ctypes.string_at(pointer, length))
            except Full:
                self.dropped_chunks += 1

        # 关键：回调对象必须持有引用，否则 _start_sdk 返回后会被 GC，
        # SDK 回调触发时踩空函数指针 -> 段错误。
        self._message_callback = on_message
        self._data_callback = on_data

        started_at = time.monotonic()
        self._trace("sdk.init.begin")
        result = init(AUTH_URLS[0], AUTH_URLS[1], self.app_key.encode("utf-8"), False)
        self._trace("sdk.init.end", result=result, elapsed=f"{time.monotonic() - started_at:.3f}")
        if result != 0:
            raise RuntimeError(f"OpenSDK_InitLib failed: {last_error()}")
        self._initialized = True
        set_config(1, 1)
        if set_token(self.access_token.encode("utf-8")) != 0:
            raise RuntimeError(f"OpenSDK_SetAccessToken failed: {last_error()}")
        session_ptr = ctypes.c_void_p()
        session_len = ctypes.c_int()
        started_at = time.monotonic()
        self._trace("session.create.begin")
        result = alloc(on_message, None, ctypes.byref(session_ptr), ctypes.byref(session_len))
        self._trace(
            "session.create.end",
            result=result,
            session_bytes=session_len.value,
            elapsed=f"{time.monotonic() - started_at:.3f}",
        )
        if result != 0 or not session_ptr.value:
            raise RuntimeError(f"OpenSDK_AllocSessionEx failed: {last_error()}")
        session = ctypes.string_at(session_ptr, session_len.value).rstrip(b"\0")
        free_data(session_ptr)
        if not session:
            raise RuntimeError("OpenSDK allocated an empty session")
        self._session = session
        self._trace("callback.register.begin")
        callback_result = set_callback(session, on_data, None)
        self._trace("callback.register.end", result=callback_result)
        if callback_result != 0:
            raise RuntimeError(f"OpenSDK_SetDataCallBack failed: {last_error()}")
        set_session_config(session, 2, 1)
        started_at = time.monotonic()
        self._trace("play.start.begin", stream_type=self.stream_type)
        result = start_play(
            session, None, self.device_serial.encode("utf-8"), 1,
            self.verify_code.encode("utf-8"), self.stream_type,
        )
        self._trace("play.start.end", result=result, elapsed=f"{time.monotonic() - started_at:.3f}")
        if result != 0:
            raise RuntimeError(f"OpenSDK_StartPlayWithStreamType failed: {last_error()}")
        self._started = True
        self._last_data_monotonic = time.monotonic()

    # ---------- ffmpeg 写入 ----------

    def _start_ffmpeg(self, out_path: Path) -> None:
        self._trace("ffmpeg.start.begin", output=out_path.name)
        process = subprocess.Popen(
            [
                self.ffmpeg, "-y", "-hide_banner", "-loglevel", "warning",
                "-fflags", "nobuffer", "-flags", "low_delay",
                "-analyzeduration", "1000000", "-probesize", "1000000",
                "-i", "pipe:0",
                "-map", "0:v:0", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-vsync", "cfr", "-r", str(OUTPUT_FPS),
                "-f", "mp4", str(out_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._process = process
        self._trace("ffmpeg.start.end", pid=process.pid)

        def write_chunks() -> None:
            assert process.stdin is not None
            while True:
                chunk = self._chunks.get()
                if chunk is None:
                    break
                try:
                    process.stdin.write(chunk)
                except (BrokenPipeError, OSError):
                    break
            try:
                process.stdin.close()
            except OSError:
                pass

        def drain_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                value = line.decode("utf-8", errors="replace").strip()
                if value:
                    self.stderr_tail = (self.stderr_tail + [value[-500:]])[-20:]

        self._writer = Thread(target=write_chunks, name="sdk-recorder-input", daemon=True)
        stderr_thread = Thread(target=drain_stderr, name="sdk-recorder-stderr", daemon=True)
        self._writer.start()
        stderr_thread.start()

    # ---------- 清理 ----------

    def close(self) -> None:
        close_started = time.monotonic()
        session = self._session
        self._trace(
            "close.begin",
            started=self._started,
            initialized=self._initialized,
            has_session=session is not None,
        )
        if self._started and session and getattr(self, "_stop_play", None) is not None:
            started_at = time.monotonic()
            self._trace("play.stop.begin")
            try:
                result = self._stop_play(session)
                self._trace(
                    "play.stop.end",
                    result=result,
                    elapsed=f"{time.monotonic() - started_at:.3f}",
                )
            except Exception as exc:  # noqa: BLE001
                self._trace("play.stop.error", error=type(exc).__name__)
        self._started = False

        if session is not None:
            if (
                getattr(self, "_set_callback", None) is not None
                and getattr(self, "_data_callback_type", None) is not None
            ):
                started_at = time.monotonic()
                self._trace("callback.unregister.begin")
                try:
                    result = self._set_callback(session, self._data_callback_type(), None)
                    self._trace(
                        "callback.unregister.end",
                        result=result,
                        elapsed=f"{time.monotonic() - started_at:.3f}",
                    )
                except Exception as exc:  # noqa: BLE001
                    self._trace("callback.unregister.error", error=type(exc).__name__)

            if getattr(self, "_free_session", None) is not None:
                started_at = time.monotonic()
                self._trace("free_session.begin")
                try:
                    result = self._free_session(session)
                    self._trace(
                        "free_session.end",
                        result=result,
                        elapsed=f"{time.monotonic() - started_at:.3f}",
                    )
                except Exception as exc:  # noqa: BLE001
                    self._trace("free_session.error", error=type(exc).__name__)
            self._session = None

        if getattr(self, "_initialized", False) and getattr(self, "_fini", None) is not None:
            started_at = time.monotonic()
            self._trace("sdk.fini.begin")
            try:
                result = self._fini()
                self._trace(
                    "sdk.fini.end",
                    result=result,
                    elapsed=f"{time.monotonic() - started_at:.3f}",
                )
            except Exception as exc:  # noqa: BLE001
                self._trace("sdk.fini.error", error=type(exc).__name__)
            self._initialized = False

        self._trace("ffmpeg.finish.begin")
        self._finish_ffmpeg()
        self._trace("ffmpeg.finish.end")
        if self._dll_dir_handle is not None:
            self._dll_dir_handle.close()
            self._dll_dir_handle = None
        self._trace(
            "close.end",
            elapsed=f"{time.monotonic() - close_started:.3f}",
            callback_chunks=self._callback_chunks,
            callback_bytes=self._callback_bytes,
            dropped_chunks=self.dropped_chunks,
        )

    def _finish_ffmpeg(self) -> None:
        if self._writer is not None:
            self._trace("ffmpeg.writer.stop_signal")
            try:
                self._chunks.put_nowait(None)
            except Full:
                self._trace("ffmpeg.writer.stop_signal_dropped", queue_size=self._chunks.qsize())
            self._writer.join(timeout=3)
            self._trace("ffmpeg.writer.joined", alive=self._writer.is_alive())
            self._writer = None
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=15)
                self._trace("ffmpeg.process.exited", returncode=process.returncode)
            except subprocess.TimeoutExpired:
                self._trace("ffmpeg.process.kill", reason="wait_timeout")
                process.kill()
                process.wait(timeout=5)
                self._trace("ffmpeg.process.killed", returncode=process.returncode)
        self._process = None

    def __enter__(self) -> "EzvizOpenSdkRecorder":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


__all__ = ["EzvizOpenSdkRecorder"]
