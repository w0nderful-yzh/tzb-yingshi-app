from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from queue import Empty, Full, Queue

from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.algorithm_runtime import (
    AdapterContext,
    AlgorithmFinding,
    RiskEventFactory,
)
from app.modules.fall.multimodal_engine.core.config import Settings
from app.modules.fall.multimodal_engine.integrations.ezviz import EzvizApiError, EzvizClient
from app.modules.fall.multimodal_engine.schemas.fall_live import (
    CameraAlignmentSnapshot,
    FallLiveInputState,
    FallLiveState,
    FallLiveStatusResponse,
    RppgLiveStatus,
)
from app.modules.fall.multimodal_engine.schemas.monitoring import (
    MonitoringMode,
    MonitoringSessionCreate,
)
from app.modules.fall.multimodal_engine.schemas.risk_event import (
    EvidenceItem,
    RiskLevel,
    RiskModule,
)
from app.modules.fall.multimodal_engine.services.monitoring import MonitoringService
from app.modules.fall.multimodal_engine.services.risk_event import RiskEventService
from app.modules.fall.multimodal_engine.database.session import SessionLocal


logger = logging.getLogger(__name__)
SessionFactory = Callable[[], Session]


class FallLiveMonitorService:
    """Supervise one persistent EZVIZ -> RTMPose3D -> BioSTGCN worker."""

    def __init__(
        self,
        settings: Settings,
        *,
        session_factory: SessionFactory = SessionLocal,
        ezviz_client_factory: Callable[[], EzvizClient] = EzvizClient,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.ezviz_client_factory = ezviz_client_factory
        initial_state = (
            FallLiveState.STOPPED if settings.fall_live_monitor_enabled else FallLiveState.DISABLED
        )
        self._status = FallLiveStatusResponse(
            enabled=settings.fall_live_monitor_enabled,
            state=initial_state,
        )
        self._status_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._last_event_monotonic = 0.0
        self._browser_frames: Queue[dict[str, str]] = Queue(maxsize=2)
        self._browser_device_id = settings.fall_live_device_id.strip()
        self._browser_received = 0
        self._browser_dropped = 0

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self) -> None:
        if not self.settings.fall_live_monitor_enabled or self.is_running:
            return
        self._stop_event.clear()
        self._browser_received = 0
        self._browser_dropped = 0
        self._set_status(
            state=FallLiveState.STARTING,
            input_state=FallLiveInputState.WAITING,
            input_message="等待摄像头输入",
            target_present=False,
            training_input_ready=False,
            risk_score=None,
            risk_level=None,
            positive_votes=None,
            torso_inclination_deg=None,
            com_proxy_relative_change=None,
            yaw_delta_deg=None,
            pose_quality=None,
            last_prediction_at=None,
            error=None,
        )
        self._worker = threading.Thread(
            target=self._run,
            name="fall-live-monitor",
            daemon=True,
        )
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=10)
        self._worker = None
        self._process = None
        self._set_status(
            state=(
                FallLiveState.STOPPED
                if self.settings.fall_live_monitor_enabled
                else FallLiveState.DISABLED
            ),
            input_state=FallLiveInputState.WAITING,
            input_message="摄像头跌倒预测未启动",
            target_present=False,
            training_input_ready=False,
            risk_score=None,
            risk_level=None,
            positive_votes=None,
            torso_inclination_deg=None,
            com_proxy_relative_change=None,
            yaw_delta_deg=None,
            pose_quality=None,
            last_prediction_at=None,
            error=None,
        )

    def get_status(self) -> FallLiveStatusResponse:
        with self._status_lock:
            return self._status.model_copy(deep=True)

    def submit_browser_frame(
        self,
        *,
        device_id: str,
        captured_at: datetime,
        frame_base64: str,
    ) -> int:
        if not self.settings.fall_live_monitor_enabled:
            raise RuntimeError("实时跌倒监测未启用")
        if self.settings.fall_live_source_mode != "browser_capture":
            raise RuntimeError("实时跌倒监测当前未使用浏览器取帧模式")
        value = frame_base64.strip()
        encoded = value.split(",", 1)[-1]
        estimated_bytes = len(encoded) * 3 // 4
        if estimated_bytes > self.settings.fall_live_browser_max_frame_kb * 1024:
            raise ValueError("浏览器截图超过允许大小")

        packet = {
            "device_id": device_id.strip(),
            "captured_at": captured_at.astimezone(timezone.utc).isoformat(),
            "frame_base64": value,
        }
        self._browser_received += 1
        self._browser_device_id = packet["device_id"]
        try:
            self._browser_frames.put_nowait(packet)
        except Full:
            try:
                self._browser_frames.get_nowait()
            except Empty:
                pass
            self._browser_dropped += 1
            self._browser_frames.put_nowait(packet)
            self._set_status(dropped_frames=self._browser_dropped)
        return self._browser_frames.qsize()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.settings.fall_live_source_mode == "browser_capture":
                    self._run_browser_worker()
                elif self.settings.fall_live_source_mode == "ezviz_opensdk":
                    device_id, worker_config = self._resolve_opensdk_source()
                    self._run_opensdk_worker(device_id, worker_config)
                else:
                    device_id, stream_url = self._resolve_stream()
                    self._run_worker(device_id, stream_url)
            except Exception as exc:
                logger.warning(
                    "Continuous fall monitor unavailable; retrying in %ss: %s: %s",
                    self.settings.fall_live_reconnect_seconds,
                    type(exc).__name__,
                    exc,
                )
                self._set_status(
                    state=FallLiveState.ERROR,
                    error=self._display_error(exc),
                )
            finally:
                self._process = None
            if not self._stop_event.is_set():
                self._stop_event.wait(self.settings.fall_live_reconnect_seconds)

    def _resolve_stream(self) -> tuple[str, str]:
        self._set_status(state=FallLiveState.CONNECTING, error=None)
        with self.ezviz_client_factory() as client:
            configured = self.settings.fall_live_device_id.strip()
            if configured:
                device_id = configured
            else:
                result = client.list_devices(page_size=50)
                online = next((item for item in result.devices if item.status == 1), None)
                if online is None:
                    raise RuntimeError("no online EZVIZ camera is available")
                device_id = online.device_serial
            play = client.get_standard_live_address(
                device_id,
                channel_no=self.settings.fall_live_channel_no,
                protocol=self.settings.fall_inference_live_protocol,
                quality=self.settings.fall_inference_live_quality,
                expire_seconds=self.settings.fall_inference_live_address_expire_seconds,
            )
        self._set_status(device_id=device_id)
        return device_id, play.play_url

    def _resolve_opensdk_source(self) -> tuple[str, dict[str, str]]:
        """Resolve cloud authentication while keeping all video bytes local."""

        self._set_status(state=FallLiveState.CONNECTING, error=None)
        with self.ezviz_client_factory() as client:
            configured = self.settings.fall_live_device_id.strip()
            if configured:
                device_id = configured
            else:
                result = client.list_devices(page_size=50)
                online = next((item for item in result.devices if item.status == 1), None)
                if online is None:
                    raise RuntimeError("no online EZVIZ camera is available")
                device_id = online.device_serial
            access_token = client.auth.get_access_token()
        verify_code = self.settings.ezviz_device_verify_code.get_secret_value().strip()
        app_key = self.settings.ezviz_app_key.get_secret_value().strip()
        if not verify_code or not app_key:
            raise RuntimeError("EZVIZ AppKey or device verification code is missing")
        self._set_status(device_id=device_id)
        return device_id, {
            "app_key": app_key,
            "access_token": access_token,
            "device_serial": device_id,
            "verify_code": verify_code,
        }

    @staticmethod
    def _display_error(exc: Exception) -> str:
        if isinstance(exc, EzvizApiError) and (
            str(exc.code) == "60019" or "加密已开启" in str(exc)
        ):
            return (
                "摄像机的视频图片加密已开启。浏览器EZOPEN播放器可以使用验证码解密，"
                "但后端实时算法需要RTMP/HLS标准流；请在萤石App的设备设置中关闭"
                "“视频图片加密”，系统随后会自动重连。"
            )
        return f"{type(exc).__name__}: {exc}"[-1000:]

    def _run_worker(self, device_id: str, stream_url: str) -> None:
        python_path = Path(self.settings.fall_inference_python)
        project_dir = Path(self.settings.fall_inference_project_dir)
        if not python_path.is_file():
            raise FileNotFoundError(f"fall inference Python not found: {python_path}")
        if not project_dir.is_dir():
            raise FileNotFoundError(f"fall inference project not found: {project_dir}")
        command = [
            str(python_path),
            "-m",
            "fall_inference.stream_worker",
            "--device",
            self.settings.fall_inference_device,
            "--stride",
            str(self.settings.fall_live_stride_frames),
            "--max-queue-frames",
            str(self.settings.fall_live_max_queue_frames),
            "--min-keypoint-confidence",
            str(self.settings.fall_live_min_keypoint_confidence),
            "--min-valid-pose-ratio",
            str(self.settings.fall_live_min_valid_pose_ratio),
            "--required-source-frames",
            str(self.settings.fall_live_required_source_frames),
            "--max-source-gap-frames",
            str(self.settings.fall_live_max_source_gap_frames),
        ]
        process = subprocess.Popen(
            command,
            cwd=project_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._process = process
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("fall inference worker pipes are unavailable")
        process.stdin.write(stream_url + "\n")
        process.stdin.flush()
        process.stdin.close()
        stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(process.stderr,),
            name="fall-live-stderr",
            daemon=True,
        )
        stderr_thread.start()
        for line in process.stdout:
            if self._stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring non-JSON fall worker output: %s", line[-500:])
                continue
            self._handle_message(device_id, message)
        return_code = process.wait()
        stderr_thread.join(timeout=1)
        if self._stop_event.is_set():
            return
        status = self.get_status()
        detail = status.error or (self._stderr_tail[-1] if self._stderr_tail else "no diagnostics")
        raise RuntimeError(f"fall inference worker exited with code {return_code}: {detail}")

    def _run_opensdk_worker(self, device_id: str, worker_config: dict[str, str]) -> None:
        python_path = Path(self.settings.fall_inference_python)
        project_dir = Path(self.settings.fall_inference_project_dir)
        sdk_root = Path(self.settings.fall_live_ezviz_opensdk_root)
        if not python_path.is_file():
            raise FileNotFoundError(f"fall inference Python not found: {python_path}")
        if not project_dir.is_dir():
            raise FileNotFoundError(f"fall inference project not found: {project_dir}")
        if not (sdk_root / "lib" / "win64" / "OpenNetStream.dll").is_file():
            raise FileNotFoundError(f"EZVIZ OpenSDK not found: {sdk_root}")
        command = [
            str(python_path),
            "-X",
            "utf8",
            "-m",
            "fall_inference.opensdk_stream_worker",
            "--sdk-root",
            str(sdk_root),
            "--device",
            self.settings.fall_inference_device,
            "--stream-type",
            str(self.settings.fall_live_ezviz_opensdk_stream_type),
            "--stride",
            str(self.settings.fall_live_stride_frames),
            "--max-queue-frames",
            str(self.settings.fall_live_max_queue_frames),
            "--min-keypoint-confidence",
            str(self.settings.fall_live_min_keypoint_confidence),
            "--min-valid-pose-ratio",
            str(self.settings.fall_live_min_valid_pose_ratio),
            "--detector-interval",
            str(self.settings.fall_live_detector_interval),
            "--pose-batch-size",
            str(self.settings.fall_live_pose_batch_size),
            "--stream-stall-timeout",
            str(self.settings.fall_live_stream_stall_timeout_seconds),
            "--stream-reconnect-seconds",
            str(self.settings.fall_live_reconnect_seconds),
        ]
        if self.settings.rppg_enabled:
            command.extend(
                [
                    "--enable-rppg",
                    "--rppg-sqi-threshold",
                    str(self.settings.rppg_sqi_threshold),
                    "--rppg-min-valid-seconds",
                    str(self.settings.rppg_min_valid_seconds),
                ]
            )
        process = subprocess.Popen(
            command,
            cwd=project_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._process = process
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("fall inference worker pipes are unavailable")
        process.stdin.write(json.dumps(worker_config, ensure_ascii=False) + "\n")
        process.stdin.flush()
        process.stdin.close()
        stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(process.stderr,),
            name="fall-live-opensdk-stderr",
            daemon=True,
        )
        stderr_thread.start()
        for line in process.stdout:
            if self._stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring non-JSON fall worker output: %s", line[-500:])
                continue
            self._handle_message(device_id, message)
        return_code = process.wait()
        stderr_thread.join(timeout=1)
        if self._stop_event.is_set():
            return
        status = self.get_status()
        detail = status.error or (self._stderr_tail[-1] if self._stderr_tail else "no diagnostics")
        raise RuntimeError(f"fall inference worker exited with code {return_code}: {detail}")

    def _run_browser_worker(self) -> None:
        python_path = Path(self.settings.fall_inference_python)
        project_dir = Path(self.settings.fall_inference_project_dir)
        if not python_path.is_file():
            raise FileNotFoundError(f"fall inference Python not found: {python_path}")
        if not project_dir.is_dir():
            raise FileNotFoundError(f"fall inference project not found: {project_dir}")
        command = [
            str(python_path),
            "-m",
            "fall_inference.browser_frame_worker",
            "--device",
            self.settings.fall_inference_device,
            "--stride",
            str(self.settings.fall_live_stride_frames),
            "--source-fps",
            str(self.settings.fall_live_browser_capture_fps),
            "--min-keypoint-confidence",
            str(self.settings.fall_live_min_keypoint_confidence),
            "--min-valid-pose-ratio",
            str(self.settings.fall_live_min_valid_pose_ratio),
            "--required-source-frames",
            str(self.settings.fall_live_required_source_frames),
            "--max-source-gap-frames",
            str(self.settings.fall_live_max_source_gap_frames),
        ]
        process = subprocess.Popen(
            command,
            cwd=project_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._process = process
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("fall inference worker pipes are unavailable")
        stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(process.stderr,),
            name="fall-live-stderr",
            daemon=True,
        )
        writer_thread = threading.Thread(
            target=self._write_browser_frames,
            args=(process,),
            name="fall-live-browser-frames",
            daemon=True,
        )
        stderr_thread.start()
        writer_thread.start()
        for line in process.stdout:
            if self._stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring non-JSON fall worker output: %s", line[-500:])
                continue
            device_id = self._browser_device_id or "ezviz-browser"
            self._handle_message(device_id, message)
        return_code = process.wait()
        writer_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        if self._stop_event.is_set():
            return
        status = self.get_status()
        detail = status.error or (self._stderr_tail[-1] if self._stderr_tail else "no diagnostics")
        raise RuntimeError(f"fall inference worker exited with code {return_code}: {detail}")

    def _write_browser_frames(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdin
        if stream is None:
            return
        try:
            while not self._stop_event.is_set() and process.poll() is None:
                try:
                    packet = self._browser_frames.get(timeout=0.5)
                except Empty:
                    continue
                stream.write(json.dumps(packet, ensure_ascii=False) + "\n")
                stream.flush()
        except (BrokenPipeError, OSError):
            return
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _drain_stderr(self, stream: object) -> None:
        for line in stream:  # type: ignore[union-attr]
            value = str(line).strip()
            if value:
                self._stderr_tail.append(value[-1000:])
                logger.info("fall worker: %s", value)

    def _handle_message(self, device_id: str, message: dict[str, object]) -> None:
        kind = str(message.get("type", ""))
        self._write_alignment_shadow_log(message, kind=kind)
        rppg_status = self._parse_rppg_status(message.get("rppg"))
        alignment_snapshot = self._parse_alignment_snapshot(message)
        if kind == "status":
            raw_state = str(message.get("state", "STARTING"))
            state = (
                FallLiveState.LOADING_MODELS
                if raw_state == "LOADING_MODELS"
                else FallLiveState.STARTING
            )
            self._set_status(
                state=state,
                model_device=self._optional_str(message.get("device")),
                model_version=self._optional_str(message.get("model_version")),
                error=None,
                **({"rppg": rppg_status} if rppg_status is not None else {}),
            )
            return

        if kind == "stream":
            raw_state = str(message.get("state", "CONNECTED"))
            if raw_state in {"CONNECTING", "RECONNECTING"}:
                detail = self._optional_str(message.get("message")) or "OpenSDK视频流已停止返回画面"
                self._set_status(
                    state=FallLiveState.CONNECTING,
                    input_state=FallLiveInputState.WAITING,
                    input_message=(
                        "视频流中断，正在自动重连"
                        if raw_state == "RECONNECTING"
                        else "正在连接OpenSDK并等待视频首帧"
                    ),
                    target_present=False,
                    training_input_ready=False,
                    risk_score=None,
                    risk_level=None,
                    positive_votes=None,
                    torso_inclination_deg=None,
                    com_proxy_relative_change=None,
                    yaw_delta_deg=None,
                    pose_quality=None,
                    last_prediction_at=None,
                    frames_ready=0,
                    source_window_frames=0,
                    valid_pose_frames=0,
                    effective_sample_fps=0.0,
                    mean_keypoint_confidence=None,
                    latest_keypoint_confidence=None,
                    queue_depth=0,
                    error=detail if raw_state == "RECONNECTING" else None,
                )
                return
            self._set_status(
                state=FallLiveState.RUNNING,
                source_fps=self._optional_float(message.get("source_fps")),
                input_state=FallLiveInputState.WAITING,
                input_message="正在建立3秒人体姿态窗口",
                training_input_ready=False,
                error=None,
                **({"rppg": rppg_status} if rppg_status is not None else {}),
            )
            return
        if kind == "progress":
            self._set_status(
                frames_ready=self._int(message.get("frames_ready")),
                captured_frames=self._browser_received,
                processed_frames=self._int(message.get("processed_frames")),
                queue_dropped_frames=self._browser_dropped,
                dropped_frames=self._int(message.get("dropped_frames")) + self._browser_dropped,
                queue_depth=self._browser_frames.qsize(),
            )
            return
        if kind == "frame_error":
            invalid = self._int(message.get("invalid"))
            self._set_status(
                invalid_image_frames=invalid,
                queue_dropped_frames=self._browser_dropped,
                dropped_frames=invalid + self._browser_dropped,
                error=self._optional_str(message.get("message")),
            )
            return
        if kind == "error":
            detail = self._optional_str(message.get("message")) or "fall worker error"
            logger.warning("fall worker reported error: %s", detail)
            self._set_status(
                state=FallLiveState.ERROR,
                error=detail,
            )
            return
        if kind == "input_status":
            try:
                input_state = FallLiveInputState(str(message.get("input_state", "WAITING")))
            except ValueError:
                input_state = FallLiveInputState.WAITING
            worker_queue_dropped = self._int(
                message.get("queue_dropped_frames", message.get("dropped_frames"))
            )
            queue_dropped = worker_queue_dropped + self._browser_dropped
            invalid = self._int(message.get("invalid_image_frames"))
            no_person = self._int(message.get("no_person_frames"))
            low_confidence = self._int(message.get("low_confidence_frames"))
            self._set_status(
                state=FallLiveState.RUNNING,
                input_state=input_state,
                input_message=(
                    self._optional_str(message.get("input_message")) or "实时输入暂不满足推理要求"
                ),
                target_present=bool(message.get("target_present", False)),
                training_input_ready=bool(message.get("training_input_ready", False)),
                risk_score=None,
                risk_level=None,
                positive_votes=None,
                torso_inclination_deg=None,
                com_proxy_relative_change=None,
                yaw_delta_deg=None,
                pose_quality=None,
                last_prediction_at=None,
                frames_ready=min(
                    max(
                        2,
                        self._int(message.get("required_source_frames"))
                        or self.settings.fall_live_required_source_frames,
                    ),
                    self._int(message.get("source_window_frames")),
                ),
                source_window_frames=self._int(message.get("source_window_frames")),
                valid_pose_frames=self._int(message.get("valid_pose_frames")),
                required_source_frames=max(
                    2,
                    self._int(message.get("required_source_frames"))
                    or self.settings.fall_live_required_source_frames,
                ),
                effective_sample_fps=(
                    self._optional_float(message.get("effective_sample_fps")) or 0.0
                ),
                mean_keypoint_confidence=self._optional_float(
                    message.get("mean_keypoint_confidence")
                ),
                latest_keypoint_confidence=self._optional_float(
                    message.get("latest_keypoint_confidence")
                ),
                max_source_gap_frames=self._int(message.get("max_source_gap_frames")),
                captured_frames=(
                    self._browser_received
                    if self.settings.fall_live_source_mode == "browser_capture"
                    else self._int(message.get("captured_frames"))
                ),
                processed_frames=self._int(message.get("processed_frames")),
                queue_dropped_frames=queue_dropped,
                invalid_image_frames=invalid,
                no_person_frames=no_person,
                low_confidence_frames=low_confidence,
                # A usable image without a qualified person is a model-quality
                # rejection, not a transport/frame drop. Keep those counters
                # separate so the operator can diagnose the real bottleneck.
                dropped_frames=queue_dropped + invalid,
                queue_depth=(
                    self._browser_frames.qsize()
                    if self.settings.fall_live_source_mode == "browser_capture"
                    else self._int(message.get("queue_depth"))
                ),
                processing_fps=self._optional_float(message.get("processing_fps")),
                pipeline_latency_seconds=self._optional_float(
                    message.get("pipeline_latency_seconds")
                ),
                alignment_snapshot=alignment_snapshot,
                **({"rppg": rppg_status} if rppg_status is not None else {}),
                error=None,
            )
            return
        if kind != "prediction":
            return
        occurred_at = self._parse_datetime(message.get("occurred_at"))
        level = RiskLevel(str(message.get("risk_level", "LOW")))
        score = float(message.get("risk_score", 0.0))
        worker_queue_dropped = self._int(
            message.get("queue_dropped_frames", message.get("dropped_frames"))
        )
        queue_dropped = worker_queue_dropped + self._browser_dropped
        invalid = self._int(message.get("invalid_image_frames"))
        no_person = self._int(message.get("no_person_frames"))
        low_confidence = self._int(message.get("low_confidence_frames"))
        self._set_status(
            state=FallLiveState.RUNNING,
            input_state=FallLiveInputState.READY,
            input_message=(
                self._optional_str(message.get("input_message")) or "输入满足实时部署模型要求"
            ),
            target_present=True,
            training_input_ready=True,
            risk_score=score,
            risk_level=level,
            positive_votes=self._int(message.get("positive_votes")),
            torso_inclination_deg=self._optional_float(message.get("torso_inclination_deg")),
            com_proxy_relative_change=self._optional_float(
                message.get("com_proxy_relative_change")
            ),
            yaw_delta_deg=self._optional_signed_float(message.get("yaw_delta_deg")),
            pose_quality=self._optional_float(message.get("pose_quality")),
            frames_ready=max(
                2,
                self._int(message.get("required_source_frames"))
                or self.settings.fall_live_required_source_frames,
            ),
            captured_frames=(
                self._browser_received
                if self.settings.fall_live_source_mode == "browser_capture"
                else self._int(message.get("captured_frames"))
            ),
            processed_frames=self._int(message.get("processed_frames")),
            source_window_frames=self._int(message.get("source_window_frames")),
            valid_pose_frames=self._int(message.get("valid_pose_frames")),
            required_source_frames=max(
                2,
                self._int(message.get("required_source_frames"))
                or self.settings.fall_live_required_source_frames,
            ),
            effective_sample_fps=(self._optional_float(message.get("effective_sample_fps")) or 0.0),
            mean_keypoint_confidence=self._optional_float(message.get("mean_keypoint_confidence")),
            latest_keypoint_confidence=self._optional_float(
                message.get("latest_keypoint_confidence")
            ),
            max_source_gap_frames=self._int(message.get("max_source_gap_frames")),
            queue_dropped_frames=queue_dropped,
            invalid_image_frames=invalid,
            no_person_frames=no_person,
            low_confidence_frames=low_confidence,
            dropped_frames=queue_dropped + invalid,
            queue_depth=(
                self._browser_frames.qsize()
                if self.settings.fall_live_source_mode == "browser_capture"
                else self._int(message.get("queue_depth"))
            ),
            processing_fps=self._optional_float(message.get("processing_fps")),
            pipeline_latency_seconds=self._optional_float(message.get("pipeline_latency_seconds")),
            last_prediction_at=occurred_at,
            alignment_snapshot=alignment_snapshot,
            **({"rppg": rppg_status} if rppg_status is not None else {}),
            error=None,
        )
        if bool(message.get("alert", False)) and self.settings.fall_live_risk_events_enabled:
            self._persist_alert(device_id, occurred_at, message)

    def _write_alignment_shadow_log(
        self,
        message: dict[str, object],
        *,
        kind: str,
    ) -> None:
        """Append Camera geometry evidence without affecting live decisions."""

        output = self.settings.fall_alignment_shadow_log_path
        if output is None or kind not in {"input_status", "prediction"}:
            return
        if message.get("frame_id") is None:
            return
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(message, ensure_ascii=False) + "\n")
        except OSError as exc:
            # Evidence logging is optional. A disk/path problem must not stop
            # Camera inference or change alert behavior.
            logger.warning("Camera alignment shadow log write failed: %s", exc)

    @classmethod
    def _parse_alignment_snapshot(
        cls,
        message: dict[str, object],
    ) -> CameraAlignmentSnapshot | None:
        frame_id = cls._optional_str(message.get("frame_id"))
        timestamp_value = message.get("source_timestamp") or message.get("captured_at")
        if frame_id is None or timestamp_value is None:
            return None

        def finite_values(value: object, length: int) -> tuple[float, ...] | None:
            if not isinstance(value, (list, tuple)) or len(value) != length:
                return None
            try:
                result = tuple(float(item) for item in value)
            except (TypeError, ValueError):
                return None
            return result if all(math.isfinite(item) for item in result) else None

        bbox = finite_values(message.get("bbox_xyxy"), 4)
        image_size_values = finite_values(message.get("image_size"), 2)
        image_size = (
            (max(1, int(image_size_values[0])), max(1, int(image_size_values[1])))
            if image_size_values is not None
            else None
        )
        footpoint: tuple[float, float] | None = None
        footpoint_source: str | None = None
        keypoints = message.get("keypoints_2d")
        if isinstance(keypoints, list) and len(keypoints) > 16:
            left = (
                finite_values(keypoints[15], len(keypoints[15]))
                if isinstance(keypoints[15], (list, tuple))
                else None
            )
            right = (
                finite_values(keypoints[16], len(keypoints[16]))
                if isinstance(keypoints[16], (list, tuple))
                else None
            )
            if left is not None and right is not None and len(left) >= 2 and len(right) >= 2:
                footpoint = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
                footpoint_source = "ANKLE_MIDPOINT_15_16"
        if footpoint is None and bbox is not None:
            footpoint = ((bbox[0] + bbox[2]) / 2.0, bbox[3])
            footpoint_source = "BBOX_BOTTOM_CENTER"

        confidence = cls._optional_float(
            message.get("camera_quality", message.get("latest_keypoint_confidence"))
        )
        try:
            return CameraAlignmentSnapshot(
                frame_id=frame_id,
                source_timestamp=cls._parse_datetime(timestamp_value),
                camera_person_id=(0 if bool(message.get("detected")) else None),
                detected=bool(message.get("detected")),
                image_size=image_size,
                bbox_xyxy=bbox,
                footpoint_uv=footpoint,
                footpoint_confidence=max(0.0, min(1.0, confidence or 0.0)),
                footpoint_source=footpoint_source,
            )
        except Exception as exc:
            logger.warning("Ignoring invalid Camera alignment snapshot: %s", str(exc)[-300:])
            return None

    @staticmethod
    def _parse_rppg_status(value: object) -> RppgLiveStatus | None:
        if not isinstance(value, dict):
            return None
        try:
            return RppgLiveStatus.model_validate(value)
        except Exception as exc:
            logger.warning("Ignoring invalid rPPG worker payload: %s", str(exc)[-500:])
            return RppgLiveStatus(
                enabled=bool(value.get("enabled", False)),
                available=False,
                quality_reason="RPPG_PAYLOAD_INVALID",
                error_hint=str(exc)[-300:],
            )

    def _persist_alert(
        self,
        device_id: str,
        occurred_at: datetime,
        message: dict[str, object],
    ) -> None:
        import time

        now = time.monotonic()
        if now - self._last_event_monotonic < self.settings.fall_live_event_cooldown_seconds:
            return
        with self.session_factory() as db:
            monitoring = MonitoringService(db)
            session = monitoring.get_current_session(device_id=device_id)
            if session is None and self.settings.fall_live_auto_create_session:
                session = monitoring.create_session(
                    MonitoringSessionCreate(
                        mode=MonitoringMode.LIVE,
                        device_id=device_id,
                        enabled_modules=[RiskModule.FALL],
                    )
                )
            if session is None:
                logger.warning("Live fall alert ignored because no session is running")
                return
            context = AdapterContext(session_id=session.id, device_id=device_id)
            finding = AlgorithmFinding(
                module=RiskModule.FALL,
                event_type="PRE_FALL_RISK",
                occurred_at=occurred_at,
                risk_score=float(message.get("risk_score", 0.0)),
                risk_level=RiskLevel.HIGH,
                summary="实时画面检测到连续失衡特征，达到跌倒预警阈值",
                evidence=[
                    EvidenceItem(
                        code="positive_votes",
                        label="集成模型阳性票数",
                        value=self._int(message.get("positive_votes")),
                    ),
                    EvidenceItem(
                        code="detected_frames",
                        label="窗口内有效人体帧",
                        value=self._int(message.get("detected_frames")),
                        unit="帧",
                    ),
                    EvidenceItem(
                        code="pipeline_latency",
                        label="当前处理延迟",
                        value=self._optional_float(message.get("pipeline_latency_seconds")),
                        unit="秒",
                    ),
                    EvidenceItem(
                        code="dropped_frames",
                        label="本轮累计丢帧",
                        value=self._int(message.get("dropped_frames")),
                        unit="帧",
                    ),
                ],
                recommended_action="立即查看实时画面并确认老人状态",
                model_version=(
                    self._optional_str(message.get("model_version"))
                    or "biostgcn-stage2-unified-v1-ensemble6"
                ),
            )
            event = RiskEventFactory().create(finding, context)
            RiskEventService(db).save_event(event)
        self._last_event_monotonic = now
        self._set_status(last_event_id=event.event_id)

    def _set_status(self, **updates: object) -> None:
        updates["checked_at"] = datetime.now(timezone.utc)
        with self._status_lock:
            self._status = self._status.model_copy(update=updates)

    @staticmethod
    def _int(value: object) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_signed_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_str(value: object) -> str | None:
        if value is None:
            return None
        result = str(value).strip()
        return result or None

    @staticmethod
    def _parse_datetime(value: object) -> datetime:
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is not None:
                    return parsed
            except ValueError:
                pass
        return datetime.now(timezone.utc)
