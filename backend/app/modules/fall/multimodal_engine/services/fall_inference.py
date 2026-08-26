from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from app.modules.fall.multimodal_engine.algorithm_runtime import AdapterContext, RiskEventFactory
from app.modules.fall.multimodal_engine.algorithm_runtime.adapters.biostgcn_fall import BioSTGCNFileAdapter
from app.modules.fall.multimodal_engine.database.session import session_scope
from app.modules.fall.multimodal_engine.integrations.ezviz.live_capture import EzvizLiveCapture
from app.modules.fall.multimodal_engine.schemas.fall_inference import (
    FallInferenceJobResponse,
    FallInferenceJobStatus,
    FallInferenceSystemStatus,
)
from app.modules.fall.multimodal_engine.services.risk_event import RiskEventService


ALLOWED_VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}


@dataclass
class _Job:
    job_id: str
    session_id: str
    device_id: str
    filename: str
    video_path: Path
    job_dir: Path
    record_non_alert_test_event: bool
    status: FallInferenceJobStatus = FallInferenceJobStatus.QUEUED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    event_id: str | None = None
    error: str | None = None


class FallInferenceBusyError(RuntimeError):
    pass


class FallInferenceJobNotFoundError(LookupError):
    pass


class FallInferenceJobService:
    """In-process job ledger for isolated BioSTGCN video inference."""

    def __init__(
        self,
        *,
        project_dir: Path,
        python_executable: Path,
        runtime_dir: Path,
        device: str,
        timeout_seconds: int,
        live_capture: EzvizLiveCapture | None = None,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.python_executable = Path(python_executable).resolve()
        self.runtime_dir = Path(runtime_dir).resolve()
        self.device = device
        self.timeout_seconds = timeout_seconds
        self.live_capture = live_capture
        self._jobs: dict[str, _Job] = {}
        self._lock = Lock()

    def system_status(self) -> FallInferenceSystemStatus:
        checkpoint_count = len(
            list((self.project_dir / "checkpoints").glob("unified_fold*/stage2_best.pt"))
        )
        project_exists = self.project_dir.is_dir()
        python_exists = self.python_executable.is_file()
        ready = project_exists and python_exists and checkpoint_count == 6
        live_capture_ready = ready and bool(self.live_capture and self.live_capture.ready)
        return FallInferenceSystemStatus(
            ready=ready,
            model_version="biostgcn-stage2-unified-v1-ensemble6",
            input_format="RTMPose3D (T,133,3) -> windows (90,33,3) + biomechanics (90,45)",
            execution_mode="isolated_subprocess_offline_video",
            project_dir_exists=project_exists,
            python_exists=python_exists,
            checkpoints_found=checkpoint_count,
            ezviz_live_capture_ready=live_capture_ready,
            note=(
                "真实离线/上传视频与萤石标准云流按需取片均已接入"
                if live_capture_ready
                else "真实离线/上传视频已接入；萤石云流采集运行环境尚未就绪"
            ),
        )

    def allocate_upload(self, filename: str) -> tuple[str, Path, Path]:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise ValueError(
                f"unsupported video extension {suffix or '(none)'}; "
                f"allowed: {', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}"
            )
        job_id = f"fall-job-{uuid4().hex}"
        job_dir = self.runtime_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        return job_id, job_dir / f"input{suffix}", job_dir

    def allocate_live_capture(self) -> tuple[str, Path, Path]:
        job_id = f"fall-live-{uuid4().hex}"
        job_dir = self.runtime_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        return job_id, job_dir / "ezviz-live.mp4", job_dir

    def create_job(
        self,
        *,
        job_id: str,
        session_id: str,
        device_id: str,
        filename: str,
        video_path: Path,
        job_dir: Path,
        record_non_alert_test_event: bool,
    ) -> FallInferenceJobResponse:
        with self._lock:
            if any(
                job.status in {FallInferenceJobStatus.QUEUED, FallInferenceJobStatus.RUNNING}
                for job in self._jobs.values()
            ):
                raise FallInferenceBusyError("another fall inference job is running")
            job = _Job(
                job_id=job_id,
                session_id=session_id,
                device_id=device_id,
                filename=filename,
                video_path=video_path,
                job_dir=job_dir,
                record_non_alert_test_event=record_non_alert_test_event,
            )
            self._jobs[job_id] = job
            return self._response(job)

    def run_job(self, job_id: str) -> None:
        job = self._mark_running(job_id)
        self._run_inference(job)

    def run_live_job(
        self,
        job_id: str,
        *,
        stream_url: str,
        capture_seconds: int,
    ) -> None:
        job = self._mark_running(job_id)
        if self.live_capture is None:
            self._fail_job(job, RuntimeError("live capture runtime is not configured"))
            return
        try:
            capture_report = self.live_capture.capture(
                stream_url,
                job.video_path,
                duration_seconds=capture_seconds,
            )
        except Exception as exc:
            self._fail_job(job, exc)
            return
        self._run_inference(job, capture_report=capture_report)

    def _mark_running(self, job_id: str) -> _Job:
        with self._lock:
            job = self._get(job_id)
            job.status = FallInferenceJobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            return job

    def _run_inference(
        self,
        job: _Job,
        *,
        capture_report: dict[str, Any] | None = None,
    ) -> None:
        adapter = BioSTGCNFileAdapter(
            project_dir=self.project_dir,
            python_executable=self.python_executable,
            job_dir=job.job_dir,
            device=self.device,
            timeout_seconds=self.timeout_seconds,
        )
        context = AdapterContext(session_id=job.session_id, device_id=job.device_id)
        try:
            adapter.load()
            adapter.start(context)
            finding = adapter.consume(job.video_path)
            report = adapter.last_report or {}
            event_id: str | None = None
            if finding.risk_level.value == "HIGH" or job.record_non_alert_test_event:
                event = RiskEventFactory().create(finding, context)
                with session_scope() as db:
                    saved = RiskEventService(db).save_event(event)
                    event_id = saved.event_id
            peak = report.get("peak") or {}
            extraction = (report.get("input") or {}).get("pose_extraction") or {}
            result = {
                "model_version": report.get("model_version"),
                "input_shape": (report.get("input") or {}).get("shape"),
                "window_count": report.get("window_count"),
                "risk_score": peak.get("risk_score"),
                "risk_level": peak.get("risk_level"),
                "alert": peak.get("alert"),
                "positive_votes": peak.get("positive_votes"),
                "detected_frames": extraction.get("detected_frames"),
                "missing_ratio": extraction.get("missing_ratio"),
                "prediction_report": str(job.job_dir / "prediction.json"),
                "pose_cache": str(job.job_dir / "rtmpose3d.npy"),
                "event_recorded": event_id is not None,
                "input_source": "EZVIZ_LIVE" if capture_report is not None else "UPLOAD",
                "live_capture": capture_report,
            }
            with self._lock:
                job.result = result
                job.event_id = event_id
                job.status = FallInferenceJobStatus.COMPLETED
                job.finished_at = datetime.now(timezone.utc)
        except Exception as exc:
            self._fail_job(job, exc)
        finally:
            if adapter.state.value == "RUNNING":
                adapter.stop()

    def _fail_job(self, job: _Job, exc: Exception) -> None:
        with self._lock:
            job.status = FallInferenceJobStatus.FAILED
            job.error = str(exc)[-2000:]
            job.finished_at = datetime.now(timezone.utc)

    def get_job(self, job_id: str) -> FallInferenceJobResponse:
        with self._lock:
            return self._response(self._get(job_id))

    def _get(self, job_id: str) -> _Job:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise FallInferenceJobNotFoundError(job_id) from exc

    @staticmethod
    def _response(job: _Job) -> FallInferenceJobResponse:
        return FallInferenceJobResponse(
            job_id=job.job_id,
            status=job.status,
            session_id=job.session_id,
            device_id=job.device_id,
            filename=job.filename,
            record_non_alert_test_event=job.record_non_alert_test_event,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            result=job.result,
            event_id=job.event_id,
            error=job.error,
        )


__all__ = [
    "ALLOWED_VIDEO_SUFFIXES",
    "FallInferenceBusyError",
    "FallInferenceJobNotFoundError",
    "FallInferenceJobService",
]
