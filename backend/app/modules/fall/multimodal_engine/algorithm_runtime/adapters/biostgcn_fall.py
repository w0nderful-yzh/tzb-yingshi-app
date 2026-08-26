from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from app.modules.fall.multimodal_engine.algorithm_runtime.adapter import AlgorithmAdapter, AdapterState
from app.modules.fall.multimodal_engine.algorithm_runtime.contracts import AlgorithmFinding
from app.modules.fall.multimodal_engine.schemas.risk_event import EvidenceItem, RiskLevel, RiskModule


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


class BioSTGCNInferenceError(RuntimeError):
    pass


class BioSTGCNFileAdapter(AlgorithmAdapter[Path]):
    """Run the frozen RTMPose3D -> BioSTGCN Stage2 file pipeline.

    The research package stays isolated in its own Python environment.  This
    adapter only translates the frozen report into the existing internal
    AlgorithmFinding contract; it does not write the database itself.
    """

    module = RiskModule.FALL
    model_version = "biostgcn-stage2-unified-v1-ensemble6"

    def __init__(
        self,
        *,
        project_dir: Path,
        python_executable: Path,
        job_dir: Path,
        device: str = "cpu",
        timeout_seconds: int = 900,
        process_runner: ProcessRunner = subprocess.run,
    ) -> None:
        super().__init__()
        self.project_dir = Path(project_dir).resolve()
        self.python_executable = Path(python_executable).resolve()
        self.job_dir = Path(job_dir).resolve()
        self.device = device
        self.timeout_seconds = timeout_seconds
        self._process_runner = process_runner
        self.last_report: dict[str, Any] | None = None
        self.command: list[str] | None = None

    def load(self) -> None:
        missing: list[str] = []
        if not self.project_dir.is_dir():
            missing.append(f"project directory: {self.project_dir}")
        if not self.python_executable.is_file():
            missing.append(f"pose Python: {self.python_executable}")
        checkpoint_dir = self.project_dir / "checkpoints"
        checkpoint_count = len(list(checkpoint_dir.glob("unified_fold*/stage2_best.pt")))
        if checkpoint_count != 6:
            missing.append(f"frozen checkpoints: expected 6, found {checkpoint_count}")
        if missing:
            self._state = AdapterState.ERROR
            raise BioSTGCNInferenceError("; ".join(missing))
        self.job_dir.mkdir(parents=True, exist_ok=True)
        super().load()

    def consume(self, input_data: Path) -> AlgorithmFinding:
        self._ensure_running()
        video_path = Path(input_data).resolve()
        if not video_path.is_file():
            raise BioSTGCNInferenceError(f"video does not exist: {video_path}")

        report_path = self.job_dir / "prediction.json"
        pose_path = self.job_dir / "rtmpose3d.npy"
        command = [
            str(self.python_executable),
            "-m",
            "fall_inference.cli",
            "--video",
            str(video_path),
            "--device",
            self.device,
            "--save-pose-npy",
            str(pose_path),
            "--output",
            str(report_path),
        ]
        self.command = command
        try:
            completed = self._process_runner(
                command,
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._state = AdapterState.ERROR
            raise BioSTGCNInferenceError(
                f"fall inference timed out after {self.timeout_seconds}s"
            ) from exc
        if completed.returncode != 0:
            self._state = AdapterState.ERROR
            detail = (completed.stderr or completed.stdout or "unknown error")[-1500:]
            raise BioSTGCNInferenceError(f"fall inference failed: {detail}")
        if not report_path.is_file():
            self._state = AdapterState.ERROR
            raise BioSTGCNInferenceError("fall inference did not create prediction.json")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.last_report = report
        peak = report["peak"]
        extraction = report.get("input", {}).get("pose_extraction") or {}
        risk_level = RiskLevel(str(peak["risk_level"]))
        is_alert = bool(peak["alert"])
        start_frame = int(peak["start_frame"])
        end_frame = int(peak["end_frame"])
        fps = float(report.get("input", {}).get("fps_for_event_timing") or 30.0)
        evidence = [
            EvidenceItem(
                code="risk_score",
                label="BioSTGCN六折平均风险分数",
                value=float(peak["risk_score"]),
            ),
            EvidenceItem(
                code="ensemble_votes",
                label="超过折内阈值的模型数",
                value=int(peak["positive_votes"]),
                unit="/6",
            ),
            EvidenceItem(
                code="window_start",
                label="风险窗口开始时间",
                value=round(start_frame / fps, 3),
                unit="s",
            ),
            EvidenceItem(
                code="window_end",
                label="风险窗口结束时间",
                value=round(end_frame / fps, 3),
                unit="s",
            ),
        ]
        if extraction:
            evidence.extend(
                [
                    EvidenceItem(
                        code="pose_detected_frames",
                        label="成功提取骨架帧数",
                        value=int(extraction.get("detected_frames", 0)),
                        unit="frame",
                    ),
                    EvidenceItem(
                        code="pose_missing_ratio",
                        label="骨架缺失比例",
                        value=float(extraction.get("missing_ratio", 0.0)),
                    ),
                ]
            )
        return AlgorithmFinding(
            module=self.module,
            event_type="PRE_FALL_RISK" if is_alert else "FALL_RISK_DIAGNOSTIC",
            occurred_at=datetime.now(timezone.utc),
            risk_score=float(peak["risk_score"]),
            risk_level=risk_level,
            summary=(
                f"真实视频BioSTGCN分析：score={float(peak['risk_score']):.3f}，"
                f"ensemble votes={int(peak['positive_votes'])}/6"
            ),
            evidence=evidence,
            recommended_action=(
                "立即关注老人状态并核查监控画面"
                if is_alert
                else "低/中风险诊断结果仅供联调记录，继续监测"
            ),
            clip_path=str(video_path),
            model_version=self.model_version,
        )


__all__ = ["BioSTGCNFileAdapter", "BioSTGCNInferenceError"]
