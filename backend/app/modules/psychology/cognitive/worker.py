"""Independent long-running Cognitive Worker for wav2vec2 MMSE estimation."""

import argparse
import importlib
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from app.modules.psychology.cognitive.result_store import CognitiveResultStore
from app.modules.psychology.cognitive.schemas import (
    CognitiveAssessmentSnapshot,
    CognitiveInferenceJob,
)

logger = logging.getLogger(__name__)
DEFAULT_RUNTIME_ROOT = Path(__file__).resolve().parent / "runtime"
UTC_TZ = timezone.utc  # noqa: UP017 - Python 3.10 worker compatibility.


@dataclass(frozen=True, slots=True)
class MmseInferenceResult:
    estimated_mmse_score: float
    audio_window_count: int


class MmseRunner(Protocol):
    def infer(self, wav_path: Path) -> MmseInferenceResult: ...


class Wav2Vec2MmseRunner:
    """Loads the verified wav2vec2 model once and reuses it for every job."""

    def __init__(
        self,
        *,
        model_dir: Path,
        device: str = "cpu",
        window_seconds: int = 15,
        step_seconds: int = 10,
        score_mean: float = 23.0280,
        score_std: float = 7.1844,
    ) -> None:
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Cognitive model directory does not exist: {model_dir}")
        self._torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        self._librosa = importlib.import_module("librosa")
        self._numpy = importlib.import_module("numpy")
        if device.startswith("cuda") and not bool(self._torch.cuda.is_available()):
            raise RuntimeError(f"Cognitive device {device} requested but CUDA is unavailable")
        self._device = self._torch.device(device)
        self._model = transformers.AutoModelForAudioClassification.from_pretrained(
            str(model_dir),
            num_labels=1,
            problem_type="regression",
        ).to(self._device)
        self._model.eval()
        self._feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(str(model_dir))
        self._sample_rate = int(self._feature_extractor.sampling_rate)
        if self._sample_rate != 16_000:
            raise RuntimeError(
                f"Cognitive model requires {self._sample_rate} Hz; expected 16000 Hz"
            )
        self._window_samples = self._sample_rate * window_seconds
        self._step_samples = self._sample_rate * step_seconds
        self._score_mean = score_mean
        self._score_std = score_std
        logger.info(
            "Cognitive model loaded model=%s requested_device=%s effective_device=%s",
            model_dir,
            device,
            self._device,
        )

    def infer(self, wav_path: Path) -> MmseInferenceResult:
        audio, _ = self._librosa.load(str(wav_path), sr=self._sample_rate, mono=True)
        starts = list(
            range(
                0,
                max(1, len(audio) - self._window_samples + 1),
                self._step_samples,
            )
        )
        if not starts:
            starts = [0]
        predictions: list[float] = []
        with self._torch.no_grad():
            for start in starts:
                segment = audio[start : start + self._window_samples]
                if len(segment) < self._window_samples:
                    segment = self._numpy.pad(segment, (0, self._window_samples - len(segment)))
                inputs = self._feature_extractor(
                    segment,
                    sampling_rate=self._sample_rate,
                    return_tensors="pt",
                    do_normalize=True,
                ).to(self._device)
                predictions.append(float(self._model(**inputs).logits.item()))
        standardized_mean = float(self._numpy.asarray(predictions).mean())
        score = standardized_mean * self._score_std + self._score_mean
        return MmseInferenceResult(
            estimated_mmse_score=score,
            audio_window_count=len(predictions),
        )


class CognitiveWorker:
    def __init__(
        self,
        *,
        runtime_root: Path,
        runner: MmseRunner,
        poll_interval_seconds: float = 0.5,
    ) -> None:
        self._store = CognitiveResultStore(runtime_root)
        self._runner = runner
        self._poll_interval_seconds = poll_interval_seconds

    def run_forever(self) -> None:
        self._store.prepare()
        logger.info("Cognitive Worker started runtime_root=%s", self._store.root)
        while True:
            if not self.process_next():
                time.sleep(self._poll_interval_seconds)

    def process_next(self) -> bool:
        self._store.prepare()
        manifest_path = self._next_manifest()
        if manifest_path is None:
            return False
        try:
            job = self._store.read_job(manifest_path)
        except (OSError, ValueError):
            logger.exception("Discarding invalid Cognitive job manifest: %s", manifest_path)
            manifest_path.unlink(missing_ok=True)
            return True
        try:
            wav_path = self._claim_audio(job)
        except RuntimeError as exc:
            logger.error("%s", exc)
            self._write_failed(
                job,
                completed_at=datetime.now(UTC_TZ),
                failure_code="audio_missing",
                failure_message=str(exc),
            )
            manifest_path.unlink(missing_ok=True)
            return True
        try:
            self._process(job, wav_path)
        finally:
            wav_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
        return True

    def _next_manifest(self) -> Path | None:
        processing = sorted(self._store.processing_dir.glob("*.json"))
        if processing:
            return processing[0]
        for source in sorted(self._store.inbox_dir.glob("*.json")):
            destination = self._store.processing_dir / source.name
            try:
                os.replace(source, destination)
            except FileNotFoundError:
                continue
            return destination
        return None

    def _claim_audio(self, job: CognitiveInferenceJob) -> Path:
        destination = self._store.processing_dir / f"{job.assessment_id}.wav"
        if destination.is_file():
            return destination
        source = self._store.inbox_dir / f"{job.assessment_id}.wav"
        try:
            os.replace(source, destination)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Cognitive job audio is missing: {job.assessment_id}") from exc
        return destination

    def _process(self, job: CognitiveInferenceJob, wav_path: Path) -> None:
        completed_at = datetime.now(UTC_TZ)
        if completed_at > job.expires_at:
            self._write_failed(
                job,
                completed_at=completed_at,
                failure_code="job_expired",
                failure_message="Cognitive inference job expired before processing",
            )
            return
        try:
            result = self._runner.infer(wav_path)
        except Exception as exc:
            logger.exception("Cognitive inference failed assessment=%s", job.assessment_id)
            self._write_failed(
                job,
                completed_at=datetime.now(UTC_TZ),
                failure_code="inference_failed",
                failure_message=str(exc),
            )
            return

        completed_at = datetime.now(UTC_TZ)
        if not 0.0 <= result.estimated_mmse_score <= 30.0:
            self._write_failed(
                job,
                completed_at=completed_at,
                failure_code="score_out_of_range",
                failure_message="Model output is outside the MMSE contract range 0-30",
                estimated_mmse_score=result.estimated_mmse_score,
                audio_window_count=result.audio_window_count,
            )
            logger.error(
                "Cognitive score rejected assessment=%s score=%s",
                job.assessment_id,
                result.estimated_mmse_score,
            )
            return

        self._store.write_snapshot(
            CognitiveAssessmentSnapshot(
                assessment_id=job.assessment_id,
                subject_key=job.subject_key,
                session_id=job.session_id,
                status="completed",
                window_started_at=job.window_started_at,
                window_ended_at=job.window_ended_at,
                effective_speech_seconds=job.effective_speech_seconds,
                estimated_mmse_score=result.estimated_mmse_score,
                audio_window_count=result.audio_window_count,
                completed_at=completed_at,
            )
        )
        logger.info(
            "Cognitive assessment completed assessment=%s score=%.3f windows=%d",
            job.assessment_id,
            result.estimated_mmse_score,
            result.audio_window_count,
        )

    def _write_failed(
        self,
        job: CognitiveInferenceJob,
        *,
        completed_at: datetime,
        failure_code: str,
        failure_message: str,
        estimated_mmse_score: float | None = None,
        audio_window_count: int = 0,
    ) -> None:
        self._store.write_snapshot(
            CognitiveAssessmentSnapshot(
                assessment_id=job.assessment_id,
                subject_key=job.subject_key,
                session_id=job.session_id,
                status="failed",
                window_started_at=job.window_started_at,
                window_ended_at=job.window_ended_at,
                effective_speech_seconds=job.effective_speech_seconds,
                estimated_mmse_score=estimated_mmse_score,
                audio_window_count=audio_window_count,
                completed_at=completed_at,
                failure_code=failure_code,
                failure_message=failure_message,
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the long-lived Cognitive Worker")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=os.getenv("COGNITIVE_MODEL_DIR"),
        required=os.getenv("COGNITIVE_MODEL_DIR") is None,
        help="Verified wav2vec2_base_adress model directory",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(os.getenv("COGNITIVE_RUNTIME_ROOT", str(DEFAULT_RUNTIME_ROOT))),
    )
    parser.add_argument("--device", default=os.getenv("COGNITIVE_DEVICE", "cpu"))
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()
    runner = Wav2Vec2MmseRunner(model_dir=args.model_dir, device=args.device)
    worker = CognitiveWorker(
        runtime_root=args.runtime_root,
        runner=runner,
        poll_interval_seconds=args.poll_seconds,
    )
    if args.once:
        return 0 if worker.process_next() else 2
    worker.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
