"""Atomic filesystem storage for Cognitive assessment snapshots and jobs."""

import hashlib
import os
import tempfile
from pathlib import Path

from app.modules.psychology.cognitive.schemas import (
    CognitiveAssessmentSnapshot,
    CognitiveInferenceJob,
)


class CognitiveResultStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def inbox_dir(self) -> Path:
        return self.root / "inbox"

    @property
    def processing_dir(self) -> Path:
        return self.root / "processing"

    @property
    def latest_dir(self) -> Path:
        return self.root / "latest"

    def prepare(self) -> None:
        for path in (self.inbox_dir, self.processing_dir, self.latest_dir):
            path.mkdir(parents=True, exist_ok=True)

    def read_latest(self, subject_key: str) -> CognitiveAssessmentSnapshot | None:
        return self._read_snapshot(self._latest_path(subject_key), subject_key)

    def read_latest_completed(self, subject_key: str) -> CognitiveAssessmentSnapshot | None:
        return self._read_snapshot(self._completed_path(subject_key), subject_key)

    def write_snapshot(self, snapshot: CognitiveAssessmentSnapshot) -> Path:
        self.prepare()
        destination = self._latest_path(snapshot.subject_key)
        completed_destination = self._completed_path(snapshot.subject_key)

        if snapshot.status != "completed" and not completed_destination.is_file():
            previous = self.read_latest(snapshot.subject_key)
            if previous is not None and previous.status == "completed":
                self._write_model_atomic(completed_destination, previous)

        self._write_model_atomic(destination, snapshot)
        if snapshot.status == "completed":
            self._write_model_atomic(completed_destination, snapshot)
        return destination

    def publish_job(self, job: CognitiveInferenceJob, wav_bytes: bytes) -> tuple[Path, Path]:
        """Publish WAV first and manifest last so the Worker never sees a partial job."""

        self.prepare()
        wav_path = self.inbox_dir / f"{job.assessment_id}.wav"
        manifest_path = self.inbox_dir / f"{job.assessment_id}.json"
        self._write_bytes_atomic(wav_path, wav_bytes)
        try:
            self._write_model_atomic(manifest_path, job)
        except BaseException:
            wav_path.unlink(missing_ok=True)
            raise
        return manifest_path, wav_path

    @staticmethod
    def read_job(path: Path) -> CognitiveInferenceJob:
        return CognitiveInferenceJob.model_validate_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_snapshot(
        path: Path,
        subject_key: str,
    ) -> CognitiveAssessmentSnapshot | None:
        if not path.is_file():
            return None
        snapshot = CognitiveAssessmentSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        if snapshot.subject_key != subject_key:
            return None
        return snapshot

    @staticmethod
    def _write_model_atomic(
        destination: Path, model: CognitiveAssessmentSnapshot | CognitiveInferenceJob
    ) -> None:
        CognitiveResultStore._write_bytes_atomic(
            destination,
            (model.model_dump_json(indent=2) + "\n").encode("utf-8"),
        )

    @staticmethod
    def _write_bytes_atomic(destination: Path, content: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _latest_path(self, subject_key: str) -> Path:
        return self.latest_dir / f"{self._subject_digest(subject_key)}.json"

    def _completed_path(self, subject_key: str) -> Path:
        return self.latest_dir / "completed" / f"{self._subject_digest(subject_key)}.json"

    @staticmethod
    def _subject_digest(subject_key: str) -> str:
        return hashlib.sha256(subject_key.encode("utf-8")).hexdigest()
