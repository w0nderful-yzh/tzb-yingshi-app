"""Atomic filesystem store for the latest assessment of each subject."""

import hashlib
import os
import tempfile
from pathlib import Path

from service.schemas import PsychologyAssessmentSnapshot


class LatestAssessmentStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def read(self, subject_key: str) -> PsychologyAssessmentSnapshot | None:
        path = self._path_for(subject_key)
        return self._read_path(path, subject_key)

    def read_latest_completed(self, subject_key: str) -> PsychologyAssessmentSnapshot | None:
        return self._read_path(
            self._completed_path_for(subject_key),
            subject_key,
        )

    def _read_path(
        self,
        path: Path,
        subject_key: str,
    ) -> PsychologyAssessmentSnapshot | None:
        if not path.is_file():
            return None
        snapshot = PsychologyAssessmentSnapshot.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if snapshot.subject_key != subject_key:
            return None
        return snapshot

    def write(self, snapshot: PsychologyAssessmentSnapshot) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        destination = self._path_for(snapshot.subject_key)
        completed_destination = self._completed_path_for(snapshot.subject_key)

        # Preserve the last completed observation before a new processing
        # snapshot replaces the current projection. This also migrates stores
        # created before the completed projection was introduced.
        if snapshot.status != "completed" and not completed_destination.is_file():
            previous = self.read(snapshot.subject_key)
            if previous is not None and previous.status == "completed":
                self._write_atomic(completed_destination, previous)

        self._write_atomic(destination, snapshot)
        if snapshot.status == "completed":
            self._write_atomic(completed_destination, snapshot)
        return destination

    def _write_atomic(
        self,
        destination: Path,
        snapshot: PsychologyAssessmentSnapshot,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                output.write(snapshot.model_dump_json(indent=2))
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _path_for(self, subject_key: str) -> Path:
        digest = hashlib.sha256(subject_key.encode("utf-8")).hexdigest()
        return self._root / f"{digest}.json"

    def _completed_path_for(self, subject_key: str) -> Path:
        digest = hashlib.sha256(subject_key.encode("utf-8")).hexdigest()
        return self._root / "completed" / f"{digest}.json"

