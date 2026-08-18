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
        if not path.is_file():
            return None
        snapshot = PsychologyAssessmentSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        if snapshot.subject_key != subject_key:
            return None
        return snapshot

    def write(self, snapshot: PsychologyAssessmentSnapshot) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        destination = self._path_for(snapshot.subject_key)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-",
            suffix=".tmp",
            dir=self._root,
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
        return destination

    def _path_for(self, subject_key: str) -> Path:
        digest = hashlib.sha256(subject_key.encode("utf-8")).hexdigest()
        return self._root / f"{digest}.json"

