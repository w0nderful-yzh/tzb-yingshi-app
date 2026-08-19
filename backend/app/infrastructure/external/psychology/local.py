"""Read-only local source for psychology latest-assessment snapshots.

Reads the latest assessment JSON directly from the algorithm package's
LatestAssessmentStore (home_out/latest/<sha256(subject_key)>.json) instead of
going through the legacy HTTP projection (:8020). Read-only by design: this
never triggers OpenFace/MCCL/XGBoost inference.
"""

import hashlib
from pathlib import Path

from app.modules.psychology.ports import PsychologySourceError
from app.modules.psychology.source_schemas import PsychologySourceSnapshot

_HOME_DETECTION_PKG = (
    Path(__file__).resolve().parents[3]
    / "modules"
    / "psychology"
    / "home_detection_pkg"
)


class LocalPsychologySource:
    """Reads the most recent completed psychology assessment from the local store."""

    def __init__(self, store_root: Path | None = None) -> None:
        self._store_root = store_root or (_HOME_DETECTION_PKG / "home_out" / "latest")

    async def get_latest_assessment(self, *, subject_key: str) -> PsychologySourceSnapshot:
        digest = hashlib.sha256(subject_key.encode("utf-8")).hexdigest()
        path = self._store_root / f"{digest}.json"
        if not path.is_file():
            raise PsychologySourceError("no psychology assessment for subject")
        try:
            snapshot = PsychologySourceSnapshot.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (ValueError, OSError) as exc:
            raise PsychologySourceError("psychology assessment snapshot is invalid") from exc
        if snapshot.subject_key != subject_key:
            raise PsychologySourceError("psychology assessment subject mismatch")
        return snapshot
