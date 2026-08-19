import hashlib
import json
from pathlib import Path

import pytest

from app.infrastructure.external.psychology.local import LocalPsychologySource
from app.modules.psychology.ports import PsychologySourceError


def _write_snapshot(store_root: Path, subject_key: str, score: float = 6.42) -> Path:
    store_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(subject_key.encode("utf-8")).hexdigest()
    path = store_root / f"{digest}.json"
    payload = {
        "schema_version": "psychology_assessment_v1",
        "assessment_id": "psy-001",
        "subject_key": subject_key,
        "status": "completed",
        "window_started_at": "2026-08-16T08:00:00+08:00",
        "window_ended_at": "2026-08-16T08:07:00+08:00",
        "estimated_phq8_score": score,
        "segment_scores": [score],
        "clip_count": 7,
        "completed_at": "2026-08-16T08:08:00+08:00",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_local_source_reads_latest_snapshot(tmp_path) -> None:
    source = LocalPsychologySource(store_root=tmp_path)
    _write_snapshot(tmp_path, "elder-001", score=7.09)

    snapshot = await source.get_latest_assessment(subject_key="elder-001")

    assert snapshot.subject_key == "elder-001"
    assert snapshot.status == "completed"
    assert snapshot.estimated_phq8_score == 7.09


@pytest.mark.asyncio
async def test_local_source_missing_snapshot_raises(tmp_path) -> None:
    source = LocalPsychologySource(store_root=tmp_path)

    with pytest.raises(PsychologySourceError):
        await source.get_latest_assessment(subject_key="nobody")


@pytest.mark.asyncio
async def test_local_source_wrong_subject_is_rejected(tmp_path) -> None:
    source = LocalPsychologySource(store_root=tmp_path)
    _write_snapshot(tmp_path, "elder-001")

    with pytest.raises(PsychologySourceError):
        await source.get_latest_assessment(subject_key="other")
