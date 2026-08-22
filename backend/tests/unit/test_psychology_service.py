import pytest

from app.modules.psychology.schemas import AssessmentState, SourceStatus
from app.modules.psychology.service import PsychologyService
from app.modules.psychology.source_schemas import PsychologySourceSnapshot


class FakePsychologySource:
    async def get_latest_assessment(self, *, subject_key: str) -> PsychologySourceSnapshot:
        assert subject_key == "elder-001"
        return PsychologySourceSnapshot.model_validate(
            {
                "schema_version": "psychology_assessment_v1",
                "assessment_id": "psy-001",
                "subject_key": subject_key,
                "status": "completed",
                "window_started_at": "2026-08-16T08:00:00+08:00",
                "window_ended_at": "2026-08-16T08:07:00+08:00",
                "estimated_phq8_score": 6.42,
                "segment_scores": [6.42],
                "clip_count": 7,
                "completed_at": "2026-08-16T08:08:00+08:00",
            }
        )

    async def get_latest_completed_assessment(
        self,
        *,
        subject_key: str,
    ) -> PsychologySourceSnapshot:
        return await self.get_latest_assessment(subject_key=subject_key)


class ProcessingPsychologySource(FakePsychologySource):
    async def get_latest_assessment(self, *, subject_key: str) -> PsychologySourceSnapshot:
        return PsychologySourceSnapshot.model_validate(
            {
                "schema_version": "psychology_assessment_v1",
                "assessment_id": "psy-002",
                "subject_key": subject_key,
                "status": "processing",
                "window_started_at": "2026-08-17T08:00:00+08:00",
                "window_ended_at": "2026-08-17T08:00:00+08:00",
                "estimated_phq8_score": None,
                "segment_scores": [],
                "clip_count": 0,
                "completed_at": None,
            }
        )

    async def get_latest_completed_assessment(
        self,
        *,
        subject_key: str,
    ) -> PsychologySourceSnapshot:
        return await FakePsychologySource.get_latest_assessment(
            self,
            subject_key=subject_key,
        )


@pytest.mark.asyncio
async def test_service_reads_latest_reference_observation() -> None:
    result = await PsychologyService(FakePsychologySource()).get_overview(subject_key="elder-001")

    assert result.source_status is SourceStatus.AVAILABLE
    assert result.assessment_state is AssessmentState.OBSERVATION_AVAILABLE
    assert result.operating_mode == "shadow"


@pytest.mark.asyncio
async def test_service_fails_closed_without_source() -> None:
    result = await PsychologyService(None).get_overview(subject_key="elder-001")

    assert result.source_status is SourceStatus.UNAVAILABLE
    assert result.attention_level == "unknown"


@pytest.mark.asyncio
async def test_service_returns_previous_completed_while_processing() -> None:
    result = await PsychologyService(ProcessingPsychologySource()).get_overview(
        subject_key="elder-001"
    )

    assert result.assessment_state is AssessmentState.COLLECTING
    assert result.latest_completed is not None
    assert result.latest_completed.estimated_phq8_score == 6.42
