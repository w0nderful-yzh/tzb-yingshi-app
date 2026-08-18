"""Read-only HTTP projection of completed/offline psychology assessments."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from service.result_store import LatestAssessmentStore
from service.schemas import HealthResponse, PsychologyAssessmentSnapshot

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE_ROOT = PACKAGE_ROOT / "home_out" / "latest"


def create_app(store: LatestAssessmentStore | None = None) -> FastAPI:
    runtime_store = store or LatestAssessmentStore(
        Path(os.environ.get("PSYCHOLOGY_LATEST_STORE", DEFAULT_STORE_ROOT))
    )
    application = FastAPI(title="Psychology Reference Assessment Projection", version="1.0.0")

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @application.get(
        "/api/psychology/assessments/latest",
        response_model=PsychologyAssessmentSnapshot,
    )
    async def latest_assessment(
        subject_key: str = Query(min_length=1, max_length=128),
    ) -> PsychologyAssessmentSnapshot:
        snapshot = runtime_store.read(subject_key)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="assessment not found")
        return snapshot

    return application


app = create_app()

