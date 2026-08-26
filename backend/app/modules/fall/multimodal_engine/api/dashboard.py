from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.database.session import get_db_session
from app.modules.fall.multimodal_engine.schemas.dashboard import DashboardSummaryResponse
from app.modules.fall.multimodal_engine.services import DashboardService


router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
DatabaseSession = Annotated[Session, Depends(get_db_session)]


@router.get("/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary(db: DatabaseSession) -> DashboardSummaryResponse:
    return DashboardService(db).get_summary()
