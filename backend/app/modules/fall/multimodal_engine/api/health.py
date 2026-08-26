import logging
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.modules.fall.multimodal_engine.database.session import engine


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["connected", "unavailable"]
    mode: Literal["simulation"]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse | JSONResponse:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        logger.exception("MySQL health check failed")
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "database": "unavailable",
                "mode": "simulation",
            },
        )

    return HealthResponse(
        status="ok",
        database="connected",
        mode="simulation",
    )
