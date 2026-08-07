from fastapi import APIRouter

from app.api.v1.routes.app_client import router as app_client_router
from app.api.v1.routes.auth import router as auth_router
from app.api.v1.routes.fraud import router as fraud_router
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.realtime import router as realtime_router
from app.api.v1.routes.ys7_signals import router as ys7_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(app_client_router)
api_router.include_router(realtime_router)
api_router.include_router(ys7_router)
api_router.include_router(fraud_router)
