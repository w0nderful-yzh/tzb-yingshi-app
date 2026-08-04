from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.request_id import RequestIdMiddleware
from app.infrastructure.database.risk_event_repository import RiskEventRepository
from app.infrastructure.database.session import Database
from app.infrastructure.event_deduplicator import EventDeduplicator
from app.infrastructure.event_queue import Ys7EventQueue
from app.infrastructure.external.ys7.event_adapter import Ys7EventAdapter
from app.infrastructure.external.ys7.event_parser import Ys7EventParser
from app.infrastructure.external.ys7.signal_listener import Ys7SignalListener
from app.infrastructure.raw_signal_store import RawSignalStore
from app.modules.fraud.service import FraudSessionService
from app.modules.fraud.visual_event_store import VisualEventStore
from app.workers.ys7_event_worker import Ys7EventWorker


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()
    configure_logging(runtime_settings.log_level)

    database = (
        Database(
            runtime_settings.database_url.get_secret_value(),
            echo=runtime_settings.database_echo,
        )
        if runtime_settings.database_enabled
        else None
    )
    event_queue = Ys7EventQueue(maxsize=runtime_settings.ys7_queue_maxsize)
    visual_event_store = VisualEventStore()
    risk_event_repository = RiskEventRepository(database) if database is not None else None
    fraud_session_service = FraudSessionService(
        visual_event_store=visual_event_store,
        risk_event_sink=risk_event_repository,
    )
    signal_listener = Ys7SignalListener(
        parser=Ys7EventParser(),
        deduplicator=EventDeduplicator(),
        raw_store=RawSignalStore(runtime_settings.ys7_raw_event_dir),
        event_queue=event_queue,
    )
    event_worker = Ys7EventWorker(
        event_queue=event_queue,
        adapter=Ys7EventAdapter(),
        visual_event_store=visual_event_store,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if database is not None:
            await database.ping()
        try:
            if runtime_settings.ys7_signal_enabled:
                await event_worker.start()
            yield
        finally:
            await event_worker.stop()
            if database is not None:
                await database.dispose()

    application = FastAPI(
        title=runtime_settings.app_name,
        version=runtime_settings.app_version,
        debug=runtime_settings.debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    application.state.settings = runtime_settings
    application.state.database = database
    application.state.ys7_signal_listener = signal_listener
    application.state.ys7_event_queue = event_queue
    application.state.ys7_event_worker = event_worker
    application.state.visual_event_store = visual_event_store
    application.state.fraud_session_service = fraud_session_service
    application.add_middleware(RequestIdMiddleware)
    register_exception_handlers(application)
    application.include_router(api_router, prefix="/api/v1")
    return application


app = create_app()
