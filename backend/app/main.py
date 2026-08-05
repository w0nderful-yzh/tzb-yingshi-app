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
from app.infrastructure.external.llm import OpenAiCompatibleFraudLlmJudge
from app.infrastructure.external.sensevoice import SenseVoiceRecognizer
from app.infrastructure.external.ys7.api_client import Ys7ApiClient
from app.infrastructure.external.ys7.event_adapter import Ys7EventAdapter
from app.infrastructure.external.ys7.event_parser import Ys7EventParser
from app.infrastructure.external.ys7.media_stream import FfmpegPcmStreamSource
from app.infrastructure.external.ys7.signal_listener import Ys7SignalListener
from app.infrastructure.raw_signal_store import RawSignalStore
from app.modules.fraud.audio import SpeechRecognizer
from app.modules.fraud.audio_service import FraudAudioService
from app.modules.fraud.llm import FraudLlmJudge, FraudLlmReviewQueue
from app.modules.fraud.service import FraudSessionService
from app.modules.fraud.visual_event_store import VisualEventStore
from app.workers.fraud_llm_review_worker import FraudLlmReviewWorker
from app.workers.ys7_event_worker import Ys7EventWorker
from app.workers.ys7_media_stream_worker import Ys7MediaStreamWorker


def create_app(
    settings: Settings | None = None,
    *,
    speech_recognizer: SpeechRecognizer | None = None,
    fraud_llm_judge: FraudLlmJudge | None = None,
) -> FastAPI:
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
    llm_api_key = (
        runtime_settings.fraud_llm_api_key.get_secret_value()
        if runtime_settings.fraud_llm_api_key is not None
        else None
    )
    llm_configured = fraud_llm_judge is not None or bool(
        runtime_settings.fraud_llm_base_url and llm_api_key and runtime_settings.fraud_llm_model
    )
    runtime_llm_judge = fraud_llm_judge
    llm_client: OpenAiCompatibleFraudLlmJudge | None = None
    if (
        runtime_settings.fraud_llm_enabled
        and runtime_llm_judge is None
        and llm_configured
        and runtime_settings.fraud_llm_base_url is not None
        and llm_api_key is not None
        and runtime_settings.fraud_llm_model is not None
    ):
        llm_client = OpenAiCompatibleFraudLlmJudge(
            base_url=runtime_settings.fraud_llm_base_url,
            api_key=llm_api_key,
            model=runtime_settings.fraud_llm_model,
            timeout_seconds=runtime_settings.fraud_llm_timeout_seconds,
            enable_thinking=runtime_settings.fraud_llm_enable_thinking,
        )
        runtime_llm_judge = llm_client
    llm_review_queue = (
        FraudLlmReviewQueue(maxsize=runtime_settings.fraud_llm_queue_maxsize)
        if runtime_settings.fraud_llm_enabled and runtime_llm_judge is not None
        else None
    )
    fraud_session_service = FraudSessionService(
        visual_event_store=visual_event_store,
        risk_event_sink=risk_event_repository,
        llm_review_queue=llm_review_queue,
        llm_trigger_state_index=runtime_settings.fraud_llm_trigger_state_index,
        llm_max_transcript_chars=runtime_settings.fraud_llm_max_transcript_chars,
        llm_vision_enabled=runtime_settings.fraud_llm_vision_enabled,
        llm_max_images=runtime_settings.fraud_llm_max_images,
    )
    llm_worker = (
        FraudLlmReviewWorker(
            queue=llm_review_queue,
            judge=runtime_llm_judge,
            fraud_session_service=fraud_session_service,
            timeout_seconds=runtime_settings.fraud_llm_timeout_seconds,
        )
        if llm_review_queue is not None and runtime_llm_judge is not None
        else None
    )
    runtime_speech_recognizer = speech_recognizer or SenseVoiceRecognizer(
        model_name=runtime_settings.sensevoice_model,
        device=runtime_settings.sensevoice_device,
    )
    fraud_audio_service = FraudAudioService(
        recognizer=runtime_speech_recognizer,
        fraud_session_service=fraud_session_service,
        max_chunk_bytes=runtime_settings.sensevoice_max_chunk_bytes,
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
    media_stream_source = FfmpegPcmStreamSource()
    ys7_api_client = Ys7ApiClient(
        app_key=(
            runtime_settings.ys7_app_key.get_secret_value()
            if runtime_settings.ys7_app_key is not None
            else None
        ),
        app_secret=(
            runtime_settings.ys7_app_secret.get_secret_value()
            if runtime_settings.ys7_app_secret is not None
            else None
        ),
        access_token=(
            runtime_settings.ys7_access_token.get_secret_value()
            if runtime_settings.ys7_access_token is not None
            else None
        ),
    )
    media_worker = Ys7MediaStreamWorker(
        address_provider=ys7_api_client,
        stream_source=media_stream_source,
        fraud_audio_service=fraud_audio_service,
        device_serial=runtime_settings.ys7_device_serial,
        channel_no=runtime_settings.ys7_channel_no,
        protocol=runtime_settings.ys7_live_protocol,
        quality=runtime_settings.ys7_live_quality,
        chunk_ms=runtime_settings.ys7_media_chunk_ms,
        queue_maxsize=runtime_settings.ys7_media_queue_maxsize,
        elder_alone=runtime_settings.ys7_elder_alone,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if database is not None:
            await database.ping()
        try:
            if llm_worker is not None:
                await llm_worker.start()
            if runtime_settings.ys7_signal_enabled:
                await event_worker.start()
            if runtime_settings.ys7_media_enabled and runtime_settings.sensevoice_enabled:
                await media_worker.start()
            elif runtime_settings.ys7_media_enabled:
                media_worker.last_error = "SenseVoice ingestion is disabled"
            yield
        finally:
            await media_worker.stop()
            await event_worker.stop()
            if llm_worker is not None:
                await llm_worker.stop()
            if llm_client is not None:
                await llm_client.close()
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
    application.state.ys7_media_worker = media_worker
    application.state.ys7_api_client = ys7_api_client
    application.state.visual_event_store = visual_event_store
    application.state.fraud_session_service = fraud_session_service
    application.state.fraud_audio_service = fraud_audio_service
    application.state.fraud_llm_configured = llm_configured
    application.state.fraud_llm_worker = llm_worker
    application.add_middleware(RequestIdMiddleware)
    register_exception_handlers(application)
    application.include_router(api_router, prefix="/api/v1")
    return application


app = create_app()
