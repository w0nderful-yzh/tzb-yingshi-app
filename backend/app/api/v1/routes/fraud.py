from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import ValidationError

from app.common.responses import ApiResponse
from app.core.request_id import get_request_id
from app.modules.fraud.audio import (
    InvalidAudioChunkError,
    SpeechRecognitionError,
    SpeechRecognitionUnavailableError,
)
from app.modules.fraud.audio_service import FraudAudioService
from app.modules.fraud.schemas import (
    FraudAnalyzeData,
    FraudAnalyzeRequest,
    FraudAudioChunkData,
    FraudAudioChunkRequest,
    FraudRiskSnapshot,
)
from app.modules.fraud.service import FraudSessionService

router = APIRouter(prefix="/fraud", tags=["fraud"])


@router.post("/analyze", response_model=ApiResponse[FraudAnalyzeData])
async def analyze_fraud_event(
    request: Request,
    payload: FraudAnalyzeRequest,
) -> ApiResponse[FraudAnalyzeData]:
    service: FraudSessionService = request.app.state.fraud_session_service
    result = await service.analyze(payload)
    return ApiResponse(data=result, request_id=get_request_id(request))


@router.post(
    "/audio/chunks",
    response_model=ApiResponse[FraudAudioChunkData],
)
async def analyze_fraud_audio_chunk(
    request: Request,
    audio: Annotated[UploadFile, File(description="Short WAV audio chunk")],
    session_id: Annotated[str, Form(min_length=1, max_length=128)],
    chunk_id: Annotated[str, Form(min_length=1, max_length=128)],
    device_id: Annotated[str, Form(min_length=1, max_length=256)],
    started_at: Annotated[datetime, Form()],
    elder_alone: Annotated[bool, Form()] = False,
) -> ApiResponse[FraudAudioChunkData]:
    settings = request.app.state.settings
    if not settings.sensevoice_enabled:
        raise HTTPException(status_code=503, detail="SenseVoice ingestion is disabled")
    if audio.content_type not in {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "application/octet-stream",
    }:
        raise HTTPException(status_code=415, detail="audio chunk must be WAV")
    try:
        metadata = FraudAudioChunkRequest(
            session_id=session_id,
            chunk_id=chunk_id,
            device_id=device_id,
            started_at=started_at,
            elder_alone=elder_alone,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail="invalid audio chunk metadata",
        ) from exc

    payload = await audio.read(settings.sensevoice_max_chunk_bytes + 1)
    service: FraudAudioService = request.app.state.fraud_audio_service
    try:
        result = await service.analyze_chunk(metadata, payload)
    except (InvalidAudioChunkError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SpeechRecognitionUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SpeechRecognitionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ApiResponse(data=result, request_id=get_request_id(request))


@router.get(
    "/sessions/{session_id}",
    response_model=ApiResponse[FraudRiskSnapshot],
)
async def get_fraud_session(
    request: Request,
    session_id: str,
    device_id: str = Query(min_length=1, max_length=256),
) -> ApiResponse[FraudRiskSnapshot]:
    service: FraudSessionService = request.app.state.fraud_session_service
    risk = await service.get_session(device_id=device_id, session_id=session_id)
    if risk is None:
        raise HTTPException(status_code=404, detail="fraud session not found")
    return ApiResponse(data=risk, request_id=get_request_id(request))
