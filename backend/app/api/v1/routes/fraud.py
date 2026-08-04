from fastapi import APIRouter, HTTPException, Query, Request

from app.common.responses import ApiResponse
from app.core.request_id import get_request_id
from app.modules.fraud.schemas import (
    FraudAnalyzeData,
    FraudAnalyzeRequest,
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
