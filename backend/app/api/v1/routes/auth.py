from fastapi import APIRouter, Request

from app.api.dependencies import CurrentIdentity, DatabaseSession
from app.common.responses import ApiResponse
from app.core.request_id import get_request_id
from app.modules.auth.schemas import AuthUser, LoginData, LoginRequest, LogoutData
from app.modules.auth.service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=ApiResponse[LoginData])
async def login(
    request: Request,
    payload: LoginRequest,
    session: DatabaseSession,
) -> ApiResponse[LoginData]:
    service = AuthService(session, ttl_hours=request.app.state.settings.auth_session_ttl_hours)
    data = await service.login(login_name=payload.login_name, password=payload.password)
    return ApiResponse(data=data, request_id=get_request_id(request))


@router.get("/me", response_model=ApiResponse[AuthUser])
async def auth_me(request: Request, identity: CurrentIdentity) -> ApiResponse[AuthUser]:
    return ApiResponse(
        data=AuthUser(
            user_id=identity.user.external_subject or str(identity.user.id),
            role=identity.role,
            name=identity.user.display_name,
        ),
        request_id=get_request_id(request),
    )


@router.post("/logout", response_model=ApiResponse[LogoutData])
async def logout(
    request: Request,
    identity: CurrentIdentity,
    session: DatabaseSession,
) -> ApiResponse[LogoutData]:
    await AuthService(
        session,
        ttl_hours=request.app.state.settings.auth_session_ttl_hours,
    ).logout(identity.token)
    return ApiResponse(data=LogoutData(), request_id=get_request_id(request))
