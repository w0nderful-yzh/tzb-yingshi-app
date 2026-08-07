from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.session import get_database_session
from app.modules.auth.service import AuthenticatedIdentity, AuthService

DatabaseSession = Annotated[AsyncSession, Depends(get_database_session)]
BearerCredentials = Annotated[
    HTTPAuthorizationCredentials | None,
    Depends(HTTPBearer(auto_error=False)),
]


async def get_current_identity(
    request: Request,
    credentials: BearerCredentials,
    session: DatabaseSession,
) -> AuthenticatedIdentity:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="请先登录")
    return await AuthService(
        session,
        ttl_hours=request.app.state.settings.auth_session_ttl_hours,
    ).authenticate(credentials.credentials)


CurrentIdentity = Annotated[AuthenticatedIdentity, Depends(get_current_identity)]
