import asyncio
import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models import AuthSessionModel, UserModel
from app.modules.auth.passwords import verify_password
from app.modules.auth.schemas import AuthUser, LoginData

AppRole = Literal["elder", "family"]
_DUMMY_PASSWORD_HASH = (
    "pbkdf2_sha256$310000$IIBrVS114FntmeD3IuL3kw$kq7Ddks6fwRG6Pa_EpVY4uNpjwYuRCeWhuRbDJNB6Ng"
)


@dataclass(frozen=True, slots=True)
class AuthenticatedIdentity:
    user: UserModel
    role: AppRole
    token: str


class AuthService:
    def __init__(self, session: AsyncSession, *, ttl_hours: int) -> None:
        self._session = session
        self._ttl = timedelta(hours=ttl_hours)

    async def login(self, *, login_name: str, password: str) -> LoginData:
        user = await self._session.scalar(
            select(UserModel).where(
                UserModel.login_name == login_name.strip().lower(),
                UserModel.is_active.is_(True),
            )
        )
        encoded_password = (
            user.password_hash if user is not None and user.password_hash else _DUMMY_PASSWORD_HASH
        )
        valid = await asyncio.to_thread(verify_password, password, encoded_password)
        if not valid or user is None:
            raise HTTPException(status_code=401, detail="账号或密码错误")
        role = _app_role(user)
        now = datetime.now(UTC)
        expires_at = now + self._ttl
        token = secrets.token_urlsafe(32)
        self._session.add(
            AuthSessionModel(
                user_id=user.id,
                token_hash=_token_hash(token),
                expires_at=expires_at,
            )
        )
        await self._session.commit()
        return LoginData(
            access_token=token,
            expires_at=expires_at,
            user=_auth_user(user, role),
        )

    async def authenticate(self, token: str) -> AuthenticatedIdentity:
        now = datetime.now(UTC)
        row = (
            await self._session.execute(
                select(AuthSessionModel, UserModel)
                .join(UserModel, UserModel.id == AuthSessionModel.user_id)
                .where(
                    AuthSessionModel.token_hash == _token_hash(token),
                    AuthSessionModel.revoked_at.is_(None),
                    AuthSessionModel.expires_at > now,
                    UserModel.is_active.is_(True),
                )
            )
        ).one_or_none()
        if row is None:
            raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
        _, user = row
        return AuthenticatedIdentity(user=user, role=_app_role(user), token=token)

    async def logout(self, token: str) -> None:
        await self._session.execute(
            update(AuthSessionModel)
            .where(
                AuthSessionModel.token_hash == _token_hash(token),
                AuthSessionModel.revoked_at.is_(None),
            )
            .values(revoked_at=datetime.now(UTC))
        )
        await self._session.commit()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _app_role(user: UserModel) -> AppRole:
    if user.role == "ELDER":
        return "elder"
    if user.role == "GUARDIAN":
        return "family"
    raise HTTPException(status_code=403, detail="该账号暂不支持 App 登录")


def _auth_user(user: UserModel, role: AppRole) -> AuthUser:
    return AuthUser(
        user_id=user.external_subject or str(user.id),
        role=role,
        name=user.display_name,
    )
