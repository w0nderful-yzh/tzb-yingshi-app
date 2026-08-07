from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LoginRequest(BaseModel):
    login_name: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("login_name")
    @classmethod
    def normalize_login_name(cls, value: str) -> str:
        return value.strip().lower()


class AuthUser(BaseModel):
    user_id: str
    role: Literal["elder", "family"]
    name: str


class LoginData(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: datetime
    user: AuthUser


class LogoutData(BaseModel):
    logged_out: bool = True
