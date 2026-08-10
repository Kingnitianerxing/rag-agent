"""Login and current-user endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.auth.db import authenticate_user, init_db, list_users
from app.auth.jwt_tokens import TokenError, create_access_token
from app.auth.models import AuthUser
from app.config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    id: int
    username: str
    roles: list[str]
    is_service: bool = False


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest):
    settings = get_settings()
    if not settings.JWT_SECRET:
        raise HTTPException(status_code=503, detail="JWT_SECRET is not configured")
    init_db()
    user = authenticate_user(body.username.strip(), body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    try:
        token = create_access_token(user)
    except TokenError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return LoginResponse(
        access_token=token,
        user={"id": user.id, "username": user.username, "roles": user.roles},
    )


@router.get("/me", response_model=UserResponse)
async def me(user: AuthUser = Depends(get_current_user)):
    return UserResponse(
        id=user.id, username=user.username, roles=user.roles, is_service=user.is_service
    )


@router.get("/users", response_model=list[UserResponse])
async def users(user: AuthUser = Depends(get_current_user)):
    """Admin-only directory for document sharing pickers."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Only admin can list users")
    return [
        UserResponse(id=u.id, username=u.username, roles=u.roles, is_service=False)
        for u in list_users()
    ]
