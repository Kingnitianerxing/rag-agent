"""JWT create/verify helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt

from app.auth.models import AuthUser
from app.config import get_settings


class TokenError(Exception):
    pass


def create_access_token(user: AuthUser) -> str:
    settings = get_settings()
    secret = settings.JWT_SECRET
    if not secret:
        raise TokenError("JWT_SECRET is not configured")
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "roles": user.roles,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(token: str) -> AuthUser:
    settings = get_settings()
    secret = settings.JWT_SECRET
    if not secret:
        raise TokenError("JWT_SECRET is not configured")
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc
    sub = payload.get("sub")
    if sub is None:
        raise TokenError("missing sub")
    roles = payload.get("roles") or []
    if not isinstance(roles, list):
        roles = []
    return AuthUser(
        id=int(sub),
        username=str(payload.get("username") or ""),
        roles=[str(r) for r in roles],
    )
