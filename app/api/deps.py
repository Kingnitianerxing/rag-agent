"""Shared API dependencies: auth, rate limiting."""
from __future__ import annotations

import hashlib
import hmac
import logging

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.auth.jwt_tokens import TokenError, decode_access_token
from app.auth.models import ROLE_ADMIN, AuthUser
from app.config import get_settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)
limiter = Limiter(key_func=get_remote_address)

# Synthetic principals
ANONYMOUS_ADMIN = AuthUser(id=0, username="anonymous", roles=[ROLE_ADMIN])
SERVICE_ADMIN = AuthUser(id=-1, username="service", roles=[ROLE_ADMIN], is_service=True)


def _cfg_str(value, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _cfg_bool(value, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _api_key_matches(token: str, key_hash: str) -> bool:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    return hmac.compare_digest(token_hash, key_hash)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str | None:
    """Legacy shared-secret check. Prefer get_current_user for new routes."""
    s = get_settings()
    key_hash = _cfg_str(getattr(s, "API_KEY_HASH", ""))
    if not key_hash:
        return None
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    if not _api_key_matches(credentials.credentials, key_hash):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> AuthUser:
    """Resolve JWT user, shared API-key service account, or open-mode anonymous admin."""
    s = get_settings()
    key_hash = _cfg_str(getattr(s, "API_KEY_HASH", ""))
    auth_enabled = _cfg_bool(getattr(s, "AUTH_ENABLED", False))
    jwt_secret = _cfg_str(getattr(s, "JWT_SECRET", ""))
    token = credentials.credentials if credentials else None

    if token:
        if key_hash and _api_key_matches(token, key_hash):
            return SERVICE_ADMIN
        if jwt_secret:
            try:
                return decode_access_token(token)
            except TokenError:
                if auth_enabled or key_hash:
                    raise HTTPException(status_code=401, detail="Invalid token") from None
        elif key_hash:
            raise HTTPException(status_code=401, detail="Invalid API key")
        elif auth_enabled:
            raise HTTPException(status_code=401, detail="Invalid token")

    if auth_enabled or key_hash:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    return ANONYMOUS_ADMIN
