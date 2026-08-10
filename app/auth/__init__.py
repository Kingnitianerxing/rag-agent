"""JWT authentication and document-level ACL."""
from app.auth.acl import (
    can_delete_document,
    can_ingest,
    effective_sources,
    resolve_allowed_sources,
)
from app.auth.models import AuthUser

__all__ = [
    "AuthUser",
    "can_delete_document",
    "can_ingest",
    "effective_sources",
    "resolve_allowed_sources",
]
