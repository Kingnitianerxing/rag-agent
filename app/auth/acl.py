"""Document ACL resolution and source-scope intersection."""
from __future__ import annotations

from app.auth.db import get_document_acl, list_readable_sources
from app.auth.models import ROLE_EDITOR, AuthUser


def resolve_allowed_sources(user: AuthUser) -> list[str] | None:
    """Return readable source paths, or None if unrestricted (admin/service)."""
    if user.is_admin:
        return None
    return list_readable_sources(user)


def effective_sources(
    requested: list[str] | None, allowed: list[str] | None
) -> list[str] | None:
    """Intersect client scope with ACL.

    Returns:
      - None: unrestricted retrieval (admin, no client scope)
      - list: must retrieve only these sources (may be empty → no access)
    """
    if allowed is None:
        return requested
    allowed_set = set(allowed)
    if requested is None:
        return list(allowed)
    return [s for s in requested if s in allowed_set]


def can_ingest(user: AuthUser) -> bool:
    return user.is_admin or user.has_role(ROLE_EDITOR)


def can_delete_document(user: AuthUser, source_hash: str) -> bool:
    if user.is_admin:
        return True
    acl = get_document_acl(source_hash)
    if acl is None:
        # Legacy docs without ACL: only admin may delete
        return False
    return int(acl["owner_id"]) == user.id


def can_share_document(user: AuthUser) -> bool:
    """Only admin (or service account) may edit document ACL shares."""
    return user.is_admin


def cache_scope_key(user: AuthUser, sources: list[str] | None) -> str:
    """Fingerprint for semantic-cache isolation across users/scopes."""
    if user.is_admin and not sources:
        return f"admin:{user.id}|*"
    src = ",".join(sorted(sources or []))
    return f"{user.id}|{src}"
