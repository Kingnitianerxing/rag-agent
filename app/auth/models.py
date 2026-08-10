from __future__ import annotations

from dataclasses import dataclass, field

ROLE_ADMIN = "admin"
ROLE_EDITOR = "editor"
ROLE_VIEWER = "viewer"
VALID_ROLES = frozenset({ROLE_ADMIN, ROLE_EDITOR, ROLE_VIEWER})


@dataclass
class AuthUser:
    """Authenticated principal (JWT user, API-key service, or open-mode anonymous)."""

    id: int
    username: str
    roles: list[str] = field(default_factory=list)
    is_service: bool = False

    def has_role(self, role: str) -> bool:
        return role in self.roles

    @property
    def is_admin(self) -> bool:
        return self.is_service or ROLE_ADMIN in self.roles
