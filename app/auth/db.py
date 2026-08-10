"""SQLite store for users and document ACLs."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import bcrypt

from app.auth.models import ROLE_ADMIN, VALID_ROLES, AuthUser
from app.config import get_settings

_lock = threading.Lock()
_initialized_paths: set[str] = set()


def _db_path() -> Path:
    settings = get_settings()
    data = Path(settings.DATA_DIR)
    data.mkdir(parents=True, exist_ok=True)
    return data / "auth.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables and bootstrap admin if configured."""
    path = str(_db_path())
    with _lock:
        if path in _initialized_paths:
            return
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    roles_json TEXT NOT NULL DEFAULT '["viewer"]',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS document_acl (
                    source_hash TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    owner_id INTEGER NOT NULL,
                    allowed_roles_json TEXT NOT NULL DEFAULT '[]',
                    allowed_user_ids_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (owner_id) REFERENCES users(id)
                );
                """
            )
            conn.commit()
            _bootstrap_admin(conn)
            _initialized_paths.add(path)
        finally:
            conn.close()


def reset_db_state_for_tests() -> None:
    """Drop init cache so tests can use a fresh DATA_DIR."""
    with _lock:
        _initialized_paths.clear()


def _bootstrap_admin(conn: sqlite3.Connection) -> None:
    settings = get_settings()
    username = (settings.BOOTSTRAP_ADMIN_USERNAME or "").strip()
    password = settings.BOOTSTRAP_ADMIN_PASSWORD or ""
    if not username or not password:
        return
    row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if row:
        return
    pw_hash = hash_password(password)
    conn.execute(
        "INSERT INTO users (username, password_hash, roles_json) VALUES (?, ?, ?)",
        (username, pw_hash, json.dumps([ROLE_ADMIN])),
    )
    conn.commit()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def get_user_by_username(username: str) -> AuthUser | None:
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, username, password_hash, roles_json FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if not row:
            return None
        return _row_to_user(row)
    finally:
        conn.close()


def list_users() -> list[AuthUser]:
    """Return all users (id, username, roles) — no password hashes."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, username, password_hash, roles_json FROM users ORDER BY id"
        ).fetchall()
        return [_row_to_user(row) for row in rows]
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> AuthUser | None:
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, username, password_hash, roles_json FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return None
        return _row_to_user(row)
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> AuthUser | None:
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, username, password_hash, roles_json FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            return None
        return _row_to_user(row)
    finally:
        conn.close()


def create_user(username: str, password: str, roles: list[str]) -> AuthUser:
    init_db()
    clean = [r for r in roles if r in VALID_ROLES] or ["viewer"]
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, roles_json) VALUES (?, ?, ?)",
            (username, hash_password(password), json.dumps(clean)),
        )
        conn.commit()
        return AuthUser(id=int(cur.lastrowid), username=username, roles=clean)
    finally:
        conn.close()


def _row_to_user(row: sqlite3.Row) -> AuthUser:
    roles = json.loads(row["roles_json"] or "[]")
    if not isinstance(roles, list):
        roles = []
    return AuthUser(id=int(row["id"]), username=row["username"], roles=[str(r) for r in roles])


def upsert_document_acl(
    source_hash: str,
    source: str,
    owner_id: int,
    allowed_roles: list[str] | None = None,
    allowed_user_ids: list[int] | None = None,
) -> None:
    init_db()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO document_acl (source_hash, source, owner_id, allowed_roles_json, allowed_user_ids_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_hash) DO UPDATE SET
                source = excluded.source,
                owner_id = excluded.owner_id,
                allowed_roles_json = excluded.allowed_roles_json,
                allowed_user_ids_json = excluded.allowed_user_ids_json
            """,
            (
                source_hash,
                source,
                owner_id,
                json.dumps(allowed_roles or []),
                json.dumps(allowed_user_ids or []),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def delete_document_acl(source_hash: str) -> None:
    init_db()
    conn = _connect()
    try:
        conn.execute("DELETE FROM document_acl WHERE source_hash = ?", (source_hash,))
        conn.commit()
    finally:
        conn.close()


def get_document_acl(source_hash: str) -> dict | None:
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT source_hash, source, owner_id, allowed_roles_json, allowed_user_ids_json "
            "FROM document_acl WHERE source_hash = ?",
            (source_hash,),
        ).fetchone()
        if not row:
            return None
        return {
            "source_hash": row["source_hash"],
            "source": row["source"],
            "owner_id": int(row["owner_id"]),
            "allowed_roles": json.loads(row["allowed_roles_json"] or "[]"),
            "allowed_user_ids": [int(x) for x in json.loads(row["allowed_user_ids_json"] or "[]")],
        }
    finally:
        conn.close()


def list_readable_sources(user: AuthUser) -> list[str]:
    """Sources the user owns or is shared on (not used for admin — they get None)."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT source, owner_id, allowed_roles_json, allowed_user_ids_json FROM document_acl"
        ).fetchall()
        out: list[str] = []
        role_set = set(user.roles)
        for row in rows:
            if int(row["owner_id"]) == user.id:
                out.append(row["source"])
                continue
            allowed_roles = set(json.loads(row["allowed_roles_json"] or "[]"))
            allowed_users = {int(x) for x in json.loads(row["allowed_user_ids_json"] or "[]")}
            if role_set & allowed_roles or user.id in allowed_users:
                out.append(row["source"])
        return out
    finally:
        conn.close()
