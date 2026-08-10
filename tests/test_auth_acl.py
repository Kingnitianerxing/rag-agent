"""JWT auth + document ACL enforcement."""
from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.auth.acl import effective_sources, resolve_allowed_sources
from app.auth.db import (
    create_user,
    init_db,
    reset_db_state_for_tests,
    upsert_document_acl,
)
from app.auth.jwt_tokens import create_access_token
from app.auth.models import ROLE_ADMIN, ROLE_EDITOR, ROLE_VIEWER, AuthUser
from app.config import get_settings
from app.main import app


@pytest.fixture
def auth_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-for-acl-tests-32b")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "")
    monkeypatch.setenv("API_KEY_HASH", "")
    get_settings.cache_clear()
    reset_db_state_for_tests()
    init_db()
    yield tmp_path
    get_settings.cache_clear()
    reset_db_state_for_tests()


@pytest.fixture
def auth_client(auth_env):
    return TestClient(app)


def _token(user: AuthUser) -> str:
    return create_access_token(user)


def test_effective_sources_intersect():
    assert effective_sources(None, None) is None
    assert effective_sources(["a"], None) == ["a"]
    assert set(effective_sources(None, ["a", "b"])) == {"a", "b"}
    assert effective_sources(["a", "c"], ["a", "b"]) == ["a"]
    assert effective_sources(["c"], ["a", "b"]) == []


def test_resolve_admin_unrestricted():
    admin = AuthUser(id=1, username="a", roles=[ROLE_ADMIN])
    assert resolve_allowed_sources(admin) is None
    svc = AuthUser(id=-1, username="service", roles=[ROLE_ADMIN], is_service=True)
    assert resolve_allowed_sources(svc) is None


def test_login_and_me(auth_client, auth_env):
    user = create_user("alice", "secret", [ROLE_VIEWER])
    res = auth_client.post("/auth/login", json={"username": "alice", "password": "secret"})
    assert res.status_code == 200
    token = res.json()["access_token"]
    me = auth_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["username"] == "alice"
    assert me.json()["id"] == user.id


def test_auth_enabled_rejects_anonymous(auth_client):
    res = auth_client.post("/chat", json={"question": "q"})
    assert res.status_code == 401


def test_viewer_cannot_ingest(auth_client, auth_env):
    user = create_user("viewer1", "pw", [ROLE_VIEWER])
    token = _token(user)
    doc = auth_env / "blocked.md"
    doc.write_text("# x\n", encoding="utf-8")
    with patch("app.api.routes_ingest.ingest_pipeline") as mock_ing:
        res = auth_client.post(
            "/ingest",
            json={"source": str(doc)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 403, res.text
        mock_ing.assert_not_called()


def test_editor_can_ingest_binds_owner(auth_client, auth_env):
    user = create_user("ed", "pw", [ROLE_EDITOR])
    token = _token(user)
    doc = auth_env / "note.md"
    doc.write_text("# hi\n", encoding="utf-8")
    with patch("app.api.routes_ingest.ingest_pipeline") as mock_ing:
        mock_ing.return_value = {"source": str(doc), "chunks": 1, "status": "ingested"}
        res = auth_client.post(
            "/ingest",
            json={"source": str(doc)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200, res.text
        assert mock_ing.call_args.args[0] == str(doc)
        assert mock_ing.call_args.kwargs.get("owner_id") == user.id or (
            len(mock_ing.call_args.args) > 2 and mock_ing.call_args.args[2] == user.id
        )


def test_list_documents_filtered(auth_client, auth_env):
    owner = create_user("owner", "pw", [ROLE_EDITOR])
    other = create_user("other", "pw", [ROLE_VIEWER])
    src_a = str(auth_env / "a.md")
    src_b = str(auth_env / "b.md")
    hash_a = hashlib.sha256(src_a.encode()).hexdigest()
    hash_b = hashlib.sha256(src_b.encode()).hexdigest()
    tracking = {
        hash_a: {"source": src_a, "chunks": 1, "ingested_at": "t"},
        hash_b: {"source": src_b, "chunks": 1, "ingested_at": "t"},
    }
    (auth_env / "ingestions.json").write_text(__import__("json").dumps(tracking), encoding="utf-8")
    upsert_document_acl(hash_a, src_a, owner.id)
    upsert_document_acl(hash_b, src_b, other.id)

    res = auth_client.get(
        "/ingest/documents", headers={"Authorization": f"Bearer {_token(owner)}"}
    )
    assert res.status_code == 200
    sources = {d["source"] for d in res.json()}
    assert src_a in sources
    assert src_b not in sources


def test_delete_forbidden_for_non_owner(auth_client, auth_env):
    owner = create_user("owner2", "pw", [ROLE_EDITOR])
    stranger = create_user("stranger", "pw", [ROLE_EDITOR])
    src = str(auth_env / "c.md")
    doc_id = hashlib.sha256(src.encode()).hexdigest()
    (auth_env / "ingestions.json").write_text(
        __import__("json").dumps({doc_id: {"source": src, "chunks": 1, "ingested_at": "t"}}),
        encoding="utf-8",
    )
    upsert_document_acl(doc_id, src, owner.id)
    res = auth_client.delete(
        f"/ingest/documents/{doc_id}",
        headers={"Authorization": f"Bearer {_token(stranger)}"},
    )
    assert res.status_code == 403


def test_chat_scopes_sources(auth_client, auth_env):
    owner = create_user("chatown", "pw", [ROLE_VIEWER])
    src = str(auth_env / "only.md")
    doc_id = hashlib.sha256(src.encode()).hexdigest()
    upsert_document_acl(doc_id, src, owner.id)

    with patch("app.api.routes_chat.query_pipeline") as mock_q:
        mock_q.return_value = {
            "answer": "ok",
            "sources": [],
            "latency_ms": 1.0,
            "usage": {},
        }
        # Client asks for unauthorized source — intersection empty → refuse without pipeline
        res = auth_client.post(
            "/chat",
            json={"question": "q", "sources": [str(auth_env / "other.md")]},
            headers={"Authorization": f"Bearer {_token(owner)}"},
        )
        assert res.status_code == 200
        assert "cannot answer" in res.json()["answer"].lower()
        mock_q.assert_not_called()

        # Allowed source passed through
        res2 = auth_client.post(
            "/chat",
            json={"question": "q", "sources": [src]},
            headers={"Authorization": f"Bearer {_token(owner)}"},
        )
        assert res2.status_code == 200
        assert mock_q.call_args.args[2] == [src]


def test_service_api_key_full_access(auth_env, monkeypatch):
    key = "service-secret"
    monkeypatch.setenv("API_KEY_HASH", hashlib.sha256(key.encode()).hexdigest())
    monkeypatch.setenv("AUTH_ENABLED", "false")
    get_settings.cache_clear()
    client = TestClient(app)
    with patch("app.api.routes_chat.query_pipeline") as mock_q:
        mock_q.return_value = {"answer": "ok", "sources": [], "latency_ms": 1.0, "usage": {}}
        res = client.post(
            "/chat",
            json={"question": "q"},
            headers={"Authorization": f"Bearer {key}"},
        )
        assert res.status_code == 200
        # unrestricted sources arg
        assert mock_q.call_args.args[2] is None
