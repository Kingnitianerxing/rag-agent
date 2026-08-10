import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.api.deps import limiter
from app.api.routes_agent import router as agent_router
from app.api.routes_auth import router as auth_router
from app.api.routes_chat import router as chat_router
from app.api.routes_ingest import router as ingest_router
from app.config import get_settings
from app.observability.logging import RequestIDMiddleware, setup_logging
from app.observability.tracing import setup_tracing

logger = logging.getLogger(__name__)

setup_logging()
setup_tracing()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from app.auth.db import init_db

    init_db()
    yield


app = FastAPI(title="Production RAG", version="0.2.0", lifespan=lifespan)

# ── Middleware ──────────────────────────────────────────
settings = get_settings()

# CORS: browsers reject credentials together with a wildcard origin,
# so only enable credentials when explicit origins are configured
cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

# Request ID
app.add_middleware(RequestIDMiddleware)

# Rate limiting
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


# ── Routes ─────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(ingest_router)
app.include_router(chat_router)
app.include_router(agent_router)


# ── Health Checks ──────────────────────────────────────
@app.get("/health/live")
async def health_live():
    return {"status": "alive"}


@app.get("/health/ready")
async def health_ready():
    checks: dict[str, str] = {"app": "ok"}

    # Check Qdrant
    try:
        from app.retrieval.vector_store import VectorStore
        vs = VectorStore()
        vs._get_store()
        checks["qdrant"] = "ok"
    except Exception as exc:
        checks["qdrant"] = f"failed: {exc}"

    # Check BM25 store
    try:
        s = get_settings()
        bm25_path = Path(s.DATA_DIR) / "bm25_index.pkl"
        checks["bm25"] = "ok" if not bm25_path.exists() or bm25_path.stat().st_size > 0 else "empty"
    except Exception as exc:
        checks["bm25"] = f"failed: {exc}"

    # Check Graph store
    try:
        s = get_settings()
        graph_path = Path(s.DATA_DIR) / "knowledge_graph.gpickle"
        checks["graph"] = "ok" if not graph_path.exists() or graph_path.stat().st_size > 0 else "empty"
    except Exception as exc:
        checks["graph"] = f"failed: {exc}"

    # Check OpenSearch (informational; only when it is the active keyword backend)
    if get_settings().KEYWORD_BACKEND == "opensearch":
        try:
            from app.core.factories import get_keyword_store
            checks["opensearch"] = "ok" if get_keyword_store().ping() else "failed: ping"
        except Exception as exc:
            checks["opensearch"] = f"failed: {exc}"

    if any("failed" in str(v) for k, v in checks.items() if k == "qdrant"):
        return JSONResponse(status_code=503, content={"status": "not_ready", "checks": checks})

    return {"status": "ready", "checks": checks}
