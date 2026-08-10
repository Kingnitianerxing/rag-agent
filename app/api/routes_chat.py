import asyncio
import json
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_current_user, limiter
from app.auth.acl import cache_scope_key, effective_sources, resolve_allowed_sources
from app.auth.models import AuthUser
from app.core.pipeline import query_pipeline, stream_query
from app.guardrails.service import apply_output, check_input

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

_NO_ACCESS_ANSWER = "I cannot answer this from the provided documents."


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    sources: list[str] | None = None
    history: list[HistoryTurn] | None = None


class SourceItem(BaseModel):
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    latency_ms: float
    total_sources: int
    usage: dict = {}
    guardrails: dict = {}
    condensed_question: str | None = None


def _scoped_sources(user: AuthUser, requested: list[str] | None) -> list[str] | None:
    allowed = resolve_allowed_sources(user)
    return effective_sources(requested, allowed)


@router.post("", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(
    request: Request, body: ChatRequest, user: AuthUser = Depends(get_current_user)
):
    blocked = check_input(body.question)
    if blocked:
        raise HTTPException(
            status_code=400, detail={"error": "blocked by input guardrails", "patterns": blocked}
        )
    sources = _scoped_sources(user, body.sources)
    if sources is not None and len(sources) == 0:
        return ChatResponse(
            answer=_NO_ACCESS_ANSWER,
            sources=[],
            latency_ms=0.0,
            total_sources=0,
            usage={},
            guardrails={"pii_redacted": [], "flags": []},
        )
    history = [t.model_dump() for t in body.history or []]
    scope = cache_scope_key(user, sources)
    result = await asyncio.to_thread(
        query_pipeline, body.question, body.top_k, sources, history, scope
    )
    guarded = apply_output(result["answer"])
    out_sources = result.get("sources", [])
    return ChatResponse(
        answer=guarded["answer"],
        sources=out_sources,
        latency_ms=result["latency_ms"],
        total_sources=len(out_sources),
        usage=result.get("usage", {}),
        guardrails={"pii_redacted": guarded["pii_redacted"], "flags": guarded["flags"]},
        condensed_question=result.get("condensed_question"),
    )


@router.post("/stream")
@limiter.limit("30/minute")
async def chat_stream(
    request: Request, body: ChatRequest, user: AuthUser = Depends(get_current_user)
):
    """Token-by-token streaming as newline-delimited JSON."""
    blocked = check_input(body.question)
    if blocked:
        raise HTTPException(
            status_code=400, detail={"error": "blocked by input guardrails", "patterns": blocked}
        )

    sources = _scoped_sources(user, body.sources)
    history = [t.model_dump() for t in body.history or []]
    scope = cache_scope_key(user, sources)

    async def event_generator():
        if sources is not None and len(sources) == 0:
            yield json.dumps({"event": "sources", "sources": [], "latency_ms": 0.0}) + "\n"
            done = {
                "event": "done",
                "answer": _NO_ACCESS_ANSWER,
                "usage": {},
                "latency_ms": 0.0,
                "guardrails": {"pii_redacted": [], "flags": []},
            }
            yield json.dumps(done) + "\n"
            return
        try:
            async for event in stream_query(
                body.question, body.top_k, sources, history, scope
            ):
                if event.get("event") == "done":
                    g = apply_output(event.get("answer", ""))
                    event["answer"] = g["answer"]
                    event["guardrails"] = {
                        "pii_redacted": g["pii_redacted"],
                        "flags": g["flags"],
                    }
                yield json.dumps(event) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("Streaming chat error: %s", exc)
            yield json.dumps({"event": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
