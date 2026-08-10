import asyncio
import json
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import run_agent, stream_agent
from app.api.deps import get_current_user, limiter
from app.auth.acl import effective_sources, resolve_allowed_sources
from app.auth.models import AuthUser
from app.guardrails.service import apply_output, check_input

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

_NO_ACCESS_ANSWER = "I cannot answer this from the provided documents."


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AgentRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    sources: list[str] | None = None
    history: list[HistoryTurn] | None = None


class AgentResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: float
    usage: dict = {}
    route: str = ""
    attempts: int = 0
    guardrails: dict = {}
    condensed_question: str | None = None


def _scoped_sources(user: AuthUser, requested: list[str] | None) -> list[str] | None:
    return effective_sources(requested, resolve_allowed_sources(user))


@router.post("", response_model=AgentResponse)
@limiter.limit("30/minute")
async def agent(
    request: Request, body: AgentRequest, user: AuthUser = Depends(get_current_user)
):
    blocked = check_input(body.question)
    if blocked:
        raise HTTPException(
            status_code=400, detail={"error": "blocked by input guardrails", "patterns": blocked}
        )
    sources = _scoped_sources(user, body.sources)
    if sources is not None and len(sources) == 0:
        return AgentResponse(
            answer=_NO_ACCESS_ANSWER,
            sources=[],
            latency_ms=0.0,
            usage={},
            route="blocked",
            attempts=0,
            guardrails={"pii_redacted": [], "flags": []},
        )
    history = [t.model_dump() for t in body.history or []]
    result = await asyncio.to_thread(run_agent, body.question, body.top_k, sources, history)
    guarded = apply_output(result["answer"])
    return AgentResponse(
        answer=guarded["answer"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
        usage=result.get("usage", {}),
        route=result.get("route", ""),
        attempts=result.get("attempts", 0),
        guardrails={"pii_redacted": guarded["pii_redacted"], "flags": guarded["flags"]},
        condensed_question=result.get("condensed_question"),
    )


@router.post("/stream")
@limiter.limit("30/minute")
async def agent_stream(
    request: Request, body: AgentRequest, user: AuthUser = Depends(get_current_user)
):
    blocked = check_input(body.question)
    if blocked:
        raise HTTPException(
            status_code=400, detail={"error": "blocked by input guardrails", "patterns": blocked}
        )

    sources = _scoped_sources(user, body.sources)
    history = [t.model_dump() for t in body.history or []]

    async def event_generator():
        if sources is not None and len(sources) == 0:
            yield json.dumps({"event": "sources", "sources": []}) + "\n"
            yield json.dumps(
                {
                    "event": "done",
                    "answer": _NO_ACCESS_ANSWER,
                    "usage": {},
                    "latency_ms": 0.0,
                    "route": "blocked",
                    "attempts": 0,
                    "guardrails": {"pii_redacted": [], "flags": []},
                }
            ) + "\n"
            return
        try:
            async for event in stream_agent(body.question, body.top_k, sources, history):
                if event.get("event") == "done":
                    g = apply_output(event.get("answer", ""))
                    event["answer"] = g["answer"]
                    event["guardrails"] = {
                        "pii_redacted": g["pii_redacted"],
                        "flags": g["flags"],
                    }
                yield json.dumps(event) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("agent stream error: %s", exc)
            yield json.dumps({"event": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
