"""A small query cache (exact + optional semantic).

On a hit, the whole query is skipped — no retrieval, no generation — which is
the single biggest latency/cost win for repeated or near-duplicate questions
(common in demos and FAQ-style traffic). Exact matches are normalized for
case/whitespace; if an ``embed_fn`` is provided, near-duplicates above a
cosine-similarity threshold also hit.

Storage is pluggable: InMemoryBackend (process-local, cleared on restart) or
RedisBackend (shared across replicas, survives restarts). Lookup semantics
are identical either way.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import OrderedDict
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryBackend:
    """Process-local FIFO entry store (the original QueryCache storage)."""

    def __init__(self, max_size: int = 256) -> None:
        self.max_size = max_size
        self._entries: OrderedDict[str, dict] = OrderedDict()

    def get(self, key: str) -> dict | None:
        return self._entries.get(key)

    def put(self, key: str, entry: dict) -> None:
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)  # evict oldest (FIFO)

    def entries(self) -> list[dict]:
        return list(self._entries.values())


class RedisBackend:
    """Redis-backed entry store with the same FIFO semantics as InMemoryBackend.

    Construction pings the server and raises if unreachable (callers fall back
    to in-memory). Any error AFTER construction degrades to a miss/no-op with
    a warning — a dead Redis must never fail a query.
    """

    def __init__(self, url: str, max_size: int = 256, namespace: str = "rag:cache") -> None:
        import redis  # local import keeps redis optional at module import time

        self.max_size = max_size
        self._ns = namespace
        self._client = redis.Redis.from_url(
            url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2,
        )
        self._client.ping()

    def _entry_key(self, key: str) -> str:
        return f"{self._ns}:entry:{key}"

    def _order_key(self) -> str:
        return f"{self._ns}:order"

    def get(self, key: str) -> dict | None:
        try:
            raw = self._client.get(self._entry_key(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_cache_get_failed: %s", exc)
            return None
        return json.loads(raw) if raw else None

    def put(self, key: str, entry: dict) -> None:
        try:
            pipe = self._client.pipeline()
            pipe.set(self._entry_key(key), json.dumps(entry))
            pipe.lrem(self._order_key(), 0, key)  # re-put refreshes position
            pipe.rpush(self._order_key(), key)
            pipe.execute()
            while self._client.llen(self._order_key()) > self.max_size:
                oldest = self._client.lpop(self._order_key())
                if oldest is None:
                    break
                self._client.delete(self._entry_key(oldest))
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_cache_put_failed: %s", exc)

    def entries(self) -> list[dict]:
        try:
            keys = self._client.lrange(self._order_key(), 0, -1)
            if not keys:
                return []
            raws = self._client.mget([self._entry_key(k) for k in keys])
            return [json.loads(r) for r in raws if r]
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_cache_entries_failed: %s", exc)
            return []


class QueryCache:
    def __init__(
        self,
        embed_fn: Callable[[str], list[float]] | None = None,
        threshold: float = 0.95,
        max_size: int = 256,
        backend=None,
    ) -> None:
        self.embed_fn = embed_fn
        self.threshold = threshold
        self._backend = backend if backend is not None else InMemoryBackend(max_size)

    @staticmethod
    def _key(question: str, scope: str = "") -> str:
        base = _normalize(question)
        return f"{scope}::{base}" if scope else base

    def get(self, question: str, scope: str = "") -> dict | None:
        key = self._key(question, scope)
        entry = self._backend.get(key)
        if entry is not None:
            return entry["result"]

        if self.embed_fn is None:
            return None
        cached_entries = self._backend.entries()
        if not cached_entries:
            return None

        # Semantic match only within the same scope.
        query_vec = self.embed_fn(question)
        best_result = None
        best_sim = self.threshold
        for cached in cached_entries:
            if scope and cached.get("scope") != scope:
                continue
            if not scope and cached.get("scope"):
                continue
            emb = cached.get("embedding")
            if emb is None:
                continue
            sim = _cosine(query_vec, emb)
            if sim >= best_sim:
                best_sim = sim
                best_result = cached["result"]
        return best_result

    def put(self, question: str, result: dict, scope: str = "") -> None:
        key = self._key(question, scope)
        embedding = self.embed_fn(question) if self.embed_fn is not None else None
        self._backend.put(
            key, {"embedding": embedding, "result": result, "scope": scope or ""}
        )
