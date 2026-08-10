"""Ingest and query pipelines with error handling, logging, and incremental ingestion."""
import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.core.cache import QueryCache
from app.core.condense import CondenseResult, attach_condense, condense_question
from app.core.factories import (
    complete_with_model,
    get_embedder,
    get_graph_store,
    get_image_vector_store,
    get_keyword_store,
    get_llm,
    get_reranker,
    get_vector_store,
    image_retrieval_ready,
)
from app.core.prompts import format_context, select_prompt
from app.graph.builder import GraphBuilder
from app.ingestion.chunkers import chunk_documents
from app.ingestion.loaders import load_documents
from app.ingestion.validation import IMAGE_SUFFIXES
from app.observability.cost import usage_for
from app.observability.tracing import trace_retrieval
from app.reranker.reranker import RerankerService
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.hybrid_retriever import HybridRetriever, rrf_fuse

logger = logging.getLogger(__name__)

# Outer retrieval pool: per-query hybrid/dense tasks + the graph leg. The
# hybrid legs themselves run on hybrid_retriever._leg_executor — a separate
# pool, so an outer task waiting on legs can never deadlock this one.
_retrieval_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="retrieval")

# ── Ingestion tracking helpers ──────────────────────────

def _load_tracking(data_dir: str) -> dict:
    path = Path(data_dir) / "ingestions.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_tracking(data_dir: str, tracking: dict) -> None:
    path = Path(data_dir) / "ingestions.json"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(tracking, f, indent=2)


def list_documents() -> list[dict]:
    tracking = _load_tracking(get_settings().DATA_DIR)
    return [
        {
            "id": k,
            "source": v["source"],
            "chunks": v["chunks"],
            "ingested_at": v.get("ingested_at", ""),
            "modality": v.get("modality", "text"),
        }
        for k, v in tracking.items()
    ]


def _record_ingestion(
    tracking: dict,
    source_hash: str,
    source: str,
    chunks: int,
    owner_id: int | None,
    modality: str,
    elapsed: float,
) -> dict:
    tracking[source_hash] = {
        "source": source,
        "chunks": chunks,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "modality": modality,
    }
    _save_tracking(get_settings().DATA_DIR, tracking)
    if owner_id is not None and owner_id > 0:
        try:
            from app.auth.db import upsert_document_acl

            upsert_document_acl(source_hash, source, owner_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("document_acl_failed: %s", exc)
    logger.info(
        "ingest_complete: source=%s chunks=%d modality=%s latency_ms=%.1f",
        source,
        chunks,
        modality,
        elapsed,
    )
    return {"source": source, "chunks": chunks, "status": "ingested"}


# ── Ingest Pipeline ─────────────────────────────────────

def ingest_pipeline(source: str, force: bool = False, owner_id: int | None = None) -> dict:
    settings = get_settings()
    start = time.time()
    logger.info("ingest_start: source=%s", source)

    # Incremental ingestion: check if already ingested
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    tracking = _load_tracking(settings.DATA_DIR)
    if source_hash in tracking and not force:
        logger.info("ingest_skipped: source=%s (already ingested)", source)
        return {
            "source": source,
            "chunks": tracking[source_hash]["chunks"],
            "status": "skipped",
        }

    # Load documents
    try:
        documents = load_documents(source)
        logger.info("documents_loaded: count=%d source=%s", len(documents), source)
    except Exception as exc:
        logger.error("document_load_failed: source=%s error=%s", source, exc)
        raise

    is_image = (
        Path(source).suffix.lower() in IMAGE_SUFFIXES
        or any(d.metadata.get("modality") == "image" for d in documents)
    )

    # Image track: qwen3-vl-embedding → rag_images (skip BM25 / graph)
    if is_image:
        if not image_retrieval_ready():
            raise ValueError(
                "Image ingestion requires IMAGE_RETRIEVAL_ENABLED=true and DASHSCOPE_API_KEY"
            )
        try:
            get_image_vector_store().upsert(documents)
            logger.info("image_vector_upserted: count=%d", len(documents))
        except Exception as exc:
            logger.error("image_vector_upsert_failed: %s", exc)
            raise
        elapsed = (time.time() - start) * 1000
        return _record_ingestion(
            tracking, source_hash, source, len(documents), owner_id, "image", elapsed
        )

    # Chunk documents
    chunks = chunk_documents(documents, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    logger.info("chunks_created: count=%d", len(chunks))

    # Vector store upsert
    try:
        vs = get_vector_store()
        vs.upsert(chunks)
        logger.info("vector_upserted: count=%d", len(chunks))
    except Exception as exc:
        logger.error("vector_upsert_failed: %s", exc)
        raise

    # Keyword index
    try:
        keyword_store = get_keyword_store()
        keyword_store.add_documents(chunks)
        logger.info("keyword_indexed: backend=%s count=%d", settings.KEYWORD_BACKEND, len(chunks))
    except Exception as exc:
        logger.error("keyword_index_failed: %s", exc)
        # Non-critical: continue without the keyword index

    # Graph extraction (optional)
    if settings.GRAPH_EXTRACTOR != "none":
        try:
            llm = get_llm() if settings.GRAPH_EXTRACTOR == "llm" else None
            builder = GraphBuilder(extractor_type=settings.GRAPH_EXTRACTOR, llm=llm)
            triples = builder.extract(chunks)
            gs = get_graph_store()
            gs.add_triples(triples)
            logger.info("graph_extracted: triples=%d", len(triples))
        except Exception as exc:
            logger.error("graph_extraction_failed: %s", exc)
            # Non-critical: continue without graph

    elapsed = (time.time() - start) * 1000
    return _record_ingestion(
        tracking, source_hash, source, len(chunks), owner_id, "text", elapsed
    )


# ── Query Pipeline ───────────────────────────────────────

def _queries_for(question: str, settings) -> list[str]:
    """Expand the question per QUERY_TRANSFORM. Only calls the LLM when an
    actual transform (multi_query / hyde) is configured."""
    mode = getattr(settings, "QUERY_TRANSFORM", "none")
    if mode not in ("multi_query", "hyde"):
        return [question]
    try:
        from app.retrieval.query_transform import build_queries
        return build_queries(question, mode, get_llm())
    except Exception as exc:
        logger.warning("query_transform_failed: %s — using original query", exc)
        return [question]


def _graph_retrieve(question: str) -> list[Document]:
    gr = GraphRetriever(graph_store=get_graph_store())
    return gr.retrieve(question, depth=1)


def _retrieve_and_rerank(
    question: str, top_k: int, settings, sources: list[str] | None = None
) -> list[Document]:
    """Retrieve (dense/hybrid + optional graph) and rerank to the final docs.

    Shared by query_pipeline (generation) and retrieve_sources (eval/UI), so
    both measure the exact same retrieval path.
    """
    # RETRIEVAL_MODE=dense is the ablation baseline (vector only); hybrid
    # additionally fuses BM25 keyword hits via RRF. With QUERY_TRANSFORM,
    # we retrieve for each expanded query and RRF-fuse across them too.
    queries = _queries_for(question, settings)

    # Graph leg is submitted first so it overlaps with the (slower) vector
    # legs; skipped when scoped — graph docs carry no per-source payload to
    # filter on.
    graph_future = None
    if settings.GRAPH_EXTRACTOR != "none" and not sources:
        graph_future = _retrieval_executor.submit(_graph_retrieve, question)

    image_future = None
    # Use `is True` so unittest MagicMocks (truthy by default) do not enable this leg.
    if (
        getattr(settings, "IMAGE_RETRIEVAL_ENABLED", False) is True
        and isinstance(getattr(settings, "DASHSCOPE_API_KEY", ""), str)
        and bool(settings.DASHSCOPE_API_KEY)
    ):
        image_future = _retrieval_executor.submit(
            get_image_vector_store().search, question, top_k, sources
        )

    hybrid_results: list[tuple[Document, float]] = []
    try:
        vs = get_vector_store()
        if settings.RETRIEVAL_MODE == "dense":
            futures = [
                _retrieval_executor.submit(vs.search, q, top_k=top_k, sources=sources)
                for q in queries
            ]
        else:
            retriever = HybridRetriever(vector_store=vs, bm25_store=get_keyword_store())
            futures = [
                _retrieval_executor.submit(retriever.retrieve, q, top_k=top_k, sources=sources)
                for q in queries
            ]
        result_lists: list[list[tuple[Document, float]]] = []
        for q, future in zip(queries, futures):
            try:
                result_lists.append(future.result())
            except Exception as exc:  # noqa: BLE001 — one bad query must not kill the rest
                logger.error("retrieval_failed: query=%s error=%s", q[:80], exc)
                result_lists.append([])
        hybrid_results = (
            result_lists[0] if len(result_lists) == 1 else rrf_fuse(result_lists)[:top_k]
        )
        logger.info(
            "retrieval_complete: mode=%s queries=%d hits=%d",
            settings.RETRIEVAL_MODE, len(queries), len(hybrid_results),
        )
    except Exception as exc:
        logger.error("retrieval_failed: %s", exc)
        # Continue with empty results

    graph_docs: list[Document] = []
    if graph_future is not None:
        try:
            graph_docs = graph_future.result()
            logger.info("graph_retrieval_complete: hits=%d", len(graph_docs))
        except Exception as exc:
            logger.error("graph_retrieval_failed: %s", exc)
            # Continue without graph results

    image_hits: list[tuple[Document, float]] = []
    if image_future is not None:
        try:
            image_hits = image_future.result()
            logger.info("image_retrieval_complete: hits=%d", len(image_hits))
        except Exception as exc:
            logger.error("image_retrieval_failed: %s", exc)
            # Continue without image results

    # Fuse text/hybrid ranks with image ranks via RRF. Previously images were
    # appended after hybrid hits and truncated away when RERANKER=none (slice
    # kept only the first RERANK_TOP_K text chunks).
    ranked_lists: list[list[tuple[Document, float]]] = []
    if hybrid_results:
        ranked_lists.append(hybrid_results)
    if image_hits:
        ranked_lists.append(image_hits)
    if len(ranked_lists) == 0:
        fused_docs: list[Document] = []
    elif len(ranked_lists) == 1:
        fused_docs = [doc for doc, _ in ranked_lists[0]]
    else:
        # Keep a wider pool for the reranker; final cut is RERANK_TOP_K.
        pool_k = max(top_k * 2, settings.RERANK_TOP_K, top_k + len(image_hits))
        fused_docs = [doc for doc, _ in rrf_fuse(ranked_lists)[:pool_k]]

    # Prefer keeping at least one image hit in the candidate pool when present.
    if image_hits:
        image_sources = {
            (d.metadata.get("source"), d.page_content) for d, _ in image_hits[:1]
        }
        have_image = any(
            (d.metadata.get("source"), d.page_content) in image_sources for d in fused_docs
        )
        if not have_image:
            fused_docs = [image_hits[0][0]] + fused_docs

    all_docs = fused_docs + graph_docs

    # Reranking
    try:
        reranker_model = get_reranker()
        reranker_svc = RerankerService(reranker=reranker_model)
        reranked = reranker_svc.rerank(question, all_docs, top_k=settings.RERANK_TOP_K)
        logger.info("rerank_complete: input=%d output=%d", len(all_docs), len(reranked))
    except Exception as exc:
        logger.warning("rerank_failed: %s — using unranked results", exc)
        reranked = all_docs[: settings.RERANK_TOP_K]

    # If rerank/truncate dropped every image but we had image hits, force-include
    # the best image so visual questions are not answered from unrelated text only.
    if image_hits and not any(d.metadata.get("modality") == "image" for d in reranked):
        best_image = image_hits[0][0]
        reranked = [best_image, *reranked][: settings.RERANK_TOP_K]
        logger.info("image_slot_preserved: source=%s", best_image.metadata.get("source"))

    return reranked


def _docs_to_sources(docs: list[Document]) -> list[dict]:
    """Render reranked docs as source dicts, stamping the 1-based citation
    number so callers can resolve the [n] markers in a grounded answer."""
    sources = []
    for i, d in enumerate(docs, 1):
        md = dict(d.metadata)
        md["citation"] = i
        sources.append({"content": d.page_content, "metadata": md})
    return sources


def retrieve_sources(
    question: str, top_k: int | None = None, sources: list[str] | None = None
) -> list[dict]:
    """Retrieval-only entry point (no generation): cited sources for a query.

    Used by the retrieval-only evaluation and the demo UI; avoids paying for
    a generation call when only the retrieved context is needed.
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.TOP_K
    start = time.time()
    reranked = _retrieve_and_rerank(question, top_k, settings, sources=sources)
    retrieval_ms = (time.time() - start) * 1000
    trace_data = [
        {"content": d.page_content[:100], "score": d.metadata.get("relevance_score", 0)}
        for d in reranked
    ]
    trace_retrieval(question, trace_data, retrieval_ms)
    return _docs_to_sources(reranked)


_query_cache: QueryCache | None = None


def _make_cache_backend(settings):
    """Redis when REDIS_URL is set and reachable; in-memory otherwise."""
    from app.core.cache import InMemoryBackend, RedisBackend

    url = getattr(settings, "REDIS_URL", "")
    if url:
        try:
            backend = RedisBackend(url)
            logger.info("query_cache_backend: redis")
            return backend
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_unavailable: %s — falling back to in-memory cache", exc)
    return InMemoryBackend()


def _get_query_cache(settings) -> QueryCache | None:
    """Return the query cache, or None when caching is off."""
    global _query_cache
    if getattr(settings, "CACHE_ENABLED", False) is not True:
        return None
    if _query_cache is None:
        _query_cache = QueryCache(
            embed_fn=lambda q: get_embedder().embed_query(q),
            threshold=getattr(settings, "CACHE_SIMILARITY_THRESHOLD", 0.95),
            backend=_make_cache_backend(settings),
        )
    return _query_cache


def query_pipeline(
    question: str,
    top_k: int | None = None,
    sources: list[str] | None = None,
    history: list[dict] | None = None,
    cache_scope: str = "",
) -> dict:
    settings = get_settings()
    if top_k is None:
        top_k = settings.TOP_K
    start = time.time()
    logger.info("query_start: question=%s", question[:100])

    # Condense BEFORE the cache lookup so the cache keys on the standalone
    # question. A condensed question is conversation-independent by
    # construction, so cross-conversation hits on it are correct.
    cq = (
        condense_question(question, history)
        if history
        else CondenseResult(question=question, applied=False)
    )
    if cq.applied:
        logger.info("condensed: %s -> %s", question[:80], cq.question[:80])
        question = cq.question

    # Cache keys include cache_scope (user/ACL fingerprint) so answers never
    # leak across principals. Unscoped admin queries still use scope "".
    cache = _get_query_cache(settings)
    if cache is not None:
        cached = cache.get(question, scope=cache_scope)
        if cached is not None:
            logger.info("cache_hit: question=%s scope=%s", question[:100], cache_scope[:40])
            return attach_condense({**cached, "cached": True}, cq)

    reranked = _retrieve_and_rerank(question, top_k, settings, sources=sources)

    retrieval_ms = (time.time() - start) * 1000
    trace_data = [
        {"content": d.page_content[:100], "score": d.metadata.get("relevance_score", 0)}
        for d in reranked
    ]
    trace_retrieval(question, trace_data, retrieval_ms)

    # LLM generation. Context is numbered so the grounded prompt's [n]
    # citations map back to the sources returned below. complete_with_model()
    # routes to the strong model and falls back to a same-provider model on
    # failure, returning the model that answered so cost is attributed correctly.
    context = format_context(reranked)
    prompt_text = select_prompt(settings.PROMPT_MODE).format(context=context, question=question)
    try:
        answer, model = complete_with_model(prompt_text)
        logger.info("llm_complete: latency_ms=%.1f", (time.time() - start) * 1000 - retrieval_ms)
    except Exception as exc:
        logger.error("llm_generation_failed: %s", exc)
        raise

    total_ms = (time.time() - start) * 1000
    logger.info("query_complete: latency_ms=%.1f", total_ms)

    usage = usage_for(prompt_text, answer, model)
    logger.info(
        "query_usage: input_tokens=%d output_tokens=%d cost_usd=%.6f",
        usage["input_tokens"], usage["output_tokens"], usage["cost_usd"],
    )

    result = {
        "answer": answer,
        "sources": _docs_to_sources(reranked),
        "latency_ms": total_ms,
        "usage": usage,
    }
    if cache is not None:
        cache.put(question, result, scope=cache_scope)
    return attach_condense(result, cq)


async def stream_query(
    question: str,
    top_k: int | None = None,
    sources: list[str] | None = None,
    history: list[dict] | None = None,
    cache_scope: str = "",
) -> AsyncIterator[dict]:
    """Stream a grounded answer token-by-token.

    Retrieval is synchronous (and not streamable), so it runs in a thread and
    is emitted as a single ``sources`` event; generation is then streamed from
    the LLM one token at a time, finishing with a ``done`` event that carries
    the assembled answer and token/cost usage. When a history-condensed
    rewrite ran, a ``condensed`` event precedes ``sources``.
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.TOP_K
    start = time.time()

    if history:
        cq = await asyncio.to_thread(condense_question, question, history)
    else:
        cq = CondenseResult(question=question, applied=False)
    if cq.applied:
        yield {"event": "condensed", "condensed_question": cq.question}
        question = cq.question

    reranked = await asyncio.to_thread(_retrieve_and_rerank, question, top_k, settings, sources)
    retrieval_ms = (time.time() - start) * 1000
    trace_data = [
        {"content": d.page_content[:100], "score": d.metadata.get("relevance_score", 0)}
        for d in reranked
    ]
    trace_retrieval(question, trace_data, retrieval_ms)
    yield {"event": "sources", "sources": _docs_to_sources(reranked), "latency_ms": retrieval_ms}

    context = format_context(reranked)
    prompt_text = select_prompt(settings.PROMPT_MODE).format(context=context, question=question)
    from app.core.factories import message_text

    llm = get_llm()
    parts: list[str] = []
    async for chunk in llm.astream(prompt_text):
        token = message_text(getattr(chunk, "content", None))
        if token:
            parts.append(token)
            yield {"event": "token", "token": token}

    answer = "".join(parts)
    usage = usage_for(prompt_text, answer, str(settings.LLM_MODEL))
    total_ms = (time.time() - start) * 1000
    logger.info("stream_complete: latency_ms=%.1f cost_usd=%.6f", total_ms, usage["cost_usd"])
    done = attach_condense({"answer": answer, "usage": usage, "latency_ms": total_ms}, cq)
    yield {"event": "done", **done}
