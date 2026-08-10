import logging

from langchain_anthropic import ChatAnthropic
from langchain_cohere import CohereRerank
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import get_settings
from app.graph.store import GraphStore
from app.retrieval.bm25_store import BM25Store
from app.retrieval.opensearch_store import OpenSearchStore

logger = logging.getLogger(__name__)

_llm_cache: dict[tuple, object] = {}
_embedder_cache: dict[tuple, object] = {}
_reranker_cache: dict[tuple, object] = {}
_vector_store_cache: dict[tuple, object] = {}
_image_vector_store_cache: dict[tuple, object] = {}
_keyword_store_cache: dict[tuple, object] = {}
_graph_store_cache: dict[tuple, object] = {}


def get_llm(model: str | None = None):
    settings = get_settings()
    model = model or settings.LLM_MODEL
    key = (settings.LLM_PROVIDER, model)
    if key in _llm_cache:
        return _llm_cache[key]
    if settings.LLM_PROVIDER == "openai":
        instance = ChatOpenAI(
            model=model,
            api_key=settings.LLM_API_KEY or None,
            base_url=settings.LLM_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
        )
    elif settings.LLM_PROVIDER == "anthropic":
        instance = ChatAnthropic(
            model=model,
            api_key=settings.LLM_API_KEY or None,
            base_url=settings.LLM_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
    _llm_cache[key] = instance
    logger.info("Created LLM client: provider=%s model=%s", settings.LLM_PROVIDER, model)
    return instance


def message_text(content) -> str:
    """Normalize LLM message content to a plain string.

    OpenAI-style models return ``str``; Anthropic (esp. with thinking) may
    return a list of blocks ``[{"type":"thinking",...},{"type":"text","text":...}]``
    or LangChain content-block objects. Callers that ``"".join`` tokens need text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in ("thinking", "redacted_thinking"):
                    continue
                text = block.get("text")
                if text:
                    parts.append(str(text))
            else:
                btype = getattr(block, "type", None)
                if btype in ("thinking", "redacted_thinking"):
                    continue
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def _content(resp) -> str:
    return message_text(getattr(resp, "content", None))


def complete_with_model(prompt: str, *, fast: bool = False) -> tuple[str, str]:
    """Invoke the routed model (fast vs strong), falling back to
    LLM_FALLBACK_MODEL (same provider) on any error. Returns
    ``(text, model_that_answered)`` so callers can attribute cost to the model
    that actually produced the answer (it differs from the configured model
    after a fallback)."""
    settings = get_settings()
    model = settings.LLM_MODEL_FAST if fast else settings.LLM_MODEL
    try:
        return _content(get_llm(model).invoke(prompt)), model
    except Exception as exc:  # noqa: BLE001
        fallback = settings.LLM_FALLBACK_MODEL
        if not fallback or fallback == model:
            raise
        logger.warning("llm_fallback: model=%s failed (%s), retrying with %s", model, exc, fallback)
        return _content(get_llm(fallback).invoke(prompt)), fallback


def complete(prompt: str, *, fast: bool = False) -> str:
    """Invoke the routed model (fast vs strong) and fall back to
    LLM_FALLBACK_MODEL (same provider) on any error."""
    return complete_with_model(prompt, fast=fast)[0]


def get_embedder():
    settings = get_settings()
    key = (settings.EMBEDDING_PROVIDER, settings.EMBEDDING_MODEL)
    if key in _embedder_cache:
        return _embedder_cache[key]
    if settings.EMBEDDING_PROVIDER == "openai":
        # Custom / OpenAI-compatible endpoints (e.g. DashScope) expect raw strings.
        # Default check_embedding_ctx_length tokenizes via tiktoken and breaks them.
        # DashScope also caps batch size at 10 texts per request.
        kwargs = {
            "model": settings.EMBEDDING_MODEL,
            "api_key": settings.EMBEDDING_API_KEY or None,
            "base_url": settings.EMBEDDING_BASE_URL,
        }
        if settings.EMBEDDING_BASE_URL:
            kwargs["check_embedding_ctx_length"] = False
            kwargs["chunk_size"] = 10
        instance = OpenAIEmbeddings(**kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")
    _embedder_cache[key] = instance
    logger.info("Created embedder client: provider=%s model=%s", settings.EMBEDDING_PROVIDER, settings.EMBEDDING_MODEL)
    return instance


def get_reranker():
    settings = get_settings()
    key = (settings.RERANKER_PROVIDER, settings.RERANKER_MODEL)
    if key in _reranker_cache:
        return _reranker_cache[key]
    if settings.RERANKER_PROVIDER == "none":
        _reranker_cache[key] = None
        return None
    if settings.RERANKER_PROVIDER == "cohere":
        instance = CohereRerank(
            model=settings.RERANKER_MODEL,
            cohere_api_key=settings.COHERE_API_KEY,
        )
    else:
        raise ValueError(f"Unsupported reranker provider: {settings.RERANKER_PROVIDER}")
    _reranker_cache[key] = instance
    logger.info("Created reranker client: provider=%s model=%s", settings.RERANKER_PROVIDER, settings.RERANKER_MODEL)
    return instance


def get_vector_store():
    """Cached VectorStore: reuses the Qdrant connection across queries."""
    # Imported lazily: vector_store.py imports get_embedder from this module.
    from app.retrieval.vector_store import VectorStore

    settings = get_settings()
    key = (settings.QDRANT_URL, settings.COLLECTION_NAME)
    if key not in _vector_store_cache:
        _vector_store_cache[key] = VectorStore()
        logger.info("Created vector store: url=%s collection=%s", key[0], key[1])
    return _vector_store_cache[key]


def get_image_vector_store():
    """Cached ImageVectorStore for the qwen3-vl-embedding dual track."""
    from app.retrieval.image_vector_store import ImageVectorStore

    settings = get_settings()
    key = (settings.QDRANT_URL, settings.VL_COLLECTION_NAME, settings.VL_EMBEDDING_DIMENSION)
    if key not in _image_vector_store_cache:
        _image_vector_store_cache[key] = ImageVectorStore()
        logger.info(
            "Created image vector store: url=%s collection=%s dim=%s",
            key[0],
            key[1],
            key[2],
        )
    return _image_vector_store_cache[key]


def image_retrieval_ready() -> bool:
    """True when image track is enabled and DashScope credentials are present."""
    settings = get_settings()
    return bool(settings.IMAGE_RETRIEVAL_ENABLED and settings.DASHSCOPE_API_KEY)


def get_keyword_store():
    """Cached keyword store: local BM25 (default) or OpenSearch, per KEYWORD_BACKEND."""
    settings = get_settings()
    if settings.KEYWORD_BACKEND == "opensearch":
        key = ("opensearch", settings.OPENSEARCH_URL, settings.OPENSEARCH_INDEX)
        if key not in _keyword_store_cache:
            _keyword_store_cache[key] = OpenSearchStore(
                url=settings.OPENSEARCH_URL, index_name=settings.OPENSEARCH_INDEX
            )
            logger.info("Created keyword store: backend=opensearch url=%s index=%s", key[1], key[2])
        return _keyword_store_cache[key]
    key = ("local", settings.DATA_DIR)
    if key not in _keyword_store_cache:
        _keyword_store_cache[key] = BM25Store(data_dir=settings.DATA_DIR)
        logger.info("Created keyword store: backend=local data_dir=%s", key[1])
    return _keyword_store_cache[key]


def get_graph_store():
    """Cached GraphStore: the knowledge graph stays in memory across queries."""
    settings = get_settings()
    key = (settings.DATA_DIR,)
    if key not in _graph_store_cache:
        _graph_store_cache[key] = GraphStore(data_dir=settings.DATA_DIR)
        logger.info("Created graph store: data_dir=%s", key[0])
    return _graph_store_cache[key]


def clear_caches() -> None:
    """Clear all factory caches. Useful for testing."""
    _llm_cache.clear()
    _embedder_cache.clear()
    _reranker_cache.clear()
    _vector_store_cache.clear()
    _image_vector_store_cache.clear()
    _keyword_store_cache.clear()
    _graph_store_cache.clear()
