from unittest.mock import MagicMock, patch

import pytest

from app.core.factories import clear_caches, get_embedder, get_llm, get_reranker


@pytest.fixture(autouse=True)
def _reset_caches():
    clear_caches()
    yield
    clear_caches()


def test_get_llm_openai():
    with patch("app.core.factories.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.LLM_PROVIDER = "openai"
            mock_settings.return_value.LLM_MODEL = "gpt-4o"
            mock_settings.return_value.LLM_API_KEY = "sk-test"
            mock_settings.return_value.LLM_BASE_URL = None
            llm = get_llm()
            mock_cls.assert_called_once()


def test_get_llm_anthropic():
    with patch("app.core.factories.ChatAnthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.LLM_PROVIDER = "anthropic"
            mock_settings.return_value.LLM_MODEL = "claude-sonnet-4-20250514"
            mock_settings.return_value.LLM_API_KEY = "sk-ant-test"
            mock_settings.return_value.LLM_BASE_URL = None
            llm = get_llm()
            mock_cls.assert_called_once()


def test_get_llm_unsupported():
    with patch("app.core.factories.get_settings") as mock_settings:
        mock_settings.return_value.LLM_PROVIDER = "ollama"
        mock_settings.return_value.LLM_MODEL = "llama3"
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm()


def test_get_embedder_openai():
    with patch("app.core.factories.OpenAIEmbeddings") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.EMBEDDING_PROVIDER = "openai"
            mock_settings.return_value.EMBEDDING_MODEL = "text-embedding-3-small"
            mock_settings.return_value.EMBEDDING_API_KEY = "sk-test"
            mock_settings.return_value.EMBEDDING_BASE_URL = None
            embedder = get_embedder()
            mock_cls.assert_called_once()


def test_get_reranker_cohere():
    with patch("app.core.factories.CohereRerank") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.RERANKER_PROVIDER = "cohere"
            mock_settings.return_value.RERANKER_MODEL = "rerank-v3"
            mock_settings.return_value.COHERE_API_KEY = "test-key"
            reranker = get_reranker()
            mock_cls.assert_called_once()


def test_get_reranker_none():
    with patch("app.core.factories.get_settings") as mock_settings:
        mock_settings.return_value.RERANKER_PROVIDER = "none"
        mock_settings.return_value.RERANKER_MODEL = "rerank-v3"
        reranker = get_reranker()
        assert reranker is None


def test_caching_returns_same_instance():
    """Verify that calling get_llm twice returns the cached instance."""
    with patch("app.core.factories.ChatOpenAI") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.LLM_PROVIDER = "openai"
            mock_settings.return_value.LLM_MODEL = "gpt-4o"
            mock_settings.return_value.LLM_API_KEY = "sk-test"
            mock_settings.return_value.LLM_BASE_URL = None
            llm1 = get_llm()
            llm2 = get_llm()
            assert llm1 is llm2
            mock_cls.assert_called_once()  # Only created once


def test_clear_caches():
    with patch("app.core.factories.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.LLM_PROVIDER = "openai"
            mock_settings.return_value.LLM_MODEL = "gpt-4o"
            mock_settings.return_value.LLM_API_KEY = "sk-test"
            mock_settings.return_value.LLM_BASE_URL = None
            get_llm()
            clear_caches()
            get_llm()
            assert mock_cls.call_count == 2  # Created twice after cache clear


def test_complete_routes_to_fast_model_when_fast():
    with patch("app.core.factories.get_llm") as mock_get, \
         patch("app.core.factories.get_settings") as mock_s:
        mock_s.return_value.LLM_MODEL = "gpt-4o"
        mock_s.return_value.LLM_MODEL_FAST = "gpt-4o-mini"
        mock_s.return_value.LLM_FALLBACK_MODEL = "gpt-4o-mini"
        mock_get.return_value.invoke.return_value = MagicMock(content="x")
        from app.core.factories import complete
        complete("p", fast=True)
        mock_get.assert_called_with("gpt-4o-mini")


def test_complete_uses_strong_model_by_default():
    with patch("app.core.factories.get_llm") as mock_get, \
         patch("app.core.factories.get_settings") as mock_s:
        mock_s.return_value.LLM_MODEL = "gpt-4o"
        mock_s.return_value.LLM_MODEL_FAST = "gpt-4o-mini"
        mock_s.return_value.LLM_FALLBACK_MODEL = "gpt-4o-mini"
        mock_get.return_value.invoke.return_value = MagicMock(content="x")
        from app.core.factories import complete
        complete("p")
        mock_get.assert_called_with("gpt-4o")


def test_complete_falls_back_on_error():
    with patch("app.core.factories.get_llm") as mock_get, \
         patch("app.core.factories.get_settings") as mock_s:
        mock_s.return_value.LLM_MODEL = "gpt-4o"
        mock_s.return_value.LLM_MODEL_FAST = "gpt-4o-mini"
        mock_s.return_value.LLM_FALLBACK_MODEL = "gpt-4o-mini"
        primary, fb = MagicMock(), MagicMock()
        primary.invoke.side_effect = RuntimeError("boom")
        fb.invoke.return_value = MagicMock(content="fallback answer")
        mock_get.side_effect = lambda m: primary if m == "gpt-4o" else fb
        from app.core.factories import complete
        assert complete("p") == "fallback answer"


def test_complete_reraises_without_fallback():
    with patch("app.core.factories.get_llm") as mock_get, \
         patch("app.core.factories.get_settings") as mock_s:
        mock_s.return_value.LLM_MODEL = "gpt-4o"
        mock_s.return_value.LLM_FALLBACK_MODEL = ""
        mock_get.return_value.invoke.side_effect = RuntimeError("boom")
        from app.core.factories import complete
        with pytest.raises(RuntimeError):
            complete("p")


def test_complete_with_model_returns_strong_model_by_default():
    with patch("app.core.factories.get_llm") as mock_get, \
         patch("app.core.factories.get_settings") as mock_s:
        mock_s.return_value.LLM_MODEL = "gpt-4o"
        mock_s.return_value.LLM_MODEL_FAST = "gpt-4o-mini"
        mock_s.return_value.LLM_FALLBACK_MODEL = "gpt-4o-mini"
        mock_get.return_value.invoke.return_value = MagicMock(content="x")
        from app.core.factories import complete_with_model
        assert complete_with_model("p") == ("x", "gpt-4o")


def test_complete_with_model_returns_fast_model_when_fast():
    with patch("app.core.factories.get_llm") as mock_get, \
         patch("app.core.factories.get_settings") as mock_s:
        mock_s.return_value.LLM_MODEL = "gpt-4o"
        mock_s.return_value.LLM_MODEL_FAST = "gpt-4o-mini"
        mock_s.return_value.LLM_FALLBACK_MODEL = "gpt-4o-mini"
        mock_get.return_value.invoke.return_value = MagicMock(content="x")
        from app.core.factories import complete_with_model
        assert complete_with_model("p", fast=True) == ("x", "gpt-4o-mini")


def test_complete_with_model_reports_fallback_model_on_error():
    """The fix: cost must attribute to the model that actually answered."""
    with patch("app.core.factories.get_llm") as mock_get, \
         patch("app.core.factories.get_settings") as mock_s:
        mock_s.return_value.LLM_MODEL = "gpt-4o"
        mock_s.return_value.LLM_MODEL_FAST = "gpt-4o-mini"
        mock_s.return_value.LLM_FALLBACK_MODEL = "gpt-4o-mini"
        primary, fb = MagicMock(), MagicMock()
        primary.invoke.side_effect = RuntimeError("boom")
        fb.invoke.return_value = MagicMock(content="fallback answer")
        mock_get.side_effect = lambda m: primary if m == "gpt-4o" else fb
        from app.core.factories import complete_with_model
        assert complete_with_model("p") == ("fallback answer", "gpt-4o-mini")


def test_get_vector_store_cached_and_cleared():
    from unittest.mock import patch

    from app.core.factories import get_vector_store

    clear_caches()
    with patch("app.retrieval.vector_store.VectorStore") as mock_cls:
        a = get_vector_store()
        b = get_vector_store()
        assert a is b
        assert mock_cls.call_count == 1
        clear_caches()
        get_vector_store()
        assert mock_cls.call_count == 2
    clear_caches()


def test_get_keyword_store_local_keyed_by_data_dir(tmp_path):
    from unittest.mock import patch

    from app.core.factories import get_keyword_store

    clear_caches()
    with patch("app.core.factories.BM25Store") as mock_cls, \
         patch("app.core.factories.get_settings") as mock_s:
        mock_cls.side_effect = lambda **kw: object()
        mock_s.return_value.KEYWORD_BACKEND = "local"
        mock_s.return_value.DATA_DIR = str(tmp_path / "a")
        a1 = get_keyword_store()
        a2 = get_keyword_store()
        assert a1 is a2
        mock_s.return_value.DATA_DIR = str(tmp_path / "b")
        b = get_keyword_store()
        assert b is not a1
    clear_caches()


def test_get_keyword_store_dispatches_to_opensearch():
    from unittest.mock import patch

    from app.core.factories import get_keyword_store

    clear_caches()
    with patch("app.core.factories.OpenSearchStore") as mock_os_cls, \
         patch("app.core.factories.BM25Store") as mock_bm_cls, \
         patch("app.core.factories.get_settings") as mock_s:
        mock_os_cls.side_effect = lambda **kw: object()
        mock_s.return_value.KEYWORD_BACKEND = "opensearch"
        mock_s.return_value.OPENSEARCH_URL = "http://test:9200"
        mock_s.return_value.OPENSEARCH_INDEX = "idx"
        s1 = get_keyword_store()
        s2 = get_keyword_store()
        assert s1 is s2
        mock_os_cls.assert_called_once_with(url="http://test:9200", index_name="idx")
        mock_bm_cls.assert_not_called()
        clear_caches()
        get_keyword_store()
        assert mock_os_cls.call_count == 2  # clear_caches() resets the store cache
    clear_caches()


def test_get_keyword_store_backends_cached_separately(tmp_path):
    from unittest.mock import patch

    from app.core.factories import get_keyword_store

    clear_caches()
    with patch("app.core.factories.OpenSearchStore") as mock_os_cls, \
         patch("app.core.factories.BM25Store") as mock_bm_cls, \
         patch("app.core.factories.get_settings") as mock_s:
        mock_os_cls.side_effect = lambda **kw: object()
        mock_bm_cls.side_effect = lambda **kw: object()
        mock_s.return_value.KEYWORD_BACKEND = "local"
        mock_s.return_value.DATA_DIR = str(tmp_path)
        local = get_keyword_store()
        mock_s.return_value.KEYWORD_BACKEND = "opensearch"
        mock_s.return_value.OPENSEARCH_URL = "http://test:9200"
        mock_s.return_value.OPENSEARCH_INDEX = "idx"
        remote = get_keyword_store()
        assert local is not remote
        mock_s.return_value.KEYWORD_BACKEND = "local"
        assert get_keyword_store() is local
    clear_caches()


def test_get_graph_store_cached(tmp_path):
    from unittest.mock import patch

    from app.core.factories import get_graph_store

    clear_caches()
    with patch("app.core.factories.GraphStore") as mock_cls, \
         patch("app.core.factories.get_settings") as mock_s:
        mock_cls.side_effect = lambda **kw: object()
        mock_s.return_value.DATA_DIR = str(tmp_path)
        g1 = get_graph_store()
        g2 = get_graph_store()
        assert g1 is g2
    clear_caches()

def test_message_text_strips_anthropic_thinking_blocks():
    from app.core.factories import message_text

    assert message_text("hello") == "hello"
    assert message_text([
        {"type": "thinking", "thinking": "plan"},
        {"type": "text", "text": "PONG"},
    ]) == "PONG"
    assert message_text(None) == ""
