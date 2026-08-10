from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def test_ingest_pipeline(tmp_path):
    with patch("app.core.pipeline.get_embedder") as mock_emb, \
         patch("app.core.pipeline.get_vector_store") as mock_vs_cls, \
         patch("app.core.pipeline.get_keyword_store") as mock_bm25_cls, \
         patch("app.core.pipeline.GraphBuilder") as mock_gb_cls, \
         patch("app.core.pipeline.get_graph_store") as mock_gs_cls, \
         patch("app.core.pipeline.load_documents") as mock_load, \
         patch("app.core.pipeline.chunk_documents") as mock_chunk, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.get_llm") as mock_llm_f, \
         patch("app.core.pipeline._load_tracking", return_value={}), \
         patch("app.core.pipeline._save_tracking"):

        mock_s.return_value.GRAPH_EXTRACTOR = "llm"
        mock_s.return_value.DATA_DIR = str(tmp_path)
        mock_s.return_value.CHUNK_SIZE = 512
        mock_s.return_value.CHUNK_OVERLAP = 64

        mock_load.return_value = [Document(page_content="test content")]
        mock_chunk.return_value = [Document(page_content="chunked content")]

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.upsert.return_value = ["id1"]

        mock_bm25 = MagicMock()
        mock_bm25_cls.return_value = mock_bm25

        mock_llm = MagicMock()
        mock_llm_f.return_value = mock_llm

        mock_gb = MagicMock()
        mock_gb_cls.return_value = mock_gb
        mock_gb.extract.return_value = [{"head": "A", "relation": "r", "tail": "B"}]

        mock_gs = MagicMock()
        mock_gs_cls.return_value = mock_gs

        from app.core.pipeline import ingest_pipeline
        result = ingest_pipeline("test.md")
        assert result["chunks"] == 1
        assert result["status"] == "ingested"


def test_query_pipeline():
    with patch("app.core.pipeline.complete_with_model", return_value=("Generated answer", "gpt-4o")), \
         patch("app.core.pipeline.get_embedder") as mock_emb, \
         patch("app.core.pipeline.get_reranker") as mock_rr_f, \
         patch("app.core.pipeline.get_vector_store") as mock_vs_cls, \
         patch("app.core.pipeline.get_keyword_store") as mock_bm25_cls, \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.GraphRetriever") as mock_gr_cls, \
         patch("app.core.pipeline.get_graph_store") as mock_gs_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval") as mock_trace:

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"

        mock_rr = MagicMock()
        mock_rr_f.return_value = mock_rr

        doc = Document(page_content="context doc")
        mock_hr = MagicMock()
        mock_hr_cls.return_value = mock_hr
        mock_hr.retrieve.return_value = [(doc, 0.9)]

        mock_rs = MagicMock()
        mock_rs_cls.return_value = mock_rs
        mock_rs.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        result = query_pipeline("What is AI?")
        assert result["answer"] == "Generated answer"
        assert len(result["sources"]) > 0


def test_query_pipeline_returns_full_source_content():
    """Sources must carry full chunk content; truncation corrupts RAGAS evaluation."""
    long_content = "x" * 600

    with patch("app.core.pipeline.complete_with_model", return_value=("Generated answer", "gpt-4o")), \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.get_vector_store"), \
         patch("app.core.pipeline.get_keyword_store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"):

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"

        doc = Document(page_content=long_content)
        mock_hr_cls.return_value.retrieve.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        result = query_pipeline("What is AI?")
        assert result["sources"][0]["content"] == long_content


def test_query_pipeline_respects_top_k_argument():
    with patch("app.core.pipeline.complete_with_model", return_value=("Generated answer", "gpt-4o")), \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.get_vector_store"), \
         patch("app.core.pipeline.get_keyword_store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"):

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"

        doc = Document(page_content="context doc")
        mock_retriever = mock_hr_cls.return_value
        mock_retriever.retrieve.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        query_pipeline("What is AI?", top_k=12)
        mock_retriever.retrieve.assert_called_once_with("What is AI?", top_k=12, sources=None)


def test_query_pipeline_short_circuits_on_cache_hit():
    """A cache hit must skip retrieval and generation entirely."""
    fake_cache = MagicMock()
    fake_cache.get.return_value = {
        "answer": "CACHED", "sources": [], "latency_ms": 1.0, "usage": {}
    }
    with patch("app.core.pipeline._get_query_cache", return_value=fake_cache), \
         patch("app.core.pipeline.complete_with_model") as mock_complete, \
         patch("app.core.pipeline._retrieve_and_rerank") as mock_rr, \
         patch("app.core.pipeline.get_settings") as mock_s:

        mock_s.return_value.TOP_K = 5
        from app.core.pipeline import query_pipeline
        result = query_pipeline("q")

        assert result["answer"] == "CACHED"
        assert result["cached"] is True
        mock_complete.assert_not_called()
        mock_rr.assert_not_called()


def test_query_pipeline_stores_result_in_cache_on_miss():
    fake_cache = MagicMock()
    fake_cache.get.return_value = None

    with patch("app.core.pipeline._get_query_cache", return_value=fake_cache), \
         patch("app.core.pipeline.complete_with_model", return_value=("fresh answer", "gpt-4o")), \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.get_vector_store"), \
         patch("app.core.pipeline.get_keyword_store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"):

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.RETRIEVAL_MODE = "hybrid"
        mock_s.return_value.PROMPT_MODE = "grounded"
        mock_s.return_value.LLM_MODEL = "gpt-4o"

        doc = Document(page_content="ctx", metadata={"source": "a.pdf"})
        mock_hr_cls.return_value.retrieve.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        query_pipeline("q")

        fake_cache.put.assert_called_once()


def test_query_pipeline_passes_numbered_context_to_llm():
    """Grounded generation needs the context numbered so [n] citations resolve."""
    with patch("app.core.pipeline.complete_with_model", return_value=("answer [1]", "gpt-4o")) as mock_complete, \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.get_vector_store"), \
         patch("app.core.pipeline.get_keyword_store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"):

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.PROMPT_MODE = "grounded"

        doc = Document(page_content="Alpha content", metadata={"source": "a.pdf"})
        mock_hr_cls.return_value.retrieve.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        query_pipeline("What is AI?")

        passed = mock_complete.call_args[0][0]
        assert "[1] (source: a.pdf)" in passed
        assert "Alpha content" in passed


def test_query_pipeline_stamps_citation_numbers_on_sources():
    with patch("app.core.pipeline.complete_with_model", return_value=("answer", "gpt-4o")), \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.get_vector_store"), \
         patch("app.core.pipeline.get_keyword_store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"):

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.PROMPT_MODE = "grounded"

        doc1 = Document(page_content="first", metadata={"source": "a.pdf"})
        doc2 = Document(page_content="second", metadata={"source": "b.pdf"})
        mock_hr_cls.return_value.retrieve.return_value = [(doc1, 0.9), (doc2, 0.8)]
        mock_rs_cls.return_value.rerank.return_value = [doc1, doc2]

        from app.core.pipeline import query_pipeline
        result = query_pipeline("What is AI?")

        assert result["sources"][0]["metadata"]["citation"] == 1
        assert result["sources"][1]["metadata"]["citation"] == 2


def test_query_pipeline_dense_mode_uses_vector_only():
    """RETRIEVAL_MODE=dense is the ablation baseline: vector search only,
    no BM25 store and no RRF fusion."""
    with patch("app.core.pipeline.complete_with_model", return_value=("answer", "gpt-4o")), \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.get_vector_store") as mock_vs_cls, \
         patch("app.core.pipeline.get_keyword_store") as mock_bm25_cls, \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"):

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.PROMPT_MODE = "grounded"
        mock_s.return_value.RETRIEVAL_MODE = "dense"

        doc = Document(page_content="context doc", metadata={"source": "a.pdf"})
        mock_vs = mock_vs_cls.return_value
        mock_vs.search.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        query_pipeline("What is AI?", top_k=7)

        mock_vs.search.assert_called_once_with("What is AI?", top_k=7, sources=None)
        mock_hr_cls.return_value.retrieve.assert_not_called()
        mock_bm25_cls.assert_not_called()


def test_retrieve_sources_returns_citations_without_generation():
    """retrieve_sources powers the cheap retrieval-only eval and the UI:
    it must return cited sources without ever calling the generation LLM."""
    with patch("app.core.pipeline.get_llm") as mock_llm_f, \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.get_vector_store"), \
         patch("app.core.pipeline.get_keyword_store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"):

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.RETRIEVAL_MODE = "hybrid"

        doc = Document(page_content="ctx", metadata={"source": "a.pdf"})
        mock_hr_cls.return_value.retrieve.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import retrieve_sources
        sources = retrieve_sources("q")

        assert sources[0]["content"] == "ctx"
        assert sources[0]["metadata"]["citation"] == 1
        mock_llm_f.assert_not_called()


def test_retrieve_sources_multi_query_fuses_per_query_results():
    """multi_query retrieves for the original + each paraphrase and fuses."""
    with patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.get_vector_store"), \
         patch("app.core.pipeline.get_keyword_store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"), \
         patch("app.core.pipeline.get_llm") as mock_llm_f:

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.RETRIEVAL_MODE = "hybrid"
        mock_s.return_value.QUERY_TRANSFORM = "multi_query"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="alt one\nalt two")
        mock_llm_f.return_value = mock_llm

        doc_a = Document(page_content="A", metadata={"source": "a"})
        doc_b = Document(page_content="B", metadata={"source": "b"})
        doc_c = Document(page_content="C", metadata={"source": "c"})
        mock_hr_cls.return_value.retrieve.side_effect = [
            [(doc_a, 0.9)], [(doc_b, 0.8)], [(doc_c, 0.7)]
        ]
        mock_rs_cls.return_value.rerank.side_effect = lambda q, docs, top_k: docs[:top_k]

        from app.core.pipeline import retrieve_sources
        sources = retrieve_sources("orig")

        assert mock_hr_cls.return_value.retrieve.call_count == 3  # orig + 2 paraphrases
        contents = {s["content"] for s in sources}
        assert {"A", "B", "C"} <= contents


def test_retrieve_sources_hyde_searches_with_hypothetical_document():
    with patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.get_vector_store"), \
         patch("app.core.pipeline.get_keyword_store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"), \
         patch("app.core.pipeline.get_llm") as mock_llm_f:

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.RETRIEVAL_MODE = "hybrid"
        mock_s.return_value.QUERY_TRANSFORM = "hyde"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Hypothetical answer passage.")
        mock_llm_f.return_value = mock_llm

        doc_a = Document(page_content="A", metadata={"source": "a"})
        mock_hr_cls.return_value.retrieve.return_value = [(doc_a, 0.9)]
        mock_rs_cls.return_value.rerank.side_effect = lambda q, docs, top_k: docs[:top_k]

        from app.core.pipeline import retrieve_sources
        retrieve_sources("orig")

        mock_hr_cls.return_value.retrieve.assert_called_once_with(
            "Hypothetical answer passage.", top_k=5, sources=None
        )


def test_query_pipeline_reports_token_usage_and_cost():
    with patch("app.core.pipeline.complete_with_model", return_value=("Generated answer", "gpt-4o")), \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.get_vector_store"), \
         patch("app.core.pipeline.get_keyword_store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"):

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.RETRIEVAL_MODE = "hybrid"
        mock_s.return_value.PROMPT_MODE = "grounded"
        mock_s.return_value.LLM_MODEL = "gpt-4o"

        doc = Document(page_content="context doc", metadata={"source": "a.pdf"})
        mock_hr_cls.return_value.retrieve.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        result = query_pipeline("What is AI?")

        assert "usage" in result
        assert result["usage"]["model"] == "gpt-4o"
        assert result["usage"]["output_tokens"] > 0
        assert result["usage"]["cost_usd"] >= 0


def test_query_pipeline_attributes_cost_to_model_that_answered():
    """On fallback, usage must reflect the model that actually answered, not LLM_MODEL."""
    with patch("app.core.pipeline.complete_with_model", return_value=("ans", "gpt-4o-mini")), \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.get_vector_store"), \
         patch("app.core.pipeline.get_keyword_store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"):

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.RETRIEVAL_MODE = "hybrid"
        mock_s.return_value.PROMPT_MODE = "grounded"
        mock_s.return_value.LLM_MODEL = "gpt-4o"  # strong configured, but mini answered

        doc = Document(page_content="ctx", metadata={"source": "a.pdf"})
        mock_hr_cls.return_value.retrieve.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        result = query_pipeline("q")
        assert result["usage"]["model"] == "gpt-4o-mini"


async def test_stream_query_streams_tokens_then_done():
    """Real streaming: emit sources first, then one event per token, then a
    terminal 'done' with the assembled answer and usage."""
    class Chunk:
        def __init__(self, content):
            self.content = content

    async def fake_astream(_text):
        for tok in ["Hello", " world"]:
            yield Chunk(tok)

    doc = Document(page_content="ctx", metadata={"source": "a.pdf"})
    with patch("app.core.pipeline._retrieve_and_rerank", return_value=[doc]), \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"), \
         patch("app.core.pipeline.get_llm") as mock_llm_f:

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.PROMPT_MODE = "grounded"
        mock_s.return_value.LLM_MODEL = "gpt-4o"

        mock_llm = MagicMock()
        mock_llm.astream = fake_astream
        mock_llm_f.return_value = mock_llm

        from app.core.pipeline import stream_query
        events = [e async for e in stream_query("q")]

    kinds = [e["event"] for e in events]
    assert kinds[0] == "sources"
    assert kinds.count("token") == 2
    assert kinds[-1] == "done"
    assert events[-1]["answer"] == "Hello world"
    assert events[-1]["usage"]["model"] == "gpt-4o"


def test_list_documents_reads_tracking(tmp_path):
    import json
    from unittest.mock import patch
    (tmp_path / "ingestions.json").write_text(json.dumps({
        "h1": {"source": "a.pdf", "chunks": 3, "ingested_at": "2026-01-01"}
    }))
    with patch("app.core.pipeline.get_settings") as mock_s:
        mock_s.return_value.DATA_DIR = str(tmp_path)
        from app.core.pipeline import list_documents
        docs = list_documents()
        assert docs == [
            {
                "id": "h1",
                "source": "a.pdf",
                "chunks": 3,
                "ingested_at": "2026-01-01",
                "modality": "text",
            }
        ]


def test_retrieve_and_rerank_skips_graph_when_scoped(monkeypatch):
    import app.core.pipeline as pipe
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "RETRIEVAL_MODE", "dense", raising=False)
    monkeypatch.setattr(settings, "GRAPH_EXTRACTOR", "nlp", raising=False)
    monkeypatch.setattr(settings, "IMAGE_RETRIEVAL_ENABLED", False, raising=False)

    class _VS:
        def search(self, q, top_k=5, sources=None):
            return []
    monkeypatch.setattr(pipe, "get_vector_store", lambda: _VS())

    called = {"graph": False}

    class _GR:
        def __init__(self, *a, **k): pass
        def retrieve(self, *a, **k):
            called["graph"] = True
            return []
    monkeypatch.setattr(pipe, "GraphRetriever", _GR)
    monkeypatch.setattr(pipe, "get_graph_store", lambda **k: object())
    monkeypatch.setattr(pipe, "RerankerService", lambda reranker: type("R", (), {"rerank": lambda self, q, d, top_k: d})())
    monkeypatch.setattr(pipe, "get_reranker", lambda: object())

    pipe._retrieve_and_rerank("q", 5, settings, sources=["only.pdf"])
    assert called["graph"] is False


def test_query_pipeline_cache_isolated_by_scope(monkeypatch):
    """Scoped lookups must pass cache_scope so ACL/users cannot share hits."""
    import app.core.pipeline as pipe

    monkeypatch.setattr(pipe, "_retrieve_and_rerank", lambda *a, **k: [])
    monkeypatch.setattr(pipe, "format_context", lambda docs: "")
    monkeypatch.setattr(pipe, "complete_with_model", lambda prompt: ("answer", "model"))
    monkeypatch.setattr(pipe, "trace_retrieval", lambda *a, **k: None)

    seen = {}

    class Cache:
        def get(self, q, scope=""):
            seen["get"] = (q, scope)
            # Simulate a hit for a different scope — must not be returned when scope differs
            if scope == "user-a|*":
                return {"answer": "CACHED", "sources": [], "latency_ms": 1.0, "usage": {}}
            return None

        def put(self, q, r, scope=""):
            seen["put"] = (q, scope)

    monkeypatch.setattr(pipe, "_get_query_cache", lambda settings: Cache())

    out = pipe.query_pipeline("q", top_k=3, sources=["only.pdf"], cache_scope="user-b|only.pdf")
    assert out["answer"] == "answer"
    assert seen["get"] == ("q", "user-b|only.pdf")
    assert seen["put"][1] == "user-b|only.pdf"


def test_retrieve_and_rerank_parallel_matches_serial_semantics():
    """Parallel refactor must be score-neutral: same docs, same order."""
    from unittest.mock import MagicMock, patch

    from langchain_core.documents import Document as Doc

    d1, d2 = Doc(page_content="v1"), Doc(page_content="g1")

    with patch("app.core.pipeline.get_vector_store") as mock_vs_f, \
         patch("app.core.pipeline.get_keyword_store") as mock_bm_f, \
         patch("app.core.pipeline.get_graph_store") as mock_gs_f, \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.GraphRetriever") as mock_gr_cls, \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls:

        mock_hr_cls.return_value.retrieve.return_value = [(d1, 0.9)]
        mock_gr_cls.return_value.retrieve.return_value = [d2]
        # Reranker passthrough: output = input, so ordering is observable
        mock_rs_cls.return_value.rerank.side_effect = lambda q, docs, top_k: docs[:top_k]

        settings = MagicMock()
        settings.RETRIEVAL_MODE = "hybrid"
        settings.GRAPH_EXTRACTOR = "llm"
        settings.QUERY_TRANSFORM = "none"
        settings.RERANK_TOP_K = 5
        settings.DATA_DIR = "./data"

        from app.core.pipeline import _retrieve_and_rerank
        result = _retrieve_and_rerank("q", 5, settings)

        # Hybrid docs first (in fused order), then graph docs — as in serial code
        assert [d.page_content for d in result] == ["v1", "g1"]


def test_query_pipeline_threads_condensed_question_through_cache_and_llm():
    from app.core.condense import CondenseResult

    fake_cache = MagicMock()
    fake_cache.get.return_value = None
    cond_usage = {"input_tokens": 20, "output_tokens": 6, "cost_usd": 0.00001, "model": "gpt-4o-mini"}
    with patch("app.core.pipeline.condense_question",
               return_value=CondenseResult("standalone?", True, cond_usage)) as mock_cond, \
         patch("app.core.pipeline._get_query_cache", return_value=fake_cache), \
         patch("app.core.pipeline._retrieve_and_rerank", return_value=[]) as mock_rr, \
         patch("app.core.pipeline.complete_with_model", return_value=("ans", "gpt-4o")) as mock_llm, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"):
        mock_s.return_value.TOP_K = 5
        mock_s.return_value.PROMPT_MODE = "grounded"
        from app.core.pipeline import query_pipeline
        result = query_pipeline("follow up?", history=[{"role": "user", "content": "What is X?"}])

        mock_cond.assert_called_once_with("follow up?", [{"role": "user", "content": "What is X?"}])
        fake_cache.get.assert_called_once_with("standalone?", scope="")
        assert mock_rr.call_args.args[0] == "standalone?"
        prompt = mock_llm.call_args.args[0]
        assert "standalone?" in prompt and "follow up?" not in prompt
        assert result["condensed_question"] == "standalone?"
        assert result["usage"]["condense"] == cond_usage
        # the cached payload stays conversation-free
        put_key, put_value = fake_cache.put.call_args.args[:2]
        assert put_key == "standalone?"
        assert fake_cache.put.call_args.kwargs.get("scope", "") == ""
        assert "condensed_question" not in put_value
        assert "condense" not in put_value["usage"]


def test_query_pipeline_no_history_never_calls_condense_llm():
    with patch("app.core.condense.complete_with_model") as mock_fast, \
         patch("app.core.pipeline._get_query_cache", return_value=None), \
         patch("app.core.pipeline._retrieve_and_rerank", return_value=[]), \
         patch("app.core.pipeline.complete_with_model", return_value=("ans", "gpt-4o")), \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"):
        mock_s.return_value.TOP_K = 5
        mock_s.return_value.PROMPT_MODE = "grounded"
        from app.core.pipeline import query_pipeline
        result = query_pipeline("plain question?")
        mock_fast.assert_not_called()
        assert result["condensed_question"] is None


def test_query_pipeline_cache_hit_carries_condense_transparency():
    from app.core.condense import CondenseResult

    fake_cache = MagicMock()
    fake_cache.get.return_value = {
        "answer": "CACHED", "sources": [], "latency_ms": 1.0, "usage": {"cost_usd": 0.001}
    }
    cond_usage = {"input_tokens": 20, "output_tokens": 6, "cost_usd": 0.00001, "model": "gpt-4o-mini"}
    with patch("app.core.pipeline.condense_question",
               return_value=CondenseResult("standalone?", True, cond_usage)), \
         patch("app.core.pipeline._get_query_cache", return_value=fake_cache), \
         patch("app.core.pipeline._retrieve_and_rerank") as mock_rr, \
         patch("app.core.pipeline.get_settings") as mock_s:
        mock_s.return_value.TOP_K = 5
        from app.core.pipeline import query_pipeline
        result = query_pipeline("f?", history=[{"role": "user", "content": "x"}])

        fake_cache.get.assert_called_once_with("standalone?", scope="")
        mock_rr.assert_not_called()
        assert result["cached"] is True
        assert result["condensed_question"] == "standalone?"
        assert result["usage"]["condense"] == cond_usage


async def test_stream_query_emits_condensed_event_before_sources():
    from app.core.condense import CondenseResult

    class Chunk:
        def __init__(self, content):
            self.content = content

    async def fake_astream(_text):
        yield Chunk("Hi")

    cond_usage = {"input_tokens": 20, "output_tokens": 6, "cost_usd": 0.00001, "model": "gpt-4o-mini"}
    with patch("app.core.pipeline.condense_question",
               return_value=CondenseResult("standalone?", True, cond_usage)), \
         patch("app.core.pipeline._retrieve_and_rerank", return_value=[]) as mock_rr, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"), \
         patch("app.core.pipeline.get_llm") as mock_llm_f:
        mock_s.return_value.TOP_K = 5
        mock_s.return_value.PROMPT_MODE = "grounded"
        mock_s.return_value.LLM_MODEL = "gpt-4o"
        mock_llm = MagicMock()
        mock_llm.astream = fake_astream
        mock_llm_f.return_value = mock_llm

        from app.core.pipeline import stream_query
        events = [e async for e in stream_query("f?", history=[{"role": "user", "content": "x"}])]

    kinds = [e["event"] for e in events]
    assert kinds[0] == "condensed"
    assert events[0]["condensed_question"] == "standalone?"
    assert kinds[1] == "sources"
    assert mock_rr.call_args.args[0] == "standalone?"
    assert events[-1]["usage"]["condense"] == cond_usage
