"""ragas 0.4.3 + langchain-community 0.4.x import compatibility."""


def test_ensure_ragas_importable_allows_import():
    from evaluation._compat import ensure_ragas_importable

    ensure_ragas_importable()
    import ragas  # noqa: F401
    from ragas.llms import LangchainLLMWrapper  # noqa: F401
    from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: F401
