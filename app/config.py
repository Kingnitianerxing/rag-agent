from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str | None = None
    LLM_MODEL_FAST: str = "gpt-4o-mini"
    LLM_FALLBACK_MODEL: str = "gpt-4o-mini"   # "" disables fallback
    LLM_TIMEOUT: int = 30

    # Embedding
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str | None = None

    # Multimodal (DashScope qwen3-vl-embedding) — image track, separate from text embeddings
    DASHSCOPE_API_KEY: str = ""
    VL_EMBEDDING_MODEL: str = "qwen3-vl-embedding"
    VL_EMBEDDING_DIMENSION: int = 1024
    VL_COLLECTION_NAME: str = "rag_images"
    VL_CAPTION_MODEL: str = "qwen-vl-plus"  # vision LLM to describe images for grounded generation
    IMAGE_RETRIEVAL_ENABLED: bool = False

    # Reranker
    RERANKER_PROVIDER: str = "cohere"
    RERANKER_MODEL: str = "rerank-v3.5"
    COHERE_API_KEY: str = ""

    # Graph
    GRAPH_EXTRACTOR: str = "llm"

    # Generation
    PROMPT_MODE: str = "grounded"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    COLLECTION_NAME: str = "rag_docs"

    # LangSmith
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "production-rag"
    LANGSMITH_TRACING: bool = False

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Retrieval
    RETRIEVAL_MODE: str = "hybrid"
    QUERY_TRANSFORM: str = "none"
    TOP_K: int = 5
    RERANK_TOP_K: int = 5

    # Keyword backend
    KEYWORD_BACKEND: str = "local"                   # local (rank_bm25) | opensearch
    OPENSEARCH_URL: str = "http://localhost:9200"
    OPENSEARCH_INDEX: str = "rag_chunks"             # scale corpus: rag_chunks_scale30

    # Cache
    CACHE_ENABLED: bool = False
    CACHE_SIMILARITY_THRESHOLD: float = 0.95
    REDIS_URL: str = ""   # e.g. redis://localhost:6379/0; empty = in-process cache

    # Chat history (multi-turn condense)
    HISTORY_CONDENSE_ENABLED: bool = True            # kill switch for the condense step
    CHAT_HISTORY_MAX_TURNS: int = 10                 # last N turns kept server-side
    CHAT_HISTORY_MAX_TURN_CHARS: int = 2000          # per-turn content cap

    # Agent
    AGENT_MAX_REWRITES: int = 2

    # Guardrails
    GUARDRAILS_ENABLED: bool = True

    # MCP
    MCP_ALLOW_INGEST: bool = True

    # Data
    DATA_DIR: str = "./data"

    # Security / Auth
    API_KEY_HASH: str = ""
    AUTH_ENABLED: bool = False
    JWT_SECRET: str = ""
    JWT_EXPIRE_MINUTES: int = 60 * 24
    BOOTSTRAP_ADMIN_USERNAME: str = ""
    BOOTSTRAP_ADMIN_PASSWORD: str = ""
    CORS_ORIGINS: str = "*"
    MAX_FILE_SIZE_MB: int = 100

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        if v not in ("openai", "anthropic"):
            raise ValueError(f"LLM_PROVIDER must be 'openai' or 'anthropic', got '{v}'")
        return v

    @field_validator("LLM_TIMEOUT")
    @classmethod
    def validate_llm_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"LLM_TIMEOUT must be > 0, got {v}")
        return v

    @field_validator("EMBEDDING_PROVIDER")
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        if v not in ("openai",):
            raise ValueError(f"EMBEDDING_PROVIDER must be 'openai', got '{v}'")
        return v

    @field_validator("RERANKER_PROVIDER")
    @classmethod
    def validate_reranker_provider(cls, v: str) -> str:
        if v not in ("none", "cohere"):
            raise ValueError(f"RERANKER_PROVIDER must be 'none' or 'cohere', got '{v}'")
        return v

    @field_validator("GRAPH_EXTRACTOR")
    @classmethod
    def validate_graph_extractor(cls, v: str) -> str:
        if v not in ("none", "llm", "nlp"):
            raise ValueError(f"GRAPH_EXTRACTOR must be 'none', 'llm', or 'nlp', got '{v}'")
        return v

    @field_validator("PROMPT_MODE")
    @classmethod
    def validate_prompt_mode(cls, v: str) -> str:
        if v not in ("basic", "grounded"):
            raise ValueError(f"PROMPT_MODE must be 'basic' or 'grounded', got '{v}'")
        return v

    @field_validator("RETRIEVAL_MODE")
    @classmethod
    def validate_retrieval_mode(cls, v: str) -> str:
        if v not in ("dense", "hybrid"):
            raise ValueError(f"RETRIEVAL_MODE must be 'dense' or 'hybrid', got '{v}'")
        return v

    @field_validator("QUERY_TRANSFORM")
    @classmethod
    def validate_query_transform(cls, v: str) -> str:
        if v not in ("none", "multi_query", "hyde"):
            raise ValueError(f"QUERY_TRANSFORM must be 'none', 'multi_query', or 'hyde', got '{v}'")
        return v

    @field_validator("KEYWORD_BACKEND")
    @classmethod
    def validate_keyword_backend(cls, v: str) -> str:
        if v not in ("local", "opensearch"):
            raise ValueError(f"KEYWORD_BACKEND must be 'local' or 'opensearch', got '{v}'")
        return v

    @field_validator("CACHE_SIMILARITY_THRESHOLD")
    @classmethod
    def validate_cache_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"CACHE_SIMILARITY_THRESHOLD must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("CHAT_HISTORY_MAX_TURNS")
    @classmethod
    def validate_chat_history_max_turns(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"CHAT_HISTORY_MAX_TURNS must be >= 0, got {v}")
        return v

    @field_validator("CHAT_HISTORY_MAX_TURN_CHARS")
    @classmethod
    def validate_chat_history_max_turn_chars(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"CHAT_HISTORY_MAX_TURN_CHARS must be > 0, got {v}")
        return v

    @field_validator("AGENT_MAX_REWRITES")
    @classmethod
    def validate_agent_max_rewrites(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"AGENT_MAX_REWRITES must be >= 0, got {v}")
        return v

    @field_validator("VL_EMBEDDING_DIMENSION")
    @classmethod
    def validate_vl_embedding_dimension(cls, v: int) -> int:
        allowed = {2560, 2048, 1536, 1024, 768, 512, 256}
        if v not in allowed:
            raise ValueError(f"VL_EMBEDDING_DIMENSION must be one of {sorted(allowed)}, got {v}")
        return v

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        if self.LLM_PROVIDER == "openai" and self.LLM_API_KEY == "":
            raise ValueError("LLM_API_KEY required when LLM_PROVIDER='openai'")
        if self.EMBEDDING_PROVIDER == "openai" and self.EMBEDDING_API_KEY == "":
            raise ValueError("EMBEDDING_API_KEY required when EMBEDDING_PROVIDER='openai'")
        if self.RERANKER_PROVIDER == "cohere" and self.COHERE_API_KEY == "":
            raise ValueError("COHERE_API_KEY required when RERANKER_PROVIDER='cohere'")
        if self.IMAGE_RETRIEVAL_ENABLED and not self.DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY required when IMAGE_RETRIEVAL_ENABLED=true")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
