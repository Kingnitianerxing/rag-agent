"""Qdrant store for image vectors produced by qwen3-vl-embedding (dual-track)."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import get_settings
from app.ingestion.vl_embedder import VLEmbedder

logger = logging.getLogger(__name__)


class ImageVectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.collection_name = settings.VL_COLLECTION_NAME
        self.url = settings.QDRANT_URL
        self.api_key = settings.QDRANT_API_KEY
        self.dimension = settings.VL_EMBEDDING_DIMENSION
        self._client: QdrantClient | None = None
        self._embedder: VLEmbedder | None = None

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    def _get_embedder(self) -> VLEmbedder:
        if self._embedder is None:
            self._embedder = VLEmbedder()
        return self._embedder

    def _ensure_collection(self) -> None:
        client = self._get_client()
        if not client.collection_exists(self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
            logger.info(
                "Created image collection: name=%s dim=%d",
                self.collection_name,
                self.dimension,
            )

    @staticmethod
    def _point_id(source: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"image::{source}"))

    def upsert(self, documents: list[Document]) -> list[str]:
        """Embed each image document and upsert into the VL collection."""
        self._ensure_collection()
        client = self._get_client()
        embedder = self._get_embedder()
        points: list[PointStruct] = []
        ids: list[str] = []
        for doc in documents:
            source = doc.metadata.get("source") or ""
            image_path = doc.metadata.get("image_path") or source
            if not image_path or not Path(image_path).is_file():
                raise FileNotFoundError(f"Image path missing for document: {source}")
            vector = embedder.embed_image_file(image_path)
            pid = self._point_id(source)
            ids.append(pid)
            payload = {
                "page_content": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "source": source,
                    "modality": "image",
                },
            }
            points.append(PointStruct(id=pid, vector=vector, payload=payload))
        if points:
            client.upsert(collection_name=self.collection_name, points=points)
            logger.info("image_vector_upserted: count=%d collection=%s", len(points), self.collection_name)
        return ids

    def search(
        self, query: str, top_k: int = 5, sources: list[str] | None = None
    ) -> list[tuple[Document, float]]:
        """Text-to-image search via VL text embedding in the image collection."""
        self._ensure_collection()
        client = self._get_client()
        vector = self._get_embedder().embed_text(query)
        query_filter = None
        if sources:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.source",
                        match=qmodels.MatchAny(any=list(sources)),
                    )
                ]
            )
        response = client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        out: list[tuple[Document, float]] = []
        for hit in response.points or []:
            payload = hit.payload or {}
            meta = dict(payload.get("metadata") or {})
            content = payload.get("page_content") or meta.get("source") or ""
            meta.setdefault("modality", "image")
            score = float(hit.score or 0.0)
            meta["score"] = score
            out.append((Document(page_content=str(content), metadata=meta), score))
        return out
