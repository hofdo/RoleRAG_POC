from __future__ import annotations

from app.domain import RetrievedChunk
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import RagCollection, RetrievalFilter
from app.rag.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        default_top_k: int = 5,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(
        self,
        *,
        query: str,
        collection: RagCollection,
        filters: RetrievalFilter,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        limit = top_k or self.default_top_k
        query_vector = self.embedding_provider.embed_text(query)
        results = self.vector_store.search(
            collection=collection,
            vector=query_vector,
            filters=filters,
            limit=limit,
        )
        return [chunk for chunk in results if chunk.visibility in filters.allowed_visibilities]
