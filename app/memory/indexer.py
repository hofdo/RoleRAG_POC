from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from app.domain import MemoryEpisode
from app.memory.store import MemoryEpisodeStore
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import RagChunk, RagCollection
from app.rag.vector_store import VectorStore


class MemoryIndexingResult(BaseModel):
    indexed_count: int


class MemoryIndexer:
    def __init__(
        self,
        *,
        memory_store: MemoryEpisodeStore | None,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.memory_store = memory_store
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def index_memories(self, memories: Sequence[MemoryEpisode]) -> MemoryIndexingResult:
        if not memories:
            return MemoryIndexingResult(indexed_count=0)

        chunks = [self._to_chunk(memory) for memory in memories]
        vectors = self.embedding_provider.embed_batch([chunk.text for chunk in chunks])
        self.vector_store.ensure_collection(
            RagCollection.SESSION_MEMORY,
            self.embedding_provider.dimension,
        )
        self.vector_store.upsert_chunks(RagCollection.SESSION_MEMORY, chunks, vectors)
        return MemoryIndexingResult(indexed_count=len(chunks))

    def reindex_session(self, session_id: str) -> MemoryIndexingResult:
        if self.memory_store is None:
            raise RuntimeError("memory store is required for session reindexing")
        return self.index_memories(self.memory_store.list_memories_for_session(session_id))

    def _to_chunk(self, memory: MemoryEpisode) -> RagChunk:
        return RagChunk(
            id=memory.id,
            source=f"memory_episode:{memory.id}",
            source_type=RagCollection.SESSION_MEMORY.value,
            text=memory.summary,
            visibility=memory.visibility,
            tags=memory.tags,
            scene_id=memory.scene_id,
            session_id=memory.session_id,
            actor_id=memory.actor_id,
            importance=memory.importance,
        )
