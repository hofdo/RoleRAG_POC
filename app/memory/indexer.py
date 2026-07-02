from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from app.domain import MemoryEpisode, Visibility
from app.memory.consolidation import CONSOLIDATED_TAG
from app.memory.store import MemoryEpisodeStore
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import RagChunk, RagCollection
from app.rag.vector_store import VectorStore

# Cross-session NPC memory: only PLAYER-visible episodes at or above this importance are
# also written to persona_memory, so a later session with the same persona can retrieve
# them (retrieval already searches persona_memory filtered by persona_id — see
# ActorContextRetriever / RetrievalFilter.player_visible(persona_id=...)).
PERSONA_MEMORY_IMPORTANCE_FLOOR = 4


class MemoryIndexingResult(BaseModel):
    indexed_count: int


class MemoryIndexer:
    def __init__(
        self,
        *,
        memory_store: MemoryEpisodeStore | None,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        importance_floor: int = 1,
        session_memory_max_episodes: int = 0,
    ) -> None:
        self.memory_store = memory_store
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.importance_floor = importance_floor
        self.session_memory_max_episodes = session_memory_max_episodes

    def index_memories(self, memories: Sequence[MemoryEpisode]) -> MemoryIndexingResult:
        eligible = [
            memory
            for memory in memories
            if memory.importance >= self.importance_floor
            and CONSOLIDATED_TAG not in memory.tags
        ]
        if not eligible:
            return MemoryIndexingResult(indexed_count=0)

        chunks = [self._to_chunk(memory) for memory in eligible]
        vectors = self.embedding_provider.embed_batch([chunk.text for chunk in chunks])
        self.vector_store.ensure_collection(
            RagCollection.SESSION_MEMORY,
            self.embedding_provider.dimension,
        )
        self.vector_store.upsert_chunks(RagCollection.SESSION_MEMORY, chunks, vectors)
        self._index_persona_memories(eligible)
        for session_id in {memory.session_id for memory in eligible}:
            self._enforce_session_cap(session_id)
        return MemoryIndexingResult(indexed_count=len(chunks))

    def reindex_session(self, session_id: str) -> MemoryIndexingResult:
        if self.memory_store is None:
            raise RuntimeError("memory store is required for session reindexing")
        return self.index_memories(self.memory_store.list_memories_for_session(session_id))

    def unindex(self, memory_ids: Sequence[str]) -> None:
        self.vector_store.delete_points(RagCollection.SESSION_MEMORY, list(memory_ids))
        try:
            self.vector_store.delete_points(RagCollection.PERSONA_MEMORY, list(memory_ids))
        except Exception:  # noqa: BLE001 - persona collection may not exist yet
            pass

    def _enforce_session_cap(self, session_id: str) -> None:
        """Bound the retrievable session-memory index to the most valuable N
        episodes (importance then recency), unindexing the rest. SQLite stays
        authoritative; nothing is deleted there. Inert when the cap is 0."""
        cap = self.session_memory_max_episodes
        if cap <= 0 or self.memory_store is None:
            return
        indexed = [
            memory
            for memory in self.memory_store.list_memories_for_session(session_id)
            if memory.importance >= self.importance_floor
        ]
        if len(indexed) <= cap:
            return
        surplus = sorted(indexed, key=self._eviction_sort_key)[cap:]
        self.vector_store.delete_points(
            RagCollection.SESSION_MEMORY,
            [memory.id for memory in surplus],
        )

    def _index_persona_memories(self, memories: Sequence[MemoryEpisode]) -> None:
        """Cross-session NPC memory: high-value PLAYER-visible episodes are also
        indexed per actor, so a later session with the same persona retrieves them
        (retrieval already searches persona_memory filtered by persona_id)."""
        lasting = [
            memory
            for memory in memories
            if memory.visibility is Visibility.PLAYER
            and memory.actor_id
            and memory.importance >= PERSONA_MEMORY_IMPORTANCE_FLOOR
        ]
        if not lasting:
            return
        chunks = [
            self._to_chunk(memory).model_copy(
                update={
                    "source_type": RagCollection.PERSONA_MEMORY.value,
                    # RetrievalFilter.player_visible(persona_id=...) filters on the
                    # chunk's persona_id payload field, not actor_id -- the actor_id
                    # is preserved for provenance/display but persona_id drives the
                    # cross-session retrieval filter (see app/rag/vector_store.py
                    # _chunk_matches_filters / _build_qdrant_filter).
                    "persona_id": memory.actor_id,
                }
            )
            for memory in lasting
        ]
        vectors = self.embedding_provider.embed_batch([chunk.text for chunk in chunks])
        self.vector_store.ensure_collection(
            RagCollection.PERSONA_MEMORY, self.embedding_provider.dimension
        )
        self.vector_store.upsert_chunks(RagCollection.PERSONA_MEMORY, chunks, vectors)

    @staticmethod
    def _eviction_sort_key(memory: MemoryEpisode) -> tuple[int, float, str]:
        # Ascending sort, so smaller key is kept: highest importance first, then
        # most recent. A missing timestamp sorts oldest (evicted first).
        created = memory.created_at.timestamp() if memory.created_at is not None else float("-inf")
        return (-memory.importance, -created, memory.id)

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
            created_at=memory.created_at,
        )
