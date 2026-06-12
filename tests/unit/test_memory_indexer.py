from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.domain import MemoryCandidate, MemoryEpisode, RetrievedChunk, SessionState, Visibility
from app.memory import MemoryEpisodeStore, MemoryIndexer
from app.persistence import SQLiteMemoryRepository, SQLiteSessionRepository
from app.persistence.sqlite import connect_sqlite, initialize_database
from app.rag import RagChunk, RagCollection, RetrievalFilter


class FakeEmbeddingProvider:
    dimension = 3

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_text(self, text: str) -> list[float]:
        return [1.0, float(len(text)), 0.0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [self.embed_text(text) for text in texts]


class RecordingVectorStore:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.ensure_calls: list[tuple[RagCollection, int]] = []
        self.upsert_calls: list[tuple[RagCollection, list[RagChunk], list[list[float]]]] = []

    def ensure_collection(self, collection: RagCollection, vector_size: int) -> None:
        self.ensure_calls.append((collection, vector_size))

    def replace_source(
        self,
        collection: RagCollection,
        source: str,
        chunks: Sequence[RagChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        raise AssertionError("replace_source should not be used for memory indexing")

    def upsert_chunks(
        self,
        collection: RagCollection,
        chunks: Sequence[RagChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if self.error is not None:
            raise self.error
        self.upsert_calls.append(
            (collection, list(chunks), [list(vector) for vector in vectors])
        )

    def delete_session_points(self, collection: RagCollection, session_id: str) -> None:
        raise AssertionError("delete_session_points should not be used for memory indexing")

    def search(
        self,
        collection: RagCollection,
        vector: Sequence[float],
        filters: RetrievalFilter,
        limit: int,
    ) -> list[RetrievedChunk]:
        raise AssertionError("search should not be used for memory indexing")


def _episode() -> MemoryEpisode:
    return MemoryEpisode(
        id="memory-1",
        session_id="session-1",
        scene_id="rose-gallery",
        actor_id="archivist",
        summary="The player promised to return before dawn.",
        importance=4,
        visibility=Visibility.PLAYER,
        tags=["promise", "dawn"],
    )


def _memory_store(tmp_path: Path) -> MemoryEpisodeStore:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    return MemoryEpisodeStore(memory_repository=SQLiteMemoryRepository(connection))


def test_memory_indexer_preserves_episode_metadata() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = RecordingVectorStore()
    indexer = MemoryIndexer(
        memory_store=None,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    result = indexer.index_memories([_episode()])

    assert result.indexed_count == 1
    assert embedding_provider.batches == [["The player promised to return before dawn."]]
    assert vector_store.ensure_calls == [(RagCollection.SESSION_MEMORY, 3)]
    collection, chunks, vectors = vector_store.upsert_calls[0]
    assert collection == RagCollection.SESSION_MEMORY
    assert vectors == [[1.0, 42.0, 0.0]]
    assert chunks == [
        RagChunk(
            id="memory-1",
            source="memory_episode:memory-1",
            source_type="session_memory",
            text="The player promised to return before dawn.",
            visibility=Visibility.PLAYER,
            tags=["promise", "dawn"],
            scene_id="rose-gallery",
            session_id="session-1",
            actor_id="archivist",
            importance=4,
        )
    ]


def test_memory_indexer_skips_vector_calls_for_empty_list() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = RecordingVectorStore()
    indexer = MemoryIndexer(
        memory_store=None,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    result = indexer.index_memories([])

    assert result.indexed_count == 0
    assert embedding_provider.batches == []
    assert vector_store.ensure_calls == []
    assert vector_store.upsert_calls == []


def test_memory_indexer_reindexes_existing_sqlite_memories(tmp_path: Path) -> None:
    memory_store = _memory_store(tmp_path)
    persisted = memory_store.persist_memories(
        session_id="session-1",
        memories=[
            MemoryCandidate(
                summary="The archive key will be ready before dawn.",
                visibility=Visibility.PLAYER,
                importance=4,
                tags=["archive", "dawn"],
                scene_id="rose-gallery",
                actor_id="archivist",
            )
        ],
    )
    vector_store = RecordingVectorStore()
    indexer = MemoryIndexer(
        memory_store=memory_store,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    result = indexer.reindex_session("session-1")

    assert result.indexed_count == 1
    assert vector_store.upsert_calls[0][1][0].id == persisted[0].id


def test_memory_indexer_failure_does_not_delete_sqlite_memory(tmp_path: Path) -> None:
    memory_store = _memory_store(tmp_path)
    memory_store.persist_memories(
        session_id="session-1",
        memories=[
            MemoryCandidate(
                summary="The archive key will be ready before dawn.",
                visibility=Visibility.PLAYER,
                importance=4,
            )
        ],
    )
    indexer = MemoryIndexer(
        memory_store=memory_store,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=RecordingVectorStore(error=RuntimeError("qdrant offline")),
    )

    with pytest.raises(RuntimeError, match="qdrant offline"):
        indexer.reindex_session("session-1")

    assert len(memory_store.list_memories_for_session("session-1")) == 1
