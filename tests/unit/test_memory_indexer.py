from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.domain import MemoryCandidate, MemoryEpisode, RetrievedChunk, SessionState, Visibility
from app.memory import MemoryEpisodeStore, MemoryIndexer
from app.persistence import SQLiteMemoryRepository, SQLiteSessionRepository
from app.persistence.sqlite import connect_sqlite, initialize_database
from app.rag import InMemoryVectorStore, RagChunk, RagCollection, RetrievalFilter


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
        self.deleted_ids: list[str] = []

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

    def delete_points(self, collection: RagCollection, chunk_ids: Sequence[str]) -> None:
        self.deleted_ids.extend(chunk_ids)

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

    # _episode() is PLAYER-visible, importance=4, actor_id="archivist" -- it clears the
    # PERSONA_MEMORY_IMPORTANCE_FLOOR, so it is embedded/upserted twice: once into
    # SESSION_MEMORY (this test's focus) and once into PERSONA_MEMORY (covered by
    # test_high_value_player_memories_are_also_indexed_per_persona).
    assert result.indexed_count == 1
    assert embedding_provider.batches == [
        ["The player promised to return before dawn."],
        ["The player promised to return before dawn."],
    ]
    assert (RagCollection.SESSION_MEMORY, 3) in vector_store.ensure_calls
    session_calls = [
        call for call in vector_store.upsert_calls if call[0] == RagCollection.SESSION_MEMORY
    ]
    assert len(session_calls) == 1
    collection, chunks, vectors = session_calls[0]
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


def _episode_with(memory_id: str, *, importance: int) -> MemoryEpisode:
    return MemoryEpisode(
        id=memory_id,
        session_id="session-1",
        scene_id="rose-gallery",
        actor_id="archivist",
        summary=f"Commitment {memory_id}.",
        importance=importance,
        visibility=Visibility.PLAYER,
        tags=["promise"],
    )


def test_index_memories_respects_importance_floor() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = RecordingVectorStore()
    indexer = MemoryIndexer(
        memory_store=None,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        importance_floor=4,
    )

    result = indexer.index_memories(
        [
            _episode_with("low-1", importance=1),
            _episode_with("low-2", importance=3),
            _episode_with("keep-1", importance=4),
            _episode_with("keep-2", importance=5),
        ]
    )

    assert result.indexed_count == 2
    _, chunks, _ = vector_store.upsert_calls[0]
    assert [chunk.id for chunk in chunks] == ["keep-1", "keep-2"]


def test_index_memories_floor_one_indexes_everything() -> None:
    vector_store = RecordingVectorStore()
    indexer = MemoryIndexer(
        memory_store=None,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        importance_floor=1,
    )

    result = indexer.index_memories(
        [_episode_with("a", importance=1), _episode_with("b", importance=2)]
    )

    assert result.indexed_count == 2


def test_reindex_session_applies_importance_floor(tmp_path: Path) -> None:
    store = _memory_store(tmp_path)
    store.persist_memories(
        session_id="session-1",
        memories=[
            MemoryCandidate(
                summary="Filler smalltalk.",
                visibility=Visibility.PLAYER,
                importance=1,
                tags=["mood"],
                scene_id="rose-gallery",
                actor_id="archivist",
            ),
            MemoryCandidate(
                summary="The player vowed to protect the archive.",
                visibility=Visibility.PLAYER,
                importance=5,
                tags=["vow"],
                scene_id="rose-gallery",
                actor_id="archivist",
            ),
        ],
    )
    vector_store = RecordingVectorStore()
    indexer = MemoryIndexer(
        memory_store=store,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        importance_floor=3,
    )

    result = indexer.reindex_session("session-1")

    assert result.indexed_count == 1
    _, chunks, _ = vector_store.upsert_calls[0]
    assert [chunk.text for chunk in chunks] == ["The player vowed to protect the archive."]


def _persist_importances(store: MemoryEpisodeStore, importances: list[int]) -> list[MemoryEpisode]:
    return store.persist_memories(
        session_id="session-1",
        memories=[
            MemoryCandidate(
                summary=f"Memory with importance {importance} #{index}",
                visibility=Visibility.PLAYER,
                importance=importance,
                tags=["promise"],
                scene_id="rose-gallery",
                actor_id="archivist",
            )
            for index, importance in enumerate(importances)
        ],
    )


def test_index_memories_evicts_lowest_priority_beyond_session_cap(tmp_path: Path) -> None:
    store = _memory_store(tmp_path)
    persisted = _persist_importances(store, [5, 4, 3, 2, 1])
    vector_store = RecordingVectorStore()
    indexer = MemoryIndexer(
        memory_store=store,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        session_memory_max_episodes=3,
    )

    indexer.index_memories(persisted)

    # Top 3 importances (5, 4, 3) stay indexed; the two lowest are unindexed.
    evicted = {memory.id for memory in persisted if memory.importance in (1, 2)}
    assert set(vector_store.deleted_ids) == evicted


def test_session_cap_eviction_leaves_persona_memory_copies_searchable(tmp_path: Path) -> None:
    # Controller decision: session-cap eviction is scoped to SESSION_MEMORY only.
    # PERSONA_MEMORY is the cross-session durable NPC memory (dual-write in
    # _index_persona_memories), so a persona copy of an evicted episode must
    # remain searchable even after its SESSION_MEMORY sibling is capped away.
    store = _memory_store(tmp_path)
    persisted = _persist_importances(store, [5, 4, 3, 2, 1])
    vector_store = InMemoryVectorStore()
    indexer = MemoryIndexer(
        memory_store=store,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        session_memory_max_episodes=3,
    )

    indexer.index_memories(persisted)

    # Top 3 importances (5, 4, 3) stay in SESSION_MEMORY; importances 1 and 2 are
    # evicted from SESSION_MEMORY by the cap.
    evicted = [memory for memory in persisted if memory.importance in (1, 2)]
    kept = [memory for memory in persisted if memory.importance not in (1, 2)]
    query_vector = FakeEmbeddingProvider().embed_text("promise")

    session_hits = vector_store.search(
        RagCollection.SESSION_MEMORY,
        query_vector,
        RetrievalFilter.player_visible(session_id="session-1"),
        limit=10,
    )
    session_hit_ids = {hit.id for hit in session_hits}
    assert session_hit_ids == {memory.id for memory in kept}
    assert not session_hit_ids & {memory.id for memory in evicted}

    # Every episode with importance >= PERSONA_MEMORY_IMPORTANCE_FLOOR (4) was
    # dual-written to PERSONA_MEMORY, including the ones later capped out of
    # SESSION_MEMORY (importance 4). Those persona copies must still be
    # searchable -- cap eviction never touches PERSONA_MEMORY.
    persona_hits = vector_store.search(
        RagCollection.PERSONA_MEMORY,
        query_vector,
        RetrievalFilter.player_visible(persona_id="archivist"),
        limit=10,
    )
    persona_hit_ids = {hit.id for hit in persona_hits}
    high_value = {memory.id for memory in persisted if memory.importance >= 4}
    assert persona_hit_ids == high_value
    # This cap (3) happens not to evict any persona-eligible (importance >= 4)
    # episode, so it only shows persona copies are unaffected by *some* cap
    # eviction. The next test tightens the cap so a persona-eligible episode is
    # itself evicted from SESSION_MEMORY, proving the exemption directly.


def test_session_cap_eviction_of_a_persona_eligible_memory_keeps_it_searchable_via_persona(
    tmp_path: Path,
) -> None:
    # A tighter cap that evicts even a persona-eligible (importance >= 4) episode
    # from SESSION_MEMORY -- confirming the PERSONA_MEMORY copy survives that
    # eviction untouched.
    store = _memory_store(tmp_path)
    persisted = _persist_importances(store, [5, 4])
    vector_store = InMemoryVectorStore()
    indexer = MemoryIndexer(
        memory_store=store,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        session_memory_max_episodes=1,
    )

    indexer.index_memories(persisted)

    evicted_memory = next(memory for memory in persisted if memory.importance == 4)
    query_vector = FakeEmbeddingProvider().embed_text("promise")

    session_hits = vector_store.search(
        RagCollection.SESSION_MEMORY,
        query_vector,
        RetrievalFilter.player_visible(session_id="session-1"),
        limit=10,
    )
    assert evicted_memory.id not in {hit.id for hit in session_hits}

    persona_hits = vector_store.search(
        RagCollection.PERSONA_MEMORY,
        query_vector,
        RetrievalFilter.player_visible(persona_id="archivist"),
        limit=10,
    )
    assert evicted_memory.id in {hit.id for hit in persona_hits}


def test_index_memories_does_not_evict_when_cap_is_zero(tmp_path: Path) -> None:
    store = _memory_store(tmp_path)
    persisted = _persist_importances(store, [5, 4, 3, 2, 1])
    vector_store = RecordingVectorStore()
    indexer = MemoryIndexer(
        memory_store=store,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        session_memory_max_episodes=0,
    )

    indexer.index_memories(persisted)

    assert vector_store.deleted_ids == []


def test_index_memories_skips_consolidated_memories() -> None:
    from app.memory.consolidation import CONSOLIDATED_TAG

    vector_store = RecordingVectorStore()
    indexer = MemoryIndexer(
        memory_store=None,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    result = indexer.index_memories(
        [
            MemoryEpisode(
                id="m1",
                session_id="session-1",
                scene_id="rose-gallery",
                summary="A consolidated original.",
                importance=4,
                visibility=Visibility.PLAYER,
                tags=[CONSOLIDATED_TAG],
            )
        ]
    )

    assert result.indexed_count == 0
    assert vector_store.upsert_calls == []


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


def test_high_value_player_memories_are_also_indexed_per_persona() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    indexer = MemoryIndexer(
        memory_store=None,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        importance_floor=1,
    )
    player_memory = MemoryEpisode(
        id="player-1",
        session_id="session-1",
        scene_id="rose-gallery",
        actor_id="innkeeper",
        summary="The player promised to return the innkeeper's ledger.",
        importance=4,
        visibility=Visibility.PLAYER,
        tags=["promise"],
    )
    gm_memory = MemoryEpisode(
        id="gm-1",
        session_id="session-1",
        scene_id="rose-gallery",
        actor_id="innkeeper",
        summary="The innkeeper secretly informs the regent.",
        importance=5,
        visibility=Visibility.GM,
        tags=["secret"],
    )

    indexer.index_memories([player_memory, gm_memory])

    query_vector = embedding_provider.embed_text("ledger")
    persona_hits = vector_store.search(
        RagCollection.PERSONA_MEMORY,
        query_vector,
        RetrievalFilter.player_visible(persona_id="innkeeper"),
        limit=10,
    )
    assert [hit.actor_id for hit in persona_hits] == ["innkeeper"]
    assert [hit.id for hit in persona_hits] == ["player-1"]

    # A different persona_id must not see another actor's persisted memory.
    other_persona_hits = vector_store.search(
        RagCollection.PERSONA_MEMORY,
        query_vector,
        RetrievalFilter.player_visible(persona_id="someone-else"),
        limit=10,
    )
    assert other_persona_hits == []


def test_low_importance_player_memory_is_not_indexed_per_persona() -> None:
    vector_store = InMemoryVectorStore()
    indexer = MemoryIndexer(
        memory_store=None,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        importance_floor=1,
    )
    low_value_memory = MemoryEpisode(
        id="player-low",
        session_id="session-1",
        scene_id="rose-gallery",
        actor_id="innkeeper",
        summary="Minor smalltalk.",
        importance=3,
        visibility=Visibility.PLAYER,
        tags=[],
    )

    indexer.index_memories([low_value_memory])

    vector_store.ensure_collection(RagCollection.PERSONA_MEMORY, FakeEmbeddingProvider().dimension)
    persona_hits = vector_store.search(
        RagCollection.PERSONA_MEMORY,
        FakeEmbeddingProvider().embed_text("smalltalk"),
        RetrievalFilter.player_visible(persona_id="innkeeper"),
        limit=10,
    )
    assert persona_hits == []


def test_unindex_also_removes_persona_memory_copies() -> None:
    vector_store = InMemoryVectorStore()
    indexer = MemoryIndexer(
        memory_store=None,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        importance_floor=1,
    )
    player_memory = MemoryEpisode(
        id="player-2",
        session_id="session-1",
        scene_id="rose-gallery",
        actor_id="innkeeper",
        summary="The player vowed to protect the inn.",
        importance=5,
        visibility=Visibility.PLAYER,
        tags=["vow"],
    )
    indexer.index_memories([player_memory])

    indexer.unindex(["player-2"])

    session_hits = vector_store.search(
        RagCollection.SESSION_MEMORY,
        FakeEmbeddingProvider().embed_text("inn"),
        RetrievalFilter.player_visible(session_id="session-1"),
        limit=10,
    )
    persona_hits = vector_store.search(
        RagCollection.PERSONA_MEMORY,
        FakeEmbeddingProvider().embed_text("inn"),
        RetrievalFilter.player_visible(persona_id="innkeeper"),
        limit=10,
    )
    assert session_hits == []
    assert persona_hits == []


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
