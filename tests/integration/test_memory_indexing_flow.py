from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.domain import MemoryCandidate, SessionState, Visibility
from app.memory import MemoryEpisodeStore, MemoryIndexer
from app.persistence import SQLiteMemoryRepository, SQLiteSessionRepository
from app.persistence.sqlite import connect_sqlite, initialize_database
from app.rag import ActorContextRetriever, InMemoryVectorStore, Retriever


class KeywordEmbeddingProvider:
    _KEYWORDS = ("archive", "dawn", "spy")

    @property
    def dimension(self) -> int:
        return len(self._KEYWORDS)

    def embed_text(self, text: str) -> list[float]:
        normalized = text.lower()
        return [float(normalized.count(keyword)) for keyword in self._KEYWORDS]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


def test_indexed_player_memory_is_retrievable_but_hidden_memory_is_not(tmp_path: Path) -> None:
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
    memory_store = MemoryEpisodeStore(
        memory_repository=SQLiteMemoryRepository(connection),
    )
    episodes = memory_store.persist_memories(
        session_id="session-1",
        memories=[
            MemoryCandidate(
                summary="The archive key will be ready before dawn.",
                visibility=Visibility.PLAYER,
                importance=4,
                tags=["archive", "dawn"],
                scene_id="rose-gallery",
                actor_id="archivist",
            ),
            MemoryCandidate(
                summary="The spy moved the hidden archive ledger.",
                visibility=Visibility.GM,
                importance=5,
                tags=["spy", "archive"],
                scene_id="rose-gallery",
            ),
            MemoryCandidate(
                summary="The archivist privately suspects the spy.",
                visibility=Visibility.CHARACTER_PRIVATE,
                importance=3,
                tags=["spy"],
                scene_id="rose-gallery",
                actor_id="archivist",
            ),
        ],
    )
    embedding_provider = KeywordEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    indexer = MemoryIndexer(
        memory_store=memory_store,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    first_result = indexer.index_memories(episodes)
    second_result = indexer.reindex_session("session-1")
    retrieved = ActorContextRetriever(
        retriever=Retriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            default_top_k=5,
        )
    ).retrieve_for_actor(
        query="archive dawn spy",
        world_id="demo_world",
        session_id="session-1",
        persona_id="archivist",
        top_k=5,
    )

    assert first_result.indexed_count == 3
    assert second_result.indexed_count == 3
    assert [chunk.text for chunk in retrieved] == [
        "The archive key will be ready before dawn.",
    ]
