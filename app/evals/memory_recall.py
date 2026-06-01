from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from app.domain import MemoryCandidate, SessionState, Visibility
from app.evals.fixtures import EvalFixture
from app.memory import MemoryEpisodeStore, MemoryIndexer
from app.persistence import SQLiteMemoryRepository, SQLiteSessionRepository
from app.persistence.sqlite import connect_sqlite, initialize_database
from app.rag import ActorContextRetriever, InMemoryVectorStore, Retriever


class MemoryRecallResult(BaseModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    retrieved_ids: list[str] = Field(default_factory=list)


def evaluate_memory_recall(fixture: EvalFixture) -> MemoryRecallResult:
    temp_dir = Path(tempfile.mkdtemp(prefix="rolerag-memory-recall-"))
    connection = connect_sqlite(temp_dir / "memory-recall.db")
    initialize_database(connection)
    SQLiteSessionRepository(connection).create_session(
        SessionState(
            id=fixture.session.id,
            world_id=fixture.session.world_id,
            active_scene_id=fixture.session.active_scene_id,
            active_persona_id=fixture.session.active_persona_id,
            player_name=fixture.session.player_name,
        )
    )
    memory_store = MemoryEpisodeStore(memory_repository=SQLiteMemoryRepository(connection))
    persisted = memory_store.persist_memories(
        session_id=fixture.session.id,
        memories=[
            MemoryCandidate(
                summary="The player promised to return before dawn for the archive key.",
                visibility=Visibility.PLAYER,
                importance=4,
                tags=["promise", "archive", "dawn"],
                scene_id=fixture.scene.id,
                actor_id=fixture.primary_persona.id,
            ),
            MemoryCandidate(
                summary="A hidden loyalist spy listens from the south alcove.",
                visibility=Visibility.GM,
                importance=5,
                tags=["spy"],
                scene_id=fixture.scene.id,
            ),
            MemoryCandidate(
                summary="Iria hid the cipher in the gallery clock.",
                visibility=Visibility.CHARACTER_PRIVATE,
                importance=4,
                tags=["clock", "cipher"],
                scene_id=fixture.scene.id,
                actor_id=fixture.primary_persona.id,
            ),
        ],
    )
    vector_store = InMemoryVectorStore()
    MemoryIndexer(
        memory_store=memory_store,
        embedding_provider=fixture.embedding_provider,
        vector_store=vector_store,
    ).index_memories(persisted)
    retrieved = ActorContextRetriever(
        retriever=Retriever(
            embedding_provider=fixture.embedding_provider,
            vector_store=vector_store,
            default_top_k=5,
        )
    ).retrieve_for_actor(
        query=fixture.build_retrieval_query(),
        world_id=fixture.world.id,
        session_id=fixture.session.id,
        persona_id=fixture.primary_persona.id,
        scene_id=fixture.scene.id,
        top_k=5,
    )
    connection.close()

    retrieved_ids = [chunk.id for chunk in retrieved]
    checks = {
        "indexed_player_memory_retrievable": persisted[0].id in retrieved_ids,
        "indexed_gm_memory_excluded": persisted[1].id not in retrieved_ids,
        "indexed_character_private_memory_excluded": persisted[2].id not in retrieved_ids,
    }
    return MemoryRecallResult(
        passed=all(checks.values()),
        checks=checks,
        retrieved_ids=retrieved_ids,
    )
