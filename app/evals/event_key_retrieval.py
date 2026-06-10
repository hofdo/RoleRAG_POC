from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from app.domain import MemoryCandidate, SessionState, Visibility
from app.evals.fixtures import EvalFixture
from app.memory import MemoryEpisodeStore, MemoryIndexer
from app.persistence import SQLiteMemoryRepository, SQLiteSessionRepository
from app.persistence.sqlite import connect_sqlite, initialize_database
from app.rag import (
    ActorContextRetriever,
    InMemoryVectorStore,
    RagChunk,
    RagCollection,
    Retriever,
)
from app.rag.retriever import build_retrieval_query


@dataclass(frozen=True)
class SeededEvent:
    """One durable story event from the June 2026 model-comparison study."""

    key: str
    memory_summary: str
    callback_message: str
    tags: tuple[str, ...]


SEEDED_EVENTS: tuple[SeededEvent, ...] = (
    SeededEvent(
        key="before_dawn_promise",
        memory_summary=(
            "The player promised to return before dawn if Iria keeps the archive "
            "door unbarred."
        ),
        callback_message="I ask whether she remembers the promise I made about returning.",
        tags=("promise", "dawn"),
    ),
    SeededEvent(
        key="silver_compass",
        memory_summary="The player entrusted a silver compass to Iria until they return.",
        callback_message="I ask Iria to return the item I entrusted to her earlier.",
        tags=("compass", "entrusted"),
    ),
    SeededEvent(
        key="blue_seal_trust_rule",
        memory_summary="Only messages carrying a blue wax seal can be trusted.",
        callback_message="I ask which seal marks the messages we agreed to trust.",
        tags=("seal", "trust"),
    ),
    SeededEvent(
        key="key_hiding_place",
        memory_summary="The spare archive key is hidden beneath the third rose pedestal.",
        callback_message="I ask where the spare archive key is hidden.",
        tags=("key", "pedestal"),
    ),
    SeededEvent(
        key="three_tap_signal",
        memory_summary="The return signal is three soft taps at the west door.",
        callback_message="I ask what tap signal will announce me at the west door.",
        tags=("signal", "tap"),
    ),
)

_EVENT_KEYWORDS = (
    "promise",
    "dawn",
    "unbarred",
    "entrust",
    "compass",
    "item",
    "seal",
    "blue",
    "wax",
    "trust",
    "message",
    "key",
    "rose",
    "pedestal",
    "hidden",
    "tap",
    "signal",
    "west",
    "door",
    "archive",
    "return",
)


class EventKeywordEmbeddingProvider:
    """Deterministic embedding over the study-event vocabulary."""

    @property
    def dimension(self) -> int:
        return len(_EVENT_KEYWORDS)

    def embed_text(self, text: str) -> list[float]:
        normalized = text.lower()
        return [float(normalized.count(keyword)) for keyword in _EVENT_KEYWORDS]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class EventKeyRetrievalResult(BaseModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    selected_ids: dict[str, list[str]] = Field(default_factory=dict)


def evaluate_event_key_retrieval(fixture: EvalFixture) -> EventKeyRetrievalResult:
    temp_dir = Path(tempfile.mkdtemp(prefix="rolerag-event-key-retrieval-"))
    connection = connect_sqlite(temp_dir / "event-key-retrieval.db")
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
                summary=event.memory_summary,
                visibility=Visibility.PLAYER,
                importance=4,
                tags=list(event.tags),
                scene_id=fixture.scene.id,
                actor_id=fixture.primary_persona.id,
            )
            for event in SEEDED_EVENTS
        ]
        + [
            MemoryCandidate(
                summary="The player admired the rose arrangements in the gallery.",
                visibility=Visibility.PLAYER,
                importance=1,
                tags=["smalltalk"],
                scene_id=fixture.scene.id,
            ),
            MemoryCandidate(
                summary="Courtiers exchanged routine pleasantries near the mirrors.",
                visibility=Visibility.PLAYER,
                importance=1,
                tags=["smalltalk"],
                scene_id=fixture.scene.id,
            ),
        ],
    )
    event_memory_ids = {
        event.key: persisted[index].id for index, event in enumerate(SEEDED_EVENTS)
    }

    embedding_provider = EventKeywordEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    MemoryIndexer(
        memory_store=memory_store,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    ).index_memories(persisted)
    lore_chunks = [
        RagChunk(
            id="lore-archive-door",
            source="demo_lore.md",
            source_type="lore",
            text="The Rose Gallery archive sits behind the locked west door.",
            visibility=Visibility.PLAYER,
            world_id=fixture.world.id,
        ),
        RagChunk(
            id="lore-court-messages",
            source="demo_lore.md",
            source_type="lore",
            text="Court messages travel by sealed courier under the regent's watch.",
            visibility=Visibility.PLAYER,
            world_id=fixture.world.id,
        ),
    ]
    vector_store.ensure_collection(RagCollection.CANON_LORE, embedding_provider.dimension)
    vector_store.upsert_chunks(
        RagCollection.CANON_LORE,
        lore_chunks,
        embedding_provider.embed_batch([chunk.text for chunk in lore_chunks]),
    )

    retriever = ActorContextRetriever(
        retriever=Retriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            default_top_k=5,
        )
    )
    checks: dict[str, bool] = {}
    selected_ids: dict[str, list[str]] = {}
    any_rejected = False
    for event in SEEDED_EVENTS:
        result = retriever.retrieve_for_actor_with_diagnostics(
            query=build_retrieval_query(
                user_message=event.callback_message,
                scene=fixture.scene,
                persona=fixture.primary_persona,
                recent_turns=(),
            ),
            world_id=fixture.world.id,
            session_id=fixture.session.id,
            persona_id=fixture.primary_persona.id,
            scene_id=fixture.scene.id,
            top_k=5,
        )
        ids = [chunk.id for chunk in result.chunks]
        selected_ids[event.key] = ids
        checks[f"{event.key}_selected"] = event_memory_ids[event.key] in ids
        any_rejected = any_rejected or bool(result.diagnostics.rejected)
    checks["diagnostics_record_rejected_candidates"] = any_rejected
    connection.close()

    return EventKeyRetrievalResult(
        passed=all(checks.values()),
        checks=checks,
        selected_ids=selected_ids,
    )
