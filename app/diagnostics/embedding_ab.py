"""Real-embedding rank A/B harness (LLM-free).

Embedding-model selection needs a measurement that isolates *retrieval rank
quality* from LLM inference: how well does a given embedding model rank the
seeded durable memories against the callback queries? This harness seeds the
shared `SEEDED_EVENTS` corpus (the same data `event_key_retrieval` /
`retrieval_miss` use) into a temp `InMemoryVectorStore` using a GIVEN embedding
provider, then for each event reports the target memory's overall rank across the
full candidate set (selected ∪ rejected) via `summarize_retrieval_miss`.

The provider is injected, so the CLI can benchmark alternative FastEmbed models
against the baseline while the unit test passes a deterministic fake — no model
downloads, no LLM calls.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from app.diagnostics.retrieval_miss import summarize_retrieval_miss
from app.domain import MemoryCandidate, SessionState, Visibility
from app.evals.event_key_retrieval import SEEDED_EVENTS, SeededEvent
from app.evals.fixtures import EvalFixture, build_eval_fixture
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
from app.rag.embeddings import EmbeddingProvider
from app.rag.retriever import build_retrieval_query


def measure_embedding_ranks(
    *,
    embedding_provider: EmbeddingProvider,
    events: Sequence[SeededEvent] = SEEDED_EVENTS,
    fixture: EvalFixture | None = None,
    top_k: int = 5,
) -> dict[str, int | None]:
    """Map each event key to its target memory's overall rank under `embedding_provider`.

    Seeds `events` (+ two filler memories and two lore chunks, matching the
    shared corpus) into a temp `InMemoryVectorStore` with the GIVEN provider,
    builds an `ActorContextRetriever`, and for each event runs
    `retrieve_for_actor_with_diagnostics` then `summarize_retrieval_miss` to read
    off the target memory's `overall_rank`. A rank of `None` means the memory did
    not appear in the candidate set at all.
    """
    resolved_fixture = fixture or build_eval_fixture()
    temp_dir = Path(tempfile.mkdtemp(prefix="rolerag-embedding-ab-"))
    connection = connect_sqlite(temp_dir / "embedding-ab.db")
    initialize_database(connection)
    SQLiteSessionRepository(connection).create_session(
        SessionState(
            id=resolved_fixture.session.id,
            world_id=resolved_fixture.session.world_id,
            active_scene_id=resolved_fixture.session.active_scene_id,
            active_persona_id=resolved_fixture.session.active_persona_id,
            player_name=resolved_fixture.session.player_name,
        )
    )
    memory_store = MemoryEpisodeStore(memory_repository=SQLiteMemoryRepository(connection))
    persisted = memory_store.persist_memories(
        session_id=resolved_fixture.session.id,
        memories=[
            MemoryCandidate(
                summary=event.memory_summary,
                visibility=Visibility.PLAYER,
                importance=4,
                tags=list(event.tags),
                scene_id=resolved_fixture.scene.id,
                actor_id=resolved_fixture.primary_persona.id,
            )
            for event in events
        ]
        + [
            MemoryCandidate(
                summary="The player admired the rose arrangements in the gallery.",
                visibility=Visibility.PLAYER,
                importance=1,
                tags=["smalltalk"],
                scene_id=resolved_fixture.scene.id,
            ),
            MemoryCandidate(
                summary="Courtiers exchanged routine pleasantries near the mirrors.",
                visibility=Visibility.PLAYER,
                importance=1,
                tags=["smalltalk"],
                scene_id=resolved_fixture.scene.id,
            ),
        ],
    )
    event_memory_ids = {event.key: persisted[index].id for index, event in enumerate(events)}

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
            world_id=resolved_fixture.world.id,
        ),
        RagChunk(
            id="lore-court-messages",
            source="demo_lore.md",
            source_type="lore",
            text="Court messages travel by sealed courier under the regent's watch.",
            visibility=Visibility.PLAYER,
            world_id=resolved_fixture.world.id,
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
            default_top_k=top_k,
        )
    )

    ranks: dict[str, int | None] = {}
    for event in events:
        result = retriever.retrieve_for_actor_with_diagnostics(
            query=build_retrieval_query(
                user_message=event.callback_message,
                scene=resolved_fixture.scene,
                persona=resolved_fixture.primary_persona,
                recent_turns=(),
            ),
            lexical_query=event.callback_message,
            world_id=resolved_fixture.world.id,
            session_id=resolved_fixture.session.id,
            persona_id=resolved_fixture.primary_persona.id,
            scene_id=resolved_fixture.scene.id,
            top_k=top_k,
        )
        memory_id = event_memory_ids[event.key]
        report = summarize_retrieval_miss(
            expected_ids=[memory_id],
            selected=[chunk.model_dump() for chunk in result.diagnostics.selected],
            rejected=[chunk.model_dump() for chunk in result.diagnostics.rejected],
        )
        ranks[event.key] = report.ranks[0].overall_rank
    connection.close()
    return ranks
