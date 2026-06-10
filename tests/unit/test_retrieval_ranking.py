from __future__ import annotations

from app.domain import RetrievedChunk, Visibility
from app.rag.diagnostics import ChunkRetrievalDiagnostic
from app.rag.models import RagCollection, RetrievalFilter
from app.rag.retriever import ActorContextRetriever


def _chunk(
    chunk_id: str,
    *,
    score: float,
    collection: RagCollection,
    text: str,
    visibility: Visibility = Visibility.PLAYER,
    importance: int | None = None,
    scene_id: str | None = None,
    persona_id: str | None = None,
    session_id: str | None = None,
    actor_id: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        source=f"{chunk_id}.md",
        source_type="memory" if collection != RagCollection.CANON_LORE else "lore",
        text=text,
        score=score,
        visibility=visibility,
        tags=[],
        scene_id=scene_id,
        persona_id=persona_id,
        session_id=session_id,
        actor_id=actor_id,
        importance=importance,
    )


class RecordingRetriever:
    def __init__(self, chunks_by_collection: dict[RagCollection, list[RetrievedChunk]]) -> None:
        self.chunks_by_collection = chunks_by_collection
        self.calls: list[tuple[RagCollection, RetrievalFilter, int | None]] = []

    def retrieve(
        self,
        *,
        query: str,
        collection: RagCollection,
        filters: RetrievalFilter,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        del query
        self.calls.append((collection, filters, top_k))
        return self.chunks_by_collection.get(collection, [])


def test_session_memory_beats_close_lore_score() -> None:
    retriever = RecordingRetriever(
        {
            RagCollection.SESSION_MEMORY: [
                _chunk(
                    "memory-1",
                    score=0.60,
                    collection=RagCollection.SESSION_MEMORY,
                    text="The player promised the archivist to return before dawn.",
                    importance=4,
                    scene_id="rose-gallery",
                    session_id="session-1",
                )
            ],
            RagCollection.CANON_LORE: [
                _chunk(
                    "lore-1",
                    score=0.63,
                    collection=RagCollection.CANON_LORE,
                    text="The archive key is kept near the west stacks.",
                )
            ],
        }
    )

    chunks = ActorContextRetriever(retriever=retriever).retrieve_for_actor(
        query="What did I promise the archivist?",
        world_id="demo_world",
        session_id="session-1",
        persona_id="archivist",
        scene_id="rose-gallery",
        top_k=2,
    )

    assert [chunk.id for chunk in chunks] == ["memory-1", "lore-1"]
    assert all(call[2] == 4 for call in retriever.calls)


def test_actor_context_retriever_keeps_canon_lore_first_when_it_is_clearly_more_relevant() -> None:
    retriever = RecordingRetriever(
        {
            RagCollection.SESSION_MEMORY: [
                _chunk(
                    "memory-1",
                    score=0.41,
                    collection=RagCollection.SESSION_MEMORY,
                    text="The player promised to return before dawn.",
                    importance=5,
                    scene_id="rose-gallery",
                    session_id="session-1",
                )
            ],
            RagCollection.CANON_LORE: [
                _chunk(
                    "lore-1",
                    score=0.82,
                    collection=RagCollection.CANON_LORE,
                    text="The Rose Gallery opens onto the winter court.",
                )
            ],
        }
    )

    chunks = ActorContextRetriever(retriever=retriever).retrieve_for_actor(
        query="Tell me about the Rose Gallery.",
        world_id="demo_world",
        session_id="session-1",
        persona_id="archivist",
        scene_id="rose-gallery",
        top_k=2,
    )

    assert [chunk.id for chunk in chunks] == ["lore-1", "memory-1"]


def test_actor_context_retriever_uses_scene_and_persona_metadata_boosts() -> None:
    retriever = RecordingRetriever(
        {
            RagCollection.SESSION_MEMORY: [
                _chunk(
                    "scene-match",
                    score=0.55,
                    collection=RagCollection.SESSION_MEMORY,
                    text="The promise was made in the gallery itself.",
                    scene_id="rose-gallery",
                    session_id="session-1",
                ),
                _chunk(
                    "persona-match",
                    score=0.55,
                    collection=RagCollection.SESSION_MEMORY,
                    text="The archivist expects the player at dawn.",
                    actor_id="archivist",
                    session_id="session-1",
                ),
                _chunk(
                    "no-match",
                    score=0.56,
                    collection=RagCollection.SESSION_MEMORY,
                    text="A lower garden fountain conversation.",
                    session_id="session-1",
                ),
            ]
        }
    )

    chunks = ActorContextRetriever(retriever=retriever).retrieve_for_actor(
        query="What matters to the archivist in this gallery scene?",
        world_id="demo_world",
        session_id="session-1",
        persona_id="archivist",
        scene_id="rose-gallery",
        top_k=3,
    )

    assert [chunk.id for chunk in chunks] == ["scene-match", "persona-match", "no-match"]


def test_actor_context_retriever_prefers_higher_importance_when_scores_are_similar() -> None:
    retriever = RecordingRetriever(
        {
            RagCollection.SESSION_MEMORY: [
                _chunk(
                    "high-importance",
                    score=0.58,
                    collection=RagCollection.SESSION_MEMORY,
                    text="The archive key promise matters later.",
                    importance=5,
                    session_id="session-1",
                ),
                _chunk(
                    "low-importance",
                    score=0.59,
                    collection=RagCollection.SESSION_MEMORY,
                    text="The player admired the roses.",
                    importance=1,
                    session_id="session-1",
                ),
            ]
        }
    )

    chunks = ActorContextRetriever(retriever=retriever).retrieve_for_actor(
        query="What am I expected to do later?",
        world_id="demo_world",
        session_id="session-1",
        persona_id="archivist",
        scene_id="rose-gallery",
        top_k=2,
    )

    assert [chunk.id for chunk in chunks] == ["high-importance", "low-importance"]


def test_actor_context_retriever_exposes_metadata_only_diagnostics_for_selected_chunks() -> None:
    retriever = RecordingRetriever(
        {
            RagCollection.SESSION_MEMORY: [
                _chunk(
                    "memory-1",
                    score=0.60,
                    collection=RagCollection.SESSION_MEMORY,
                    text="The player promised to return before dawn.",
                    importance=4,
                    scene_id="rose-gallery",
                    session_id="session-1",
                )
            ]
        }
    )

    result = ActorContextRetriever(retriever=retriever).retrieve_for_actor_with_diagnostics(
        query="What did I promise?",
        world_id="demo_world",
        session_id="session-1",
        persona_id="archivist",
        scene_id="rose-gallery",
        top_k=1,
    )

    assert [chunk.id for chunk in result.chunks] == ["memory-1"]
    assert len(result.diagnostics.selected) == 1
    diagnostic = result.diagnostics.selected[0]
    assert isinstance(diagnostic, ChunkRetrievalDiagnostic)
    assert diagnostic.id == "memory-1"
    assert diagnostic.collection == RagCollection.SESSION_MEMORY
    assert diagnostic.original_score == 0.60
    assert diagnostic.adjusted_score > diagnostic.original_score
    assert diagnostic.visibility == Visibility.PLAYER
    assert diagnostic.tags == []
    assert not hasattr(diagnostic, "text")


def test_actor_context_retriever_exposes_rejected_candidates_in_diagnostics() -> None:
    retriever = RecordingRetriever(
        {
            RagCollection.SESSION_MEMORY: [
                _chunk(
                    "memory-1",
                    score=0.60,
                    collection=RagCollection.SESSION_MEMORY,
                    text="The player promised to return before dawn.",
                    session_id="session-1",
                ),
                _chunk(
                    "memory-2",
                    score=0.30,
                    collection=RagCollection.SESSION_MEMORY,
                    text="The player admired the roses.",
                    session_id="session-1",
                ),
            ],
            RagCollection.CANON_LORE: [
                _chunk(
                    "lore-1",
                    score=0.20,
                    collection=RagCollection.CANON_LORE,
                    text="The archive key is kept near the west stacks.",
                )
            ],
        }
    )

    result = ActorContextRetriever(retriever=retriever).retrieve_for_actor_with_diagnostics(
        query="What did I promise?",
        world_id="demo_world",
        session_id="session-1",
        persona_id="archivist",
        scene_id="rose-gallery",
        top_k=1,
    )

    assert [chunk.id for chunk in result.chunks] == ["memory-1"]
    assert [entry.id for entry in result.diagnostics.selected] == ["memory-1"]
    assert [entry.id for entry in result.diagnostics.rejected] == ["memory-2", "lore-1"]
    rejected = result.diagnostics.rejected[0]
    assert isinstance(rejected, ChunkRetrievalDiagnostic)
    assert rejected.selected_rank is None
    assert rejected.original_score == 0.30
    assert rejected.applied_boosts
    assert not hasattr(rejected, "text")
