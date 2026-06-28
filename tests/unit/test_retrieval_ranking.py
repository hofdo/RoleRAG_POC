from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.domain import RetrievedChunk, Visibility
from app.rag.diagnostics import ChunkRetrievalDiagnostic
from app.rag.models import RagCollection, RetrievalFilter
from app.rag.ranking import RankingWeights, RetrievalRankingContext, rerank_chunks
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
    created_at: datetime | None = None,
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
        created_at=created_at,
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


def test_lexical_overlap_boost_rescues_paraphrased_event_memory() -> None:
    retriever = RecordingRetriever(
        {
            RagCollection.SESSION_MEMORY: [
                _chunk(
                    "event-memory",
                    score=0.50,
                    collection=RagCollection.SESSION_MEMORY,
                    text="The player promised to return before dawn.",
                    session_id="session-1",
                ),
                _chunk(
                    "distractor-memory",
                    score=0.55,
                    collection=RagCollection.SESSION_MEMORY,
                    text="Courtiers exchanged routine pleasantries near the mirrors.",
                    session_id="session-1",
                ),
            ]
        }
    )

    result = ActorContextRetriever(retriever=retriever).retrieve_for_actor_with_diagnostics(
        query=(
            "Scene: Rose Gallery\n"
            "User message: I ask whether she remembers the promise I made about returning."
        ),
        lexical_query="I ask whether she remembers the promise I made about returning.",
        world_id="world-1",
        session_id="session-1",
        persona_id="archivist",
        top_k=1,
    )

    assert [chunk.id for chunk in result.chunks] == ["event-memory"]
    selected = result.diagnostics.selected[0]
    assert selected.applied_boosts["lexical"] > 0.0


def test_lexical_boost_uses_player_message_not_scene_boilerplate() -> None:
    retriever = RecordingRetriever(
        {
            RagCollection.CANON_LORE: [
                _chunk(
                    "lore-1",
                    score=0.40,
                    collection=RagCollection.CANON_LORE,
                    text="The Rose Gallery archive sits behind the locked west door.",
                )
            ]
        }
    )

    result = ActorContextRetriever(retriever=retriever).retrieve_for_actor_with_diagnostics(
        query="Scene: Rose Gallery\nLocation: Winter Palace\nUser message: I greet her warmly.",
        lexical_query="I greet her warmly.",
        world_id="world-1",
        session_id="session-1",
        persona_id="archivist",
        top_k=1,
    )

    assert "lexical" not in result.diagnostics.selected[0].applied_boosts


def test_lexical_boost_counts_tag_matches_and_is_capped() -> None:
    chunk = _chunk(
        "event-memory",
        score=0.10,
        collection=RagCollection.SESSION_MEMORY,
        text="The promise, the compass, the seal, the key, and the signal all return at dawn.",
        session_id="session-1",
    )
    chunk = chunk.model_copy(update={"tags": ["promise", "dawn"]})
    retriever = RecordingRetriever({RagCollection.SESSION_MEMORY: [chunk]})

    result = ActorContextRetriever(retriever=retriever).retrieve_for_actor_with_diagnostics(
        query="promise compass seal key signal return dawn",
        lexical_query="promise compass seal key signal return dawn",
        world_id="world-1",
        session_id="session-1",
        persona_id="archivist",
        top_k=1,
    )

    assert result.diagnostics.selected[0].applied_boosts["lexical"] == 0.25


def _session_candidate(
    chunk_id: str,
    *,
    created_at: datetime | None,
    score: float = 0.50,
    importance: int | None = None,
) -> tuple[RagCollection, RetrievedChunk]:
    return (
        RagCollection.SESSION_MEMORY,
        _chunk(
            chunk_id,
            score=score,
            collection=RagCollection.SESSION_MEMORY,
            text="The player made a durable commitment.",
            session_id="session-1",
            created_at=created_at,
            importance=importance,
        ),
    )


_RANK_CONTEXT = RetrievalRankingContext(
    query="commitment", session_id="session-1", persona_id="archivist"
)


def test_recency_boost_prefers_newer_among_distinct_timestamps() -> None:
    older = _session_candidate(
        "older", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), importance=5
    )
    newer = _session_candidate(
        "newer", created_at=datetime(2026, 6, 1, tzinfo=timezone.utc), importance=5
    )

    chunks, diagnostics = rerank_chunks(
        context=_RANK_CONTEXT,
        candidates=[older, newer],
        top_k=2,
        weights=RankingWeights(recency_weight=0.1),
    )

    assert [chunk.id for chunk in chunks] == ["newer", "older"]
    boosts = {diag.id: diag.applied_boosts for diag in diagnostics.selected}
    assert boosts["newer"]["recency"] == 0.1  # importance 5 -> full factor
    assert "recency" not in boosts["older"]


def test_recency_boost_scales_with_importance() -> None:
    # Two equally-recent memories: the important one earns a larger recency lift than the trivial
    # one, and an unimportant memory cannot ride recency over an older high-importance memory.
    important_new = _session_candidate(
        "important-new", created_at=datetime(2026, 6, 1, tzinfo=timezone.utc), importance=5
    )
    trivial_new = _session_candidate(
        "trivial-new", created_at=datetime(2026, 6, 1, tzinfo=timezone.utc), importance=1
    )
    older = _session_candidate(
        "older", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), importance=5
    )

    _, diagnostics = rerank_chunks(
        context=_RANK_CONTEXT,
        candidates=[important_new, trivial_new, older],
        top_k=3,
        weights=RankingWeights(recency_weight=0.1),
    )

    boosts = {
        diag.id: diag.applied_boosts
        for diag in [*diagnostics.selected, *diagnostics.rejected]
    }
    assert boosts["important-new"]["recency"] == pytest.approx(0.1)  # 0.1 * 1.0 * (5/5)
    assert boosts["trivial-new"]["recency"] == pytest.approx(0.02)  # 0.1 * 1.0 * (1/5)
    assert "recency" not in boosts["older"]  # oldest distinct timestamp -> rank 0


def test_recency_no_boost_when_all_timestamps_equal() -> None:
    stamp = datetime(2026, 3, 1, tzinfo=timezone.utc)
    candidates = [
        _session_candidate("a", created_at=stamp),
        _session_candidate("b", created_at=stamp),
        _session_candidate("c", created_at=stamp),
    ]

    _, diagnostics = rerank_chunks(
        context=_RANK_CONTEXT,
        candidates=candidates,
        top_k=3,
        weights=RankingWeights(recency_weight=0.1),
    )

    for diag in [*diagnostics.selected, *diagnostics.rejected]:
        assert "recency" not in diag.applied_boosts


def test_recency_no_boost_for_chunks_without_created_at() -> None:
    dated_new = _session_candidate(
        "dated-new", created_at=datetime(2026, 6, 1, tzinfo=timezone.utc), importance=5
    )
    dated_old = _session_candidate(
        "dated-old", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), importance=5
    )
    undated = _session_candidate("undated", created_at=None, importance=5)

    _, diagnostics = rerank_chunks(
        context=_RANK_CONTEXT,
        candidates=[dated_new, dated_old, undated],
        top_k=3,
        weights=RankingWeights(recency_weight=0.1),
    )

    boosts = {
        diag.id: diag.applied_boosts
        for diag in [*diagnostics.selected, *diagnostics.rejected]
    }
    assert boosts["dated-new"]["recency"] == 0.1
    assert "recency" not in boosts["undated"]


def test_weights_override_changes_collection_boost() -> None:
    candidate = _session_candidate("memory-1", created_at=None)

    _, diagnostics = rerank_chunks(
        context=_RANK_CONTEXT,
        candidates=[candidate],
        top_k=1,
        weights=RankingWeights(session_memory_weight=0.5),
    )

    assert diagnostics.selected[0].applied_boosts["collection"] == 0.5
