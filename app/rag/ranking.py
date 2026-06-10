from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from app.domain import RetrievedChunk
from app.rag.diagnostics import ChunkRetrievalDiagnostic, RetrievalDiagnostics
from app.rag.models import RagCollection

SESSION_MEMORY_WEIGHT: Final[float] = 0.08
PERSONA_MEMORY_WEIGHT: Final[float] = 0.04
CANON_LORE_WEIGHT: Final[float] = 0.0
SESSION_ID_MATCH_BOOST: Final[float] = 0.02
SCENE_ID_MATCH_BOOST: Final[float] = 0.04
PERSONA_ID_MATCH_BOOST: Final[float] = 0.03
IMPORTANCE_STEP_BOOST: Final[float] = 0.015
COLLECTION_PRIORITY: Final[dict[RagCollection, int]] = {
    RagCollection.SESSION_MEMORY: 0,
    RagCollection.PERSONA_MEMORY: 1,
    RagCollection.CANON_LORE: 2,
}
COLLECTION_WEIGHTS: Final[dict[RagCollection, float]] = {
    RagCollection.SESSION_MEMORY: SESSION_MEMORY_WEIGHT,
    RagCollection.PERSONA_MEMORY: PERSONA_MEMORY_WEIGHT,
    RagCollection.CANON_LORE: CANON_LORE_WEIGHT,
}


@dataclass(frozen=True)
class RetrievalRankingContext:
    query: str
    session_id: str
    persona_id: str
    scene_id: str | None = None


@dataclass(frozen=True)
class RankedChunk:
    chunk: RetrievedChunk
    collection: RagCollection
    original_score: float
    adjusted_score: float
    applied_boosts: dict[str, float]


def candidate_limit(top_k: int) -> int:
    return max(top_k, top_k * 2)


def rerank_chunks(
    *,
    context: RetrievalRankingContext,
    candidates: list[tuple[RagCollection, RetrievedChunk]],
    top_k: int,
) -> tuple[list[RetrievedChunk], RetrievalDiagnostics]:
    ranked = [
        _rank_chunk(context=context, collection=collection, chunk=chunk)
        for collection, chunk in candidates
    ]
    deduplicated = _deduplicate_ranked_chunks(ranked)
    selected = deduplicated[:top_k]
    rejected = deduplicated[top_k:]
    diagnostics = RetrievalDiagnostics(
        query=context.query,
        selected=[
            _to_chunk_diagnostic(ranked_chunk, selected_rank=index)
            for index, ranked_chunk in enumerate(selected, start=1)
        ],
        rejected=[_to_chunk_diagnostic(ranked_chunk) for ranked_chunk in rejected],
    )
    return [ranked_chunk.chunk for ranked_chunk in selected], diagnostics


def _to_chunk_diagnostic(
    ranked_chunk: RankedChunk,
    *,
    selected_rank: int | None = None,
) -> ChunkRetrievalDiagnostic:
    return ChunkRetrievalDiagnostic(
        id=ranked_chunk.chunk.id,
        source=ranked_chunk.chunk.source,
        source_type=ranked_chunk.chunk.source_type,
        collection=ranked_chunk.collection,
        visibility=ranked_chunk.chunk.visibility,
        tags=ranked_chunk.chunk.tags,
        original_score=ranked_chunk.original_score,
        adjusted_score=ranked_chunk.adjusted_score,
        applied_boosts=ranked_chunk.applied_boosts,
        selected_rank=selected_rank,
    )


def _rank_chunk(
    *,
    context: RetrievalRankingContext,
    collection: RagCollection,
    chunk: RetrievedChunk,
) -> RankedChunk:
    applied_boosts: dict[str, float] = {}
    collection_boost = COLLECTION_WEIGHTS[collection]
    if collection_boost != 0.0:
        applied_boosts["collection"] = collection_boost
    if chunk.session_id is not None and chunk.session_id == context.session_id:
        applied_boosts["session"] = SESSION_ID_MATCH_BOOST
    if context.scene_id is not None and chunk.scene_id == context.scene_id:
        applied_boosts["scene"] = SCENE_ID_MATCH_BOOST
    if chunk.persona_id == context.persona_id or chunk.actor_id == context.persona_id:
        applied_boosts["persona"] = PERSONA_ID_MATCH_BOOST
    if chunk.importance is not None and chunk.importance > 1:
        applied_boosts["importance"] = (chunk.importance - 1) * IMPORTANCE_STEP_BOOST
    adjusted_score = chunk.score + sum(applied_boosts.values())
    return RankedChunk(
        chunk=chunk,
        collection=collection,
        original_score=chunk.score,
        adjusted_score=adjusted_score,
        applied_boosts=applied_boosts,
    )


def _deduplicate_ranked_chunks(chunks: list[RankedChunk]) -> list[RankedChunk]:
    deduplicated: dict[str, RankedChunk] = {}
    for ranked_chunk in sorted(chunks, key=_sort_key):
        deduplicated.setdefault(ranked_chunk.chunk.id, ranked_chunk)
    return list(deduplicated.values())


def _sort_key(ranked_chunk: RankedChunk) -> tuple[float, float, int, str]:
    return (
        -ranked_chunk.adjusted_score,
        -ranked_chunk.original_score,
        COLLECTION_PRIORITY[ranked_chunk.collection],
        ranked_chunk.chunk.id,
    )
