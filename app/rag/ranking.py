from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
LEXICAL_MATCH_STEP_BOOST: Final[float] = 0.05
LEXICAL_MATCH_MAX_BOOST: Final[float] = 0.25

# Function words excluded from lexical overlap so that conversational framing
# ("I ask whether she...") does not boost unrelated chunks.
_LEXICAL_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "the", "and", "for", "are", "was", "were", "with", "that", "this",
        "what", "which", "who", "whom", "whether", "about", "into", "onto",
        "from", "they", "them", "their", "she", "her", "hers", "him", "his",
        "you", "your", "yours", "our", "ours", "have", "has", "had", "does",
        "did", "will", "would", "should", "could", "can", "may", "might",
        "not", "all", "any", "some", "one", "two", "where", "when", "how",
        "why", "ask", "asks", "asked", "tell", "tells", "told", "say", "says",
        "said",
    }
)
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
class RankingWeights:
    """Tunable reranking weights. Defaults equal the canonical module constants
    above, so an omitted/`DEFAULT_RANKING_WEIGHTS` argument reproduces the
    pre-config behavior byte-for-byte."""

    session_memory_weight: float = SESSION_MEMORY_WEIGHT
    persona_memory_weight: float = PERSONA_MEMORY_WEIGHT
    canon_lore_weight: float = CANON_LORE_WEIGHT
    session_id_match_boost: float = SESSION_ID_MATCH_BOOST
    scene_id_match_boost: float = SCENE_ID_MATCH_BOOST
    persona_id_match_boost: float = PERSONA_ID_MATCH_BOOST
    importance_step_boost: float = IMPORTANCE_STEP_BOOST
    lexical_match_step_boost: float = LEXICAL_MATCH_STEP_BOOST
    lexical_match_max_boost: float = LEXICAL_MATCH_MAX_BOOST
    recency_weight: float = 0.0
    candidate_oversample_factor: int = 2

    def collection_weight(self, collection: RagCollection) -> float:
        return {
            RagCollection.SESSION_MEMORY: self.session_memory_weight,
            RagCollection.PERSONA_MEMORY: self.persona_memory_weight,
            RagCollection.CANON_LORE: self.canon_lore_weight,
        }[collection]


DEFAULT_RANKING_WEIGHTS: Final[RankingWeights] = RankingWeights()


@dataclass(frozen=True)
class RetrievalRankingContext:
    query: str
    session_id: str
    persona_id: str
    scene_id: str | None = None
    # Player-message-only text for lexical overlap; falls back to `query`,
    # which also contains scene/persona framing that would otherwise boost
    # any chunk sharing scene vocabulary.
    lexical_query: str | None = None


@dataclass(frozen=True)
class RankedChunk:
    chunk: RetrievedChunk
    collection: RagCollection
    original_score: float
    adjusted_score: float
    applied_boosts: dict[str, float]


def candidate_limit(top_k: int, *, oversample_factor: int = 2) -> int:
    """Oversample each collection so reranking can promote boosted candidates."""
    return top_k * oversample_factor


def rerank_chunks(
    *,
    context: RetrievalRankingContext,
    candidates: list[tuple[RagCollection, RetrievedChunk]],
    top_k: int,
    weights: RankingWeights = DEFAULT_RANKING_WEIGHTS,
) -> tuple[list[RetrievedChunk], RetrievalDiagnostics]:
    recency_ranks = _compute_recency_ranks(candidates)
    ranked = [
        _rank_chunk(
            context=context,
            collection=collection,
            chunk=chunk,
            weights=weights,
            recency_rank=recency_ranks.get(chunk.id, 0.0),
        )
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


def _compute_recency_ranks(
    candidates: list[tuple[RagCollection, RetrievedChunk]],
) -> dict[str, float]:
    """Map chunk id -> normalized recency rank in [0, 1] (newest -> 1.0).

    Ranks by *distinct* ``created_at`` so memories written in one batch share a
    value and cannot reorder among themselves. Returns an empty mapping when
    fewer than two distinct timestamps exist, so a single-batch candidate set
    (e.g. the event_key_retrieval seed) receives no differential boost. Chunks
    without a timestamp (canon lore, legacy rows) are simply absent and get no
    recency boost.
    """
    created_by_id: dict[str, datetime] = {}
    for _collection, chunk in candidates:
        if chunk.created_at is not None:
            created_by_id.setdefault(chunk.id, chunk.created_at)
    distinct = sorted(set(created_by_id.values()))
    if len(distinct) < 2:
        return {}
    rank_by_timestamp = {
        timestamp: index / (len(distinct) - 1)
        for index, timestamp in enumerate(distinct)
    }
    return {
        chunk_id: rank_by_timestamp[created]
        for chunk_id, created in created_by_id.items()
    }


def _rank_chunk(
    *,
    context: RetrievalRankingContext,
    collection: RagCollection,
    chunk: RetrievedChunk,
    weights: RankingWeights = DEFAULT_RANKING_WEIGHTS,
    recency_rank: float = 0.0,
) -> RankedChunk:
    applied_boosts: dict[str, float] = {}
    collection_boost = weights.collection_weight(collection)
    if collection_boost != 0.0:
        applied_boosts["collection"] = collection_boost
    if chunk.session_id is not None and chunk.session_id == context.session_id:
        applied_boosts["session"] = weights.session_id_match_boost
    if context.scene_id is not None and chunk.scene_id == context.scene_id:
        applied_boosts["scene"] = weights.scene_id_match_boost
    if chunk.persona_id == context.persona_id or chunk.actor_id == context.persona_id:
        applied_boosts["persona"] = weights.persona_id_match_boost
    if chunk.importance is not None and chunk.importance > 1:
        applied_boosts["importance"] = (chunk.importance - 1) * weights.importance_step_boost
    lexical_boost = _lexical_overlap_boost(
        query_text=context.lexical_query or context.query,
        chunk=chunk,
        weights=weights,
    )
    if lexical_boost > 0.0:
        applied_boosts["lexical"] = lexical_boost
    recency_boost = weights.recency_weight * recency_rank * _recency_importance_factor(chunk)
    if recency_boost > 0.0:
        applied_boosts["recency"] = recency_boost
    adjusted_score = chunk.score + sum(applied_boosts.values())
    return RankedChunk(
        chunk=chunk,
        collection=collection,
        original_score=chunk.score,
        adjusted_score=adjusted_score,
        applied_boosts=applied_boosts,
    )


def _recency_importance_factor(chunk: RetrievedChunk) -> float:
    """Scale the recency boost by importance so recency lifts a recent *important* memory more
    than a recent trivial one, and never lifts an unimportant or timeless (lore) chunk over an
    older high-importance memory. Importance 1-5 -> 0.2-1.0; missing importance (canon lore,
    legacy rows) -> 0.0, so recency only ever reorders scored episodic memories.

    This is what makes the recency boost recall-safe: a newer low-value memory cannot out-rank an
    older promise/key fact, because its importance factor shrinks the recency lift toward zero.
    """
    if chunk.importance is None:
        return 0.0
    return min(chunk.importance, 5) / 5.0


def _lexical_overlap_boost(
    *,
    query_text: str,
    chunk: RetrievedChunk,
    weights: RankingWeights = DEFAULT_RANKING_WEIGHTS,
) -> float:
    query_terms = content_terms(query_text)
    if not query_terms:
        return 0.0
    chunk_terms = content_terms(chunk.text) | content_terms(" ".join(chunk.tags))
    matches = len(query_terms & chunk_terms)
    return min(matches * weights.lexical_match_step_boost, weights.lexical_match_max_boost)


def content_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw_token in text.lower().split():
        token = "".join(char for char in raw_token if char.isalpha())
        if len(token) < 3 or token in _LEXICAL_STOPWORDS:
            continue
        terms.add(_stem(token))
    return terms


def _stem(token: str) -> str:
    """Light deterministic suffix stripping so 'promised' matches 'promise'."""
    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]
    elif token.endswith("ed") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("es") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("s") and len(token) > 3:
        token = token[:-1]
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


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
