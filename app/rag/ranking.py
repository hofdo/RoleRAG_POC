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


def candidate_limit(top_k: int) -> int:
    """Oversample each collection so reranking can promote boosted candidates."""
    return top_k * 2


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
    lexical_boost = _lexical_overlap_boost(
        query_text=context.lexical_query or context.query,
        chunk=chunk,
    )
    if lexical_boost > 0.0:
        applied_boosts["lexical"] = lexical_boost
    adjusted_score = chunk.score + sum(applied_boosts.values())
    return RankedChunk(
        chunk=chunk,
        collection=collection,
        original_score=chunk.score,
        adjusted_score=adjusted_score,
        applied_boosts=applied_boosts,
    )


def _lexical_overlap_boost(*, query_text: str, chunk: RetrievedChunk) -> float:
    query_terms = content_terms(query_text)
    if not query_terms:
        return 0.0
    chunk_terms = content_terms(chunk.text) | content_terms(" ".join(chunk.tags))
    matches = len(query_terms & chunk_terms)
    return min(matches * LEXICAL_MATCH_STEP_BOOST, LEXICAL_MATCH_MAX_BOOST)


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
