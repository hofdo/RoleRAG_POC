from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Final

from app.domain import MemoryEpisode, RetrievedChunk
from app.domain.visibility import Visibility
from app.rag.diagnostics import ChunkRetrievalDiagnostic, RetrievalDiagnostics
from app.rag.models import RagCollection

if TYPE_CHECKING:
    from app.rag.lexical import LexicalHit

SESSION_MEMORY_WEIGHT: Final[float] = 0.08
PERSONA_MEMORY_WEIGHT: Final[float] = 0.04
CANON_LORE_WEIGHT: Final[float] = 0.0
SESSION_ID_MATCH_BOOST: Final[float] = 0.02
SCENE_ID_MATCH_BOOST: Final[float] = 0.04
PERSONA_ID_MATCH_BOOST: Final[float] = 0.03
IMPORTANCE_STEP_BOOST: Final[float] = 0.015
LEXICAL_MATCH_STEP_BOOST: Final[float] = 0.05
LEXICAL_MATCH_MAX_BOOST: Final[float] = 0.25

# Confidence-equivalent contributed to retrieval_confidence by a chunk guaranteed
# into the prompt by the Lane B lexical slice (docs/26 §3.4, backlog #79). A hit's
# raw summed-IDF slice_score is on a session-relative scale that is NOT comparable
# to cosine similarity, so a slice member instead floors its confidence
# contribution at this fixed cosine-equivalent. 0.5 clears the default
# low_retrieval_confidence threshold (0.45, app/config.py) so a turn rescued
# ENTIRELY by the lexical slice is not treated as low-confidence and does not
# trigger hedging on exactly the turn the slice exists to rescue. Pinned by the
# named slice-confidence test.
SLICE_CONFIDENCE_EQUIVALENT: Final[float] = 0.5

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


@dataclass(frozen=True)
class SliceQuotas:
    """Lane B lexical slice-quota policy (docs/26 §3.4, backlog #79).

    ``lexical`` reserves that many prompt slots for the highest session-pool-IDF
    lexical matches to the player's message; ``0`` is a byte-identical no-op (the
    house opt-in pattern). ``min_slice_score`` is fix 2's floor: only hits whose
    summed-IDF score clears it may claim a reserved slot, so weak/common-term hits
    spill to the fused fill exactly as if the slice were empty. ``None`` means NO
    floor -- the mechanism ships, but docs/26 deliberately refuses to guess a
    numeral (Stage 5 measures it), and ``None`` must behave exactly like pre-fix
    semantics so the fix's mere presence never silently changes behavior.
    """

    lexical: int = 0
    min_slice_score: float | None = None


DEFAULT_SLICE_QUOTAS: Final[SliceQuotas] = SliceQuotas()


@dataclass(frozen=True)
class SliceQuotaResult:
    """Output of ``apply_lexical_slice_quotas``: the reordered/injected chunk list
    feeding ``select_retrieved_chunks_for_prompt``, its diagnostics, and the set of
    chunk ids that occupy a guaranteed lexical slot (so the stage can compute
    slice-aware ``retrieval_confidence`` independently of the diagnostics)."""

    chunks: list[RetrievedChunk]
    diagnostics: RetrievalDiagnostics | None
    slice_member_ids: frozenset[str] = field(default_factory=frozenset)


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


def _eligible_quota_hits(
    lexical_hits: Sequence[LexicalHit], quotas: SliceQuotas
) -> list[LexicalHit]:
    """The top ``quotas.lexical`` hits that clear ``min_slice_score``.

    ``min_slice_score is None`` (fix 2 ships unset) applies no floor -- identical
    to pre-fix "any matched term qualifies" semantics. ``lexical_hits`` is already
    in deterministic descending-score order (see ``score_memories_lexical``).
    """
    floor = quotas.min_slice_score
    eligible = [hit for hit in lexical_hits if floor is None or hit.score >= floor]
    return eligible[: quotas.lexical]


def apply_lexical_slice_quotas(
    *,
    chunks: Sequence[RetrievedChunk],
    diagnostics: RetrievalDiagnostics | None,
    lexical_hits: Sequence[LexicalHit],
    memories_by_id: Mapping[str, MemoryEpisode],
    quotas: SliceQuotas,
    prompt_window: int,
) -> SliceQuotaResult:
    """Reorder/inject so guaranteed lexical members reach the FINAL prompt window.

    This operates so that ``select_retrieved_chunks_for_prompt``'s UNCHANGED walk
    (which selects the first ``prompt_window`` PLAYER-visible, non-excluded,
    distinct chunks front-to-back) lands the quota members in the actor prompt --
    docs/26 §3.4's single most load-bearing detail. The mechanism is
    ordering, not simulation: the guaranteed members are moved to the FRONT of the
    list in quota order (a dense hit already in the natural top window is merely
    reordered to the front, so no dense chunk is evicted; a hit deeper than the
    window, or one not dense-fetched at all, is pulled in, evicting the lowest
    dense chunk). ``prompt_window`` is accepted for interface clarity and future
    caps; the front-load ordering makes the guarantee hold for any window size.

    Injected members (a lexical hit whose memory was not dense-fetched) are built
    from ``memories_by_id`` as ``original_score=0.0`` / ``applied_boosts={}`` with
    a dedicated ``slice_score`` on the diagnostic (fix 3) -- so the
    ``adjusted_score == original_score + sum(applied_boosts)`` identity holds for
    every chunk (0.0 == 0.0 + 0). A dense-fetched chunk that also wins a slot keeps
    its real score/boosts and merely gains ``slice_score``/matched-terms/guaranteed
    labels. A memory never appears twice: a hit already present as a dense chunk is
    promoted in place, never re-injected.

    ``quotas.lexical <= 0`` (default) is a byte-identical no-op: the inputs are
    returned unchanged with an empty ``slice_member_ids``. Passing ``chunks=()``
    with ``diagnostics=None`` yields the lexical-only degradation the stage uses
    when dense retrieval fails.
    """
    if quotas.lexical <= 0:
        return SliceQuotaResult(chunks=list(chunks), diagnostics=diagnostics)
    quota_hits = _eligible_quota_hits(lexical_hits, quotas)
    if not quota_hits:
        return SliceQuotaResult(chunks=list(chunks), diagnostics=diagnostics)

    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    diag_by_id = (
        {entry.id: entry for entry in diagnostics.selected} if diagnostics is not None else {}
    )
    guaranteed_chunks: list[RetrievedChunk] = []
    guaranteed_diags: list[ChunkRetrievalDiagnostic] = []
    consumed: set[str] = set()
    for hit in quota_hits:
        memory_id = hit.memory_id
        if memory_id in consumed:
            continue
        dense_chunk = chunk_by_id.get(memory_id)
        if dense_chunk is not None:
            guaranteed_chunks.append(dense_chunk)
            if diagnostics is not None:
                base = diag_by_id.get(memory_id) or _diagnostic_from_chunk(dense_chunk)
                guaranteed_diags.append(_mark_slice_diagnostic(base, hit))
        else:
            memory = memories_by_id.get(memory_id)
            if memory is None:
                continue
            guaranteed_chunks.append(_memory_to_chunk(memory))
            if diagnostics is not None:
                guaranteed_diags.append(_injected_diagnostic(memory, hit))
        consumed.add(memory_id)

    if not consumed:
        return SliceQuotaResult(chunks=list(chunks), diagnostics=diagnostics)

    rest_chunks = [chunk for chunk in chunks if chunk.id not in consumed]
    ordered_chunks = guaranteed_chunks + rest_chunks

    new_diagnostics = diagnostics
    if diagnostics is not None:
        rest_diags = [
            diag_by_id.get(chunk.id) or _diagnostic_from_chunk(chunk) for chunk in rest_chunks
        ]
        ordered_diags = guaranteed_diags + rest_diags
        renumbered = [
            entry.model_copy(update={"selected_rank": rank})
            for rank, entry in enumerate(ordered_diags, start=1)
        ]
        # A hit injected from beyond the reranked window may also appear in
        # ``rejected``; drop it there so a guaranteed member is never double-listed.
        new_rejected = [entry for entry in diagnostics.rejected if entry.id not in consumed]
        new_diagnostics = RetrievalDiagnostics(
            query=diagnostics.query, selected=renumbered, rejected=new_rejected
        )

    return SliceQuotaResult(
        chunks=ordered_chunks,
        diagnostics=new_diagnostics,
        slice_member_ids=frozenset(consumed),
    )


def slice_aware_confidence(
    chunks: Sequence[RetrievedChunk], slice_member_ids: frozenset[str]
) -> float:
    """``retrieval_confidence`` over PLAYER-visible chunks, slice-aware (docs/26
    §3.4). Each chunk contributes ``max(original_score, SLICE_CONFIDENCE_EQUIVALENT)``
    when it holds a guaranteed lexical slot, else its raw ``original_score`` -- so a
    turn rescued entirely by the lexical slice (injected chunks carry
    ``original_score == 0.0``) is not read as low-confidence. With an empty
    ``slice_member_ids`` this is byte-identical to the pre-Lane-B
    ``max(chunk.score for player-visible chunks) if any else 0.0``.
    """
    contributions = [
        (
            max(chunk.score, SLICE_CONFIDENCE_EQUIVALENT)
            if chunk.id in slice_member_ids
            else chunk.score
        )
        for chunk in chunks
        if chunk.visibility == Visibility.PLAYER
    ]
    return max(contributions) if contributions else 0.0


def _memory_to_chunk(memory: MemoryEpisode) -> RetrievedChunk:
    """Build the injected session-memory chunk, mirroring
    ``app.memory.indexer.MemoryIndexer._to_chunk`` so an injected chunk is
    indistinguishable from its dense-fetched form (same id, source, fields)."""
    return RetrievedChunk(
        id=memory.id,
        source=f"memory_episode:{memory.id}",
        source_type=RagCollection.SESSION_MEMORY.value,
        text=memory.summary,
        score=0.0,
        visibility=memory.visibility,
        tags=memory.tags,
        scene_id=memory.scene_id,
        session_id=memory.session_id,
        actor_id=memory.actor_id,
        importance=memory.importance,
        created_at=memory.created_at,
    )


def _injected_diagnostic(memory: MemoryEpisode, hit: LexicalHit) -> ChunkRetrievalDiagnostic:
    return ChunkRetrievalDiagnostic(
        id=memory.id,
        source=f"memory_episode:{memory.id}",
        source_type=RagCollection.SESSION_MEMORY.value,
        collection=RagCollection.SESSION_MEMORY,
        visibility=memory.visibility,
        tags=list(memory.tags),
        original_score=0.0,
        adjusted_score=0.0,
        applied_boosts={},
        selected_rank=None,
        slice_score=hit.score,
        slice_matched_terms=list(hit.matched),
        slice_guaranteed=True,
    )


def _mark_slice_diagnostic(
    entry: ChunkRetrievalDiagnostic, hit: LexicalHit
) -> ChunkRetrievalDiagnostic:
    return entry.model_copy(
        update={
            "slice_score": hit.score,
            "slice_matched_terms": list(hit.matched),
            "slice_guaranteed": True,
        }
    )


def _diagnostic_from_chunk(chunk: RetrievedChunk) -> ChunkRetrievalDiagnostic:
    """Fallback diagnostic for a chunk with no reranked entry (defensive: the
    reranked ``chunks``/``diagnostics.selected`` lists are parallel in the real
    retriever, so this is only reached if a caller supplies a mismatched pair)."""
    return ChunkRetrievalDiagnostic(
        id=chunk.id,
        source=chunk.source,
        source_type=chunk.source_type,
        collection=RagCollection.SESSION_MEMORY,
        visibility=chunk.visibility,
        tags=list(chunk.tags),
        original_score=chunk.score,
        adjusted_score=chunk.score,
        applied_boosts={},
    )
