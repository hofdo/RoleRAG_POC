"""Lane B lexical slice scorer (docs/26 §3.4, backlog #79).

Pure, deterministic, session-relative lexical scoring of the player's message
against the session's own memory pool. Reuses ``content_terms`` from
``app.rag.ranking`` as the tokenizer (no new tokenizer) so lexical overlap here
and the additive-boost rerank there stay on the exact same term surface.

IDF is computed over the session's own pool, so conversational frame vocabulary
(``{player, iria, vale, keep, return}`` — the #72 poison set) is automatically
cheap (in many docs -> low IDF) and genuinely rare terms (``blue, wax, seal,
compass``) are automatically expensive. This is self-calibrating and needs no
language-specific stopword list, the one general (if partial) German mitigation
available without a German stemmer (N2 stays deferred, docs/26 §8 Q5).

This module lives OUTSIDE the vector store: it runs on ``MemoryEpisode`` rows the
turn pipeline already loads every turn for the Standing-facts block, so Lane B
adds zero queries and needs no ``InMemoryVectorStore`` parity work.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.domain import MemoryEpisode, Visibility
from app.memory.consolidation import CONSOLIDATED_TAG
from app.rag.ranking import content_terms


@dataclass(frozen=True)
class LexicalHit:
    """One session memory's lexical match against the query.

    ``score`` is the summed session-pool IDF of the matched terms; ``matched`` is
    the sorted tuple of matched (stemmed) terms, recorded for diagnostics so a
    slice pick is never a silent lexical mystery. ``importance`` and
    ``created_ordinal`` carry the deterministic tie-break keys (see
    ``score_memories_lexical``'s ordering).
    """

    memory_id: str
    score: float
    importance: int
    created_ordinal: float
    matched: tuple[str, ...]


def _created_ordinal(memory: MemoryEpisode) -> float:
    """Recency tie-break key. Missing timestamps sort oldest (lowest), mirroring
    ``app.orchestration.canon_builder._created_ordinal`` -- this repo has a known
    ``created_at``-None sort hazard, so None is handled explicitly here (never
    left to ``min``/``max`` over a set containing ``None``) and pinned by a test."""
    if memory.created_at is None:
        return float("-inf")
    return memory.created_at.timestamp()


def _doc_terms(memory: MemoryEpisode) -> set[str]:
    return content_terms(memory.summary) | content_terms(" ".join(memory.tags))


def idf_over(docs: Mapping[str, set[str]]) -> dict[str, float]:
    """Session-pool inverse document frequency: ``log(N / df)`` over the pool.

    A term in every pool doc gets IDF 0 (frame vocabulary contributes nothing);
    a term in one doc gets ``log(N)`` (rare terms are expensive). Natural log, so
    the value is deterministic and hand-computable in tests.
    """
    total = len(docs)
    if total == 0:
        return {}
    document_frequency: dict[str, int] = {}
    for terms in docs.values():
        for term in terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    return {term: math.log(total / count) for term, count in document_frequency.items()}


def score_memories_lexical(
    *,
    query_text: str,
    memories: Sequence[MemoryEpisode],
) -> list[LexicalHit]:
    """Score the pool of PLAYER-visible, non-consolidated session memories by the
    summed session-pool IDF of the terms they share with ``query_text``.

    Pool filter (docs/26 fix 1): ``visibility == PLAYER`` AND ``CONSOLIDATED_TAG``
    not in tags -- a consolidation-retired original must never be resurrected into
    the actor prompt via the lexical injection path, which would silently bypass
    the consolidation policy.

    Ordering is deterministic: ``(-score, -importance, -created_ordinal,
    memory_id)`` -- highest score first, then more-important, then newer, then a
    stable id tie-break so the order is total even when every other key ties.
    """
    pool = [
        memory
        for memory in memories
        if memory.visibility == Visibility.PLAYER and CONSOLIDATED_TAG not in memory.tags
    ]
    docs = {memory.id: _doc_terms(memory) for memory in pool}
    idf = idf_over(docs)
    query_terms = content_terms(query_text)
    hits: list[LexicalHit] = []
    for memory in pool:
        matched = query_terms & docs[memory.id]
        if not matched:
            continue
        hits.append(
            LexicalHit(
                memory_id=memory.id,
                score=sum(idf[term] for term in matched),
                importance=memory.importance,
                created_ordinal=_created_ordinal(memory),
                matched=tuple(sorted(matched)),
            )
        )
    hits.sort(key=lambda hit: (-hit.score, -hit.importance, -hit.created_ordinal, hit.memory_id))
    return hits
