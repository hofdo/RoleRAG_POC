"""Recall@k / nDCG@k / MRR over graded-relevance judgments (docs/22 P0.4).

Pure functions over `(ranked_ids, judgments)` -- no embedding, retriever, or I/O
dependency, so the metric math itself is fully unit-testable against hand-computed
values independent of any embedding model (deterministic keyword provider or
real). `judgments` maps chunk id -> graded relevance (0=irrelevant, 1=related,
2=directly relevant); a chunk id absent from the mapping is treated as judgment 0,
matching the corpus convention in `app.evals.semantic_corpus` (only chunks scoring
>=1 are listed).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(
    ranked_ids: Sequence[str],
    judgments: Mapping[str, int],
    k: int,
    *,
    relevance_floor: int = 1,
) -> float:
    """Fraction of relevant chunks (judgment >= `relevance_floor`) present in the
    top `k` ranked ids, out of every relevant chunk in `judgments`.

    `relevance_floor=1` (the default) is standard binary recall: judgment 1 or 2
    both count as relevant. `relevance_floor=2` is the strict variant docs/22 asks
    for: only directly-relevant (judgment 2) chunks count, so a strong "related but
    not the answer" retrieval no longer inflates the score.

    Returns 1.0 (vacuously satisfied) when `judgments` has no chunk at or above
    `relevance_floor` -- there is nothing to recall, so top-k trivially recalls
    all of it. Every corpus query in `semantic_corpus` has at least one judgment,
    so this only matters for hand-built edge-case tests.
    """
    relevant_ids = {
        chunk_id for chunk_id, grade in judgments.items() if grade >= relevance_floor
    }
    if not relevant_ids:
        return 1.0
    top_k = set(ranked_ids[:k])
    return len(relevant_ids & top_k) / len(relevant_ids)


def _dcg(gains: Sequence[float], k: int) -> float:
    """Standard log2-discounted DCG: gain_i / log2(i + 1), 1-indexed rank `i`."""
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains[:k]))


def ndcg_at_k(ranked_ids: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    """Normalized DCG@k using the graded (0/1/2) judgments and a log2 rank discount.

    DCG@k = sum_{i=1}^{k} judgment(ranked_ids[i]) / log2(i + 1) (chunks absent
    from `judgments` contribute a gain of 0). IDCG@k is the same formula applied
    to the ideal ordering -- every judged chunk's own grade, sorted descending,
    independent of `ranked_ids` (a perfect retriever ranks every relevant chunk
    first regardless of how many chunks exist total).

    Returns 0.0 when `judgments` is empty (no relevant chunk exists, so no
    ordering can be scored) -- there is no ideal DCG to normalize against, unlike
    `recall_at_k`'s vacuous-1.0 convention (recall asks "did we find everything
    relevant"; nDCG asks "how good is this ranking", and there is no ranking
    quality to award when nothing is judged relevant).
    """
    gains = [float(judgments.get(chunk_id, 0)) for chunk_id in ranked_ids]
    dcg = _dcg(gains, k)
    ideal_gains = sorted((float(grade) for grade in judgments.values()), reverse=True)
    idcg = _dcg(ideal_gains, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr(
    ranked_ids: Sequence[str],
    judgments: Mapping[str, int],
    *,
    relevance_floor: int = 1,
) -> float:
    """Reciprocal rank of the first chunk judged >= `relevance_floor`, or 0.0 if
    none of `ranked_ids` meets the floor (including an empty ranking)."""
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if judgments.get(chunk_id, 0) >= relevance_floor:
            return 1.0 / rank
    return 0.0
