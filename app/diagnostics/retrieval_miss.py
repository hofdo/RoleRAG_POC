"""Retrieval-miss observability.

When a durable memory is indexed but does not make the top-k selection, the
checkpoint already fails in strict mode — but pass/fail alone does not say *how
far off* the memory ranked. This module ranks expected chunk ids across the full
candidate set (selected first, then the rejected tail in ranking order) so a
tuning run can see whether a recency/floor change moved a wanted memory from,
say, rank 7 toward the top-5 cut.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExpectedChunkRank:
    chunk_id: str
    selected: bool
    overall_rank: int | None
    adjusted_score: float | None


@dataclass(frozen=True)
class RetrievalMissReport:
    expected_ids: tuple[str, ...]
    ranks: tuple[ExpectedChunkRank, ...]
    missed_ids: tuple[str, ...]
    absent_ids: tuple[str, ...]


def _ordered_candidates(
    selected: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    ranked_selected = sorted(selected, key=lambda item: item.get("selected_rank") or 0)
    return [*ranked_selected, *rejected]


def summarize_retrieval_miss(
    *,
    expected_ids: Sequence[str],
    selected: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
) -> RetrievalMissReport:
    """Rank each expected chunk id across selected ∪ rejected candidates.

    `missed_ids` = expected ids that were not selected (whether they ranked in
    the rejected tail or were absent from the candidate set entirely).
    `absent_ids` = the subset that did not appear in the candidate set at all.
    """
    selected_ids = {str(item["id"]) for item in selected}
    rank_by_id: dict[str, int] = {}
    score_by_id: dict[str, float | None] = {}
    for position, item in enumerate(_ordered_candidates(selected, rejected), start=1):
        chunk_id = str(item["id"])
        if chunk_id in rank_by_id:
            continue
        rank_by_id[chunk_id] = position
        raw_score = item.get("adjusted_score")
        score_by_id[chunk_id] = float(raw_score) if raw_score is not None else None

    ranks = tuple(
        ExpectedChunkRank(
            chunk_id=str(chunk_id),
            selected=str(chunk_id) in selected_ids,
            overall_rank=rank_by_id.get(str(chunk_id)),
            adjusted_score=score_by_id.get(str(chunk_id)),
        )
        for chunk_id in expected_ids
    )
    missed_ids = tuple(rank.chunk_id for rank in ranks if not rank.selected)
    absent_ids = tuple(rank.chunk_id for rank in ranks if rank.overall_rank is None)
    return RetrievalMissReport(
        expected_ids=tuple(str(chunk_id) for chunk_id in expected_ids),
        ranks=ranks,
        missed_ids=missed_ids,
        absent_ids=absent_ids,
    )
