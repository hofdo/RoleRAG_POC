"""Embedding-cosine near-duplicate detection for memory write-dedup.

The term-overlap check (`is_covered_by_summaries`) catches lexical duplicates but
misses paraphrases that share intent without sharing words. This adds a semantic
pass over candidate summaries. It is gated by a cosine threshold that defaults to
1.0 (effectively off — only exact-direction matches drop, which the lexical pass
already handles), so it adds no cost until a tuning run lowers it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_magnitude = math.sqrt(sum(value * value for value in left))
    right_magnitude = math.sqrt(sum(value * value for value in right))
    if left_magnitude == 0.0 or right_magnitude == 0.0:
        return 0.0
    return numerator / (left_magnitude * right_magnitude)


def is_semantic_duplicate(
    vector: Sequence[float],
    references: Sequence[Sequence[float]],
    *,
    threshold: float,
) -> bool:
    """True when `vector` is at least `threshold` cosine-similar to any reference."""
    return any(
        cosine_similarity(vector, reference) >= threshold for reference in references
    )
