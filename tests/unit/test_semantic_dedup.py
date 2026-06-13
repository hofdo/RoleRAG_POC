from __future__ import annotations

from app.memory.semantic_dedup import cosine_similarity, is_semantic_duplicate


def test_cosine_similarity_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_is_semantic_duplicate_when_above_threshold() -> None:
    assert is_semantic_duplicate(
        [1.0, 0.0],
        [[0.0, 1.0], [0.99, 0.01]],
        threshold=0.9,
    )


def test_is_not_semantic_duplicate_when_below_threshold() -> None:
    assert not is_semantic_duplicate(
        [1.0, 0.0],
        [[0.0, 1.0], [0.5, 0.5]],
        threshold=0.9,
    )


def test_threshold_one_only_matches_identical_direction() -> None:
    assert is_semantic_duplicate([2.0, 0.0], [[1.0, 0.0]], threshold=1.0)
    assert not is_semantic_duplicate([1.0, 0.1], [[1.0, 0.0]], threshold=1.0)
