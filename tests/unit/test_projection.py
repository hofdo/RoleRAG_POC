from __future__ import annotations

import math
import random

import pytest

from app.rag.projection import fit_pca_2d, project_2d


def test_fit_pca_2d_analytic_diagonal_case() -> None:
    """Three points on the (1,1,0) diagonal: the first component must align with
    that diagonal and coordinates must be proportional to [-sqrt(2), 0, sqrt(2)].
    The sign convention (largest-magnitude element forced positive) fixes the
    component to (1/sqrt(2), 1/sqrt(2), 0), so the expected coordinates are exact.
    """
    vectors = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 2.0, 0.0)]

    pca = fit_pca_2d(vectors)
    coords = project_2d(pca, vectors)

    expected_x = [-math.sqrt(2), 0.0, math.sqrt(2)]
    for (x, y), expected in zip(coords, expected_x, strict=True):
        assert x == pytest.approx(expected, abs=1e-9)
        assert y == pytest.approx(0.0, abs=1e-9)

    assert pca.components[0] == pytest.approx((1 / math.sqrt(2), 1 / math.sqrt(2), 0.0))
    assert pca.explained_variance_ratio[0] == pytest.approx(1.0, abs=1e-9)
    assert pca.explained_variance_ratio[1] == pytest.approx(0.0, abs=1e-9)


def test_fit_pca_2d_is_deterministic_under_row_shuffle() -> None:
    vectors = [
        (0.0, 0.0, 1.0),
        (1.0, 2.0, 0.0),
        (2.0, 1.0, 3.0),
        (-1.0, 0.5, 2.0),
        (3.0, -2.0, 1.0),
    ]
    shuffled = list(vectors)
    random.Random(42).shuffle(shuffled)

    pca_a = fit_pca_2d(vectors)
    pca_b = fit_pca_2d(shuffled)

    assert pca_a.components == pytest.approx(pca_b.components)
    assert pca_a.explained_variance_ratio == pytest.approx(pca_b.explained_variance_ratio)

    coords_a = sorted(project_2d(pca_a, vectors))
    coords_b = sorted(project_2d(pca_b, shuffled))
    for (x_a, y_a), (x_b, y_b) in zip(coords_a, coords_b, strict=True):
        assert x_a == pytest.approx(x_b)
        assert y_a == pytest.approx(y_b)


def test_fit_pca_2d_single_point() -> None:
    pca = fit_pca_2d([(1.0, 2.0, 3.0)])

    assert project_2d(pca, [(1.0, 2.0, 3.0)]) == [(0.0, 0.0)]
    assert pca.explained_variance_ratio == (0.0, 0.0)
    assert not pca.components.any()


def test_fit_pca_2d_rank_one_collinear_points() -> None:
    vectors = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (-1.0, -2.0, -3.0)]

    pca = fit_pca_2d(vectors)
    coords = project_2d(pca, vectors)

    assert all(y == 0.0 for _x, y in coords)
    assert pca.explained_variance_ratio[1] == 0.0
    assert not pca.components[1].any()


def test_fit_pca_2d_all_identical_points() -> None:
    vectors = [(1.0, 1.0, 1.0)] * 4

    pca = fit_pca_2d(vectors)
    coords = project_2d(pca, vectors)

    assert coords == [(0.0, 0.0)] * 4
    assert pca.explained_variance_ratio == (0.0, 0.0)


def test_project_2d_reprojection_is_consistent_and_finite() -> None:
    vectors = [
        (0.0, 1.0, 2.0),
        (3.0, 1.0, 0.0),
        (2.0, 2.0, 2.0),
        (-1.0, 4.0, 1.0),
    ]
    pca = fit_pca_2d(vectors)

    first = project_2d(pca, vectors)
    second = project_2d(pca, vectors)

    assert first == second
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in first)


def test_project_2d_query_overlay_matches_training_point() -> None:
    vectors = [
        (0.0, 1.0, 2.0),
        (3.0, 1.0, 0.0),
        (2.0, 2.0, 2.0),
        (-1.0, 4.0, 1.0),
    ]
    pca = fit_pca_2d(vectors)
    coords = project_2d(pca, vectors)

    query_coords = project_2d(pca, [vectors[2]])

    assert len(query_coords) == 1
    assert query_coords[0] == pytest.approx(coords[2])


def test_fit_pca_2d_raises_on_empty_input() -> None:
    with pytest.raises(ValueError):
        fit_pca_2d([])


def test_project_2d_returns_empty_list_for_empty_input() -> None:
    pca = fit_pca_2d([(1.0, 2.0), (3.0, 4.0)])

    assert project_2d(pca, []) == []
