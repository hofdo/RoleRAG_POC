"""Deterministic 2-D PCA used by the RAG debug/vector-map surface (backlog #85).

Pure numpy, no scikit-learn; recomputed per request over the full point set so
layouts are stable and filters never reshuffle them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Pca2D:
    mean: NDArray[np.float64]  # shape (dim,)
    components: NDArray[np.float64]  # shape (2, dim); rows zero-padded when rank < 2
    explained_variance_ratio: tuple[float, float]


def fit_pca_2d(vectors: Sequence[Sequence[float]]) -> Pca2D:
    """Fit a deterministic 2-component PCA over ``vectors``.

    Raises ``ValueError`` on empty input; callers must guard for that case rather
    than receiving a degenerate fit. Otherwise centers on the column mean and
    takes the top two right-singular-vectors via ``numpy.linalg.svd``, zero-padding
    ``components`` to shape (2, dim) when fewer than two are available (a single
    point, or fewer input dimensions than 2). Component signs are fixed
    (svd_flip-equivalent: the largest-magnitude element of each row is forced
    positive) so repeated fits over the same set -- regardless of row order --
    produce identical components and, therefore, identical projected coordinates.
    """
    if len(vectors) == 0:
        raise ValueError("cannot fit PCA on empty input")

    matrix: NDArray[np.float64] = np.asarray(vectors, dtype=np.float64)
    mean: NDArray[np.float64] = matrix.mean(axis=0)
    centered = matrix - mean
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    dim = matrix.shape[1]
    components = np.zeros((2, dim), dtype=np.float64)
    kept = min(2, _effective_rank(singular_values, shape=centered.shape))
    components[:kept, :] = vt[:kept, :]
    components = _flip_signs(components)

    total_variance = float(np.sum(singular_values**2))
    ratios = [0.0, 0.0]
    if total_variance > 0.0:
        for index in range(kept):
            ratios[index] = float(singular_values[index] ** 2 / total_variance)

    return Pca2D(
        mean=mean,
        components=components,
        explained_variance_ratio=(ratios[0], ratios[1]),
    )


def project_2d(pca: Pca2D, vectors: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Project ``vectors`` into the 2-D space defined by ``pca``.

    Works both for the set ``pca`` was fitted on and for new vectors (e.g. a
    query overlay), since the fitted mean/components are reused unchanged.
    """
    if len(vectors) == 0:
        return []

    matrix: NDArray[np.float64] = np.asarray(vectors, dtype=np.float64)
    projected = (matrix - pca.mean) @ pca.components.T
    return [(float(row[0]), float(row[1])) for row in projected]


def _effective_rank(singular_values: NDArray[np.float64], *, shape: tuple[int, int]) -> int:
    """Count singular values distinguishable from zero (same tolerance convention
    as ``numpy.linalg.matrix_rank``), so a rank-deficient fit (a single point, fully
    collinear points, or all-identical points) reports zeroed components/ratios for
    the directions with no real variance instead of an arbitrary orthonormal
    completion vector."""
    if singular_values.size == 0:
        return 0
    tolerance = singular_values.max() * max(shape) * np.finfo(np.float64).eps
    return int(np.sum(singular_values > tolerance))


def _flip_signs(components: NDArray[np.float64]) -> NDArray[np.float64]:
    """Deterministic sign convention (svd_flip-equivalent): for each row, the
    element with the largest absolute value is forced positive."""
    flipped = components.copy()
    for row_index in range(flipped.shape[0]):
        row = flipped[row_index]
        if not np.any(row):
            continue
        max_abs_index = int(np.argmax(np.abs(row)))
        if row[max_abs_index] < 0.0:
            flipped[row_index] = -row
    return flipped
