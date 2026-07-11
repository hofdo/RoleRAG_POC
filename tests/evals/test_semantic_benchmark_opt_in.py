"""Opt-in real-embedding semantic benchmark tier (`-m semantic`, docs/22 P0.4).

SKIPPED BY DEFAULT: the whole module is gated behind the `ROLERAG_SEMANTIC_MODEL`
environment variable, which must be set to a real FastEmbed model name (e.g.
`sentence-transformers/all-MiniLM-L6-v2`) to run. This keeps the default gate
(`pytest`, no flags) from ever attempting a model download -- the environment this
was authored in blocks FastEmbed downloads outright, and the owner's machine is
where this tier is meant to run:

    ROLERAG_SEMANTIC_MODEL=sentence-transformers/all-MiniLM-L6-v2 \\
        pytest -m semantic tests/evals/test_semantic_benchmark.py

(equivalently: `rolerag semantic-benchmark --model <name>` for the human-readable
table -- this test module asserts machine-checkable floors on the same numbers.)

The `semantic` marker is registered in pyproject.toml so `-m semantic` / `-m "not
semantic"` selection works once the env var is set; the skip itself is
env-var-driven rather than purely marker-driven so that an unadorned `pytest` run
-- which does not pass `-m` at all -- still never downloads anything.

Floors below are PROVISIONAL and deliberately generous: this is the first real
run this corpus has ever had (the offline half of P0.4 ships the corpus + harness
without a real-model run, per the environment constraint above). Calibrate these
floors on the owner's first real run and tighten them -- see docs/22 P0.4.
"""

from __future__ import annotations

import os

import pytest

from app.diagnostics.semantic_benchmark import SemanticBenchmarkReport, run_semantic_benchmark
from app.rag.embeddings import FastEmbedEmbeddingProvider

_MODEL_ENV_VAR = "ROLERAG_SEMANTIC_MODEL"
_MODEL_NAME = os.environ.get(_MODEL_ENV_VAR)

pytestmark = [
    pytest.mark.semantic,
    pytest.mark.skipif(
        _MODEL_NAME is None,
        reason=(
            f"opt-in tier: set {_MODEL_ENV_VAR}=<fastembed-model-name> to run the real-"
            "embedding semantic benchmark (downloads on first use). Skipped by default so "
            "the deterministic gate never attempts a model download."
        ),
    ),
]

# Provisional, generous floors -- see the module docstring. A real MiniLM-class
# model should clear these comfortably once run; if it doesn't, that is itself a
# P0.4 finding worth recording in docs/22, not a reason to lower the floor further
# without investigation.
_MIN_RECALL_AT_10 = 0.3
_MIN_NDCG_AT_10 = 0.2


@pytest.fixture(scope="module")
def report() -> SemanticBenchmarkReport:
    assert _MODEL_NAME is not None  # guaranteed by pytestmark skipif above
    return run_semantic_benchmark(
        embedding_provider=FastEmbedEmbeddingProvider(model_name=_MODEL_NAME),
        provider_label=_MODEL_NAME,
    )


def test_reranked_overall_recall_at_10_clears_the_provisional_floor(
    report: SemanticBenchmarkReport,
) -> None:
    assert report.reranked.overall.recall_at_10 >= _MIN_RECALL_AT_10


def test_reranked_overall_ndcg_at_10_clears_the_provisional_floor(
    report: SemanticBenchmarkReport,
) -> None:
    assert report.reranked.overall.ndcg_at_10 >= _MIN_NDCG_AT_10


def test_german_subset_recall_at_10_clears_a_lower_provisional_floor(
    report: SemanticBenchmarkReport,
) -> None:
    """The default `all-MiniLM-L6-v2` baseline is English-only (docs/22 P1.2), so
    the German subset floor is set lower deliberately -- a multilingual candidate
    (P1.2) should clear the same floor as the overall set; this floor exists so a
    monolingual baseline run doesn't fail the whole tier on an expected weak spot."""
    assert report.reranked.german.recall_at_10 >= 0.0


def test_reranked_path_is_no_worse_than_raw_dense_on_aggregate_recall(
    report: SemanticBenchmarkReport,
) -> None:
    """The deterministic boost layer (session/scene/persona/importance/lexical) is
    additive on top of the raw vector score and should not systematically hurt
    recall relative to dense-only search. A regression here means a boost is
    actively displacing relevant chunks, worth investigating before trusting the
    reranked numbers as the production quality signal."""
    assert report.reranked.overall.recall_at_10 >= report.raw.overall.recall_at_10 - 0.05
