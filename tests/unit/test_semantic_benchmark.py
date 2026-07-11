"""Deterministic plumbing smoke test for the semantic benchmark harness (docs/22
P0.4), using `SemanticCorpusKeywordEmbeddingProvider` -- a pure vocabulary-overlap
counter with no semantic understanding at all. The numbers this test asserts on
are NOT a measure of retrieval quality; they only prove the harness (corpus
indexing, the production `ActorContextRetriever` path, the raw dense-only path,
and the recall@k/nDCG@k aggregation) runs end-to-end and produces well-formed
output. Real quality measurement is the opt-in `-m semantic` tier in
tests/evals/test_semantic_benchmark.py, run against a real FastEmbed model.
"""

from __future__ import annotations

from app.diagnostics.semantic_benchmark import run_semantic_benchmark
from app.evals.semantic_corpus import SEMANTIC_QUERIES, SemanticCorpusKeywordEmbeddingProvider


def test_semantic_benchmark_runs_end_to_end_with_keyword_provider() -> None:
    report = run_semantic_benchmark(
        embedding_provider=SemanticCorpusKeywordEmbeddingProvider(),
        provider_label="keyword",
    )

    assert report.provider_label == "keyword"
    assert len(report.reranked.per_query) == len(SEMANTIC_QUERIES)
    assert len(report.raw.per_query) == len(SEMANTIC_QUERIES)


def test_aggregate_metrics_are_valid_fractions() -> None:
    report = run_semantic_benchmark(
        embedding_provider=SemanticCorpusKeywordEmbeddingProvider(),
        provider_label="keyword",
    )

    for path_report in (report.reranked, report.raw):
        for aggregate in (path_report.overall, path_report.german):
            assert 0.0 <= aggregate.recall_at_5 <= 1.0
            assert 0.0 <= aggregate.recall_at_10 <= 1.0
            assert 0.0 <= aggregate.recall_at_5_strict <= 1.0
            assert 0.0 <= aggregate.ndcg_at_10 <= 1.0
            assert 0.0 <= aggregate.mrr <= 1.0


def test_german_subset_aggregate_covers_only_german_queries() -> None:
    report = run_semantic_benchmark(
        embedding_provider=SemanticCorpusKeywordEmbeddingProvider(),
        provider_label="keyword",
    )
    german_query_count = sum(1 for query in SEMANTIC_QUERIES if query.language == "de")

    assert report.reranked.german.query_count == german_query_count
    assert report.raw.german.query_count == german_query_count
    assert report.reranked.overall.query_count == len(SEMANTIC_QUERIES)


def test_per_query_metrics_carry_the_query_language() -> None:
    report = run_semantic_benchmark(
        embedding_provider=SemanticCorpusKeywordEmbeddingProvider(),
        provider_label="keyword",
    )
    languages_by_id = {query.id: query.language for query in SEMANTIC_QUERIES}

    for metrics in report.reranked.per_query:
        assert metrics.language == languages_by_id[metrics.query_id]


def test_reranked_path_finds_at_least_some_directly_relevant_chunks() -> None:
    """Even a semantically meaningless keyword count should surface *some*
    directly-relevant chunk in the top 10 for most queries, since every corpus
    chunk's judged-relevant text shares literal vocabulary with its query by
    construction (e.g. "Amber Ring" appears in both). This is a plumbing sanity
    check, not a quality bar -- see the module docstring."""
    report = run_semantic_benchmark(
        embedding_provider=SemanticCorpusKeywordEmbeddingProvider(),
        provider_label="keyword",
    )

    assert report.reranked.overall.recall_at_10 > 0.0
