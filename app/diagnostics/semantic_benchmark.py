"""Graded-relevance recall@k / nDCG@k benchmark harness (docs/22 P0.4, offline half).

Sibling to `embedding_ab.py` (same LLM-free, real-embedding-injectable shape) but
scores the ~85-chunk graded corpus in `app.evals.semantic_corpus` instead of the
9-item `SEEDED_EVENTS` pool, using the metric layer in `app.evals.retrieval_metrics`
(recall@k, nDCG@k, MRR) instead of raw ranks.

Two retrieval paths are measured per query, both against the identical indexed
corpus:

- **reranked** -- the production path, `ActorContextRetriever.retrieve_for_actor_with_diagnostics`
  (dual query, per-collection oversampling, the full additive boost rerank in
  `app.rag.ranking`). This is what a real turn actually returns, so docs/22's
  "prefer measuring through ActorContextRetriever so boosts are included" guidance
  makes this the primary number.
- **raw** -- a single-query dense search per collection (`Retriever.retrieve`,
  cosine score only, no dual-query, no boosts), merged and sorted by raw score.
  Cheap to compute from the same indexed store, so it is always included alongside
  `reranked` rather than gated behind a flag: it isolates how much of the
  reranked score comes from the embedding model itself versus the deterministic
  boost layer, which matters when comparing embedding models (P1.2) since the
  boost layer is identical across all of them.

Both paths share the same corpus/query set, so the caller (CLI `semantic-benchmark`,
the `-m semantic` opt-in test tier, and the deterministic keyword-provider smoke
test) gets recall@5, recall@10, strict recall@5 (judgment==2 only), nDCG@10, and
MRR, aggregated overall and over the German query subset separately.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from app.evals.retrieval_metrics import mrr as mrr_metric
from app.evals.retrieval_metrics import ndcg_at_k, recall_at_k
from app.evals.semantic_corpus import (
    SCENE_ID,
    SEMANTIC_CORPUS_CHUNKS,
    SEMANTIC_PERSONAS,
    SEMANTIC_QUERIES,
    SEMANTIC_SCENE,
    SESSION_ID,
    WORLD_ID,
    SemanticCorpusChunk,
    SemanticQuery,
)
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import RagChunk, RagCollection, RetrievalFilter
from app.rag.retriever import ActorContextRetriever, Retriever, build_retrieval_query
from app.rag.vector_store import InMemoryVectorStore

_DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class SemanticQueryMetrics:
    query_id: str
    language: str
    recall_at_5: float
    recall_at_10: float
    recall_at_5_strict: float
    ndcg_at_10: float
    mrr: float


@dataclass(frozen=True)
class SemanticAggregate:
    query_count: int
    recall_at_5: float
    recall_at_10: float
    recall_at_5_strict: float
    ndcg_at_10: float
    mrr: float


@dataclass(frozen=True)
class SemanticPathReport:
    label: str
    per_query: tuple[SemanticQueryMetrics, ...]
    overall: SemanticAggregate
    german: SemanticAggregate


@dataclass(frozen=True)
class SemanticBenchmarkReport:
    provider_label: str
    reranked: SemanticPathReport
    raw: SemanticPathReport


def run_semantic_benchmark(
    *,
    embedding_provider: EmbeddingProvider,
    provider_label: str = "embedding-provider",
    corpus: Sequence[SemanticCorpusChunk] = SEMANTIC_CORPUS_CHUNKS,
    queries: Sequence[SemanticQuery] = SEMANTIC_QUERIES,
    top_k: int = _DEFAULT_TOP_K,
) -> SemanticBenchmarkReport:
    """Index `corpus` under `embedding_provider` and score `queries` on both the
    reranked production path and the raw dense-only path. See the module docstring
    for what each path measures. `top_k` bounds both the `reranked` retrieval call
    and the `raw` per-collection search; recall@5/@10 and nDCG@10 all read off the
    same top-`top_k` ranking, so `top_k` must be >= 10 for the @10 metrics to be
    meaningful (the default, 10, matches that requirement)."""
    vector_store = _index_corpus(embedding_provider=embedding_provider, corpus=corpus)
    inner_retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        default_top_k=top_k,
    )
    actor_retriever = ActorContextRetriever(retriever=inner_retriever)

    reranked_metrics: list[SemanticQueryMetrics] = []
    raw_metrics: list[SemanticQueryMetrics] = []
    for query in queries:
        persona = SEMANTIC_PERSONAS[query.active_persona_id]
        framed_query = build_retrieval_query(
            user_message=query.text,
            scene=SEMANTIC_SCENE,
            persona=persona,
            recent_turns=(),
        )
        reranked_result = actor_retriever.retrieve_for_actor_with_diagnostics(
            query=framed_query,
            lexical_query=query.text,
            world_id=WORLD_ID,
            session_id=SESSION_ID,
            persona_id=query.active_persona_id,
            scene_id=SCENE_ID,
            top_k=top_k,
        )
        reranked_metrics.append(
            _metrics_for_query(query, [chunk.id for chunk in reranked_result.chunks])
        )

        raw_ranked_ids = _raw_dense_ranked_ids(
            retriever=inner_retriever,
            query_text=framed_query,
            persona_id=query.active_persona_id,
            top_k=top_k,
        )
        raw_metrics.append(_metrics_for_query(query, raw_ranked_ids))

    return SemanticBenchmarkReport(
        provider_label=provider_label,
        reranked=_build_path_report(
            label="reranked (production ActorContextRetriever)",
            per_query=reranked_metrics,
        ),
        raw=_build_path_report(
            label="raw dense (single-query, no rerank boosts)",
            per_query=raw_metrics,
        ),
    )


def _index_corpus(
    *,
    embedding_provider: EmbeddingProvider,
    corpus: Sequence[SemanticCorpusChunk],
) -> InMemoryVectorStore:
    vector_store = InMemoryVectorStore()
    by_collection: dict[RagCollection, list[RagChunk]] = defaultdict(list)
    for chunk in corpus:
        by_collection[chunk.collection].append(chunk.to_rag_chunk())
    for collection, chunks in by_collection.items():
        vector_store.ensure_collection(collection, embedding_provider.dimension)
        vectors = embedding_provider.embed_batch([chunk.text for chunk in chunks])
        vector_store.upsert_chunks(collection, chunks, vectors)
    return vector_store


def _raw_dense_ranked_ids(
    *,
    retriever: Retriever,
    query_text: str,
    persona_id: str,
    top_k: int,
) -> list[str]:
    searches = [
        (RagCollection.SESSION_MEMORY, RetrievalFilter.player_visible(session_id=SESSION_ID)),
        (RagCollection.PERSONA_MEMORY, RetrievalFilter.player_visible(persona_id=persona_id)),
        (RagCollection.CANON_LORE, RetrievalFilter.player_visible(world_id=WORLD_ID)),
    ]
    candidates = [
        chunk
        for collection, filters in searches
        for chunk in retriever.retrieve(
            query=query_text, collection=collection, filters=filters, top_k=top_k
        )
    ]
    candidates.sort(key=lambda chunk: chunk.score, reverse=True)
    ranked_ids: list[str] = []
    seen: set[str] = set()
    for chunk in candidates:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        ranked_ids.append(chunk.id)
    return ranked_ids[:top_k]


def _metrics_for_query(query: SemanticQuery, ranked_ids: Sequence[str]) -> SemanticQueryMetrics:
    return SemanticQueryMetrics(
        query_id=query.id,
        language=query.language,
        recall_at_5=recall_at_k(ranked_ids, query.judgments, 5),
        recall_at_10=recall_at_k(ranked_ids, query.judgments, 10),
        recall_at_5_strict=recall_at_k(ranked_ids, query.judgments, 5, relevance_floor=2),
        ndcg_at_10=ndcg_at_k(ranked_ids, query.judgments, 10),
        mrr=mrr_metric(ranked_ids, query.judgments),
    )


def _build_path_report(
    *,
    label: str,
    per_query: Sequence[SemanticQueryMetrics],
) -> SemanticPathReport:
    german = [metrics for metrics in per_query if metrics.language == "de"]
    return SemanticPathReport(
        label=label,
        per_query=tuple(per_query),
        overall=_aggregate(per_query),
        german=_aggregate(german),
    )


def _aggregate(metrics: Sequence[SemanticQueryMetrics]) -> SemanticAggregate:
    count = len(metrics)
    if count == 0:
        return SemanticAggregate(
            query_count=0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            recall_at_5_strict=0.0,
            ndcg_at_10=0.0,
            mrr=0.0,
        )
    return SemanticAggregate(
        query_count=count,
        recall_at_5=sum(m.recall_at_5 for m in metrics) / count,
        recall_at_10=sum(m.recall_at_10 for m in metrics) / count,
        recall_at_5_strict=sum(m.recall_at_5_strict for m in metrics) / count,
        ndcg_at_10=sum(m.ndcg_at_10 for m in metrics) / count,
        mrr=sum(m.mrr for m in metrics) / count,
    )


__all__ = [
    "SemanticAggregate",
    "SemanticBenchmarkReport",
    "SemanticPathReport",
    "SemanticQueryMetrics",
    "run_semantic_benchmark",
]
