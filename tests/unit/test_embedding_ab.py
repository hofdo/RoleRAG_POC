from __future__ import annotations

from app.diagnostics.embedding_ab import measure_embedding_ranks
from app.evals.event_key_retrieval import SEEDED_EVENTS, EventKeywordEmbeddingProvider


def test_measure_embedding_ranks_covers_every_event() -> None:
    ranks = measure_embedding_ranks(embedding_provider=EventKeywordEmbeddingProvider())

    assert set(ranks) == {event.key for event in SEEDED_EVENTS}


def test_measure_embedding_ranks_returns_int_or_none() -> None:
    ranks = measure_embedding_ranks(embedding_provider=EventKeywordEmbeddingProvider())

    for value in ranks.values():
        assert value is None or isinstance(value, int)


def test_measure_embedding_ranks_finds_every_seeded_memory() -> None:
    # With the deterministic keyword embedding every durable memory is present in
    # the candidate set, so no event should miss (rank None).
    ranks = measure_embedding_ranks(embedding_provider=EventKeywordEmbeddingProvider())

    assert all(rank is not None for rank in ranks.values())
