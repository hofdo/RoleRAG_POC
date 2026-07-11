"""Integrity checks for the graded semantic corpus (docs/22 P0.4).

These are structural/string checks over the corpus and query data itself -- no
embedding provider or retrieval involved -- so a future edit to
`app.evals.semantic_corpus` that breaks an id reference, drops the German subset,
or accidentally makes a "distractor" cluster mutually reinforcing (all facts
judged relevant together, defeating the point of a distractor) fails here rather
than silently degrading the benchmark's discriminative power.
"""

from __future__ import annotations

from app.evals.semantic_corpus import (
    DISTRACTOR_CLUSTERS,
    SEMANTIC_CORPUS_CHUNKS,
    SEMANTIC_PERSONAS,
    SEMANTIC_QUERIES,
    corpus_chunk_ids,
)
from app.rag.models import RagCollection


def test_chunk_ids_are_unique() -> None:
    ids = [chunk.id for chunk in SEMANTIC_CORPUS_CHUNKS]

    assert len(ids) == len(set(ids))


def test_corpus_size_is_roughly_ten_times_the_old_nine_item_pool() -> None:
    # docs/22 P0.4 asks for "~10x bride-for-sarnhold" scale; the concrete floor
    # here pins against the old event_key_retrieval/embedding_ab pool (9 items:
    # 5 seeded events + 2 filler memories + 2 lore chunks) that BACKLOG #10 found
    # too small to discriminate embedding models.
    assert 80 <= len(SEMANTIC_CORPUS_CHUNKS) <= 100


def test_query_ids_are_unique() -> None:
    ids = [query.id for query in SEMANTIC_QUERIES]

    assert len(ids) == len(set(ids))


def test_query_count_is_in_the_docs22_target_range() -> None:
    assert 30 <= len(SEMANTIC_QUERIES) <= 40


def test_every_judgment_references_an_existing_chunk() -> None:
    known_ids = corpus_chunk_ids()

    for query in SEMANTIC_QUERIES:
        unknown = set(query.judgments) - known_ids
        assert not unknown, f"{query.id} judges unknown chunk ids: {sorted(unknown)}"


def test_judgments_are_graded_one_or_two_never_zero() -> None:
    # 0 is implicit (absence from the mapping); an explicit 0 would be redundant
    # and signals a fixture-authoring mistake.
    for query in SEMANTIC_QUERIES:
        for chunk_id, grade in query.judgments.items():
            assert grade in (1, 2), f"{query.id}/{chunk_id} has out-of-range grade {grade}"


def test_every_query_has_at_least_one_directly_relevant_judgment() -> None:
    for query in SEMANTIC_QUERIES:
        assert any(grade == 2 for grade in query.judgments.values()), (
            f"{query.id} has no judgment==2 chunk -- nothing can ever be a perfect answer"
        )


def test_every_query_targets_a_known_persona() -> None:
    for query in SEMANTIC_QUERIES:
        assert query.active_persona_id in SEMANTIC_PERSONAS


def test_chunk_scope_fields_match_their_collection() -> None:
    """canon_lore chunks carry world_id (only); session_memory chunks carry
    session_id (only); persona_memory chunks carry persona_id (only) -- matching
    how `ActorContextRetriever` scopes each collection's search."""
    for chunk in SEMANTIC_CORPUS_CHUNKS:
        if chunk.collection == RagCollection.CANON_LORE:
            assert chunk.world_id is not None
            assert chunk.session_id is None
            assert chunk.persona_id is None
        elif chunk.collection == RagCollection.SESSION_MEMORY:
            assert chunk.session_id is not None
            assert chunk.world_id is None
            assert chunk.persona_id is None
        elif chunk.collection == RagCollection.PERSONA_MEMORY:
            assert chunk.persona_id is not None
            assert chunk.world_id is None
            assert chunk.session_id is None


def test_german_subset_is_non_empty() -> None:
    german_chunks = [chunk for chunk in SEMANTIC_CORPUS_CHUNKS if chunk.language == "de"]
    german_queries = [query for query in SEMANTIC_QUERIES if query.language == "de"]

    assert len(german_chunks) >= 8
    assert 8 <= len(german_queries) <= 12


def test_german_queries_target_at_least_one_german_chunk() -> None:
    german_chunk_ids = {
        chunk.id for chunk in SEMANTIC_CORPUS_CHUNKS if chunk.language == "de"
    }

    for query in SEMANTIC_QUERIES:
        if query.language != "de":
            continue
        assert query.judgments.keys() & german_chunk_ids, (
            f"German query {query.id} judges no German chunk as relevant"
        )


def test_distractor_clusters_share_the_proper_noun_across_different_facts() -> None:
    """Every chunk in a cluster mentions the shared proper noun, but the facts
    themselves are pairwise distinct -- the property that makes them adversarial
    (an embedding matching only the proper noun cannot discriminate between them)."""
    chunk_by_id = {chunk.id: chunk for chunk in SEMANTIC_CORPUS_CHUNKS}

    for proper_noun, chunk_ids in DISTRACTOR_CLUSTERS:
        assert len(chunk_ids) >= 3, f"{proper_noun} cluster has fewer than 3 facts"
        texts = []
        for chunk_id in chunk_ids:
            chunk = chunk_by_id[chunk_id]
            assert proper_noun in chunk.text, (
                f"{chunk_id} does not mention '{proper_noun}': {chunk.text!r}"
            )
            texts.append(chunk.text)
        assert len(texts) == len(set(texts)), f"{proper_noun} cluster has duplicate facts"


def test_distractor_clusters_are_discriminated_by_at_least_one_query_each() -> None:
    """For every distractor-cluster chunk that some query judges directly relevant
    (==2), the *other* chunks in the same cluster must NOT also be judged relevant
    for that query -- otherwise the query cannot tell the facts apart and the
    cluster is not actually adversarial for it."""
    for proper_noun, chunk_ids in DISTRACTOR_CLUSTERS:
        cluster = set(chunk_ids)
        for target_id in chunk_ids:
            targeting_queries = [
                query for query in SEMANTIC_QUERIES if query.judgments.get(target_id) == 2
            ]
            assert targeting_queries, (
                f"no query judges {target_id} ({proper_noun} cluster) as directly relevant"
            )
            for query in targeting_queries:
                siblings_also_relevant = {
                    chunk_id
                    for chunk_id in cluster - {target_id}
                    if query.judgments.get(chunk_id, 0) >= 1
                }
                assert not siblings_also_relevant, (
                    f"{query.id} judges {target_id} AND sibling(s) "
                    f"{sorted(siblings_also_relevant)} in the {proper_noun} cluster relevant "
                    "-- distractors are not discriminated"
                )
