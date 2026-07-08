"""Paired InMemory↔Qdrant filter-parity harness (#50).

``InMemoryVectorStore`` (tests) and ``QdrantVectorStore`` (live) maintain two hand-written
filter implementations — ``_chunk_matches_filters`` and ``_build_qdrant_filter``. A future
hybrid/sparse retrieval leg that forgets to carry the same ``query_filter`` would break the
visibility boundary (invariant #2) while deterministic tests stayed green. This harness runs
one fixture set through *both* ``.search()`` paths for every filter dimension and asserts the
matched id sets are identical, so any divergence fails the gate.

The Qdrant store runs against an embedded ``QdrantClient(":memory:")`` (local mode, in-process,
deterministic — not a live server), so the real ``_build_qdrant_filter`` matcher is exercised.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient

from app.domain import Visibility
from app.rag.models import RagChunk, RagCollection, RetrievalFilter
from app.rag.vector_store import InMemoryVectorStore, QdrantVectorStore

_COLLECTION = RagCollection.SESSION_MEMORY
_VECTOR_SIZE = 3

# One chunk per interesting combination of dimensions. Vectors are arbitrary but distinct.
_CHUNKS: list[tuple[RagChunk, list[float]]] = [
    (
        RagChunk(
            id="c1",
            source="s",
            source_type="memory",
            text="alpha",
            visibility=Visibility.PLAYER,
            tags=["quest", "promise"],
            world_id="w1",
            scene_id="sc1",
            persona_id="p1",
            session_id="sess1",
        ),
        [1.0, 0.0, 0.0],
    ),
    (
        RagChunk(
            id="c2",
            source="s",
            source_type="memory",
            text="beta",
            visibility=Visibility.PLAYER,
            tags=["quest"],
            world_id="w1",
            scene_id="sc2",
            persona_id="p2",
            session_id="sess1",
        ),
        [0.9, 0.1, 0.0],
    ),
    (
        RagChunk(
            id="c3",
            source="s",
            source_type="memory",
            text="gamma",
            visibility=Visibility.GM,
            tags=["quest", "promise"],
            world_id="w1",
            scene_id="sc1",
            persona_id="p1",
            session_id="sess2",
        ),
        [0.0, 1.0, 0.0],
    ),
    (
        RagChunk(
            id="c4",
            source="s",
            source_type="memory",
            text="delta",
            visibility=Visibility.CHARACTER_PRIVATE,
            tags=[],
            world_id="w2",
            scene_id="sc3",
            persona_id="p3",
            session_id="sess2",
        ),
        [0.0, 0.0, 1.0],
    ),
]

_QUERY_VECTOR = [1.0, 0.0, 0.0]

_FILTERS: dict[str, RetrievalFilter] = {
    "player_only": RetrievalFilter(allowed_visibilities=[Visibility.PLAYER]),
    "gm_and_player": RetrievalFilter(
        allowed_visibilities=[Visibility.PLAYER, Visibility.GM]
    ),
    "by_world": RetrievalFilter(allowed_visibilities=list(Visibility), world_id="w1"),
    "by_scene": RetrievalFilter(allowed_visibilities=list(Visibility), scene_id="sc1"),
    "by_persona": RetrievalFilter(allowed_visibilities=list(Visibility), persona_id="p1"),
    "by_session": RetrievalFilter(allowed_visibilities=list(Visibility), session_id="sess1"),
    "single_tag": RetrievalFilter(allowed_visibilities=list(Visibility), tags=["quest"]),
    "two_tags_and": RetrievalFilter(
        allowed_visibilities=list(Visibility), tags=["quest", "promise"]
    ),
    "tag_absent": RetrievalFilter(allowed_visibilities=list(Visibility), tags=["missing"]),
    "visibility_and_session": RetrievalFilter(
        allowed_visibilities=[Visibility.PLAYER], session_id="sess1"
    ),
    "matches_nothing": RetrievalFilter(
        allowed_visibilities=[Visibility.PLAYER], world_id="w2"
    ),
}


def _ids(store: InMemoryVectorStore | QdrantVectorStore, filters: RetrievalFilter) -> set[str]:
    results = store.search(_COLLECTION, _QUERY_VECTOR, filters, limit=50)
    return {chunk.id for chunk in results}


def _populate(store: InMemoryVectorStore | QdrantVectorStore) -> None:
    store.ensure_collection(_COLLECTION, _VECTOR_SIZE)
    chunks: Sequence[RagChunk] = [chunk for chunk, _ in _CHUNKS]
    vectors: Sequence[list[float]] = [vector for _, vector in _CHUNKS]
    store.upsert_chunks(_COLLECTION, chunks, vectors)


@pytest.fixture
def in_memory_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    _populate(store)
    return store


@pytest.fixture
def qdrant_store() -> QdrantVectorStore:
    store = QdrantVectorStore(url="local")
    store._client = QdrantClient(":memory:")
    _populate(store)
    return store


@pytest.mark.parametrize("filter_name", list(_FILTERS))
def test_filter_parity_across_stores(
    filter_name: str,
    in_memory_store: InMemoryVectorStore,
    qdrant_store: QdrantVectorStore,
) -> None:
    filters = _FILTERS[filter_name]
    assert _ids(in_memory_store, filters) == _ids(qdrant_store, filters)


def test_two_tag_and_semantics_is_conjunctive(
    in_memory_store: InMemoryVectorStore,
    qdrant_store: QdrantVectorStore,
) -> None:
    # Guards the specific divergence this harness was written for: requiring both tags must
    # match only the chunk carrying *both* (c1), not every chunk carrying either (c1, c2, c3).
    both = RetrievalFilter(allowed_visibilities=list(Visibility), tags=["quest", "promise"])
    assert _ids(in_memory_store, both) == {"c1", "c3"}
    assert _ids(qdrant_store, both) == {"c1", "c3"}
