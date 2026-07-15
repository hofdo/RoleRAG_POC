"""``scroll_points`` / ``get_chunks`` -- paired InMemory/Qdrant coverage.

These are read/dump methods used by debug and visualization surfaces (not the turn
pipeline): ``scroll_points`` returns every real point (chunk + vector) in a collection,
excluding the P1.4 embedding-model fingerprint sentinel; ``get_chunks`` fetches chunks by
id, silently dropping ids that aren't present. Both stores must agree on the same fixture
set, same pattern as tests/unit/test_vector_store_parity.py and
tests/unit/test_vector_store_fingerprint.py: the Qdrant half runs against an embedded
``QdrantClient(":memory:")`` (local mode, in-process, deterministic -- not a live server).
"""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient

from app.domain import Visibility
from app.rag.models import RagChunk, RagCollection
from app.rag.vector_store import InMemoryVectorStore, QdrantVectorStore, StoredPoint

_COLLECTION = RagCollection.SESSION_MEMORY
_VECTOR_SIZE = 3

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
            visibility=Visibility.GM,
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
            visibility=Visibility.CHARACTER_PRIVATE,
            tags=[],
            world_id="w2",
            scene_id="sc3",
            persona_id="p3",
            session_id="sess2",
        ),
        [0.0, 1.0, 0.0],
    ),
]


def _populate(store: InMemoryVectorStore | QdrantVectorStore, *, model_key: str | None) -> None:
    store.ensure_collection(_COLLECTION, _VECTOR_SIZE, model_key=model_key)
    chunks: Sequence[RagChunk] = [chunk for chunk, _ in _CHUNKS]
    vectors: Sequence[list[float]] = [vector for _, vector in _CHUNKS]
    store.upsert_chunks(_COLLECTION, chunks, vectors)


def _qdrant_store() -> QdrantVectorStore:
    store = QdrantVectorStore(url="local")
    store._client = QdrantClient(":memory:")
    return store


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    magnitude = sum(component * component for component in vector) ** 0.5
    if magnitude == 0.0:
        return tuple(vector)
    return tuple(component / magnitude for component in vector)


def _summarize(points: list[StoredPoint]) -> dict[str, tuple[tuple[float, ...], str, str]]:
    # (normalized+rounded vector, source_type, visibility) is enough to compare the two
    # stores' payload round-trips without depending on field ordering. Normalizing first
    # is required for the Qdrant side: a COSINE-distance collection stores/returns
    # L2-normalized vectors, not the raw ones passed to upsert_chunks, so a byte-exact
    # comparison against InMemoryVectorStore's raw vectors would fail even though both
    # stores agree on the chunk's direction.
    return {
        point.chunk.id: (
            tuple(round(component, 6) for component in _normalize(point.vector)),
            point.chunk.source_type,
            point.chunk.visibility.value,
        )
        for point in points
    }


def test_scroll_points_parity_across_stores_excludes_sentinel() -> None:
    in_memory = InMemoryVectorStore()
    _populate(in_memory, model_key="all-MiniLM-L6-v2")

    qdrant = _qdrant_store()
    _populate(qdrant, model_key="all-MiniLM-L6-v2")

    in_memory_points = in_memory.scroll_points(_COLLECTION)
    qdrant_points = qdrant.scroll_points(_COLLECTION)

    assert _summarize(in_memory_points) == _summarize(qdrant_points)
    assert {point.chunk.id for point in in_memory_points} == {"c1", "c2", "c3"}
    assert {point.chunk.id for point in qdrant_points} == {"c1", "c2", "c3"}


def test_qdrant_scroll_points_paginates_past_the_page_size() -> None:
    store = _qdrant_store()
    store.ensure_collection(_COLLECTION, 1)
    total = 300
    chunks = [
        RagChunk(
            id=f"bulk-{index}",
            source="s",
            source_type="memory",
            text=f"chunk {index}",
            visibility=Visibility.PLAYER,
            session_id="sess1",
        )
        for index in range(total)
    ]
    vectors = [[float(index % 7)] for index in range(total)]
    store.upsert_chunks(_COLLECTION, chunks, vectors)

    points = store.scroll_points(_COLLECTION)

    assert len(points) == total
    assert {point.chunk.id for point in points} == {chunk.id for chunk in chunks}


def test_scroll_points_empty_on_missing_collection_both_stores() -> None:
    assert InMemoryVectorStore().scroll_points(_COLLECTION) == []
    assert _qdrant_store().scroll_points(_COLLECTION) == []


def test_get_chunks_parity_preserves_order_and_drops_missing_ids() -> None:
    in_memory = InMemoryVectorStore()
    _populate(in_memory, model_key=None)

    qdrant = _qdrant_store()
    _populate(qdrant, model_key=None)

    requested = ["c3", "missing-1", "c1", "missing-2", "c2"]

    in_memory_chunks = in_memory.get_chunks(_COLLECTION, requested)
    qdrant_chunks = qdrant.get_chunks(_COLLECTION, requested)

    assert [chunk.id for chunk in in_memory_chunks] == ["c3", "c1", "c2"]
    assert [chunk.id for chunk in qdrant_chunks] == ["c3", "c1", "c2"]


def test_get_chunks_empty_on_missing_collection_both_stores() -> None:
    assert InMemoryVectorStore().get_chunks(_COLLECTION, ["c1"]) == []
    assert _qdrant_store().get_chunks(_COLLECTION, ["c1"]) == []


def test_get_chunks_returns_empty_list_for_empty_id_sequence() -> None:
    in_memory = InMemoryVectorStore()
    _populate(in_memory, model_key=None)
    qdrant = _qdrant_store()
    _populate(qdrant, model_key=None)

    assert in_memory.get_chunks(_COLLECTION, []) == []
    assert qdrant.get_chunks(_COLLECTION, []) == []
