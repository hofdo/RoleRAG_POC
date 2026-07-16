"""``list_source_chunk_ids`` / ``delete_source_points`` -- paired InMemory/Qdrant coverage.

Both methods are new source-keyed store operations added together: ``list_source_chunk_ids``
backs ``ingest_document``'s content-fingerprint skip (backlog #86), and ``delete_source_points``
backs ``ingest-scenario-lore --prune`` (backlog #87). Same pattern as
tests/unit/test_vector_store_parity.py and tests/unit/test_vector_store_scroll_dump.py: the
Qdrant half runs against an embedded ``QdrantClient(":memory:")`` (local mode, in-process,
deterministic -- not a live server), so the real implementation is exercised, not a stub.
"""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient

from app.domain import Visibility
from app.rag.models import RagChunk, RagCollection
from app.rag.vector_store import InMemoryVectorStore, QdrantVectorStore

_COLLECTION = RagCollection.CANON_LORE
_VECTOR_SIZE = 3

_SOURCE_A = "pack/documents/lore_a.md"
_SOURCE_B = "pack/documents/lore_b.md"

_CHUNKS: list[tuple[RagChunk, list[float]]] = [
    (
        RagChunk(
            id="a1",
            source=_SOURCE_A,
            source_type="lore",
            text="alpha one",
            visibility=Visibility.PLAYER,
        ),
        [1.0, 0.0, 0.0],
    ),
    (
        RagChunk(
            id="a2",
            source=_SOURCE_A,
            source_type="lore",
            text="alpha two",
            visibility=Visibility.PLAYER,
        ),
        [0.9, 0.1, 0.0],
    ),
    (
        RagChunk(
            id="b1",
            source=_SOURCE_B,
            source_type="lore",
            text="beta one",
            visibility=Visibility.GM,
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


# ---------------------------------------------------------------------------
# list_source_chunk_ids
# ---------------------------------------------------------------------------


def test_list_source_chunk_ids_parity_across_stores() -> None:
    in_memory = InMemoryVectorStore()
    _populate(in_memory, model_key=None)
    qdrant = _qdrant_store()
    _populate(qdrant, model_key=None)

    assert in_memory.list_source_chunk_ids(_COLLECTION, _SOURCE_A) == {"a1", "a2"}
    assert qdrant.list_source_chunk_ids(_COLLECTION, _SOURCE_A) == {"a1", "a2"}
    assert in_memory.list_source_chunk_ids(_COLLECTION, _SOURCE_B) == {"b1"}
    assert qdrant.list_source_chunk_ids(_COLLECTION, _SOURCE_B) == {"b1"}


def test_list_source_chunk_ids_unknown_source_is_empty_both_stores() -> None:
    in_memory = InMemoryVectorStore()
    _populate(in_memory, model_key=None)
    qdrant = _qdrant_store()
    _populate(qdrant, model_key=None)

    assert in_memory.list_source_chunk_ids(_COLLECTION, "no/such/source.md") == set()
    assert qdrant.list_source_chunk_ids(_COLLECTION, "no/such/source.md") == set()


def test_list_source_chunk_ids_missing_collection_is_empty_both_stores() -> None:
    assert InMemoryVectorStore().list_source_chunk_ids(_COLLECTION, _SOURCE_A) == set()
    assert _qdrant_store().list_source_chunk_ids(_COLLECTION, _SOURCE_A) == set()


def test_list_source_chunk_ids_never_returns_the_fingerprint_sentinel() -> None:
    # The P1.4 sentinel meta point never carries a "source" field, so it can never match the
    # source filter -- but assert the invariant directly rather than relying on that being
    # incidental: fingerprinting first (writes the sentinel), then confirm neither store's
    # per-source id sets ever surface it, for a source string chosen to coincide with nothing
    # real in the fixture.
    in_memory = InMemoryVectorStore()
    _populate(in_memory, model_key="all-MiniLM-L6-v2")
    qdrant = _qdrant_store()
    _populate(qdrant, model_key="all-MiniLM-L6-v2")

    for store in (in_memory, qdrant):
        for source in (_SOURCE_A, _SOURCE_B, "unrelated"):
            assert "__rolerag_sentinel__" not in store.list_source_chunk_ids(
                _COLLECTION, source
            )


def test_qdrant_list_source_chunk_ids_paginates_past_the_page_size() -> None:
    store = _qdrant_store()
    store.ensure_collection(_COLLECTION, 1)
    total = 300
    source = "pack/documents/big_lore.md"
    chunks = [
        RagChunk(
            id=f"bulk-{index}",
            source=source,
            source_type="lore",
            text=f"chunk {index}",
            visibility=Visibility.PLAYER,
        )
        for index in range(total)
    ]
    vectors = [[float(index % 7)] for index in range(total)]
    store.upsert_chunks(_COLLECTION, chunks, vectors)

    ids = store.list_source_chunk_ids(_COLLECTION, source)

    assert ids == {chunk.id for chunk in chunks}


# ---------------------------------------------------------------------------
# delete_source_points
# ---------------------------------------------------------------------------


def test_delete_source_points_removes_only_the_named_source_both_stores() -> None:
    in_memory = InMemoryVectorStore()
    _populate(in_memory, model_key=None)
    qdrant = _qdrant_store()
    _populate(qdrant, model_key=None)

    in_memory.delete_source_points(_COLLECTION, _SOURCE_A)
    qdrant.delete_source_points(_COLLECTION, _SOURCE_A)

    assert in_memory.list_source_chunk_ids(_COLLECTION, _SOURCE_A) == set()
    assert qdrant.list_source_chunk_ids(_COLLECTION, _SOURCE_A) == set()
    # The untouched source survives in both stores.
    assert in_memory.list_source_chunk_ids(_COLLECTION, _SOURCE_B) == {"b1"}
    assert qdrant.list_source_chunk_ids(_COLLECTION, _SOURCE_B) == {"b1"}
    assert {point.chunk.id for point in in_memory.scroll_points(_COLLECTION)} == {"b1"}
    assert {point.chunk.id for point in qdrant.scroll_points(_COLLECTION)} == {"b1"}


def test_delete_source_points_unknown_source_is_a_no_op_both_stores() -> None:
    in_memory = InMemoryVectorStore()
    _populate(in_memory, model_key=None)
    qdrant = _qdrant_store()
    _populate(qdrant, model_key=None)

    in_memory.delete_source_points(_COLLECTION, "no/such/source.md")
    qdrant.delete_source_points(_COLLECTION, "no/such/source.md")

    assert {point.chunk.id for point in in_memory.scroll_points(_COLLECTION)} == {
        "a1",
        "a2",
        "b1",
    }
    assert {point.chunk.id for point in qdrant.scroll_points(_COLLECTION)} == {"a1", "a2", "b1"}


def test_delete_source_points_missing_collection_does_not_raise_either_store() -> None:
    InMemoryVectorStore().delete_source_points(_COLLECTION, _SOURCE_A)  # must not raise
    _qdrant_store().delete_source_points(_COLLECTION, _SOURCE_A)  # must not raise
