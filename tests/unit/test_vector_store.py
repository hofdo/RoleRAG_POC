from __future__ import annotations

from collections import Counter
from uuid import UUID

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.domain import Visibility
from app.rag.models import RagChunk, RagCollection, RetrievalFilter
from app.rag.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorStoreDimensionMismatch,
    VectorStoreModelMismatch,
    _qdrant_point_id,
    _search_qdrant_points,
)

# P2.1 (docs/22): the keyword payload-index fields every search filters on.
_EXPECTED_PAYLOAD_INDEX_FIELDS: set[str] = {
    "visibility",
    "world_id",
    "session_id",
    "persona_id",
    "scene_id",
    "tags",
}


def test_in_memory_delete_points_removes_only_listed_chunks() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(RagCollection.SESSION_MEMORY, 2)
    chunks = [
        RagChunk(
            id=f"m{index}",
            source="memory",
            source_type="session_memory",
            text=f"memory {index}",
            visibility=Visibility.PLAYER,
            session_id="session-1",
        )
        for index in range(3)
    ]
    store.upsert_chunks(
        RagCollection.SESSION_MEMORY,
        chunks,
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )

    store.delete_points(RagCollection.SESSION_MEMORY, ["m1"])

    remaining = store.search(
        RagCollection.SESSION_MEMORY,
        [1.0, 1.0],
        RetrievalFilter.player_visible(session_id="session-1"),
        limit=10,
    )
    assert {chunk.id for chunk in remaining} == {"m0", "m2"}


def test_qdrant_point_id_normalizes_app_chunk_ids_to_stable_uuid() -> None:
    chunk_id = "chunk-a93e5df9b3fd8b25"

    point_id = _qdrant_point_id(RagCollection.CANON_LORE, chunk_id)

    assert str(UUID(point_id)) == point_id
    assert point_id == _qdrant_point_id(RagCollection.CANON_LORE, chunk_id)
    assert point_id != chunk_id


def test_qdrant_point_id_keeps_collections_separate() -> None:
    chunk_id = "same-logical-id"

    assert _qdrant_point_id(
        RagCollection.CANON_LORE,
        chunk_id,
    ) != _qdrant_point_id(
        RagCollection.SESSION_MEMORY,
        chunk_id,
    )


def test_search_qdrant_points_uses_legacy_search_when_available() -> None:
    class LegacyClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def search(self, **kwargs: object) -> list[str]:
            self.calls.append(kwargs)
            return ["legacy-point"]

    client = LegacyClient()

    result = _search_qdrant_points(
        client,
        collection_name="canon_lore",
        query_vector=[1.0, 2.0],
        query_filter="filter",
        limit=3,
    )

    assert result == ["legacy-point"]
    assert client.calls == [
        {
            "collection_name": "canon_lore",
            "query_vector": [1.0, 2.0],
            "query_filter": "filter",
            "limit": 3,
            "with_payload": True,
        }
    ]


def test_search_qdrant_points_uses_query_points_for_current_client() -> None:
    class QueryResponse:
        points = ["query-point"]

    class CurrentClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def query_points(self, **kwargs: object) -> QueryResponse:
            self.calls.append(kwargs)
            return QueryResponse()

    client = CurrentClient()

    result = _search_qdrant_points(
        client,
        collection_name="canon_lore",
        query_vector=[1.0, 2.0],
        query_filter="filter",
        limit=3,
    )

    assert result == ["query-point"]
    assert client.calls == [
        {
            "collection_name": "canon_lore",
            "query": [1.0, 2.0],
            "query_filter": "filter",
            "limit": 3,
            "with_payload": True,
        }
    ]


def test_in_memory_store_deletes_points_for_session() -> None:
    from app.domain import Visibility
    from app.rag.models import RagChunk, RetrievalFilter
    from app.rag.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    store.ensure_collection(RagCollection.SESSION_MEMORY, 2)
    chunks = [
        RagChunk(
            id="memory-1",
            source="memory",
            source_type="memory",
            text="promise one",
            visibility=Visibility.PLAYER,
            session_id="session-1",
        ),
        RagChunk(
            id="memory-2",
            source="memory",
            source_type="memory",
            text="promise two",
            visibility=Visibility.PLAYER,
            session_id="session-2",
        ),
    ]
    store.upsert_chunks(RagCollection.SESSION_MEMORY, chunks, [[1.0, 0.0], [0.0, 1.0]])

    store.delete_session_points(RagCollection.SESSION_MEMORY, "session-1")

    from app.domain import Visibility as _V

    results = store.search(
        RagCollection.SESSION_MEMORY,
        [1.0, 1.0],
        RetrievalFilter(allowed_visibilities=[_V.PLAYER]),
        10,
    )
    assert [chunk.id for chunk in results] == ["memory-2"]


def test_in_memory_ensure_collection_raises_dimension_mismatch() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(RagCollection.SESSION_MEMORY, 2)

    with pytest.raises(VectorStoreDimensionMismatch) as exc_info:
        store.ensure_collection(RagCollection.SESSION_MEMORY, 3)

    message = str(exc_info.value)
    assert "session_memory" in message
    assert "2" in message
    assert "3" in message


def test_in_memory_drop_collection_clears_state() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(RagCollection.CANON_LORE, 2)
    store.ensure_collection(RagCollection.SESSION_MEMORY, 2)
    canon_chunk = RagChunk(
        id="lore-1",
        source="lore",
        source_type="lore",
        text="canon",
        visibility=Visibility.PLAYER,
        world_id="demo_world",
    )
    session_chunk = RagChunk(
        id="memory-1",
        source="memory",
        source_type="session_memory",
        text="memory",
        visibility=Visibility.PLAYER,
        session_id="session-1",
    )
    store.upsert_chunks(RagCollection.CANON_LORE, [canon_chunk], [[1.0, 0.0]])
    store.upsert_chunks(RagCollection.SESSION_MEMORY, [session_chunk], [[0.0, 1.0]])

    store.drop_collection(RagCollection.CANON_LORE)

    assert (
        store.search(
            RagCollection.CANON_LORE,
            [1.0, 0.0],
            RetrievalFilter.player_visible(world_id="demo_world"),
            limit=10,
        )
        == []
    )
    # The dropped collection can be re-initialized with a different size.
    store.ensure_collection(RagCollection.CANON_LORE, 5)
    remaining = store.search(
        RagCollection.SESSION_MEMORY,
        [0.0, 1.0],
        RetrievalFilter.player_visible(session_id="session-1"),
        limit=10,
    )
    assert [chunk.id for chunk in remaining] == ["memory-1"]


def test_in_memory_ensure_collection_without_model_key_never_fingerprints() -> None:
    # Byte-identical legacy behavior: callers that never pass model_key (evals, older
    # callers) get no fingerprint check/adopt at all, no matter how many times called.
    store = InMemoryVectorStore()
    store.ensure_collection(RagCollection.CANON_LORE, 3)
    store.ensure_collection(RagCollection.CANON_LORE, 3)  # would raise if it fingerprinted
    assert store._model_fingerprints == {}


def test_in_memory_ensure_collection_adopts_fingerprint_on_first_fingerprinted_contact() -> None:
    # Backward compatibility: an existing (or brand-new) collection with no fingerprint
    # yet adopts the current model identity instead of failing.
    store = InMemoryVectorStore()
    store.ensure_collection(RagCollection.CANON_LORE, 3)  # unfingerprinted contact first

    store.ensure_collection(RagCollection.CANON_LORE, 3, model_key="model-a")

    assert store._model_fingerprints[RagCollection.CANON_LORE] == "model-a"
    # Once adopted, the same model_key keeps passing.
    store.ensure_collection(RagCollection.CANON_LORE, 3, model_key="model-a")


def test_in_memory_ensure_collection_raises_model_mismatch() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(RagCollection.CANON_LORE, 3, model_key="model-a")

    with pytest.raises(VectorStoreModelMismatch) as exc_info:
        store.ensure_collection(RagCollection.CANON_LORE, 3, model_key="model-b")

    message = str(exc_info.value)
    assert "canon_lore" in message
    assert "model-a" in message
    assert "model-b" in message


def test_in_memory_drop_then_recreate_clears_fingerprint() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(RagCollection.CANON_LORE, 3, model_key="model-a")

    store.drop_collection(RagCollection.CANON_LORE)

    # A stale fingerprint would otherwise brick the embedding-migration runbook.
    store.ensure_collection(RagCollection.CANON_LORE, 3, model_key="model-b")
    assert store._model_fingerprints[RagCollection.CANON_LORE] == "model-b"


class _FakeVectors:
    def __init__(self, size: int) -> None:
        self.size = size


class _FakeParams:
    def __init__(self, size: int) -> None:
        self.vectors = _FakeVectors(size)


class _FakeConfig:
    def __init__(self, size: int) -> None:
        self.params = _FakeParams(size)


class _FakeCollectionInfo:
    def __init__(self, size: int) -> None:
        self.config = _FakeConfig(size)


class _FakeQdrantClient:
    def __init__(self, *, existing_size: int | None) -> None:
        self._existing_size = existing_size
        self.deleted: list[str] = []
        self.created: list[dict[str, object]] = []
        self.payload_indexes: list[dict[str, object]] = []

    def collection_exists(self, *, collection_name: str) -> bool:
        return self._existing_size is not None

    def get_collection(self, *, collection_name: str) -> _FakeCollectionInfo:
        assert self._existing_size is not None  # only called when the collection exists
        return _FakeCollectionInfo(self._existing_size)

    def create_collection(self, **kwargs: object) -> None:
        self.created.append(kwargs)

    def delete_collection(self, *, collection_name: str) -> None:
        self.deleted.append(collection_name)

    def create_payload_index(self, **kwargs: object) -> None:
        self.payload_indexes.append(kwargs)


def test_qdrant_ensure_collection_raises_dimension_mismatch_on_existing_collection() -> None:
    store = QdrantVectorStore(url="http://localhost:6333")
    store._client = _FakeQdrantClient(existing_size=1536)

    with pytest.raises(VectorStoreDimensionMismatch) as exc_info:
        store.ensure_collection(RagCollection.CANON_LORE, 768)

    message = str(exc_info.value)
    assert "canon_lore" in message
    assert "1536" in message
    assert "768" in message


def test_qdrant_drop_collection_calls_delete() -> None:
    store = QdrantVectorStore(url="http://localhost:6333")
    client = _FakeQdrantClient(existing_size=1536)
    store._client = client

    store.drop_collection(RagCollection.SESSION_MEMORY)

    assert client.deleted == ["session_memory"]


# ---------------------------------------------------------------------------
# P2.1: payload indexes + opt-in scalar quantization
# ---------------------------------------------------------------------------


def test_qdrant_ensure_collection_creates_payload_indexes_on_new_collection() -> None:
    store = QdrantVectorStore(url="http://localhost:6333")
    client = _FakeQdrantClient(existing_size=None)
    store._client = client

    store.ensure_collection(RagCollection.CANON_LORE, 3)

    assert len(client.created) == 1  # went through the create-collection path
    field_counts = Counter(call["field_name"] for call in client.payload_indexes)
    assert field_counts == Counter(dict.fromkeys(_EXPECTED_PAYLOAD_INDEX_FIELDS, 1))


def test_qdrant_ensure_collection_indexes_already_existing_collection_on_repeat_contact() -> None:
    # docs/22 P2.1 explicitly calls this out: the early-return path for an already-existing
    # collection must also (re-)create indexes, or a pre-P2.1 collection would never get
    # them. existing_size makes every call here -- including the first -- take the
    # early-return branch, never the create-collection branch.
    store = QdrantVectorStore(url="http://localhost:6333")
    client = _FakeQdrantClient(existing_size=3)
    store._client = client

    store.ensure_collection(RagCollection.CANON_LORE, 3)
    store.ensure_collection(RagCollection.CANON_LORE, 3)  # idempotency: must not raise

    assert client.created == []  # never touched the create-collection path
    field_counts = Counter(call["field_name"] for call in client.payload_indexes)
    assert field_counts == Counter(dict.fromkeys(_EXPECTED_PAYLOAD_INDEX_FIELDS, 2))


def test_qdrant_create_collection_passes_int8_quantization_when_enabled() -> None:
    store = QdrantVectorStore(url="http://localhost:6333", scalar_quantization=True)
    client = _FakeQdrantClient(existing_size=None)
    store._client = client

    store.ensure_collection(RagCollection.CANON_LORE, 3)

    assert len(client.created) == 1
    quantization_config = client.created[0]["quantization_config"]
    assert isinstance(quantization_config, qdrant_models.ScalarQuantization)
    assert quantization_config.scalar.type == qdrant_models.ScalarType.INT8


def test_qdrant_create_collection_omits_quantization_by_default() -> None:
    store = QdrantVectorStore(url="http://localhost:6333")  # scalar_quantization defaults False
    client = _FakeQdrantClient(existing_size=None)
    store._client = client

    store.ensure_collection(RagCollection.CANON_LORE, 3)

    assert len(client.created) == 1
    # Byte-identical to the pre-P2.1 call: no quantization_config key at all, not just None.
    assert "quantization_config" not in client.created[0]


def test_qdrant_payload_index_failure_is_non_fatal_and_store_stays_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = QdrantVectorStore(url="local")
    store._client = QdrantClient(":memory:")

    def _raise(**_kwargs: object) -> None:
        raise RuntimeError("simulated create_payload_index failure")

    monkeypatch.setattr(store._client, "create_payload_index", _raise)

    store.ensure_collection(RagCollection.CANON_LORE, 3)  # must not raise

    chunk = RagChunk(
        id="lore-1",
        source="lore.md",
        source_type="lore",
        text="canon fact",
        visibility=Visibility.PLAYER,
        world_id="w1",
    )
    store.upsert_chunks(RagCollection.CANON_LORE, [chunk], [[1.0, 0.0, 0.0]])

    results = store.search(
        RagCollection.CANON_LORE,
        [1.0, 0.0, 0.0],
        RetrievalFilter.player_visible(world_id="w1"),
        limit=10,
    )
    assert [result.id for result in results] == ["lore-1"]
