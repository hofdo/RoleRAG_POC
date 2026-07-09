"""Embedding-model identity fingerprint (P1.4) -- paired InMemory/Qdrant coverage.

Both ``VectorStore`` implementations store the embedding-model identity *inside* the
vector store so fingerprint and collection lifecycles are atomic: a sentinel meta point per
Qdrant collection, a dict in ``InMemoryVectorStore`` (parity). ``ensure_collection`` gains a
model-identity check (``VectorStoreModelMismatch``) that is a strict opt-in -- a caller that
never passes ``model_key`` gets byte-identical pre-P1.4 behavior.

The Qdrant half runs against an embedded ``QdrantClient(":memory:")`` (local mode, in-process,
deterministic -- not a live server, same pattern as tests/unit/test_vector_store_parity.py) so
the real sentinel/filter code path is exercised, including the hard requirement that the
sentinel point never surfaces in ``.search()`` results.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from app.domain import Visibility
from app.rag.models import RagChunk, RagCollection, RetrievalFilter
from app.rag.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorStoreModelMismatch,
)

_COLLECTION = RagCollection.CANON_LORE
_VECTOR_SIZE = 3


def _qdrant_store() -> QdrantVectorStore:
    store = QdrantVectorStore(url="local")
    store._client = QdrantClient(":memory:")
    return store


# ---------------------------------------------------------------------------
# Mismatch raises
# ---------------------------------------------------------------------------


def test_in_memory_model_mismatch_raises() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(_COLLECTION, _VECTOR_SIZE, model_key="all-MiniLM-L6-v2")

    with pytest.raises(VectorStoreModelMismatch) as exc_info:
        store.ensure_collection(
            _COLLECTION, _VECTOR_SIZE, model_key="paraphrase-multilingual-MiniLM-L12-v2"
        )

    message = str(exc_info.value)
    assert "canon_lore" in message
    assert "all-MiniLM-L6-v2" in message
    assert "paraphrase-multilingual-MiniLM-L12-v2" in message
    assert "docs/22_rag_scaling_roadmap.md" in message  # points at the migration runbook


def test_qdrant_model_mismatch_raises() -> None:
    store = _qdrant_store()
    store.ensure_collection(_COLLECTION, _VECTOR_SIZE, model_key="all-MiniLM-L6-v2")

    with pytest.raises(VectorStoreModelMismatch) as exc_info:
        store.ensure_collection(
            _COLLECTION, _VECTOR_SIZE, model_key="paraphrase-multilingual-MiniLM-L12-v2"
        )

    message = str(exc_info.value)
    assert "canon_lore" in message
    assert "all-MiniLM-L6-v2" in message
    assert "paraphrase-multilingual-MiniLM-L12-v2" in message
    assert "docs/22_rag_scaling_roadmap.md" in message


# ---------------------------------------------------------------------------
# Drop-then-recreate clears the fingerprint
# ---------------------------------------------------------------------------


def test_in_memory_drop_then_recreate_clears_fingerprint() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(_COLLECTION, _VECTOR_SIZE, model_key="all-MiniLM-L6-v2")

    store.drop_collection(_COLLECTION)

    # A stale fingerprint would otherwise brick the embedding-migration runbook: dropping
    # the collection must let a genuinely different model be adopted cleanly.
    store.ensure_collection(
        _COLLECTION, _VECTOR_SIZE, model_key="paraphrase-multilingual-MiniLM-L12-v2"
    )
    store.ensure_collection(
        _COLLECTION, _VECTOR_SIZE, model_key="paraphrase-multilingual-MiniLM-L12-v2"
    )  # still consistent, no raise


def test_qdrant_drop_then_recreate_clears_fingerprint() -> None:
    store = _qdrant_store()
    store.ensure_collection(_COLLECTION, _VECTOR_SIZE, model_key="all-MiniLM-L6-v2")

    store.drop_collection(_COLLECTION)

    store.ensure_collection(
        _COLLECTION, _VECTOR_SIZE, model_key="paraphrase-multilingual-MiniLM-L12-v2"
    )
    store.ensure_collection(
        _COLLECTION, _VECTOR_SIZE, model_key="paraphrase-multilingual-MiniLM-L12-v2"
    )
    assert (
        store.read_model_fingerprint(_COLLECTION) == "paraphrase-multilingual-MiniLM-L12-v2"
    )


# ---------------------------------------------------------------------------
# Fingerprint adoption on unfingerprinted collections (backward compatibility)
# ---------------------------------------------------------------------------


def test_in_memory_adopts_fingerprint_on_unfingerprinted_collection() -> None:
    store = InMemoryVectorStore()
    # Simulates a pre-P1.4 install / a caller that never passed model_key.
    store.ensure_collection(_COLLECTION, _VECTOR_SIZE)

    # First fingerprinted contact must adopt, not raise.
    store.ensure_collection(_COLLECTION, _VECTOR_SIZE, model_key="all-MiniLM-L6-v2")

    with pytest.raises(VectorStoreModelMismatch):
        store.ensure_collection(_COLLECTION, _VECTOR_SIZE, model_key="a-different-model")


def test_qdrant_adopts_fingerprint_on_unfingerprinted_collection() -> None:
    store = _qdrant_store()
    # Simulates a pre-P1.4 install: the collection exists with vectors already, no sentinel.
    store.ensure_collection(_COLLECTION, _VECTOR_SIZE)
    assert store.read_model_fingerprint(_COLLECTION) is None

    store.ensure_collection(_COLLECTION, _VECTOR_SIZE, model_key="all-MiniLM-L6-v2")

    assert store.read_model_fingerprint(_COLLECTION) == "all-MiniLM-L6-v2"
    with pytest.raises(VectorStoreModelMismatch):
        store.ensure_collection(_COLLECTION, _VECTOR_SIZE, model_key="a-different-model")


def test_qdrant_read_model_fingerprint_is_read_only_and_does_not_adopt() -> None:
    store = _qdrant_store()
    store.ensure_collection(_COLLECTION, _VECTOR_SIZE)  # unfingerprinted

    # Reading must never write a fingerprint as a side effect (doctor must stay read-only).
    assert store.read_model_fingerprint(_COLLECTION) is None
    assert store.read_model_fingerprint(_COLLECTION) is None

    # A later real fingerprinted contact still adopts cleanly -- proves the reads above
    # never wrote a sentinel that would otherwise collide/short-circuit adoption.
    store.ensure_collection(_COLLECTION, _VECTOR_SIZE, model_key="all-MiniLM-L6-v2")
    assert store.read_model_fingerprint(_COLLECTION) == "all-MiniLM-L6-v2"


def test_qdrant_read_model_fingerprint_missing_collection_returns_none() -> None:
    store = _qdrant_store()
    assert store.read_model_fingerprint(_COLLECTION) is None


# ---------------------------------------------------------------------------
# The sentinel must never surface in search results
# ---------------------------------------------------------------------------


def test_qdrant_sentinel_never_appears_in_search_results() -> None:
    store = _qdrant_store()
    store.ensure_collection(_COLLECTION, _VECTOR_SIZE, model_key="all-MiniLM-L6-v2")
    chunk = RagChunk(
        id="lore-1",
        source="lore.md",
        source_type="lore",
        text="canon fact",
        visibility=Visibility.PLAYER,
        world_id="w1",
    )
    store.upsert_chunks(_COLLECTION, [chunk], [[1.0, 0.0, 0.0]])

    # Broad, unscoped query over every visibility -- the sentinel would surface here first
    # if it were not excluded, since its zero vector still participates in ANN search.
    results = store.search(
        _COLLECTION,
        [1.0, 0.0, 0.0],
        RetrievalFilter(allowed_visibilities=list(Visibility)),
        limit=50,
    )

    assert [result.id for result in results] == ["lore-1"]


def test_qdrant_sentinel_never_appears_across_repeated_ensure_collection_calls() -> None:
    # Multiple ensure_collection calls (as real callers do every ingest/index) must not
    # multiply sentinel points or ever surface one.
    store = _qdrant_store()
    for _ in range(3):
        store.ensure_collection(_COLLECTION, _VECTOR_SIZE, model_key="all-MiniLM-L6-v2")

    results = store.search(
        _COLLECTION,
        [0.0, 0.0, 0.0],
        RetrievalFilter(allowed_visibilities=list(Visibility)),
        limit=50,
    )

    assert results == []
