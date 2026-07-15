"""Integration tests for the RAG debug/visualization surface (backlog #85).

Covers ``POST /diagnostics/rag-map`` and ``POST /diagnostics/chunk-texts``:
point coverage, coordinate stability, the redaction policy pin (hidden text
never leaks unless ``include_hidden=True``), the query-point overlay, and the
strict (``extra="forbid"``) request validation.
"""

from __future__ import annotations

import math
from collections.abc import Generator, Sequence
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.api.debug_routes import get_debug_embedding_provider, get_debug_vector_store
from app.domain import Visibility
from app.main import app
from app.rag import InMemoryVectorStore, RagChunk, RagCollection

_CREATED_AT = datetime(2026, 7, 1, tzinfo=UTC)


class _FakeEmbeddingProvider:
    """Deterministic 4-dim embedding: one count per fixed letter."""

    dimension = 4
    _LETTERS = ("a", "b", "c", "d")

    def embed_text(self, text: str) -> list[float]:
        normalized = text.lower()
        return [float(normalized.count(letter)) for letter in self._LETTERS]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


def _chunk(
    *,
    chunk_id: str,
    text: str,
    visibility: Visibility,
    source: str = "lore-pack",
) -> RagChunk:
    return RagChunk(
        id=chunk_id,
        source=source,
        source_type="lore",
        text=text,
        visibility=visibility,
        tags=["seed"],
        world_id="demo_world",
        scene_id="rose-gallery",
        persona_id="archivist",
        session_id="session-1",
        actor_id=None,
        importance=3,
        created_at=_CREATED_AT,
    )


def _seeded_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    canon_chunks = [
        _chunk(chunk_id="canon-player", text="Public canon lore.", visibility=Visibility.PLAYER),
        _chunk(chunk_id="canon-gm", text="GM-only canon secret.", visibility=Visibility.GM),
        _chunk(
            chunk_id="canon-private",
            text="Character-private canon detail.",
            visibility=Visibility.CHARACTER_PRIVATE,
        ),
    ]
    canon_vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    session_chunks = [
        _chunk(
            chunk_id="session-player",
            text="Player-visible session memory.",
            visibility=Visibility.PLAYER,
        ),
        _chunk(
            chunk_id="session-gm",
            text="Hidden GM session memory.",
            visibility=Visibility.GM,
        ),
    ]
    session_vectors = [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]]

    store.ensure_collection(RagCollection.CANON_LORE, vector_size=4)
    store.upsert_chunks(RagCollection.CANON_LORE, canon_chunks, canon_vectors)
    store.ensure_collection(RagCollection.SESSION_MEMORY, vector_size=4)
    store.upsert_chunks(RagCollection.SESSION_MEMORY, session_chunks, session_vectors)
    return store


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    store = _seeded_store()
    app.dependency_overrides[get_debug_vector_store] = lambda: store
    app.dependency_overrides[get_debug_embedding_provider] = lambda: _FakeEmbeddingProvider()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def empty_client() -> Generator[TestClient, None, None]:
    store = InMemoryVectorStore()
    app.dependency_overrides[get_debug_vector_store] = lambda: store
    app.dependency_overrides[get_debug_embedding_provider] = lambda: _FakeEmbeddingProvider()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_rag_map_returns_all_seeded_points_with_stable_coordinates(
    client: TestClient,
) -> None:
    first = client.post("/diagnostics/rag-map", json={})
    second = client.post("/diagnostics/rag-map", json={})

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()

    assert first_payload["point_count"] == 5
    assert len(first_payload["points"]) == 5
    assert first_payload["truncated"] is False
    assert len(first_payload["explained_variance"]) == 2

    for point in first_payload["points"]:
        assert math.isfinite(point["x"])
        assert math.isfinite(point["y"])

    # Coordinates are stable across repeated calls over the same data.
    first_by_id = {point["id"]: (point["x"], point["y"]) for point in first_payload["points"]}
    second_by_id = {point["id"]: (point["x"], point["y"]) for point in second_payload["points"]}
    assert first_by_id == second_by_id


def test_rag_map_redacts_hidden_text_by_default(client: TestClient) -> None:
    response = client.post("/diagnostics/rag-map", json={})
    assert response.status_code == 200
    points = {point["id"]: point for point in response.json()["points"]}

    for chunk_id in ("canon-gm", "canon-private", "session-gm"):
        point = points[chunk_id]
        assert point["visibility"] != "player"
        assert point["text_preview"] is None
        assert point["text_redacted"] is True

    for chunk_id in ("canon-player", "session-player"):
        point = points[chunk_id]
        assert point["visibility"] == "player"
        assert point["text_preview"] is not None
        assert point["text_redacted"] is False


def test_rag_map_include_hidden_reveals_preview(client: TestClient) -> None:
    response = client.post("/diagnostics/rag-map", json={"include_hidden": True})
    assert response.status_code == 200
    points = {point["id"]: point for point in response.json()["points"]}

    for chunk_id in ("canon-gm", "canon-private", "session-gm"):
        point = points[chunk_id]
        assert point["text_preview"] is not None
        assert point["text_redacted"] is False


def test_rag_map_query_text_produces_query_point(client: TestClient) -> None:
    with_query = client.post("/diagnostics/rag-map", json={"query_text": "aabbcc"})
    without_query = client.post("/diagnostics/rag-map", json={})

    assert with_query.status_code == 200
    assert without_query.status_code == 200
    query_point = with_query.json()["query_point"]
    assert isinstance(query_point, list)
    assert len(query_point) == 2
    assert all(isinstance(value, float) for value in query_point)
    assert without_query.json()["query_point"] is None


def test_rag_map_empty_store_returns_empty_response(empty_client: TestClient) -> None:
    response = empty_client.post("/diagnostics/rag-map", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["points"] == []
    assert payload["point_count"] == 0
    assert payload["explained_variance"] == [0.0, 0.0]
    assert payload["query_point"] is None


def test_rag_map_rejects_extra_field(client: TestClient) -> None:
    response = client.post("/diagnostics/rag-map", json={"unexpected": "nope"})
    assert response.status_code == 422


def test_chunk_texts_mixed_ids_preserve_order_and_redact_by_default(
    client: TestClient,
) -> None:
    response = client.post(
        "/diagnostics/chunk-texts",
        json={
            "items": [
                {"collection": "canon_lore", "id": "canon-player"},
                {"collection": "canon_lore", "id": "missing-id"},
                {"collection": "canon_lore", "id": "canon-gm"},
                {"collection": "session_memory", "id": "session-gm"},
            ]
        },
    )
    assert response.status_code == 200
    chunks = response.json()["chunks"]
    assert [item["id"] for item in chunks] == [
        "canon-player",
        "missing-id",
        "canon-gm",
        "session-gm",
    ]

    player_chunk, missing_chunk, gm_chunk, session_gm_chunk = chunks
    assert player_chunk["found"] is True
    assert player_chunk["text"] == "Public canon lore."
    assert player_chunk["text_redacted"] is False

    assert missing_chunk["found"] is False
    assert missing_chunk["text"] is None
    assert missing_chunk["visibility"] is None
    assert missing_chunk["text_redacted"] is False

    assert gm_chunk["found"] is True
    assert gm_chunk["text"] is None
    assert gm_chunk["text_redacted"] is True

    assert session_gm_chunk["found"] is True
    assert session_gm_chunk["text"] is None
    assert session_gm_chunk["text_redacted"] is True


def test_chunk_texts_include_hidden_reveals_full_text(client: TestClient) -> None:
    response = client.post(
        "/diagnostics/chunk-texts",
        json={
            "items": [{"collection": "canon_lore", "id": "canon-gm"}],
            "include_hidden": True,
        },
    )
    assert response.status_code == 200
    chunk = response.json()["chunks"][0]
    assert chunk["found"] is True
    assert chunk["text"] == "GM-only canon secret."
    assert chunk["text_redacted"] is False


def test_chunk_texts_rejects_extra_field(client: TestClient) -> None:
    response = client.post(
        "/diagnostics/chunk-texts",
        json={"items": [{"collection": "canon_lore", "id": "canon-player"}], "unexpected": True},
    )
    assert response.status_code == 422


def test_chunk_texts_rejects_invalid_collection(client: TestClient) -> None:
    response = client.post(
        "/diagnostics/chunk-texts",
        json={"items": [{"collection": "not_a_collection", "id": "canon-player"}]},
    )
    assert response.status_code == 422
