from __future__ import annotations

from uuid import UUID

from app.rag.models import RagCollection
from app.rag.vector_store import _qdrant_point_id, _search_qdrant_points


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
