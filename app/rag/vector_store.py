from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from app.domain import RetrievedChunk
from app.rag.models import RagChunk, RagCollection, RetrievalFilter

QdrantClientType: Any | None
qdrant_models: Any | None
try:
    from qdrant_client import QdrantClient as QdrantClientType
    from qdrant_client.http import models as qdrant_models
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    QdrantClientType = None
    qdrant_models = None


class VectorStoreDimensionMismatch(ValueError):
    """Raised when a collection already exists with a different vector size."""

    def __init__(self, collection: RagCollection, existing: int, requested: int) -> None:
        self.collection = collection
        self.existing = existing
        self.requested = requested
        super().__init__(
            f"collection {collection.value} already initialized with size {existing}, "
            f"not {requested}"
        )


class VectorStore(Protocol):
    def ensure_collection(self, collection: RagCollection, vector_size: int) -> None: ...

    def replace_source(
        self,
        collection: RagCollection,
        source: str,
        chunks: Sequence[RagChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    def upsert_chunks(
        self,
        collection: RagCollection,
        chunks: Sequence[RagChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    def search(
        self,
        collection: RagCollection,
        vector: Sequence[float],
        filters: RetrievalFilter,
        limit: int,
    ) -> list[RetrievedChunk]: ...

    def delete_session_points(self, collection: RagCollection, session_id: str) -> None: ...

    def delete_points(self, collection: RagCollection, chunk_ids: Sequence[str]) -> None: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._vector_sizes: dict[RagCollection, int] = {}
        self._entries: dict[RagCollection, list[tuple[RagChunk, list[float]]]] = defaultdict(list)

    def ensure_collection(self, collection: RagCollection, vector_size: int) -> None:
        existing = self._vector_sizes.get(collection)
        if existing is not None and existing != vector_size:
            raise VectorStoreDimensionMismatch(collection, existing, vector_size)
        self._vector_sizes[collection] = vector_size

    def drop_collection(self, collection: RagCollection) -> None:
        self._vector_sizes.pop(collection, None)
        self._entries.pop(collection, None)

    def replace_source(
        self,
        collection: RagCollection,
        source: str,
        chunks: Sequence[RagChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        retained = [
            (chunk, vector)
            for chunk, vector in self._entries[collection]
            if chunk.source != source
        ]
        retained.extend(
            (chunk, list(vector))
            for chunk, vector in zip(chunks, vectors, strict=True)
        )
        self._entries[collection] = retained

    def upsert_chunks(
        self,
        collection: RagCollection,
        chunks: Sequence[RagChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        chunk_ids = {chunk.id for chunk in chunks}
        retained = [
            (chunk, vector)
            for chunk, vector in self._entries[collection]
            if chunk.id not in chunk_ids
        ]
        retained.extend(
            (chunk, list(vector))
            for chunk, vector in zip(chunks, vectors, strict=True)
        )
        self._entries[collection] = retained

    def delete_session_points(self, collection: RagCollection, session_id: str) -> None:
        self._entries[collection] = [
            (chunk, vector)
            for chunk, vector in self._entries[collection]
            if chunk.session_id != session_id
        ]

    def delete_points(self, collection: RagCollection, chunk_ids: Sequence[str]) -> None:
        drop = set(chunk_ids)
        if not drop:
            return
        self._entries[collection] = [
            (chunk, vector)
            for chunk, vector in self._entries[collection]
            if chunk.id not in drop
        ]

    def search(
        self,
        collection: RagCollection,
        vector: Sequence[float],
        filters: RetrievalFilter,
        limit: int,
    ) -> list[RetrievedChunk]:
        matches: list[tuple[float, RagChunk]] = []
        for chunk, stored_vector in self._entries[collection]:
            if not _chunk_matches_filters(chunk, filters):
                continue
            score = _cosine_similarity(vector, stored_vector)
            matches.append((score, chunk))

        matches.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                id=chunk.id,
                source=chunk.source,
                source_type=chunk.source_type,
                text=chunk.text,
                score=score,
                visibility=chunk.visibility,
                tags=chunk.tags,
                world_id=chunk.world_id,
                scene_id=chunk.scene_id,
                persona_id=chunk.persona_id,
                session_id=chunk.session_id,
                actor_id=chunk.actor_id,
                importance=chunk.importance,
                created_at=chunk.created_at,
            )
            for score, chunk in matches[:limit]
        ]


class QdrantVectorStore:
    def __init__(self, *, url: str) -> None:
        self._url = url
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if QdrantClientType is None or qdrant_models is None:
            raise ImportError("qdrant-client is required for QdrantVectorStore")
        if self._client is None:
            self._client = QdrantClientType(url=self._url)
        return self._client

    def ensure_collection(self, collection: RagCollection, vector_size: int) -> None:
        models = _require_qdrant_models()
        if self.client.collection_exists(collection_name=collection.value):
            existing = self.client.get_collection(
                collection_name=collection.value
            ).config.params.vectors.size
            if existing != vector_size:
                raise VectorStoreDimensionMismatch(collection, existing, vector_size)
            return
        self.client.create_collection(
            collection_name=collection.value,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def drop_collection(self, collection: RagCollection) -> None:
        if self.client.collection_exists(collection_name=collection.value):
            self.client.delete_collection(collection_name=collection.value)

    def replace_source(
        self,
        collection: RagCollection,
        source: str,
        chunks: Sequence[RagChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        models = _require_qdrant_models()
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        self.client.delete(
            collection_name=collection.value,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source),
                        )
                    ]
                )
            ),
        )

        if not chunks:
            return

        points = [
            models.PointStruct(
                id=_qdrant_point_id(collection, chunk.id),
                vector=list(vector),
                payload=_chunk_to_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=collection.value, points=points)

    def upsert_chunks(
        self,
        collection: RagCollection,
        chunks: Sequence[RagChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        models = _require_qdrant_models()
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        points = [
            models.PointStruct(
                id=_qdrant_point_id(collection, chunk.id),
                vector=list(vector),
                payload=_chunk_to_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=collection.value, points=points)

    def delete_session_points(self, collection: RagCollection, session_id: str) -> None:
        models = _require_qdrant_models()
        if not self.client.collection_exists(collection_name=collection.value):
            return
        self.client.delete(
            collection_name=collection.value,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=session_id),
                        )
                    ]
                )
            ),
        )

    def delete_points(self, collection: RagCollection, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        models = _require_qdrant_models()
        if not self.client.collection_exists(collection_name=collection.value):
            return
        self.client.delete(
            collection_name=collection.value,
            points_selector=models.PointIdsList(
                points=[_qdrant_point_id(collection, chunk_id) for chunk_id in chunk_ids]
            ),
        )

    def search(
        self,
        collection: RagCollection,
        vector: Sequence[float],
        filters: RetrievalFilter,
        limit: int,
    ) -> list[RetrievedChunk]:
        if not self.client.collection_exists(collection_name=collection.value):
            return []
        results = _search_qdrant_points(
            self.client,
            collection_name=collection.value,
            query_vector=list(vector),
            query_filter=_build_qdrant_filter(filters),
            limit=limit,
        )
        return [
            _payload_to_retrieved_chunk(result.payload, score=result.score)
            for result in results
        ]


def _build_qdrant_filter(filters: RetrievalFilter) -> Any:
    models = _require_qdrant_models()
    must: list[Any] = [
        models.FieldCondition(
            key="visibility",
            match=models.MatchAny(
                any=[visibility.value for visibility in filters.allowed_visibilities]
            ),
        )
    ]
    if filters.world_id is not None:
        must.append(
            models.FieldCondition(
                key="world_id",
                match=models.MatchValue(value=filters.world_id),
            )
        )
    if filters.scene_id is not None:
        must.append(
            models.FieldCondition(
                key="scene_id",
                match=models.MatchValue(value=filters.scene_id),
            )
        )
    if filters.persona_id is not None:
        must.append(
            models.FieldCondition(
                key="persona_id",
                match=models.MatchValue(value=filters.persona_id),
            )
        )
    if filters.session_id is not None:
        must.append(
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=filters.session_id),
            )
        )
    # AND semantics: a chunk must carry *every* filter tag, matching
    # ``InMemoryVectorStore``'s ``issubset`` check (#50). One ``MatchValue`` condition per
    # tag; multiple ``must`` entries are ANDed. (``MatchAny`` here would be OR and diverge
    # from the in-memory store — see tests/unit/test_vector_store_parity.py.)
    for tag in filters.tags:
        must.append(
            models.FieldCondition(
                key="tags",
                match=models.MatchValue(value=tag),
            )
        )
    return models.Filter(must=must)


def _require_qdrant_models() -> Any:
    if qdrant_models is None:
        raise ImportError("qdrant-client is required for QdrantVectorStore")
    return qdrant_models


def _chunk_matches_filters(chunk: RagChunk, filters: RetrievalFilter) -> bool:
    if chunk.visibility not in filters.allowed_visibilities:
        return False
    if filters.world_id is not None and chunk.world_id != filters.world_id:
        return False
    if filters.scene_id is not None and chunk.scene_id != filters.scene_id:
        return False
    if filters.persona_id is not None and chunk.persona_id != filters.persona_id:
        return False
    if filters.session_id is not None and chunk.session_id != filters.session_id:
        return False
    if filters.tags and not set(filters.tags).issubset(chunk.tags):
        return False
    return True


def _chunk_to_payload(chunk: RagChunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "source": chunk.source,
        "source_type": chunk.source_type,
        "text": chunk.text,
        "visibility": chunk.visibility.value,
        "tags": chunk.tags,
        "world_id": chunk.world_id,
        "scene_id": chunk.scene_id,
        "persona_id": chunk.persona_id,
        "session_id": chunk.session_id,
        "actor_id": chunk.actor_id,
        "importance": chunk.importance,
        "created_at": chunk.created_at.isoformat() if chunk.created_at is not None else None,
    }


def _payload_to_retrieved_chunk(payload: dict[str, Any] | None, *, score: float) -> RetrievedChunk:
    raw_payload = payload or {}
    return RetrievedChunk(
        id=str(raw_payload["id"]),
        source=str(raw_payload["source"]),
        source_type=str(raw_payload["source_type"]),
        text=str(raw_payload["text"]),
        score=score,
        visibility=raw_payload["visibility"],
        tags=list(raw_payload.get("tags", [])),
        world_id=raw_payload.get("world_id"),
        scene_id=raw_payload.get("scene_id"),
        persona_id=raw_payload.get("persona_id"),
        session_id=raw_payload.get("session_id"),
        actor_id=raw_payload.get("actor_id"),
        importance=raw_payload.get("importance"),
        created_at=_parse_payload_datetime(raw_payload.get("created_at")),
    )


def _parse_payload_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value))


def _qdrant_point_id(collection: RagCollection, chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"rolerag:{collection.value}:{chunk_id}"))


def _search_qdrant_points(
    client: Any,
    *,
    collection_name: str,
    query_vector: list[float],
    query_filter: Any,
    limit: int,
) -> list[Any]:
    if hasattr(client, "search"):
        return list(
            client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        )
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return list(response.points)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")

    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_magnitude = math.sqrt(sum(value * value for value in left))
    right_magnitude = math.sqrt(sum(value * value for value in right))
    if left_magnitude == 0.0 or right_magnitude == 0.0:
        return 0.0
    return numerator / (left_magnitude * right_magnitude)
