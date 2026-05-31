from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Protocol

try:
    from qdrant_client import QdrantClient  # type: ignore[import-not-found]
    from qdrant_client.http import models as qdrant_models  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    QdrantClient = None
    qdrant_models = None

from app.domain import RetrievedChunk
from app.rag.models import RagChunk, RagCollection, RetrievalFilter


class VectorStore(Protocol):
    def ensure_collection(self, collection: RagCollection, vector_size: int) -> None: ...

    def replace_source(
        self,
        collection: RagCollection,
        source: str,
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


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._vector_sizes: dict[RagCollection, int] = {}
        self._entries: dict[RagCollection, list[tuple[RagChunk, list[float]]]] = defaultdict(list)

    def ensure_collection(self, collection: RagCollection, vector_size: int) -> None:
        existing = self._vector_sizes.get(collection)
        if existing is not None and existing != vector_size:
            raise ValueError(
                "collection "
                f"{collection.value} already initialized with size {existing}, "
                f"not {vector_size}"
            )
        self._vector_sizes[collection] = vector_size

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
            )
            for score, chunk in matches[:limit]
        ]


class QdrantVectorStore:
    def __init__(self, *, url: str) -> None:
        if QdrantClient is None or qdrant_models is None:
            raise ImportError("qdrant-client is required for QdrantVectorStore")
        self.client = QdrantClient(url=url)

    def ensure_collection(self, collection: RagCollection, vector_size: int) -> None:
        if not self.client.collection_exists(collection_name=collection.value):
            self.client.create_collection(
                collection_name=collection.value,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )

    def replace_source(
        self,
        collection: RagCollection,
        source: str,
        chunks: Sequence[RagChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        self.client.delete(
            collection_name=collection.value,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="source",
                            match=qdrant_models.MatchValue(value=source),
                        )
                    ]
                )
            ),
        )

        if not chunks:
            return

        points = [
            qdrant_models.PointStruct(
                id=chunk.id,
                vector=list(vector),
                payload=_chunk_to_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=collection.value, points=points)

    def search(
        self,
        collection: RagCollection,
        vector: Sequence[float],
        filters: RetrievalFilter,
        limit: int,
    ) -> list[RetrievedChunk]:
        results = self.client.search(
            collection_name=collection.value,
            query_vector=list(vector),
            query_filter=_build_qdrant_filter(filters),
            limit=limit,
            with_payload=True,
        )
        return [
            _payload_to_retrieved_chunk(result.payload, score=result.score)
            for result in results
        ]


def _build_qdrant_filter(filters: RetrievalFilter) -> Any:
    assert qdrant_models is not None
    must: list[Any] = [
        qdrant_models.FieldCondition(
            key="visibility",
            match=qdrant_models.MatchAny(
                any=[visibility.value for visibility in filters.allowed_visibilities]
            ),
        )
    ]
    if filters.world_id is not None:
        must.append(
            qdrant_models.FieldCondition(
                key="world_id",
                match=qdrant_models.MatchValue(value=filters.world_id),
            )
        )
    if filters.scene_id is not None:
        must.append(
            qdrant_models.FieldCondition(
                key="scene_id",
                match=qdrant_models.MatchValue(value=filters.scene_id),
            )
        )
    if filters.persona_id is not None:
        must.append(
            qdrant_models.FieldCondition(
                key="persona_id",
                match=qdrant_models.MatchValue(value=filters.persona_id),
            )
        )
    if filters.session_id is not None:
        must.append(
            qdrant_models.FieldCondition(
                key="session_id",
                match=qdrant_models.MatchValue(value=filters.session_id),
            )
        )
    if filters.tags:
        must.append(
            qdrant_models.FieldCondition(
                key="tags",
                match=qdrant_models.MatchAny(any=filters.tags),
            )
        )
    return qdrant_models.Filter(must=must)


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
    )


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")

    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_magnitude = math.sqrt(sum(value * value for value in left))
    right_magnitude = math.sqrt(sum(value * value for value in right))
    if left_magnitude == 0.0 or right_magnitude == 0.0:
        return 0.0
    return numerator / (left_magnitude * right_magnitude)
