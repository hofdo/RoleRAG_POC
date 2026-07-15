from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
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

# Embedding-model identity fingerprint (P1.4): a reserved point id/payload marker for the
# per-collection sentinel meta point. Fixed and namespaced away from real chunk ids (which
# route through _qdrant_point_id / uuid5) so it can never collide with a chunk. Every query
# path (_build_qdrant_filter) excludes _SENTINEL_PAYLOAD_KEY so the sentinel never surfaces
# in search results.
_FINGERPRINT_SENTINEL_ID = str(uuid5(NAMESPACE_URL, "rolerag:__embedding_model_fingerprint__"))
_SENTINEL_PAYLOAD_KEY = "__rolerag_sentinel__"
_FINGERPRINT_PAYLOAD_KEY = "embedding_model"


@dataclass(frozen=True)
class StoredPoint:
    """A point read back from a vector store, for debug/visualization surfaces."""

    chunk: RagChunk
    vector: list[float]


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


class VectorStoreModelMismatch(ValueError):
    """Raised when a collection's stored embedding-model fingerprint differs from the
    model identity the caller is about to write with.

    Same-dimension model swaps (e.g. ``all-MiniLM-L6-v2`` -> a same-384-dim multilingual
    model, P1.2) pass the existing size check silently and mix incompatible vector spaces.
    See docs/22_rag_scaling_roadmap.md#p12-embedding-model-upgrade-path-multilingual for the
    migration runbook (reset-index -> reindex-memories / re-ingest) this error points to.
    """

    RUNBOOK_HINT = (
        "Run the embedding-migration runbook before switching EMBEDDING_MODEL: "
        "docs/22_rag_scaling_roadmap.md#p12-embedding-model-upgrade-path-multilingual "
        "(reset-index the affected collection(s), then reindex-memories / re-ingest)."
    )

    def __init__(self, collection: RagCollection, existing: str, requested: str) -> None:
        self.collection = collection
        self.existing = existing
        self.requested = requested
        super().__init__(
            f"collection {collection.value} was indexed with embedding model {existing!r}, "
            f"not {requested!r}. {self.RUNBOOK_HINT}"
        )


class VectorStore(Protocol):
    def ensure_collection(
        self, collection: RagCollection, vector_size: int, model_key: str | None = None
    ) -> None: ...

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

    def scroll_points(self, collection: RagCollection) -> list[StoredPoint]:
        """Return every real point with its vector, excluding the internal sentinel meta point."""
        ...

    def get_chunks(self, collection: RagCollection, chunk_ids: Sequence[str]) -> list[RagChunk]:
        """Fetch chunks by chunk id. Missing ids are silently absent from the result."""
        ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._vector_sizes: dict[RagCollection, int] = {}
        self._entries: dict[RagCollection, list[tuple[RagChunk, list[float]]]] = defaultdict(list)
        # Embedding-model identity fingerprint (P1.4), parity with QdrantVectorStore's
        # sentinel meta point. Absent key == unfingerprinted collection (pre-P1.4 or a
        # caller that never passed model_key) -- adopted on first fingerprinted contact,
        # never invented from a bare ensure_collection(..., model_key=None) call.
        self._model_fingerprints: dict[RagCollection, str] = {}

    def ensure_collection(
        self, collection: RagCollection, vector_size: int, model_key: str | None = None
    ) -> None:
        existing = self._vector_sizes.get(collection)
        if existing is not None and existing != vector_size:
            raise VectorStoreDimensionMismatch(collection, existing, vector_size)
        self._vector_sizes[collection] = vector_size
        if model_key is None:
            return
        existing_fingerprint = self._model_fingerprints.get(collection)
        if existing_fingerprint is None:
            # Backward compatibility: adopt the current model identity on first
            # fingerprinted contact with an unfingerprinted (or brand-new) collection.
            self._model_fingerprints[collection] = model_key
        elif existing_fingerprint != model_key:
            raise VectorStoreModelMismatch(collection, existing_fingerprint, model_key)

    def drop_collection(self, collection: RagCollection) -> None:
        self._vector_sizes.pop(collection, None)
        self._entries.pop(collection, None)
        self._model_fingerprints.pop(collection, None)

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

    def scroll_points(self, collection: RagCollection) -> list[StoredPoint]:
        # No sentinel to filter here: unlike QdrantVectorStore (which stores the P1.4
        # fingerprint as an in-collection sentinel point), this store keeps the model
        # fingerprint in a separate dict (_model_fingerprints), so every entry in
        # self._entries is a real chunk.
        return [
            StoredPoint(chunk=chunk, vector=list(vector))
            for chunk, vector in self._entries[collection]
        ]

    def get_chunks(self, collection: RagCollection, chunk_ids: Sequence[str]) -> list[RagChunk]:
        by_id = {chunk.id: chunk for chunk, _ in self._entries[collection]}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]


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

    def ensure_collection(
        self, collection: RagCollection, vector_size: int, model_key: str | None = None
    ) -> None:
        models = _require_qdrant_models()
        if self.client.collection_exists(collection_name=collection.value):
            existing = self.client.get_collection(
                collection_name=collection.value
            ).config.params.vectors.size
            if existing != vector_size:
                raise VectorStoreDimensionMismatch(collection, existing, vector_size)
            self._check_or_adopt_fingerprint(collection, vector_size, model_key)
            return
        self.client.create_collection(
            collection_name=collection.value,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        self._check_or_adopt_fingerprint(collection, vector_size, model_key)

    def drop_collection(self, collection: RagCollection) -> None:
        if self.client.collection_exists(collection_name=collection.value):
            self.client.delete_collection(collection_name=collection.value)

    def read_model_fingerprint(self, collection: RagCollection) -> str | None:
        """Read-only lookup of a collection's embedding-model fingerprint, if any.

        Never writes/adopts -- used by ``doctor --check-qdrant`` to surface a mismatch
        loudly without mutating state as a side effect of an inspection command. Returns
        ``None`` for a missing collection or an unfingerprinted (pre-P1.4) collection.
        """
        if not self.client.collection_exists(collection_name=collection.value):
            return None
        return self._read_fingerprint(collection)

    def _check_or_adopt_fingerprint(
        self, collection: RagCollection, vector_size: int, model_key: str | None
    ) -> None:
        """Store/verify the embedding-model identity via a sentinel meta point (P1.4).

        Fingerprint and collection lifecycles are atomic: the sentinel lives inside the
        same Qdrant collection it fingerprints, so ``drop_collection`` clears it for free
        and there is no separate store to go stale. A collection with no sentinel yet
        (pre-P1.4 install, or a caller that passes ``model_key=None``) is left alone --
        the fingerprint is adopted lazily on the first call that *does* pass a model_key,
        so existing installs keep working byte-identically.
        """
        if model_key is None:
            return
        existing = self._read_fingerprint(collection)
        if existing is None:
            self._write_fingerprint(collection, vector_size, model_key)
        elif existing != model_key:
            raise VectorStoreModelMismatch(collection, existing, model_key)

    def _read_fingerprint(self, collection: RagCollection) -> str | None:
        records = self.client.retrieve(
            collection_name=collection.value,
            ids=[_FINGERPRINT_SENTINEL_ID],
            with_payload=True,
        )
        if not records:
            return None
        payload = records[0].payload or {}
        model_key = payload.get(_FINGERPRINT_PAYLOAD_KEY)
        return str(model_key) if model_key is not None else None

    def _write_fingerprint(
        self, collection: RagCollection, vector_size: int, model_key: str
    ) -> None:
        models = _require_qdrant_models()
        self.client.upsert(
            collection_name=collection.value,
            points=[
                models.PointStruct(
                    id=_FINGERPRINT_SENTINEL_ID,
                    vector=[0.0] * vector_size,
                    payload={
                        _SENTINEL_PAYLOAD_KEY: True,
                        _FINGERPRINT_PAYLOAD_KEY: model_key,
                    },
                )
            ],
        )

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

    def scroll_points(self, collection: RagCollection) -> list[StoredPoint]:
        if not self.client.collection_exists(collection_name=collection.value):
            return []
        models = _require_qdrant_models()
        scroll_filter = models.Filter(
            must_not=[
                models.FieldCondition(
                    key=_SENTINEL_PAYLOAD_KEY,
                    match=models.MatchValue(value=True),
                )
            ]
        )
        points: list[StoredPoint] = []
        offset: Any = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=collection.value,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
                scroll_filter=scroll_filter,
            )
            for record in records:
                payload = record.payload or {}
                # Defensive: the filter above already excludes the sentinel; skip it again
                # in case a payload carrying the sentinel key ever slips through.
                if payload.get(_SENTINEL_PAYLOAD_KEY):
                    continue
                # Defensive: only accept an unnamed single-vector config's plain vector
                # list; skip records with named vectors or no vector.
                if not isinstance(record.vector, list):
                    continue
                points.append(
                    StoredPoint(
                        chunk=_payload_to_rag_chunk(payload),
                        vector=list(record.vector),
                    )
                )
            if next_offset is None:
                break
            offset = next_offset
        return points

    def get_chunks(self, collection: RagCollection, chunk_ids: Sequence[str]) -> list[RagChunk]:
        if not chunk_ids:
            return []
        if not self.client.collection_exists(collection_name=collection.value):
            return []
        records = self.client.retrieve(
            collection_name=collection.value,
            ids=[_qdrant_point_id(collection, chunk_id) for chunk_id in chunk_ids],
            with_payload=True,
        )
        by_chunk_id: dict[str, RagChunk] = {}
        for record in records:
            payload = record.payload or {}
            if payload.get(_SENTINEL_PAYLOAD_KEY):
                continue
            chunk = _payload_to_rag_chunk(payload)
            by_chunk_id[chunk.id] = chunk
        return [by_chunk_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_chunk_id]


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
    # Defense in depth: the sentinel meta point (P1.4 embedding-model fingerprint) already
    # carries no `visibility` field, so the `must` clause above never matches it -- this
    # `must_not` makes the exclusion explicit and independent of that incidental fact, so it
    # keeps holding even if the visibility filter's shape ever changes.
    must_not = [
        models.FieldCondition(
            key=_SENTINEL_PAYLOAD_KEY,
            match=models.MatchValue(value=True),
        )
    ]
    return models.Filter(must=must, must_not=must_not)


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


def _payload_to_rag_chunk(payload: dict[str, Any]) -> RagChunk:
    return RagChunk(
        id=str(payload["id"]),
        source=str(payload["source"]),
        source_type=str(payload["source_type"]),
        text=str(payload["text"]),
        visibility=payload["visibility"],
        tags=list(payload.get("tags", [])),
        world_id=payload.get("world_id"),
        scene_id=payload.get("scene_id"),
        persona_id=payload.get("persona_id"),
        session_id=payload.get("session_id"),
        actor_id=payload.get("actor_id"),
        importance=payload.get("importance"),
        created_at=_parse_payload_datetime(payload.get("created_at")),
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
