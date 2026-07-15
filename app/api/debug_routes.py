"""Text-bearing local RAG debug surface (backlog #85, docs/28).

Persisted turn diagnostics (``app.api.routes``, ``TurnDetailResponse``) stay
metadata-only by design -- no chunk text, no vectors. This module is the
deliberate exception: a local-only debug/visualization surface that reads
chunk text and vector coordinates back out of the configured vector store, for
inspecting what actually got indexed and how it clusters.

The visibility invariant (hidden, non-``player`` text never reaches a cloud
provider) is untouched by this module: these endpoints never construct an LLM
request and never call a provider. Hidden text is only ever returned to the
*caller of this local API* -- and only when the request explicitly opts in
with ``include_hidden=True``. Every response path applies the same redaction
rule (see ``_is_visible``): a chunk's text/preview is present iff its
visibility is ``player`` or the caller asked for hidden content; otherwise the
text field is ``None`` and ``text_redacted`` is ``True``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.api.errors import ApiError
from app.api.schemas import (
    ChunkTextResponse,
    ChunkTextsRequest,
    ChunkTextsResponse,
    ErrorResponse,
    RagMapPointResponse,
    RagMapRequest,
    RagMapResponse,
)
from app.composition import build_embedding_provider, build_vector_store
from app.config import Settings, get_settings
from app.domain import Visibility
from app.rag import RagChunk, RagCollection, StoredPoint, VectorStore
from app.rag.embeddings import EmbeddingProvider
from app.rag.projection import fit_pca_2d, project_2d

router = APIRouter()

# Hard cap on points fit/projected/returned by /diagnostics/rag-map: PCA is O(n * dim^2)
# and the response is JSON, not paginated, so an unbounded scroll over a large index
# could otherwise make this local debug endpoint expensive or return a multi-MB payload.
_MAP_MAX_POINTS = 20_000
_PREVIEW_CHARS = 120

ERROR_422_RESPONSE: dict[int | str, dict[str, object]] = {422: {"model": ErrorResponse}}
ERROR_503_RESPONSE: dict[int | str, dict[str, object]] = {503: {"model": ErrorResponse}}


def get_debug_vector_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> VectorStore:
    return build_vector_store(settings)


def get_debug_embedding_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingProvider:
    return build_embedding_provider(settings)


def _is_visible(visibility: Visibility, *, include_hidden: bool) -> bool:
    return include_hidden or visibility == Visibility.PLAYER


def _wrap_store_error(exc: Exception) -> ApiError:
    return ApiError(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        code="retrieval_unavailable",
        message=f"Vector store is unavailable: {exc}",
    )


def _to_map_point(
    collection: RagCollection,
    point: StoredPoint,
    coordinates: tuple[float, float],
    *,
    include_hidden: bool,
) -> RagMapPointResponse:
    chunk = point.chunk
    visible = _is_visible(chunk.visibility, include_hidden=include_hidden)
    x, y = coordinates
    return RagMapPointResponse(
        id=chunk.id,
        collection=collection.value,
        x=x,
        y=y,
        source=chunk.source,
        source_type=chunk.source_type,
        visibility=chunk.visibility.value,
        tags=chunk.tags,
        world_id=chunk.world_id,
        scene_id=chunk.scene_id,
        persona_id=chunk.persona_id,
        session_id=chunk.session_id,
        actor_id=chunk.actor_id,
        importance=chunk.importance,
        created_at=chunk.created_at,
        text_preview=chunk.text[:_PREVIEW_CHARS] if visible else None,
        text_redacted=not visible,
    )


@router.post(
    "/diagnostics/rag-map",
    response_model=RagMapResponse,
    responses={**ERROR_422_RESPONSE, **ERROR_503_RESPONSE},
)
def get_rag_map(
    request: RagMapRequest,
    vector_store: Annotated[VectorStore, Depends(get_debug_vector_store)],
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_debug_embedding_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RagMapResponse:
    entries: list[tuple[RagCollection, StoredPoint]] = []
    try:
        for collection in RagCollection:
            entries.extend((collection, point) for point in vector_store.scroll_points(collection))
    except Exception as exc:
        raise _wrap_store_error(exc) from exc

    if not entries:
        return RagMapResponse(
            points=[],
            point_count=0,
            truncated=False,
            explained_variance=[0.0, 0.0],
            query_point=None,
            embedding_model=settings.embedding_model,
        )

    truncated = False
    if len(entries) > _MAP_MAX_POINTS:
        entries = entries[:_MAP_MAX_POINTS]
        truncated = True

    vectors = [point.vector for _, point in entries]
    pca = fit_pca_2d(vectors)
    coordinates = project_2d(pca, vectors)

    points = [
        _to_map_point(collection, point, xy, include_hidden=request.include_hidden)
        for (collection, point), xy in zip(entries, coordinates, strict=True)
    ]

    query_point: list[float] | None = None
    if request.query_text is not None and request.query_text.strip():
        try:
            query_vector = embedding_provider.embed_text(request.query_text)
        except Exception as exc:
            raise _wrap_store_error(exc) from exc
        query_x, query_y = project_2d(pca, [query_vector])[0]
        query_point = [query_x, query_y]

    return RagMapResponse(
        points=points,
        point_count=len(points),
        truncated=truncated,
        explained_variance=list(pca.explained_variance_ratio),
        query_point=query_point,
        embedding_model=settings.embedding_model,
    )


@router.post(
    "/diagnostics/chunk-texts",
    response_model=ChunkTextsResponse,
    responses={**ERROR_422_RESPONSE, **ERROR_503_RESPONSE},
)
def get_chunk_texts(
    request: ChunkTextsRequest,
    vector_store: Annotated[VectorStore, Depends(get_debug_vector_store)],
) -> ChunkTextsResponse:
    ids_by_collection: dict[RagCollection, list[str]] = defaultdict(list)
    for item in request.items:
        ids_by_collection[RagCollection(item.collection)].append(item.id)

    chunks_by_key: dict[tuple[RagCollection, str], RagChunk] = {}
    try:
        for collection, chunk_ids in ids_by_collection.items():
            for chunk in vector_store.get_chunks(collection, chunk_ids):
                chunks_by_key[(collection, chunk.id)] = chunk
    except Exception as exc:
        raise _wrap_store_error(exc) from exc

    results: list[ChunkTextResponse] = []
    for item in request.items:
        collection = RagCollection(item.collection)
        found_chunk = chunks_by_key.get((collection, item.id))
        if found_chunk is None:
            results.append(
                ChunkTextResponse(
                    id=item.id,
                    collection=item.collection,
                    visibility=None,
                    text=None,
                    text_redacted=False,
                    found=False,
                )
            )
            continue
        visible = _is_visible(found_chunk.visibility, include_hidden=request.include_hidden)
        results.append(
            ChunkTextResponse(
                id=found_chunk.id,
                collection=item.collection,
                visibility=found_chunk.visibility.value,
                text=found_chunk.text if visible else None,
                text_redacted=not visible,
                found=True,
            )
        )
    return ChunkTextsResponse(chunks=results)
