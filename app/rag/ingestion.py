from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field

from app.domain import Visibility
from app.rag.chunking import ChunkingConfig, chunk_text
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import RagChunk, RagCollection
from app.rag.vector_store import VectorStore

SUPPORTED_DOCUMENT_SUFFIXES = {".md", ".txt"}


class IngestionRequest(BaseModel):
    path: Path
    collection: RagCollection
    source_type: str
    visibility: Visibility
    tags: list[str] = Field(default_factory=list)
    world_id: str | None = None
    scene_id: str | None = None
    persona_id: str | None = None
    session_id: str | None = None


class IngestionResult(BaseModel):
    source: str
    collection: RagCollection
    chunk_count: int


def ingest_document(
    request: IngestionRequest,
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chunking_config: ChunkingConfig | None = None,
) -> IngestionResult:
    path = request.path
    if not path.exists():
        raise FileNotFoundError(f"missing document: {path}")
    if path.suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise ValueError(f"unsupported document type: {path.suffix}")

    text = path.read_text(encoding="utf-8")
    chunks_text = chunk_text(text, config=chunking_config)
    if not chunks_text:
        raise ValueError(f"empty document: {path}")

    source = str(path)
    chunks = [
        RagChunk(
            id=_chunk_id(source=source, text=chunk_text_value, index=index),
            source=source,
            source_type=request.source_type,
            text=chunk_text_value,
            visibility=request.visibility,
            tags=request.tags,
            world_id=request.world_id,
            scene_id=request.scene_id,
            persona_id=request.persona_id,
            session_id=request.session_id,
        )
        for index, chunk_text_value in enumerate(chunks_text)
    ]
    vectors = embedding_provider.embed_batch([chunk.text for chunk in chunks])
    vector_store.ensure_collection(request.collection, embedding_provider.dimension)
    vector_store.replace_source(request.collection, source, chunks, vectors)
    return IngestionResult(
        source=source,
        collection=request.collection,
        chunk_count=len(chunks),
    )


def ingest_lore_manifest(
    content_root: Path,
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chunking_config: ChunkingConfig | None = None,
) -> list[IngestionResult]:
    """Ingest every document listed in ``<content_root>/documents/manifest.json`` into CANON_LORE.

    Idempotent: ``ingest_document`` replaces a source's chunks by path, so re-running on every
    session start cannot duplicate lore. Raises ``FileNotFoundError`` when no manifest exists
    (the caller decides whether a manifest-less scenario is an error or simply has no lore).
    """
    # Local import keeps the app.content -> app.rag dependency one-directional at module load.
    from app.content.validator import LoreManifest

    manifest_path = content_root / "documents" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing lore manifest: {manifest_path}")
    manifest = LoreManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    return [
        ingest_document(
            IngestionRequest(
                path=content_root / "documents" / document.path,
                collection=RagCollection.CANON_LORE,
                source_type=document.source_type,
                visibility=document.visibility,
                tags=document.tags,
                world_id=document.world_id,
                scene_id=document.scene_id,
                persona_id=document.persona_id,
            ),
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            chunking_config=chunking_config,
        )
        for document in manifest.documents
    ]


def _chunk_id(*, source: str, text: str, index: int) -> str:
    digest = sha256(f"{source}:{index}:{text}".encode("utf-8")).hexdigest()
    return f"chunk-{digest[:16]}"
