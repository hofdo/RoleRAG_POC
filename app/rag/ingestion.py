from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field

from app.domain import Visibility
from app.rag.chunking import ChunkingConfig, chunk_text
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import RagChunk, RagCollection
from app.rag.vector_store import VectorStore

SUPPORTED_DOCUMENT_SUFFIXES = {".md", ".txt"}

# First markdown H1 line (docs/22 P1.3 contextual chunk header doc_title). Deliberately
# just the H1 level -- app.rag.chunking's own ATX heading regex (levels 1-6) is a
# separate, section-splitting concern.
_FIRST_H1_RE = re.compile(r"^# (.+)$", re.MULTILINE)


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
    skipped: bool = False


def ingest_document(
    request: IngestionRequest,
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chunking_config: ChunkingConfig | None = None,
    model_key: str | None = None,
    force: bool = False,
) -> IngestionResult:
    path = request.path
    if not path.exists():
        raise FileNotFoundError(f"missing document: {path}")
    if path.suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise ValueError(f"unsupported document type: {path.suffix}")

    text = path.read_text(encoding="utf-8")
    # Always derived and always passed: chunk_text ignores doc_title entirely on the
    # legacy (structure_aware=False, default) path, so this is harmless when the flag is
    # off and gives the structure-aware path (docs/22 P1.3) its contextual header for free
    # when it's on -- no branch needed here on the flag itself.
    doc_title = _derive_doc_title(text, path)
    chunks_text = chunk_text(text, config=chunking_config, doc_title=doc_title)
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

    # ensure_collection runs before any skip decision, deliberately: its dimension and P1.4
    # model-fingerprint guards must fire even when the content-fingerprint check below would
    # otherwise skip the document entirely (a stale/mismatched embedding model must never be
    # masked by a lucky unchanged-content skip).
    vector_store.ensure_collection(
        request.collection, embedding_provider.dimension, model_key=model_key
    )

    if not force:
        # Content-fingerprint skip (backlog #86): chunk ids are sha256(source:index:text)[:16]
        # (_chunk_id below), so an unchanged document always reproduces exactly the same id
        # set. If the store already holds exactly that set for this source, embedding +
        # replace_source would end in an identical state -- skip both. Either-direction
        # mismatch (extra or missing ids) falls through to a full re-ingest via plain set
        # equality. Fail-open: a read failure here must never break ingestion, so an
        # unexpected exception from the store falls back to the full path below instead of
        # propagating (mirrors the repo's fail-open-for-optimizations stance, e.g.
        # QdrantVectorStore._ensure_payload_indexes).
        existing_ids: set[str] | None
        try:
            existing_ids = vector_store.list_source_chunk_ids(request.collection, source)
        except Exception:
            existing_ids = None
        if existing_ids is not None and existing_ids == {chunk.id for chunk in chunks}:
            return IngestionResult(
                source=source,
                collection=request.collection,
                chunk_count=len(chunks),
                skipped=True,
            )

    vectors = embedding_provider.embed_batch([chunk.text for chunk in chunks])
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
    model_key: str | None = None,
    force: bool = False,
) -> list[IngestionResult]:
    """Ingest every document listed in ``<content_root>/documents/manifest.json`` into CANON_LORE.

    Idempotent: ``ingest_document`` replaces a source's chunks by path, so re-running on every
    session start cannot duplicate lore. Raises ``FileNotFoundError`` when no manifest exists
    (the caller decides whether a manifest-less scenario is an error or simply has no lore).
    Unless ``force=True``, a document whose content is unchanged since the last ingest is
    skipped without re-embedding (backlog #86) -- see ``ingest_document.skipped``.
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
            model_key=model_key,
            force=force,
        )
        for document in manifest.documents
    ]


def _chunk_id(*, source: str, text: str, index: int) -> str:
    digest = sha256(f"{source}:{index}:{text}".encode("utf-8")).hexdigest()
    return f"chunk-{digest[:16]}"


def _derive_doc_title(text: str, path: Path) -> str:
    """docs/22 P1.3 contextual-header doc_title: the document's first markdown H1 line's
    text, or the filename stem when it has none."""
    match = _FIRST_H1_RE.search(text.replace("\r\n", "\n"))
    if match:
        title = match.group(1).strip()
        if title:
            return title
    return path.stem
