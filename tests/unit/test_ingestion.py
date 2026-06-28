from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.content.validator import LoreDocumentMetadata, LoreManifest
from app.domain import RetrievedChunk, Visibility
from app.rag.ingestion import IngestionRequest, ingest_document, ingest_lore_manifest
from app.rag.models import RagChunk, RagCollection, RetrievalFilter


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.dimension = 3

    def embed_text(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 0.0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(index + 1), float(len(text)), 0.0] for index, text in enumerate(texts)]


class RecordingVectorStore:
    def __init__(self) -> None:
        self.ensure_calls: list[tuple[RagCollection, int]] = []
        self.replace_calls: list[tuple[RagCollection, str, list[RagChunk], list[list[float]]]] = []

    def ensure_collection(self, collection: RagCollection, vector_size: int) -> None:
        self.ensure_calls.append((collection, vector_size))

    def replace_source(
        self,
        collection: RagCollection,
        source: str,
        chunks: Sequence[RagChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        self.replace_calls.append(
            (collection, source, list(chunks), [list(vector) for vector in vectors])
        )

    def upsert_chunks(
        self,
        collection: RagCollection,
        chunks: Sequence[RagChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        raise AssertionError("upsert_chunks should not be called during ingestion")

    def delete_session_points(self, collection: RagCollection, session_id: str) -> None:
        raise AssertionError("delete_session_points should not be called during ingestion")

    def delete_points(self, collection: RagCollection, chunk_ids: Sequence[str]) -> None:
        raise AssertionError("delete_points should not be called during ingestion")

    def search(
        self,
        collection: RagCollection,
        vector: Sequence[float],
        filters: RetrievalFilter,
        limit: int,
    ) -> list[RetrievedChunk]:
        raise AssertionError("search should not be called during ingestion")


def test_ingest_document_attaches_required_metadata_and_replaces_source(tmp_path: Path) -> None:
    document = tmp_path / "demo_lore.md"
    document.write_text(
        (
            "# Rose Gallery\n\nCourtiers gather around mirrored columns.\n\n"
            "The west door stays guarded."
        ),
        encoding="utf-8",
    )
    embedding_provider = FakeEmbeddingProvider()
    vector_store = RecordingVectorStore()

    result = ingest_document(
        IngestionRequest(
            path=document,
            collection=RagCollection.CANON_LORE,
            source_type="lore",
            visibility=Visibility.PLAYER,
            tags=["palace", "social"],
            world_id="demo_world",
        ),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    assert result.chunk_count >= 1
    assert result.source == str(document)
    assert vector_store.ensure_calls == [(RagCollection.CANON_LORE, 3)]
    assert len(vector_store.replace_calls) == 1
    _, source, chunks, vectors = vector_store.replace_calls[0]
    assert source == str(document)
    assert len(chunks) == result.chunk_count
    assert len(vectors) == result.chunk_count
    assert all(chunk.visibility == Visibility.PLAYER for chunk in chunks)
    assert all(chunk.source_type == "lore" for chunk in chunks)
    assert all(chunk.tags == ["palace", "social"] for chunk in chunks)
    assert all(chunk.world_id == "demo_world" for chunk in chunks)
    assert all(chunk.source == str(document) for chunk in chunks)
    assert all(chunk.id for chunk in chunks)


def _write_scenario(tmp_path: Path) -> Path:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "lore_a.md").write_text(
        "# A\n\nThe gallery has mirrored columns.", encoding="utf-8"
    )
    (documents / "lore_b.md").write_text(
        "# B\n\nThe west door stays guarded.", encoding="utf-8"
    )
    manifest = LoreManifest(
        documents=[
            LoreDocumentMetadata(
                path="lore_a.md",
                visibility=Visibility.PLAYER,
                source_type="lore",
                world_id="demo_world",
            ),
            LoreDocumentMetadata(
                path="lore_b.md",
                visibility=Visibility.GM,
                source_type="lore",
                world_id="demo_world",
            ),
        ]
    )
    (documents / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    return tmp_path


def test_ingest_lore_manifest_ingests_every_document(tmp_path: Path) -> None:
    content_root = _write_scenario(tmp_path)
    vector_store = RecordingVectorStore()

    results = ingest_lore_manifest(
        content_root,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    assert len(results) == 2
    assert len(vector_store.replace_calls) == 2
    sources = {source for _, source, _, _ in vector_store.replace_calls}
    assert sources == {
        str(content_root / "documents" / "lore_a.md"),
        str(content_root / "documents" / "lore_b.md"),
    }
    # Per-document visibility from the manifest is carried onto the chunks.
    visibilities = {
        chunk.visibility for _, _, chunks, _ in vector_store.replace_calls for chunk in chunks
    }
    assert visibilities == {Visibility.PLAYER, Visibility.GM}


def test_ingest_lore_manifest_raises_when_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing lore manifest"):
        ingest_lore_manifest(
            tmp_path,
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=RecordingVectorStore(),
        )


def test_ingest_document_rejects_empty_documents(tmp_path: Path) -> None:
    document = tmp_path / "empty.txt"
    document.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(ValueError, match="empty document"):
        ingest_document(
            IngestionRequest(
                path=document,
                collection=RagCollection.CANON_LORE,
                source_type="lore",
                visibility=Visibility.PLAYER,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=RecordingVectorStore(),
        )
