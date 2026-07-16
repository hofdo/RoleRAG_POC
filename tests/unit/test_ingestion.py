from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.content.validator import LoreDocumentMetadata, LoreManifest
from app.domain import RetrievedChunk, Visibility
from app.rag.ingestion import IngestionRequest, ingest_document, ingest_lore_manifest
from app.rag.models import RagChunk, RagCollection, RetrievalFilter
from app.rag.vector_store import InMemoryVectorStore, StoredPoint, VectorStoreModelMismatch


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
        self.list_source_calls: list[tuple[RagCollection, str]] = []

    def ensure_collection(
        self, collection: RagCollection, vector_size: int, model_key: str | None = None
    ) -> None:
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

    def delete_source_points(self, collection: RagCollection, source: str) -> None:
        raise AssertionError("delete_source_points should not be called during ingestion")

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

    def scroll_points(self, collection: RagCollection) -> list[StoredPoint]:
        raise AssertionError("scroll_points should not be called during ingestion")

    def get_chunks(self, collection: RagCollection, chunk_ids: Sequence[str]) -> list[RagChunk]:
        raise AssertionError("get_chunks should not be called during ingestion")

    def list_source_chunk_ids(self, collection: RagCollection, source: str) -> set[str]:
        # Real ingest_document behavior (backlog #86): called on every non-force ingest to
        # decide whether the document is unchanged. This recorder never has anything on file,
        # so every ingest in these tests takes the full embed+replace path, matching their
        # existing assertions.
        self.list_source_calls.append((collection, source))
        return set()


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


# ---------------------------------------------------------------------------
# backlog #86: content-fingerprint skip for unchanged documents
# ---------------------------------------------------------------------------


def _request(document: Path) -> IngestionRequest:
    return IngestionRequest(
        path=document,
        collection=RagCollection.CANON_LORE,
        source_type="lore",
        visibility=Visibility.PLAYER,
    )


def test_ingest_document_second_call_with_unchanged_content_is_skipped(tmp_path: Path) -> None:
    document = tmp_path / "demo_lore.md"
    document.write_text(
        "# Rose Gallery\n\nCourtiers gather around mirrored columns.", encoding="utf-8"
    )
    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()

    first = ingest_document(
        _request(document), embedding_provider=embedding_provider, vector_store=vector_store
    )
    second = ingest_document(
        _request(document), embedding_provider=embedding_provider, vector_store=vector_store
    )

    assert first.skipped is False
    assert second.skipped is True
    assert second.chunk_count == first.chunk_count
    assert second.source == first.source
    # embed_batch ran exactly once -- the second, unchanged-content call never re-embedded.
    assert len(embedding_provider.batches) == 1


def test_ingest_document_changed_content_reingests(tmp_path: Path) -> None:
    document = tmp_path / "demo_lore.md"
    document.write_text("# Rose Gallery\n\nCourtiers gather.", encoding="utf-8")
    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()

    ingest_document(
        _request(document), embedding_provider=embedding_provider, vector_store=vector_store
    )
    document.write_text(
        "# Rose Gallery\n\nCourtiers gather near the fountain now.", encoding="utf-8"
    )
    second = ingest_document(
        _request(document), embedding_provider=embedding_provider, vector_store=vector_store
    )

    assert second.skipped is False
    assert len(embedding_provider.batches) == 2


def test_ingest_document_force_reingests_despite_identical_content(tmp_path: Path) -> None:
    document = tmp_path / "demo_lore.md"
    document.write_text("# Rose Gallery\n\nCourtiers gather.", encoding="utf-8")
    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()

    ingest_document(
        _request(document), embedding_provider=embedding_provider, vector_store=vector_store
    )
    forced = ingest_document(
        _request(document),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        force=True,
    )

    assert forced.skipped is False
    assert len(embedding_provider.batches) == 2


def test_ingest_document_store_read_failure_falls_back_to_full_ingest(tmp_path: Path) -> None:
    document = tmp_path / "demo_lore.md"
    document.write_text("# Rose Gallery\n\nCourtiers gather.", encoding="utf-8")
    embedding_provider = FakeEmbeddingProvider()

    class RaisingListVectorStore(InMemoryVectorStore):
        def list_source_chunk_ids(self, collection: RagCollection, source: str) -> set[str]:
            raise RuntimeError("simulated store-read failure")

    vector_store = RaisingListVectorStore()

    result = ingest_document(
        _request(document), embedding_provider=embedding_provider, vector_store=vector_store
    )

    assert result.skipped is False
    assert len(embedding_provider.batches) == 1
    # The fail-open fallback still completed the real ingest -- the chunk is retrievable.
    assert vector_store.scroll_points(RagCollection.CANON_LORE)


def test_ingest_document_ensure_collection_guards_fire_before_skip_decision(
    tmp_path: Path,
) -> None:
    """ensure_collection's dimension/model-fingerprint guards (P1.4) must run -- and can
    raise -- before the content-fingerprint skip decision is ever evaluated, even when the
    document's content (and therefore its chunk-id set) is completely unchanged."""
    document = tmp_path / "demo_lore.md"
    document.write_text("# Rose Gallery\n\nCourtiers gather.", encoding="utf-8")
    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()

    ingest_document(
        _request(document),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        model_key="model-a",
    )

    with pytest.raises(VectorStoreModelMismatch):
        ingest_document(
            _request(document),
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            model_key="model-b",
        )
