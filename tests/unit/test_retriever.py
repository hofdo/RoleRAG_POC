from __future__ import annotations

from collections.abc import Sequence

from app.domain import RetrievedChunk, Visibility
from app.rag.models import RagCollection, RetrievalFilter
from app.rag.retriever import Retriever


class FakeEmbeddingProvider:
    dimension = 3

    def embed_text(self, text: str) -> list[float]:
        return [1.0, float(len(text)), 0.0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, float(len(text)), 0.0] for text in texts]


class FakeVectorStore:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.search_calls: list[tuple[RagCollection, list[float], RetrievalFilter, int]] = []

    def ensure_collection(self, collection: RagCollection, vector_size: int) -> None:
        raise AssertionError("ensure_collection should not be called during retrieval")

    def replace_source(
        self,
        collection: RagCollection,
        source: str,
        chunks: Sequence[object],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        raise AssertionError("replace_source should not be called during retrieval")

    def search(
        self,
        collection: RagCollection,
        vector: Sequence[float],
        filters: RetrievalFilter,
        limit: int,
    ) -> list[RetrievedChunk]:
        self.search_calls.append((collection, list(vector), filters, limit))
        return self.chunks


def test_retriever_filters_out_non_player_visible_chunks() -> None:
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(
            [
                RetrievedChunk(
                    id="public-1",
                    source="demo.md",
                    source_type="lore",
                    text="The rose gallery opens onto the winter court.",
                    score=0.9,
                    visibility=Visibility.PLAYER,
                ),
                RetrievedChunk(
                    id="gm-1",
                    source="demo.md",
                    source_type="lore",
                    text="The regent's spy waits behind the west screen.",
                    score=0.99,
                    visibility=Visibility.GM,
                ),
            ]
        ),
        default_top_k=4,
    )

    results = retriever.retrieve(
        query="What is visible in the rose gallery?",
        collection=RagCollection.CANON_LORE,
        filters=RetrievalFilter.player_visible(world_id="demo_world"),
    )

    assert [chunk.id for chunk in results] == ["public-1"]


def test_retriever_passes_filters_and_limit_to_vector_store() -> None:
    vector_store = FakeVectorStore([])
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        default_top_k=6,
    )
    filters = RetrievalFilter(
        allowed_visibilities=[Visibility.PLAYER],
        world_id="demo_world",
        scene_id="rose-gallery",
    )

    retriever.retrieve(
        query="Where is the west door?",
        collection=RagCollection.CANON_LORE,
        filters=filters,
        top_k=2,
    )

    assert len(vector_store.search_calls) == 1
    collection, vector, passed_filters, limit = vector_store.search_calls[0]
    assert collection == RagCollection.CANON_LORE
    assert vector == [1.0, float(len("Where is the west door?")), 0.0]
    assert passed_filters == filters
    assert limit == 2
