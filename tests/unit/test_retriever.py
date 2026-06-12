from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from app.domain import PersonaCard, RetrievedChunk, SceneState, StoredTurn, Visibility
from app.llm.router import ModelProviderName, ModelRoute
from app.rag.models import RagChunk, RagCollection, RetrievalFilter
from app.rag.retriever import ActorContextRetriever, Retriever, build_retrieval_query
from app.rag.vector_store import InMemoryVectorStore


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

    def upsert_chunks(
        self,
        collection: RagCollection,
        chunks: Sequence[object],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        raise AssertionError("upsert_chunks should not be called during retrieval")

    def delete_session_points(self, collection: RagCollection, session_id: str) -> None:
        raise AssertionError("delete_session_points should not be called during retrieval")

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


def test_in_memory_actor_retrieval_filters_scope_and_hidden_visibility() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    canon_chunks = [
        RagChunk(
            id="expected-lore",
            source="expected-lore.md",
            source_type="lore",
            text="gallery",
            visibility=Visibility.PLAYER,
            world_id="demo_world",
        ),
        RagChunk(
            id="wrong-world",
            source="wrong-world.md",
            source_type="lore",
            text="gallery",
            visibility=Visibility.PLAYER,
            world_id="other_world",
        ),
        RagChunk(
            id="gm-only",
            source="gm-only.md",
            source_type="lore",
            text="gallery",
            visibility=Visibility.GM,
            world_id="demo_world",
        ),
    ]
    vector_store.ensure_collection(RagCollection.CANON_LORE, embedding_provider.dimension)
    vector_store.upsert_chunks(
        RagCollection.CANON_LORE,
        canon_chunks,
        embedding_provider.embed_batch([chunk.text for chunk in canon_chunks]),
    )
    session_chunks = [
        RagChunk(
            id="expected-session",
            source="expected-session.md",
            source_type="memory",
            text="gallery",
            visibility=Visibility.PLAYER,
            session_id="session-1",
        ),
        RagChunk(
            id="wrong-session",
            source="wrong-session.md",
            source_type="memory",
            text="gallery",
            visibility=Visibility.PLAYER,
            session_id="session-2",
        ),
    ]
    vector_store.ensure_collection(RagCollection.SESSION_MEMORY, embedding_provider.dimension)
    vector_store.upsert_chunks(
        RagCollection.SESSION_MEMORY,
        session_chunks,
        embedding_provider.embed_batch([chunk.text for chunk in session_chunks]),
    )
    persona_chunks = [
        RagChunk(
            id="expected-persona",
            source="expected-persona.md",
            source_type="memory",
            text="gallery",
            visibility=Visibility.PLAYER,
            persona_id="archivist",
        ),
        RagChunk(
            id="wrong-persona",
            source="wrong-persona.md",
            source_type="memory",
            text="gallery",
            visibility=Visibility.PLAYER,
            persona_id="envoy",
        ),
        RagChunk(
            id="character-private",
            source="character-private.md",
            source_type="memory",
            text="gallery",
            visibility=Visibility.CHARACTER_PRIVATE,
            persona_id="archivist",
        ),
    ]
    vector_store.ensure_collection(RagCollection.PERSONA_MEMORY, embedding_provider.dimension)
    vector_store.upsert_chunks(
        RagCollection.PERSONA_MEMORY,
        persona_chunks,
        embedding_provider.embed_batch([chunk.text for chunk in persona_chunks]),
    )

    results = ActorContextRetriever(
        retriever=Retriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
    ).retrieve_for_actor(
        query="gallery",
        world_id="demo_world",
        session_id="session-1",
        persona_id="archivist",
        top_k=10,
    )

    assert {chunk.id for chunk in results} == {
        "expected-lore",
        "expected-session",
        "expected-persona",
    }


class RecordingRetriever:
    def __init__(self, chunks_by_collection: dict[RagCollection, list[RetrievedChunk]]) -> None:
        self.chunks_by_collection = chunks_by_collection
        self.calls: list[tuple[str, RagCollection, RetrievalFilter, int | None]] = []

    def retrieve(
        self,
        *,
        query: str,
        collection: RagCollection,
        filters: RetrievalFilter,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, collection, filters, top_k))
        return self.chunks_by_collection.get(collection, [])


def test_actor_context_retriever_aggregates_collections_by_score_and_scope() -> None:
    duplicate = RetrievedChunk(
        id="duplicate",
        source="memory",
        source_type="memory",
        text="The player promised to return.",
        score=0.8,
        visibility=Visibility.PLAYER,
    )
    retriever = RecordingRetriever(
        {
            RagCollection.SESSION_MEMORY: [duplicate],
            RagCollection.PERSONA_MEMORY: [
                RetrievedChunk(
                    id="persona-1",
                    source="persona",
                    source_type="memory",
                    text="Iria trusts careful questions.",
                    score=0.95,
                    visibility=Visibility.PLAYER,
                )
            ],
            RagCollection.CANON_LORE: [
                duplicate,
                RetrievedChunk(
                    id="lore-1",
                    source="lore",
                    source_type="lore",
                    text="The gallery has mirrored columns.",
                    score=0.7,
                    visibility=Visibility.PLAYER,
                ),
            ],
        }
    )

    chunks = ActorContextRetriever(retriever=retriever).retrieve_for_actor(
        query="gallery",
        world_id="demo_world",
        session_id="session-1",
        persona_id="archivist",
        top_k=2,
    )

    assert [chunk.id for chunk in chunks] == ["persona-1", "duplicate"]
    assert [call[1] for call in retriever.calls] == [
        RagCollection.SESSION_MEMORY,
        RagCollection.PERSONA_MEMORY,
        RagCollection.CANON_LORE,
    ]
    assert retriever.calls[0][2] == RetrievalFilter.player_visible(session_id="session-1")
    assert retriever.calls[1][2] == RetrievalFilter.player_visible(persona_id="archivist")
    assert retriever.calls[2][2] == RetrievalFilter.player_visible(world_id="demo_world")
    assert all(call[3] == 4 for call in retriever.calls)


def test_build_retrieval_query_uses_visible_context_and_latest_two_turns() -> None:
    route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=700,
        temperature=0.75,
        reason="default local route",
    )
    recent_turns = [
        StoredTurn(
            id=index,
            session_id="session-1",
            turn_index=index,
            scene_id="rose-gallery",
            persona_id="archivist",
            user_message=f"Question {index}",
            assistant_message=f"Answer {index}",
            route=route,
            created_at=datetime(2026, 5, 27, 11, index, tzinfo=UTC),
        )
        for index in range(1, 4)
    ]

    query = build_retrieval_query(
        user_message="What do I notice?",
        scene=SceneState(
            id="rose-gallery",
            title="Rose Gallery",
            location="Winter Palace",
            player_visible_summary="Courtiers drift between mirrors and roses.",
            gm_private_summary="The spy waits nearby.",
        ),
        persona=PersonaCard(
            id="archivist",
            name="Iria Vale",
            role="npc",
            public_description="A composed archivist.",
            private_description="She is aiding the coup.",
            speaking_style="Precise and dry.",
            goals=["protect the archive"],
            secrets=["She forged a ledger."],
        ),
        recent_turns=recent_turns,
    )

    assert "Scene: Rose Gallery" in query
    assert "Location: Winter Palace" in query
    assert "Active persona: Iria Vale" in query
    assert "Persona goals: protect the archive" in query
    assert "Question 1" not in query
    assert "Question 2" in query
    assert "Answer 3" in query
    assert "What do I notice?" in query
    assert "spy waits nearby" not in query
    assert "aiding the coup" not in query
    assert "forged a ledger" not in query
