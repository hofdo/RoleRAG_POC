from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.domain import PersonaCard, RetrievedChunk, SceneState, StoredTurn
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import RagCollection, RetrievalFilter
from app.rag.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        default_top_k: int = 5,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(
        self,
        *,
        query: str,
        collection: RagCollection,
        filters: RetrievalFilter,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        limit = top_k or self.default_top_k
        query_vector = self.embedding_provider.embed_text(query)
        results = self.vector_store.search(
            collection=collection,
            vector=query_vector,
            filters=filters,
            limit=limit,
        )
        return [chunk for chunk in results if chunk.visibility in filters.allowed_visibilities]


class ChunkRetrieving(Protocol):
    def retrieve(
        self,
        *,
        query: str,
        collection: RagCollection,
        filters: RetrievalFilter,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]: ...


class ActorContextRetriever:
    def __init__(self, *, retriever: ChunkRetrieving) -> None:
        self.retriever = retriever

    def retrieve_for_actor(
        self,
        *,
        query: str,
        world_id: str,
        session_id: str,
        persona_id: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        searches = [
            (RagCollection.SESSION_MEMORY, RetrievalFilter.player_visible(session_id=session_id)),
            (RagCollection.PERSONA_MEMORY, RetrievalFilter.player_visible(persona_id=persona_id)),
            (RagCollection.CANON_LORE, RetrievalFilter.player_visible(world_id=world_id)),
        ]
        candidates = [
            chunk
            for collection, filters in searches
            for chunk in self.retriever.retrieve(
                query=query,
                collection=collection,
                filters=filters,
                top_k=top_k,
            )
        ]
        deduplicated: dict[str, RetrievedChunk] = {}
        for chunk in sorted(candidates, key=lambda item: item.score, reverse=True):
            deduplicated.setdefault(chunk.id, chunk)
        return list(deduplicated.values())[:top_k]


def build_retrieval_query(
    *,
    user_message: str,
    scene: SceneState,
    persona: PersonaCard,
    recent_turns: Sequence[StoredTurn] = (),
) -> str:
    lines = [
        f"Scene: {scene.title}",
        f"Location: {scene.location}",
        f"Active persona: {persona.name}",
    ]
    if persona.goals:
        lines.append(f"Persona goals: {', '.join(persona.goals)}")

    recent_context = recent_turns[-2:]
    if recent_context:
        lines.append("Recent dialogue:")
        for turn in recent_context:
            lines.append(f"User: {_clip_line(turn.user_message)}")
            lines.append(f"Assistant: {_clip_line(turn.assistant_message)}")

    lines.append(f"User message: {user_message}")
    return "\n".join(lines)


def _clip_line(text: str, max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."
