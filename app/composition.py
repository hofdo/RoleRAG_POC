from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.agents import CriticAgent, MemoryCurator
from app.config import Settings, is_usable_cloud_api_key
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.provider import LlmProvider
from app.memory import MemoryEpisodeStore, MemoryIndexer, RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator
from app.persistence import (
    FileDataLoader,
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
    connect_sqlite,
    initialize_database,
)
from app.persistence.repositories import SessionRepository
from app.rag import (
    ActorContextRetriever,
    EmbeddingProvider,
    FastEmbedEmbeddingProvider,
    QdrantVectorStore,
    Retriever,
    VectorStore,
)


@dataclass
class AppServices:
    connection: sqlite3.Connection
    session_repository: SessionRepository
    orchestrator: TurnOrchestrator
    recent_dialogue_store: RecentDialogueStore

    def close(self) -> None:
        self.connection.close()


def redact_settings(settings: Settings) -> dict[str, object]:
    values = settings.model_dump(mode="json")
    values["local_llm_api_key"] = "***"
    values["cloud_llm_api_key"] = "***"
    return values


def build_local_provider(settings: Settings) -> LlmProvider:
    return OpenAICompatibleProvider(
        provider_name="local",
        base_url=settings.local_llm_base_url,
        api_key=settings.local_llm_api_key,
    )


def build_cloud_provider(settings: Settings) -> LlmProvider | None:
    if not is_usable_cloud_api_key(settings.cloud_llm_api_key):
        return None
    return OpenAICompatibleProvider(
        provider_name="cloud",
        base_url=settings.cloud_llm_base_url,
        api_key=settings.cloud_llm_api_key,
    )


def build_file_loader(content_root: Path | str = "data") -> FileDataLoader:
    return FileDataLoader(base_path=content_root)


def build_critic_agent() -> CriticAgent:
    return CriticAgent()


def build_memory_curator() -> MemoryCurator:
    return MemoryCurator()


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    return FastEmbedEmbeddingProvider(model_name=settings.embedding_model)


def build_vector_store(settings: Settings) -> VectorStore:
    return QdrantVectorStore(url=settings.qdrant_url)


def build_actor_context_retriever(
    settings: Settings,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
) -> ActorContextRetriever:
    return ActorContextRetriever(
        retriever=Retriever(
            embedding_provider=embedding_provider or build_embedding_provider(settings),
            vector_store=vector_store or build_vector_store(settings),
            default_top_k=settings.rag_default_top_k,
        )
    )


def build_services(
    settings: Settings,
    *,
    enable_retrieval: bool,
    content_root: Path | str | None = None,
) -> AppServices:
    resolved_content_root = content_root or settings.content_root
    connection = connect_sqlite(settings.database_path)
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    turn_repository = SQLiteTurnRepository(connection)
    memory_repository = SQLiteMemoryRepository(connection)
    memory_store = MemoryEpisodeStore(memory_repository=memory_repository)
    recent_dialogue_store = RecentDialogueStore(
        turn_repository=turn_repository,
        recent_turns=settings.recent_dialogue_turns,
    )
    embedding_provider = build_embedding_provider(settings) if enable_retrieval else None
    vector_store = build_vector_store(settings) if enable_retrieval else None
    orchestrator = TurnOrchestrator(
        loader=build_file_loader(resolved_content_root),
        loader_factory=build_file_loader,
        content_root=str(resolved_content_root),
        provider=build_local_provider(settings),
        cloud_provider=build_cloud_provider(settings),
        critic_agent=build_critic_agent(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=recent_dialogue_store,
        memory_store=memory_store,
        memory_curator=build_memory_curator(),
        memory_indexer=(
            MemoryIndexer(
                memory_store=memory_store,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
            )
            if embedding_provider is not None and vector_store is not None
            else None
        ),
        actor_context_retriever=(
            build_actor_context_retriever(
                settings,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
            )
            if embedding_provider is not None and vector_store is not None
            else None
        ),
        retrieval_top_k=settings.rag_default_top_k,
        max_retrieved_chunk_chars=settings.rag_max_retrieved_chunk_chars,
        local_model=settings.local_llm_model,
        cloud_model=settings.cloud_llm_model,
        local_max_tokens=settings.local_llm_max_tokens,
        cloud_max_tokens=settings.cloud_llm_max_tokens,
        local_temperature=settings.local_llm_temperature,
        cloud_temperature=settings.cloud_llm_temperature,
        cloud_mode=settings.cloud_mode,
    )
    return AppServices(
        connection=connection,
        session_repository=session_repository,
        orchestrator=orchestrator,
        recent_dialogue_store=recent_dialogue_store,
    )
