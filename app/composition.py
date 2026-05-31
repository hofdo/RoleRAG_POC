from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from app.agents import CriticAgent, MemoryCurator
from app.config import Settings
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.provider import LlmProvider
from app.memory import MemoryEpisodeStore, RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator
from app.persistence import (
    FileDataLoader,
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
    connect_sqlite,
    initialize_database,
)
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
    orchestrator: TurnOrchestrator
    recent_dialogue_store: RecentDialogueStore

    def close(self) -> None:
        self.connection.close()


def redact_settings(settings: Settings) -> dict[str, object]:
    values = settings.model_dump()
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
    if settings.cloud_llm_api_key == "replace_me":
        return None
    return OpenAICompatibleProvider(
        provider_name="cloud",
        base_url=settings.cloud_llm_base_url,
        api_key=settings.cloud_llm_api_key,
    )


def build_file_loader() -> FileDataLoader:
    return FileDataLoader()


def build_critic_agent() -> CriticAgent:
    return CriticAgent()


def build_memory_curator() -> MemoryCurator:
    return MemoryCurator()


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    return FastEmbedEmbeddingProvider(model_name=settings.embedding_model)


def build_vector_store(settings: Settings) -> VectorStore:
    return QdrantVectorStore(url=settings.qdrant_url)


def build_actor_context_retriever(settings: Settings) -> ActorContextRetriever:
    return ActorContextRetriever(
        retriever=Retriever(
            embedding_provider=build_embedding_provider(settings),
            vector_store=build_vector_store(settings),
            default_top_k=settings.rag_default_top_k,
        )
    )


def build_services(
    settings: Settings,
    *,
    enable_retrieval: bool,
) -> AppServices:
    connection = connect_sqlite(settings.database_path)
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    turn_repository = SQLiteTurnRepository(connection)
    memory_repository = SQLiteMemoryRepository(connection)
    recent_dialogue_store = RecentDialogueStore(
        turn_repository=turn_repository,
        recent_turns=settings.recent_dialogue_turns,
    )
    orchestrator = TurnOrchestrator(
        loader=build_file_loader(),
        provider=build_local_provider(settings),
        cloud_provider=build_cloud_provider(settings),
        critic_agent=build_critic_agent(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=recent_dialogue_store,
        memory_store=MemoryEpisodeStore(memory_repository=memory_repository),
        memory_curator=build_memory_curator(),
        actor_context_retriever=(
            build_actor_context_retriever(settings) if enable_retrieval else None
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
        orchestrator=orchestrator,
        recent_dialogue_store=recent_dialogue_store,
    )
