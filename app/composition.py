from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.agents import CriticAgent, MemoryCurator
from app.config import Settings, is_usable_cloud_api_key
from app.diagnostics.structured_failures import StructuredOutputFailureSink
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.provider import LlmProvider
from app.memory import MemoryEpisodeStore, MemoryIndexer, RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator
from app.persistence import (
    FileDataLoader,
    SQLiteCanonRepository,
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
    connect_sqlite,
    initialize_database,
)
from app.persistence.repositories import CanonRepository, MemoryRepository, SessionRepository
from app.rag import (
    ActorContextRetriever,
    EmbeddingProvider,
    FastEmbedEmbeddingProvider,
    QdrantVectorStore,
    Retriever,
    VectorStore,
)
from app.rag.ranking import RankingWeights


@dataclass
class AppServices:
    connection: sqlite3.Connection
    session_repository: SessionRepository
    orchestrator: TurnOrchestrator
    recent_dialogue_store: RecentDialogueStore
    memory_repository: MemoryRepository | None = None
    canon_repository: CanonRepository | None = None

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
        timeout_seconds=settings.local_llm_timeout_seconds,
        max_retries=settings.local_llm_max_retries,
    )


def build_cloud_provider(settings: Settings) -> LlmProvider | None:
    if not is_usable_cloud_api_key(settings.cloud_llm_api_key):
        return None
    return OpenAICompatibleProvider(
        provider_name="cloud",
        base_url=settings.cloud_llm_base_url,
        api_key=settings.cloud_llm_api_key,
        timeout_seconds=settings.cloud_llm_timeout_seconds,
        max_retries=settings.cloud_llm_max_retries,
    )


def build_file_loader(content_root: Path | str = "data") -> FileDataLoader:
    return FileDataLoader(base_path=content_root)


def build_critic_agent() -> CriticAgent:
    return CriticAgent()


def build_memory_curator() -> MemoryCurator:
    return MemoryCurator()


def build_structured_failure_sink(settings: Settings) -> StructuredOutputFailureSink | None:
    if settings.structured_output_failure_log_dir is None:
        return None
    return StructuredOutputFailureSink(directory=settings.structured_output_failure_log_dir)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    return FastEmbedEmbeddingProvider(model_name=settings.embedding_model)


def build_vector_store(settings: Settings) -> VectorStore:
    return QdrantVectorStore(url=settings.qdrant_url)


def build_ranking_weights(settings: Settings) -> RankingWeights:
    return RankingWeights(
        session_memory_weight=settings.rag_session_memory_weight,
        persona_memory_weight=settings.rag_persona_memory_weight,
        canon_lore_weight=settings.rag_canon_lore_weight,
        session_id_match_boost=settings.rag_session_id_match_boost,
        scene_id_match_boost=settings.rag_scene_id_match_boost,
        persona_id_match_boost=settings.rag_persona_id_match_boost,
        importance_step_boost=settings.rag_importance_step_boost,
        lexical_match_step_boost=settings.rag_lexical_match_step_boost,
        lexical_match_max_boost=settings.rag_lexical_match_max_boost,
        recency_weight=settings.rag_recency_weight,
        candidate_oversample_factor=settings.rag_candidate_oversample_factor,
    )


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
        ),
        weights=build_ranking_weights(settings),
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
    canon_repository = SQLiteCanonRepository(connection)
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
                importance_floor=settings.rag_index_importance_floor,
                session_memory_max_episodes=settings.session_memory_max_episodes,
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
        local_structured_max_tokens=settings.local_structured_max_tokens,
        cloud_max_tokens=settings.cloud_llm_max_tokens,
        local_temperature=settings.local_llm_temperature,
        cloud_temperature=settings.cloud_llm_temperature,
        cloud_mode=settings.cloud_mode,
        recent_dialogue_max_message_chars=settings.recent_dialogue_max_message_chars,
        structured_failure_sink=build_structured_failure_sink(settings),
        critic_gating=settings.critic_gating,
        curator_gating=settings.curator_gating,
        canon_importance_floor=settings.canon_importance_floor,
        canon_max_items=settings.canon_max_items,
        canon_max_chars=settings.canon_max_chars,
        memory_embedding_provider=embedding_provider,
        write_dedup_cosine_threshold=settings.rag_write_dedup_cosine_threshold,
        low_retrieval_confidence=settings.low_retrieval_confidence,
        high_scene_complexity=settings.high_scene_complexity,
        truncation_retry_budget_multiplier=settings.truncation_retry_budget_multiplier,
        containment_overlap_threshold=settings.containment_overlap_threshold,
        memory_consolidation_threshold=settings.memory_consolidation_threshold,
        memory_consolidation_max_importance=settings.memory_consolidation_max_importance,
        canon_repository=canon_repository,
    )
    return AppServices(
        connection=connection,
        session_repository=session_repository,
        orchestrator=orchestrator,
        recent_dialogue_store=recent_dialogue_store,
        memory_repository=memory_repository,
        canon_repository=canon_repository,
    )
