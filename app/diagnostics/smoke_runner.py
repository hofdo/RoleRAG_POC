from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.agents.critic_agent import CriticAgent
from app.agents.memory_curator import MemoryCurator
from app.config import Settings
from app.diagnostics.runtime_checks import (
    DiagnosticCheck,
    DiagnosticStatus,
    RuntimeDiagnosticsReport,
    build_runtime_diagnostics,
)
from app.domain import TurnInput, Visibility
from app.llm.provider import LlmProvider, LlmRequest, LlmResponse
from app.memory import MemoryEpisodeStore, MemoryIndexer, RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator, TurnOrchestratorConfig
from app.persistence import (
    FileDataLoader,
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
)
from app.persistence.sqlite import connect_sqlite, initialize_database
from app.rag import (
    ActorContextRetriever,
    ChunkingConfig,
    IngestionRequest,
    InMemoryVectorStore,
    RagCollection,
    RetrievalDiagnostics,
    Retriever,
    ingest_document,
)
from app.rag.embeddings import EmbeddingProvider
from app.rag.retriever import build_retrieval_query


class RetrievalSelectionSummary(BaseModel):
    id: str
    source: str
    source_type: str
    collection: str
    visibility: str
    selected_rank: int
    original_score: float
    adjusted_score: float
    tags: list[str] = Field(default_factory=list)


class SmokeRunSummary(BaseModel):
    status: DiagnosticStatus
    mode: Literal["safe", "real-runtime"]
    session_id: str | None
    temp_database_path: str | None
    ingested_chunk_count: int
    persisted_turns: int
    memory_written: bool
    persisted_memories: int
    indexed_memories: int
    retrieval_selected_count: int
    actor_response: str
    warnings: list[str] = Field(default_factory=list)
    steps: list[DiagnosticCheck] = Field(default_factory=list)
    retrieval_selected: list[RetrievalSelectionSummary] = Field(default_factory=list)
    live_diagnostics: RuntimeDiagnosticsReport | None = None


class DeterministicKeywordEmbeddingProvider(EmbeddingProvider):
    _KEYWORDS = ("rose", "gallery", "archive", "dawn", "regent", "door", "promise")

    @property
    def dimension(self) -> int:
        return len(self._KEYWORDS)

    def embed_text(self, text: str) -> list[float]:
        normalized = text.lower()
        return [float(normalized.count(keyword)) for keyword in self._KEYWORDS]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class SmokeTestProvider(LlmProvider):
    async def generate(self, request: LlmRequest) -> LlmResponse:
        task = request.metadata.get("task")
        if task == "critic":
            text = '{"accepted": true, "issues": [], "repair_instruction": null}'
        elif task == "memory_extraction":
            text = (
                '{"write_memory": true, "memories": [{"summary": '
                '"The player promised to return before dawn for archive access.", '
                '"visibility": "player", "importance": 4, "tags": ["promise", "archive", "dawn"], '
                '"scene_id": "rose-gallery", "actor_id": "archivist"}], '
                '"reason": "The promise changes future access."}'
            )
        else:
            text = (
                "Iria Vale folds her hands by the mirrored column. "
                "\"Return before dawn, and the archive door will open to you.\""
            )
        return LlmResponse(
            text=text,
            provider="fake",
            model=request.model,
            usage={"total_tokens": 0},
            finish_reason="stop",
        )


def run_smoke(*, settings: Settings | None = None, real_runtime: bool = False) -> SmokeRunSummary:
    resolved_settings = settings or Settings()
    temp_dir = Path(tempfile.mkdtemp(prefix="rolerag-smoke-"))
    database_path = temp_dir / "smoke.db"
    loader = FileDataLoader()
    embedding_provider = DeterministicKeywordEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    provider = SmokeTestProvider()
    connection = connect_sqlite(database_path)
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    turn_repository = SQLiteTurnRepository(connection)
    memory_repository = SQLiteMemoryRepository(connection)
    memory_store = MemoryEpisodeStore(memory_repository=memory_repository)
    recent_dialogue_store = RecentDialogueStore(
        turn_repository=turn_repository,
        recent_turns=resolved_settings.recent_dialogue_turns,
    )
    retriever = ActorContextRetriever(
        retriever=Retriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            default_top_k=resolved_settings.rag_default_top_k,
        )
    )
    orchestrator = TurnOrchestrator(
        loader=loader,
        provider=provider,
        cloud_provider=None,
        critic_agent=CriticAgent(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=recent_dialogue_store,
        memory_store=memory_store,
        memory_curator=MemoryCurator(),
        memory_indexer=MemoryIndexer(
            memory_store=memory_store,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        ),
        actor_context_retriever=retriever,
        config=TurnOrchestratorConfig(
            local_model=resolved_settings.local_llm_model,
            cloud_model=resolved_settings.cloud_llm_model,
            local_max_tokens=resolved_settings.local_llm_max_tokens,
            cloud_max_tokens=resolved_settings.cloud_llm_max_tokens,
            local_temperature=resolved_settings.local_llm_temperature,
            cloud_temperature=resolved_settings.cloud_llm_temperature,
            cloud_mode=resolved_settings.cloud_mode,
            retrieval_top_k=resolved_settings.rag_default_top_k,
            max_retrieved_chunk_chars=resolved_settings.rag_max_retrieved_chunk_chars,
        ),
    )
    steps: list[DiagnosticCheck] = []
    session_id: str | None = None
    actor_response = ""
    warnings: list[str] = []
    memory_written = False
    persisted_turns = 0
    persisted_memories = 0
    indexed_memories = 0
    ingested_chunk_count = 0
    retrieval_selected: list[RetrievalSelectionSummary] = []
    live_diagnostics = None
    try:
        loader.load_world("demo_world")
        loader.load_persona("archivist")
        loader.load_scene("rose-gallery")
        steps.append(_pass_step("demo_data", "Demo world, persona, and scene loaded."))

        ingestion_result = ingest_document(
            IngestionRequest(
                path=Path("data/documents/demo_lore.md"),
                collection=RagCollection.CANON_LORE,
                source_type="lore",
                visibility=Visibility.PLAYER,
                world_id="demo_world",
                tags=["demo", "palace"],
            ),
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            chunking_config=ChunkingConfig(
                chunk_size_chars=resolved_settings.rag_chunk_size_chars,
                chunk_overlap_chars=resolved_settings.rag_chunk_overlap_chars,
            ),
        )
        ingested_chunk_count = ingestion_result.chunk_count
        steps.append(_pass_step("ingestion", f"Ingested {ingested_chunk_count} lore chunks."))

        session = orchestrator.create_session(
            world_id="demo_world",
            scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
            session_id="smoke-session",
        )
        session_id = session.id
        steps.append(_pass_step("session", f"Created session {session.id}."))

        result = asyncio.run(
            orchestrator.run_turn(
                turn_input=TurnInput(
                    session_id=session.id,
                    message="I promise I will return before dawn. What do I need to know?",
                )
            )
        )
        actor_response = result.text
        warnings = result.warnings
        memory_written = result.memory_written
        persisted_turns = turn_repository.count_turns(session.id)
        persisted_memories = len(memory_repository.list_memories_for_session(session.id))
        steps.append(_pass_step("turn", "Completed one fake-provider turn and persisted it."))

        indexed_memories = _count_collection_entries(vector_store, RagCollection.SESSION_MEMORY)
        if memory_written:
            steps.append(
                _pass_step(
                    "memory",
                    f"Persisted {persisted_memories} memories and indexed {indexed_memories}.",
                )
            )
        else:
            steps.append(
                DiagnosticCheck(
                    name="memory",
                    status=DiagnosticStatus.WARN,
                    message="Smoke turn completed, but no memory was written.",
                    hint="Inspect memory curator output and the fake smoke provider behavior.",
                )
            )

        persona = loader.load_persona("archivist")
        scene = loader.load_scene("rose-gallery")
        query = build_retrieval_query(
            user_message="What did I promise the archivist?",
            scene=scene,
            persona=persona,
            recent_turns=recent_dialogue_store.load_recent_dialogue(session.id),
        )
        retrieval_result = retriever.retrieve_for_actor_with_diagnostics(
            query=query,
            lexical_query="What did I promise the archivist?",
            world_id="demo_world",
            session_id=session.id,
            persona_id="archivist",
            scene_id="rose-gallery",
            top_k=resolved_settings.rag_default_top_k,
        )
        retrieval_selected = _to_retrieval_summary(retrieval_result.diagnostics)
        steps.append(
            _pass_step(
                "retrieval",
                f"Retrieved {len(retrieval_selected)} metadata-only diagnostic selections.",
            )
        )

        if real_runtime:
            live_diagnostics = build_runtime_diagnostics(
                resolved_settings,
                check_qdrant=True,
                check_local_provider=True,
            )
            steps.extend(live_diagnostics.checks)
    except Exception as exc:
        steps.append(
            DiagnosticCheck(
                name="smoke_run",
                status=DiagnosticStatus.FAIL,
                message=f"Smoke run failed: {exc}",
                hint="Run `python -m app.cli doctor` to isolate local environment issues.",
            )
        )
    finally:
        connection.close()

    all_checks = list(steps)
    status = _overall_status(all_checks)
    return SmokeRunSummary(
        status=status,
        mode="real-runtime" if real_runtime else "safe",
        session_id=session_id,
        temp_database_path=str(database_path),
        ingested_chunk_count=ingested_chunk_count,
        persisted_turns=persisted_turns,
        memory_written=memory_written,
        persisted_memories=persisted_memories,
        indexed_memories=indexed_memories,
        retrieval_selected_count=len(retrieval_selected),
        actor_response=actor_response,
        warnings=warnings,
        steps=steps,
        retrieval_selected=retrieval_selected,
        live_diagnostics=live_diagnostics,
    )


def _to_retrieval_summary(diagnostics: RetrievalDiagnostics) -> list[RetrievalSelectionSummary]:
    return [
        RetrievalSelectionSummary(
            id=item.id,
            source=item.source,
            source_type=item.source_type,
            collection=item.collection.value,
            visibility=item.visibility.value,
            selected_rank=item.selected_rank if item.selected_rank is not None else rank,
            original_score=item.original_score,
            adjusted_score=item.adjusted_score,
            tags=item.tags,
        )
        for rank, item in enumerate(diagnostics.selected, start=1)
    ]


def _count_collection_entries(vector_store: InMemoryVectorStore, collection: RagCollection) -> int:
    entries = getattr(vector_store, "_entries", {})
    collection_entries = entries.get(collection, [])
    return len(collection_entries)


def _pass_step(name: str, message: str) -> DiagnosticCheck:
    return DiagnosticCheck(name=name, status=DiagnosticStatus.PASS, message=message)


def _overall_status(checks: list[DiagnosticCheck]) -> DiagnosticStatus:
    if any(check.status == DiagnosticStatus.FAIL for check in checks):
        return DiagnosticStatus.FAIL
    if any(check.status == DiagnosticStatus.WARN for check in checks):
        return DiagnosticStatus.WARN
    return DiagnosticStatus.PASS
