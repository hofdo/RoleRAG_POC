from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from app.agents.critic_agent import CriticAgentOutputError
from app.agents.memory_curator import MemoryCuratorOutputError
from app.domain import (
    CriticResult,
    MemoryCuratorResult,
    PersonaCard,
    RetrievedChunk,
    SceneState,
    SessionState,
    StoredTurn,
    TurnInput,
    Visibility,
)
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from app.llm.router import CloudMode, ModelProviderName, ModelRoute
from app.memory import MemoryEpisodeStore
from app.orchestration.context_budget import ContextBudget
from app.orchestration.stages import (
    LoadedTurnContext,
    TurnCritiqueStage,
    TurnMemoryStage,
    TurnPersistenceStage,
    TurnRetrievalStage,
    TurnRoutingStage,
)


def _context() -> LoadedTurnContext:
    return LoadedTurnContext(
        session=SessionState(
            id="session",
            world_id="world",
            active_scene_id="scene",
            active_persona_id="persona",
            player_name="Player",
        ),
        persona=PersonaCard(
            id="persona",
            name="Archivist",
            role="npc",
            public_description="A careful archivist.",
            speaking_style="Precise.",
        ),
        scene=SceneState(
            id="scene",
            title="Gallery",
            location="Palace",
            player_visible_summary="A mirrored gallery.",
        ),
        recent_turns=(),
    )


def _routing(*, cloud_mode: CloudMode = CloudMode.ASK) -> TurnRoutingStage:
    return TurnRoutingStage(
        local_model="local",
        cloud_model="cloud",
        local_max_tokens=700,
        local_structured_max_tokens=350,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        cloud_mode=cloud_mode,
    )


def test_retrieval_stage_uses_only_player_visible_scores_for_confidence() -> None:
    chunks = [
        RetrievedChunk(
            id="visible",
            source="lore.md",
            source_type="lore",
            text="Visible",
            score=0.6,
            visibility=Visibility.PLAYER,
        ),
        RetrievedChunk(
            id="hidden",
            source="gm.md",
            source_type="lore",
            text="Hidden",
            score=0.99,
            visibility=Visibility.GM,
        ),
    ]
    retriever = SimpleNamespace(retrieve_for_actor=lambda **_: chunks)
    stage = TurnRetrievalStage(
        actor_context_retriever=retriever,
        context_budget=ContextBudget(retrieved_chunks=3),
    )

    result = stage.run(
        turn_input=TurnInput(session_id="session", message="Look around."),
        context=_context(),
    )

    assert result.chunks == tuple(chunks)
    assert result.confidence == 0.6
    assert result.warnings == ()
    with pytest.raises(AttributeError):
        result.confidence = 0.1  # type: ignore[misc]


def test_retrieval_stage_degrades_to_warning() -> None:
    def fail(**_: object) -> list[RetrievedChunk]:
        raise RuntimeError("offline")

    stage = TurnRetrievalStage(
        actor_context_retriever=SimpleNamespace(retrieve_for_actor=fail),
        context_budget=ContextBudget(),
    )

    result = stage.run(
        turn_input=TurnInput(session_id="session", message="Look around."),
        context=_context(),
    )

    assert result.chunks == ()
    assert result.confidence is None
    assert result.warnings == ("retrieval skipped: offline",)


def test_routing_stage_normalizes_confirmation_required_actor_route() -> None:
    result = _routing().actor(
        turn_input=TurnInput(
            session_id="session",
            message="Use cloud.",
            user_requested_cloud=True,
        ),
        scene=_context().scene,
        retrieval_confidence=None,
    )

    assert result.route.provider == ModelProviderName.LOCAL
    assert result.route.reason == "confirmation required before cloud route: user requested cloud"
    assert result.warnings == (
        "cloud actor skipped: confirmation required for cloud (user requested cloud)",
    )


def test_persistence_stage_appends_before_updating_session_activity() -> None:
    calls: list[tuple[str, object]] = []
    created_at = datetime(2026, 1, 2, tzinfo=UTC)
    route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local",
        max_tokens=700,
        temperature=0.75,
        reason="default local route",
    )
    stored_turn = StoredTurn(
        id=1,
        session_id="session",
        turn_index=1,
        scene_id="scene",
        persona_id="persona",
        user_message="Question",
        assistant_message="Answer",
        route=route,
        created_at=created_at,
    )

    class TurnRepository:
        def append_turn(self, **_: object) -> StoredTurn:
            calls.append(("append", created_at))
            return stored_turn

    class SessionRepository:
        def update_session_activity(
            self,
            session_id: str,
            *,
            updated_at: datetime,
        ) -> None:
            calls.append((session_id, updated_at))

    stage = TurnPersistenceStage(
        session_repository=SessionRepository(),  # type: ignore[arg-type]
        turn_repository=TurnRepository(),  # type: ignore[arg-type]
    )
    stage.run(
        session=_context().session,
        user_message="Question",
        assistant_message="Answer",
        route=route,
    )

    assert calls == [("append", created_at), ("session", created_at)]


class RecordingFailureSink:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def record(
        self,
        *,
        task: str,
        category: str,
        raw_text: str,
        model: str,
        session_id: str | None = None,
    ) -> None:
        self.records.append(
            {
                "task": task,
                "category": category,
                "raw_text": raw_text,
                "model": model,
                "session_id": session_id,
            }
        )


class UnusedProvider(LlmProvider):
    async def generate(self, request: LlmRequest) -> LlmResponse:
        raise AssertionError("provider must not be called")


class FailingCriticAgent:
    def __init__(self, error: CriticAgentOutputError) -> None:
        self.error = error

    async def evaluate(
        self,
        *,
        provider: LlmProvider,
        route: ModelRoute,
        persona: PersonaCard,
        scene: SceneState,
        user_message: str,
        draft: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> CriticResult:
        raise self.error

    def build_local_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        rejected_draft: str,
        issues: list[str],
        repair_instruction: str | None,
    ) -> list[LlmMessage]:
        return []

    def build_cloud_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        issues: list[str],
    ) -> list[LlmMessage]:
        return []


class FailingCurator:
    def __init__(self, error: MemoryCuratorOutputError) -> None:
        self.error = error

    async def curate(
        self,
        *,
        provider: LlmProvider,
        route: ModelRoute,
        session: SessionState,
        scene: SceneState,
        persona: PersonaCard,
        user_message: str,
        assistant_message: str,
    ) -> MemoryCuratorResult:
        raise self.error


@pytest.mark.asyncio
async def test_critique_stage_categorizes_structured_failure_and_records_raw_text() -> None:
    sink = RecordingFailureSink()
    stage = TurnCritiqueStage(
        provider=UnusedProvider(),
        critic_agent=FailingCriticAgent(
            CriticAgentOutputError(
                "invalid structured output",
                category="schema",
                raw_text='{"accepted": "maybe"}',
            )
        ),
        routing_stage=_routing(),
        failure_sink=sink,
    )
    context = _context()

    result = await stage.run(
        persona=context.persona,
        scene=context.scene,
        user_message="Hello.",
        draft="Good evening.",
        retrieved_chunks=(),
    )

    assert result.critique is None
    assert result.warnings == ("critic skipped: invalid structured output (schema)",)
    assert sink.records == [
        {
            "task": "critic",
            "category": "schema",
            "raw_text": '{"accepted": "maybe"}',
            "model": "local",
            "session_id": None,
        }
    ]


@pytest.mark.asyncio
async def test_critique_stage_tolerates_failure_sink_errors() -> None:
    class BrokenSink:
        def record(
            self,
            *,
            task: str,
            category: str,
            raw_text: str,
            model: str,
            session_id: str | None = None,
        ) -> None:
            raise OSError("disk full")

    stage = TurnCritiqueStage(
        provider=UnusedProvider(),
        critic_agent=FailingCriticAgent(
            CriticAgentOutputError(
                "invalid structured output", category="parse", raw_text="not json"
            )
        ),
        routing_stage=_routing(),
        failure_sink=BrokenSink(),
    )
    context = _context()

    result = await stage.run(
        persona=context.persona,
        scene=context.scene,
        user_message="Hello.",
        draft="Good evening.",
        retrieved_chunks=(),
    )

    assert result.critique is None
    assert result.warnings == (
        "critic skipped: invalid structured output (parse)",
        "structured failure log skipped: disk full",
    )


@pytest.mark.asyncio
async def test_memory_stage_categorizes_structured_failure_and_records_raw_text() -> None:
    sink = RecordingFailureSink()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, SimpleNamespace()),
        memory_curator=FailingCurator(
            MemoryCuratorOutputError(
                "invalid structured output", category="parse", raw_text="not json"
            )
        ),
        memory_indexer=None,
        routing_stage=_routing(),
        failure_sink=sink,
    )
    context = _context()

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="Hello.",
        assistant_message="Good evening.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert result.memory_written is False
    assert result.warnings == ("memory curation skipped: invalid structured output (parse)",)
    assert sink.records == [
        {
            "task": "memory_extraction",
            "category": "parse",
            "raw_text": "not json",
            "model": "local",
            "session_id": "session",
        }
    ]


def test_retrieval_stage_exposes_candidate_diagnostics_when_retriever_supports_them() -> None:
    from app.rag.diagnostics import (
        ChunkRetrievalDiagnostic,
        RetrievalDiagnostics,
        RetrievalResult,
    )
    from app.rag.models import RagCollection

    chunk = RetrievedChunk(
        id="memory-1",
        source="memory_episode:memory-1",
        source_type="session_memory",
        text="The player promised to return before dawn.",
        score=0.6,
        visibility=Visibility.PLAYER,
    )
    rag_diagnostics = RetrievalDiagnostics(
        query="What did I promise?",
        selected=[
            ChunkRetrievalDiagnostic(
                id="memory-1",
                source="memory_episode:memory-1",
                source_type="session_memory",
                collection=RagCollection.SESSION_MEMORY,
                visibility=Visibility.PLAYER,
                tags=["promise"],
                original_score=0.6,
                adjusted_score=0.7,
                applied_boosts={"collection": 0.08, "session": 0.02},
                selected_rank=1,
            )
        ],
        rejected=[
            ChunkRetrievalDiagnostic(
                id="lore-1",
                source="lore.md",
                source_type="lore",
                collection=RagCollection.CANON_LORE,
                visibility=Visibility.PLAYER,
                tags=[],
                original_score=0.2,
                adjusted_score=0.2,
                applied_boosts={},
                selected_rank=None,
            )
        ],
    )
    retriever = SimpleNamespace(
        retrieve_for_actor_with_diagnostics=lambda **_: RetrievalResult(
            chunks=[chunk], diagnostics=rag_diagnostics
        )
    )
    stage = TurnRetrievalStage(
        actor_context_retriever=retriever,
        context_budget=ContextBudget(retrieved_chunks=3),
    )

    result = stage.run(
        turn_input=TurnInput(session_id="session", message="What did I promise?"),
        context=_context(),
    )

    assert result.chunks == (chunk,)
    assert result.confidence == 0.6
    assert result.diagnostics is not None
    assert result.diagnostics.query == "What did I promise?"
    selected = result.diagnostics.selected[0]
    assert selected.id == "memory-1"
    assert selected.collection == "session_memory"
    assert selected.selected_rank == 1
    assert selected.applied_boosts == {"collection": 0.08, "session": 0.02}
    rejected = result.diagnostics.rejected[0]
    assert rejected.id == "lore-1"
    assert rejected.collection == "canon_lore"
    assert rejected.selected_rank is None


def test_retrieval_stage_returns_no_diagnostics_for_plain_retriever() -> None:
    chunks = [
        RetrievedChunk(
            id="visible",
            source="lore.md",
            source_type="lore",
            text="Visible",
            score=0.6,
            visibility=Visibility.PLAYER,
        )
    ]
    stage = TurnRetrievalStage(
        actor_context_retriever=SimpleNamespace(retrieve_for_actor=lambda **_: chunks),
        context_budget=ContextBudget(retrieved_chunks=3),
    )

    result = stage.run(
        turn_input=TurnInput(session_id="session", message="Look around."),
        context=_context(),
    )

    assert result.chunks == tuple(chunks)
    assert result.diagnostics is None


def test_retrieval_stage_passes_player_message_as_lexical_query() -> None:
    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> list[RetrievedChunk]:
        captured.update(kwargs)
        return []

    stage = TurnRetrievalStage(
        actor_context_retriever=SimpleNamespace(retrieve_for_actor=capture),
        context_budget=ContextBudget(retrieved_chunks=3),
    )

    stage.run(
        turn_input=TurnInput(session_id="session", message="What did I promise?"),
        context=_context(),
    )

    assert captured["lexical_query"] == "What did I promise?"
    assert "What did I promise?" in str(captured["query"])


class RecordingMemoryStore:
    def __init__(self) -> None:
        self.persisted: list[Any] = []

    def persist_memories(self, *, session_id: str, memories: list[Any]) -> list[Any]:
        from app.domain import MemoryEpisode

        episodes = [
            MemoryEpisode(
                id=f"memory-{len(self.persisted) + index + 1}",
                session_id=session_id,
                scene_id=candidate.scene_id or "scene",
                actor_id=candidate.actor_id,
                summary=candidate.summary,
                importance=candidate.importance,
                visibility=candidate.visibility,
                tags=list(candidate.tags),
            )
            for index, candidate in enumerate(memories)
        ]
        self.persisted.extend(episodes)
        return episodes


class DecliningCurator:
    async def curate(self, **_: object) -> Any:
        from app.domain import MemoryCuratorResult

        return MemoryCuratorResult(write_memory=False, memories=[], reason="nothing durable")


class CoveringCurator:
    async def curate(self, **_: object) -> Any:
        from app.domain import MemoryCandidate, MemoryCuratorResult, Visibility

        return MemoryCuratorResult(
            write_memory=True,
            memories=[
                MemoryCandidate(
                    summary="The player promised to return before dawn.",
                    visibility=Visibility.PLAYER,
                    importance=4,
                    tags=["promise"],
                    scene_id="scene",
                    actor_id="persona",
                )
            ],
            reason="explicit promise",
        )


@pytest.mark.asyncio
async def test_memory_stage_falls_back_to_deterministic_extraction_when_curator_fails() -> None:
    store = RecordingMemoryStore()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=FailingCurator(
            MemoryCuratorOutputError(
                "invalid structured output", category="parse", raw_text="not json"
            )
        ),
        memory_indexer=None,
        routing_stage=_routing(),
        failure_sink=None,
    )
    context = _context()

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I promise to return before dawn.",
        assistant_message="Iria nods once.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert result.memory_written is True
    assert len(store.persisted) == 1
    assert "return before dawn" in store.persisted[0].summary
    assert any("deterministic" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_memory_stage_adds_explicit_event_when_curator_declines_to_write() -> None:
    store = RecordingMemoryStore()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=DecliningCurator(),
        memory_indexer=None,
        routing_stage=_routing(),
    )
    context = _context()

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I promise to return before dawn.",
        assistant_message="Iria nods once.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert result.memory_written is True
    assert len(store.persisted) == 1
    assert "return before dawn" in store.persisted[0].summary


@pytest.mark.asyncio
async def test_memory_stage_does_not_duplicate_event_already_curated() -> None:
    store = RecordingMemoryStore()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=CoveringCurator(),
        memory_indexer=None,
        routing_stage=_routing(),
    )
    context = _context()

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I promise to return before dawn.",
        assistant_message="Iria nods once.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert result.memory_written is True
    assert len(store.persisted) == 1
    assert store.persisted[0].summary == "The player promised to return before dawn."
