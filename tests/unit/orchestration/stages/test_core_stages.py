from __future__ import annotations

from collections.abc import Sequence
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


def test_routing_stage_keeps_confirmation_required_cloud_route() -> None:
    result = _routing().actor(
        turn_input=TurnInput(
            session_id="session",
            message="Use cloud.",
            user_requested_cloud=True,
        ),
        scene=_context().scene,
        retrieval_confidence=None,
    )

    assert result.route.provider == ModelProviderName.CLOUD
    assert result.route.requires_user_confirmation is True
    assert result.route.reason == "user requested cloud"
    assert result.warnings == ()


def test_routing_stage_cloud_confirmed_clears_confirmation_flag() -> None:
    result = _routing().actor(
        turn_input=TurnInput(
            session_id="session",
            message="Use cloud.",
            user_requested_cloud=True,
            cloud_confirmed=True,
        ),
        scene=_context().scene,
        retrieval_confidence=None,
    )

    assert result.route.provider == ModelProviderName.CLOUD
    assert result.route.requires_user_confirmation is False


def test_routing_stage_force_local_overrides_cloud_request() -> None:
    result = _routing().actor(
        turn_input=TurnInput(
            session_id="session",
            message="Use cloud.",
            user_requested_cloud=True,
            force_local=True,
        ),
        scene=_context().scene,
        retrieval_confidence=None,
    )

    assert result.route.provider == ModelProviderName.LOCAL
    assert result.route.reason == "user declined cloud"
    assert result.warnings == ()


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


class _EchoingCriticAgent:
    def __init__(self, result: CriticResult) -> None:
        self.result = result

    async def evaluate(self, **_: object) -> CriticResult:
        return self.result

    def build_local_repair_messages(self, **_: object) -> list[Any]:
        return []

    def build_cloud_repair_messages(self, **_: object) -> list[Any]:
        return []


@pytest.mark.asyncio
async def test_critique_stage_redacts_hidden_fact_leak_from_critic_output() -> None:
    persona = PersonaCard(
        id="archivist",
        name="Iria",
        role="npc",
        public_description="A composed archivist.",
        speaking_style="Dry.",
        secrets=["She forged one inventory ledger."],
    )
    scene = SceneState(
        id="rose-gallery",
        title="Rose Gallery",
        location="Palace",
        player_visible_summary="Courtiers drift between mirrors.",
        gm_private_summary="The regent's spy is already in the room.",
    )
    leaking = CriticResult(
        accepted=False,
        issues=["The draft nearly admits she forged one inventory ledger."],
        repair_instruction="Do not reveal that the regent's spy is already in the room.",
    )
    stage = TurnCritiqueStage(
        provider=UnusedProvider(),
        critic_agent=_EchoingCriticAgent(leaking),
        routing_stage=_routing(),
    )

    result = await stage.run(
        persona=persona,
        scene=scene,
        user_message="What are you hiding?",
        draft="...",
        retrieved_chunks=(),
    )

    assert result.warnings == ("critic output redacted: prevented hidden-fact leak",)
    assert result.critique is not None
    assert result.critique.accepted is False
    assert "forged one inventory ledger" not in result.critique.issues[0].lower()
    assert result.critique.repair_instruction is not None
    assert "spy is already in the room" not in result.critique.repair_instruction.lower()


@pytest.mark.asyncio
async def test_critique_stage_passes_clean_critic_output_through_unchanged() -> None:
    persona = PersonaCard(
        id="archivist",
        name="Iria",
        role="npc",
        public_description="A composed archivist.",
        speaking_style="Dry.",
        secrets=["She forged one inventory ledger."],
    )
    clean = CriticResult(
        accepted=False,
        issues=["The tone is too generic."],
        repair_instruction="Answer the question concretely.",
    )
    stage = TurnCritiqueStage(
        provider=UnusedProvider(),
        critic_agent=_EchoingCriticAgent(clean),
        routing_stage=_routing(),
    )

    result = await stage.run(
        persona=persona,
        scene=_context().scene,
        user_message="What are you hiding?",
        draft="...",
        retrieved_chunks=(),
    )

    assert result.warnings == ()
    assert result.critique == clean


def test_critique_stage_risk_gate_honors_high_scene_complexity_override() -> None:
    from app.llm.router import ModelProviderName

    clean = CriticResult(accepted=True)
    default_stage = TurnCritiqueStage(
        provider=UnusedProvider(),
        critic_agent=_EchoingCriticAgent(clean),
        routing_stage=_routing(),
    )
    raised_stage = TurnCritiqueStage(
        provider=UnusedProvider(),
        critic_agent=_EchoingCriticAgent(clean),
        routing_stage=_routing(),
        high_scene_complexity=5,
    )

    # Complexity 4 is risky at the default threshold (4) but not when raised to 5.
    assert default_stage._is_risky_turn(
        validator_flagged=False,
        retrieval_confidence=0.9,
        scene_complexity=4,
        route_provider=ModelProviderName.LOCAL,
    ) is True
    assert raised_stage._is_risky_turn(
        validator_flagged=False,
        retrieval_confidence=0.9,
        scene_complexity=4,
        route_provider=ModelProviderName.LOCAL,
    ) is False


def test_generation_stage_warns_on_silent_prompt_truncation() -> None:
    import dataclasses
    from datetime import UTC, datetime

    from app.domain import RetrievedChunk, Visibility
    from app.orchestration.context_budget import ContextBudget
    from app.orchestration.stages.generation import TurnGenerationStage
    from app.orchestration.stages.retrieval import RetrievalStageResult

    route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local",
        max_tokens=100,
        temperature=0.5,
        reason="x",
    )
    long_turn = StoredTurn(
        id=1,
        session_id="session",
        turn_index=1,
        scene_id="scene",
        persona_id="persona",
        user_message="u" * 50,
        assistant_message="a",
        route=route,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    context = dataclasses.replace(_context(), recent_turns=(long_turn,))
    retrieval = RetrievalStageResult(
        chunks=(
            RetrievedChunk(
                id="c1",
                source="lore",
                source_type="canon_lore",
                text="x" * 50,
                score=0.9,
                visibility=Visibility.PLAYER,
            ),
        ),
        confidence=0.9,
        warnings=(),
    )
    stage = TurnGenerationStage(
        provider=UnusedProvider(),
        cloud_provider=None,
        routing_stage=_routing(),
        context_budget=ContextBudget(retrieved_chunks=5, max_retrieved_chunk_chars=20),
        recent_dialogue_max_message_chars=20,
    )

    warnings = stage._context_truncation_warnings(context=context, retrieval=retrieval)

    assert any("recent dialogue truncated" in warning for warning in warnings)
    assert any("retrieved context truncated" in warning for warning in warnings)


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

    def list_memories_for_session(self, session_id: str) -> list[Any]:
        return [episode for episode in self.persisted if episode.session_id == session_id]


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


class ParaphraseCurator:
    async def curate(self, **_: object) -> Any:
        from app.domain import MemoryCandidate, MemoryCuratorResult, Visibility

        return MemoryCuratorResult(
            write_memory=True,
            memories=[
                MemoryCandidate(
                    # Shares almost no content terms with the seeded memory, so
                    # the lexical pass keeps it; only the semantic pass can drop it.
                    summary="At first light the visitor will come back.",
                    visibility=Visibility.PLAYER,
                    importance=4,
                    tags=["promise"],
                    scene_id="scene",
                    actor_id="persona",
                )
            ],
            reason="paraphrase",
        )


class _MappingEmbeddingProvider:
    dimension = 2

    def embed_text(self, text: str) -> list[float]:
        intent_group = {
            "The player promised to return before dawn.",
            "At first light the visitor will come back.",
        }
        return [1.0, 0.0] if text in intent_group else [0.0, 1.0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


def _seed_promise(store: RecordingMemoryStore, session_id: str) -> None:
    from app.domain import MemoryCandidate, Visibility

    store.persist_memories(
        session_id=session_id,
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
    )


@pytest.mark.asyncio
async def test_memory_stage_semantic_dedup_drops_paraphrase_when_enabled() -> None:
    store = RecordingMemoryStore()
    context = _context()
    _seed_promise(store, context.session.id)
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=ParaphraseCurator(),
        memory_indexer=None,
        routing_stage=_routing(),
        embedding_provider=_MappingEmbeddingProvider(),
        write_dedup_cosine_threshold=0.9,
    )

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I ask Iria about the weather outside.",
        assistant_message="Iria glances toward the frosted glass.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert result.memory_written is False
    assert len(store.persisted) == 1
    assert any("semantic memory dedup dropped" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_memory_stage_semantic_dedup_is_inert_by_default() -> None:
    store = RecordingMemoryStore()
    context = _context()
    _seed_promise(store, context.session.id)
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=ParaphraseCurator(),
        memory_indexer=None,
        routing_stage=_routing(),
    )

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I ask Iria about the weather outside.",
        assistant_message="Iria glances toward the frosted glass.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert result.memory_written is True
    assert len(store.persisted) == 2


@pytest.mark.asyncio
async def test_memory_stage_drops_candidates_covered_by_persisted_memories() -> None:
    store = RecordingMemoryStore()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=CoveringCurator(),
        memory_indexer=None,
        routing_stage=_routing(),
    )
    context = _context()

    async def run_turn() -> Any:
        return await stage.run(
            session=context.session,
            scene=context.scene,
            persona=context.persona,
            user_message="I promise to return before dawn.",
            assistant_message="Iria nods once.",
            retrieval_confidence=None,
            scene_complexity=1,
        )

    first = await run_turn()
    assert first.memory_written is True
    assert len(store.persisted) == 1

    second = await run_turn()
    assert second.memory_written is False
    assert len(store.persisted) == 1
    assert any("memory dedup dropped" in warning for warning in second.warnings)


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


class AcceptingCriticAgent(FailingCriticAgent):
    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(self, **_: object) -> CriticResult:
        self.calls += 1
        return CriticResult(accepted=True)


@pytest.mark.asyncio
async def test_critique_stage_auto_gating_skips_low_risk_turn() -> None:
    critic = AcceptingCriticAgent()
    stage = TurnCritiqueStage(
        provider=UnusedProvider(),
        critic_agent=critic,
        routing_stage=_routing(),
        gating="auto",
    )
    context = _context()

    result = await stage.run(
        persona=context.persona,
        scene=context.scene,
        user_message="Hello.",
        draft="Good evening.",
        retrieved_chunks=(),
        validator_flagged=False,
        retrieval_confidence=0.9,
        scene_complexity=1,
        route_provider=ModelProviderName.LOCAL,
    )

    assert result.critique is None
    assert result.warnings == ("critic gated: low-risk turn",)
    assert critic.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "signals",
    [
        {"validator_flagged": True},
        {"retrieval_confidence": 0.2},
        {"retrieval_confidence": None},
        {"scene_complexity": 4},
        {"route_provider": ModelProviderName.CLOUD},
    ],
)
async def test_critique_stage_auto_gating_runs_critic_on_risk_signals(
    signals: dict[str, object],
) -> None:
    critic = AcceptingCriticAgent()
    stage = TurnCritiqueStage(
        provider=UnusedProvider(),
        critic_agent=critic,
        routing_stage=_routing(),
        gating="auto",
    )
    context = _context()
    arguments: dict[str, Any] = {
        "validator_flagged": False,
        "retrieval_confidence": 0.9,
        "scene_complexity": 1,
        "route_provider": ModelProviderName.LOCAL,
    }
    arguments.update(signals)

    result = await stage.run(
        persona=context.persona,
        scene=context.scene,
        user_message="Hello.",
        draft="Good evening.",
        retrieved_chunks=(),
        **arguments,
    )

    assert result.critique is not None
    assert critic.calls == 1


@pytest.mark.asyncio
async def test_critique_stage_always_mode_runs_critic_without_signals() -> None:
    critic = AcceptingCriticAgent()
    stage = TurnCritiqueStage(
        provider=UnusedProvider(),
        critic_agent=critic,
        routing_stage=_routing(),
    )
    context = _context()

    result = await stage.run(
        persona=context.persona,
        scene=context.scene,
        user_message="Hello.",
        draft="Good evening.",
        retrieved_chunks=(),
        validator_flagged=False,
        retrieval_confidence=0.9,
        scene_complexity=1,
        route_provider=ModelProviderName.LOCAL,
    )

    assert result.critique is not None
    assert critic.calls == 1


class CountingCurator:
    def __init__(self) -> None:
        self.calls = 0

    async def curate(self, **_: object) -> Any:
        from app.domain import MemoryCuratorResult

        self.calls += 1
        return MemoryCuratorResult(write_memory=False, memories=[], reason="nothing durable")


@pytest.mark.asyncio
async def test_memory_stage_auto_gating_skips_curator_without_durable_signals() -> None:
    store = RecordingMemoryStore()
    curator = CountingCurator()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=curator,
        memory_indexer=None,
        routing_stage=_routing(),
        gating="auto",
    )
    context = _context()

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="Good evening, Iria.",
        assistant_message="Good evening to you as well.",
        retrieval_confidence=0.9,
        scene_complexity=1,
    )

    assert curator.calls == 0
    assert result.memory_written is False
    assert result.warnings == ("memory curation gated: no durable-event signals",)
    assert store.persisted == []


@pytest.mark.asyncio
async def test_memory_stage_auto_gating_runs_curator_for_explicit_promise() -> None:
    store = RecordingMemoryStore()
    curator = CountingCurator()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=curator,
        memory_indexer=None,
        routing_stage=_routing(),
        gating="auto",
    )
    context = _context()

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I promise I will return before dawn.",
        assistant_message="I will hold you to it.",
        retrieval_confidence=0.9,
        scene_complexity=1,
    )

    assert curator.calls == 1
    assert result.memory_written is True


@pytest.mark.asyncio
async def test_memory_stage_auto_gating_runs_curator_for_assistant_durable_terms() -> None:
    store = RecordingMemoryStore()
    curator = CountingCurator()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=curator,
        memory_indexer=None,
        routing_stage=_routing(),
        gating="auto",
    )
    context = _context()

    await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="And then?",
        assistant_message="I swear on the archive that the ledger stays sealed.",
        retrieval_confidence=0.9,
        scene_complexity=1,
    )

    assert curator.calls == 1
