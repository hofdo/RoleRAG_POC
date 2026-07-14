from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from app.agents.critic_agent import CriticAgentOutputError
from app.agents.memory_curator import MemoryCuratorOutputError
from app.domain import (
    CriticResult,
    MemoryCandidate,
    MemoryCuratorResult,
    MemoryEpisode,
    PersonaCard,
    RetrievedChunk,
    SceneState,
    SessionState,
    StoredTurn,
    TurnInput,
    Visibility,
)
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from app.llm.router import ModelProviderName, ModelRoute
from app.memory import MemoryEpisodeStore, RecentDialogueStore
from app.memory.consolidation import CONSOLIDATED_TAG
from app.memory.deterministic_extractor import DETERMINISTIC_EVENT_IMPORTANCE
from app.orchestration.context_budget import ContextBudget
from app.orchestration.stages import (
    LoadedTurnContext,
    TurnCritiqueStage,
    TurnMemoryStage,
    TurnPersistenceStage,
    TurnRetrievalStage,
    TurnRoutingStage,
    TurnSessionLoader,
)
from app.orchestration.stages.generation import GenerationStageResult, TurnGenerationStage
from app.orchestration.stages.retrieval import RetrievalStageResult
from app.orchestration.stages.routing import RoutingStageResult
from app.persistence import DemoWorldRecord
from app.rag.ranking import SliceQuotas


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


def _routing() -> TurnRoutingStage:
    return TurnRoutingStage(
        local_model="local",
        cloud_model="cloud",
        local_max_tokens=700,
        local_structured_max_tokens=350,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
    )


class _FakeSessionDataLoader:
    """Minimal TurnDataLoader: one world/persona/scene, real nominal domain types
    (a Protocol structurally matches, but its method RETURN types are concrete
    classes -- DemoWorldRecord/PersonaCard/SceneState -- so a SimpleNamespace stand-in
    would not satisfy mypy --strict there)."""

    def load_world(self, world_id: str) -> DemoWorldRecord:
        return DemoWorldRecord(
            id=world_id,
            name="Test World",
            default_scene_id="scene",
            persona_ids=["persona"],
            scene_ids=["scene"],
        )

    def load_persona(self, persona_id: str) -> PersonaCard:
        return PersonaCard(
            id=persona_id,
            name="Archivist",
            role="npc",
            public_description="A careful archivist.",
            speaking_style="Precise.",
        )

    def load_scene(self, scene_id: str) -> SceneState:
        return SceneState(
            id=scene_id,
            title="Gallery",
            location="Palace",
            player_visible_summary="A mirrored gallery.",
        )


def _build_session_loader(
    *,
    memories: Sequence[MemoryEpisode] = (),
    canon_tag_pinning: bool = False,
) -> TurnSessionLoader:
    """docs/26 §3.3 (#78) TurnSessionLoader wiring fixture. The memory store is a
    fake returning exactly the given ``memories`` -- full, deterministic control over
    each MemoryEpisode's created_at (unlike a real SQLite-backed store, whose
    wall-clock timestamps would make the stale-fact safeguard's ordering
    non-deterministic to test)."""
    session = SessionState(
        id="session",
        world_id="world",
        active_scene_id="scene",
        active_persona_id="persona",
        player_name="Player",
    )
    memory_store = MemoryEpisodeStore(
        memory_repository=cast(
            Any, SimpleNamespace(list_memories_for_session=lambda session_id: list(memories))
        )
    )
    return TurnSessionLoader(
        loader=_FakeSessionDataLoader(),
        loader_factory=None,
        content_root="data",
        session_repository=cast(Any, SimpleNamespace(get_session=lambda session_id: session)),
        recent_dialogue_store=RecentDialogueStore(
            turn_repository=cast(
                Any, SimpleNamespace(list_recent_turns=lambda session_id, limit: [])
            ),
            recent_turns=8,
        ),
        memory_store=memory_store,
        canon_repository=None,
        canon_importance_floor=4,
        canon_max_items=8,
        canon_max_chars=900,
        canon_tag_pinning=canon_tag_pinning,
    )


def _tagged_memory(
    memory_id: str,
    *,
    summary: str,
    importance: int = 2,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
) -> MemoryEpisode:
    return MemoryEpisode(
        id=memory_id,
        session_id="session",
        scene_id="scene",
        summary=summary,
        importance=importance,
        visibility=Visibility.PLAYER,
        tags=tags if tags is not None else ["rule"],
        created_at=created_at,
    )


def test_session_loader_flag_off_leaves_sub_floor_tagged_memory_unpinned() -> None:
    loader = _build_session_loader(
        memories=[
            _tagged_memory(
                "blue-seal", summary="The player will only trust messages with a blue wax seal."
            )
        ]
    )

    context = loader.load(TurnInput(session_id="session", message="hello"))

    assert context.standing_facts == ()
    assert context.warnings == ()


def test_session_loader_flag_on_widens_eligibility_into_context() -> None:
    loader = _build_session_loader(
        memories=[
            _tagged_memory(
                "blue-seal", summary="The player will only trust messages with a blue wax seal."
            )
        ],
        canon_tag_pinning=True,
    )

    context = loader.load(TurnInput(session_id="session", message="hello"))

    assert context.standing_facts == (
        "The player will only trust messages with a blue wax seal.",
    )
    assert context.warnings == ()


def test_session_loader_flag_on_surfaces_safeguard_drop_as_context_warning() -> None:
    loader = _build_session_loader(
        memories=[
            _tagged_memory(
                "seal-old",
                summary="The trust rule is a blue wax seal.",
                created_at=datetime(2026, 6, 1, tzinfo=UTC),
            ),
            _tagged_memory(
                "seal-new",
                summary="The trust rule is now a blue wax seal plus a knock at the door.",
                created_at=datetime(2026, 6, 2, tzinfo=UTC),
            ),
        ],
        canon_tag_pinning=True,
    )

    context = loader.load(TurnInput(session_id="session", message="hello"))

    assert context.standing_facts == (
        "The trust rule is now a blue wax seal plus a knock at the door.",
    )
    assert len(context.warnings) == 1
    assert "stale canon fact superseded" in context.warnings[0]


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


def test_routing_stage_actor_routes_on_the_given_provider() -> None:
    local_result = _routing().actor(provider=ModelProviderName.LOCAL, scene=_context().scene)
    cloud_result = _routing().actor(provider=ModelProviderName.CLOUD, scene=_context().scene)

    assert local_result.route.provider == ModelProviderName.LOCAL
    assert local_result.route.reason == "session provider: local"
    assert local_result.warnings == ()
    assert cloud_result.route.provider == ModelProviderName.CLOUD
    assert cloud_result.route.reason == "session provider: cloud"
    assert cloud_result.warnings == ()


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


def test_record_structured_failure_redacts_hidden_facts_from_raw_text() -> None:
    from app.llm.structured_output import StructuredOutputError
    from app.orchestration.stages.critique import record_structured_failure

    secret = "the regent ordered the poisoning at midnight"
    sink = RecordingFailureSink()
    error = StructuredOutputError(
        "bad", category="parse", raw_text=f'{{"issues": ["He admitted {secret}"]}}'
    )

    warnings = record_structured_failure(
        sink=sink, task="critic", error=error, model="local-model", hidden_facts=[secret]
    )

    assert warnings == ()
    recorded = sink.records[0]["raw_text"]
    assert isinstance(recorded, str)
    assert secret not in recorded
    assert "[redacted]" in recorded


def test_record_structured_failure_keeps_raw_text_without_hidden_facts() -> None:
    from app.llm.structured_output import StructuredOutputError
    from app.orchestration.stages.critique import record_structured_failure

    sink = RecordingFailureSink()
    error = StructuredOutputError("bad", category="parse", raw_text='{"issues": ["x"]}')

    record_structured_failure(sink=sink, task="critic", error=error, model="m")

    assert sink.records[0]["raw_text"] == '{"issues": ["x"]}'


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
        include_hidden: bool = True,
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

    async def consolidate(self, **_: object) -> str:
        raise AssertionError("consolidate not expected in this test")


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
        route_provider=ModelProviderName.LOCAL,
    )

    assert result.critique is None
    assert result.failed is True  # an errored critic fails the turn closed downstream
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
        route_provider=ModelProviderName.LOCAL,
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
        route_provider=ModelProviderName.LOCAL,
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


class _UsageScriptedProvider(LlmProvider):
    """Answers every request with a fixed piece of text and a scripted usage dict,
    so tests can assert exactly what GenerationStageResult.usage / warnings carry."""

    def __init__(self, *, usage: dict[str, int], text: str = "A short reply.") -> None:
        self.usage = usage
        self.text = text
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            text=self.text,
            provider="fake",
            model=request.model,
            usage=self.usage,
            finish_reason="stop",
        )


@dataclass(frozen=True)
class _GenerationRunArgs:
    turn_input: TurnInput
    context: LoadedTurnContext
    retrieval: RetrievalStageResult
    routing: RoutingStageResult


def _generation_run_args(*, message: str = "Hello there.") -> _GenerationRunArgs:
    route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=50,
        temperature=0.5,
        reason="test route",
    )
    return _GenerationRunArgs(
        turn_input=TurnInput(session_id="session", message=message),
        context=_context(),
        retrieval=RetrievalStageResult(chunks=(), confidence=0.9, warnings=()),
        routing=RoutingStageResult(route=route, scene_complexity=1, warnings=()),
    )


async def _run_generation_stage(
    stage: TurnGenerationStage, args: _GenerationRunArgs
) -> GenerationStageResult:
    return await stage.run(
        turn_input=args.turn_input,
        context=args.context,
        retrieval=args.retrieval,
        routing=args.routing,
    )


@pytest.mark.asyncio
async def test_generation_stage_run_threads_usage_from_response() -> None:
    provider = _UsageScriptedProvider(
        usage={"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50}
    )
    stage = TurnGenerationStage(
        provider=provider,
        cloud_provider=None,
        routing_stage=_routing(),
        context_budget=ContextBudget(),
        recent_dialogue_max_message_chars=900,
    )

    result = await _run_generation_stage(stage, _generation_run_args())

    assert result.usage == {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50}
    assert result.text == "A short reply."


@pytest.mark.asyncio
async def test_generation_stage_context_preflight_disabled_by_default() -> None:
    # model_context_window_tokens defaults to 0: no estimate, no warning, no
    # behavior change, no matter how large the prompt is.
    provider = _UsageScriptedProvider(usage={"total_tokens": 0})
    stage = TurnGenerationStage(
        provider=provider,
        cloud_provider=None,
        routing_stage=_routing(),
        context_budget=ContextBudget(),
        recent_dialogue_max_message_chars=900,
    )
    args = _generation_run_args(message="x" * 100_000)

    result = await _run_generation_stage(stage, args)

    assert not any("context preflight" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_generation_stage_context_preflight_fires_above_threshold() -> None:
    provider = _UsageScriptedProvider(usage={"total_tokens": 0})
    stage = TurnGenerationStage(
        provider=provider,
        cloud_provider=None,
        routing_stage=_routing(),
        context_budget=ContextBudget(),
        recent_dialogue_max_message_chars=900,
        model_context_window_tokens=100,
        context_warn_ratio=0.5,
    )
    # System prompt alone (persona/scene boilerplate) plus this is easily >200 chars
    # (~>50 estimated tokens vs a 50-token threshold).
    args = _generation_run_args(message="x" * 1000)

    result = await _run_generation_stage(stage, args)

    assert any("context preflight: estimated" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_generation_stage_context_preflight_silent_below_threshold() -> None:
    provider = _UsageScriptedProvider(usage={"total_tokens": 0})
    stage = TurnGenerationStage(
        provider=provider,
        cloud_provider=None,
        routing_stage=_routing(),
        context_budget=ContextBudget(),
        recent_dialogue_max_message_chars=900,
        model_context_window_tokens=1_000_000,
        context_warn_ratio=0.85,
    )

    result = await _run_generation_stage(stage, _generation_run_args())

    assert not any("context preflight" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_generation_stage_actual_usage_warning_fires_when_configured() -> None:
    provider = _UsageScriptedProvider(usage={"prompt_tokens": 900, "total_tokens": 950})
    stage = TurnGenerationStage(
        provider=provider,
        cloud_provider=None,
        routing_stage=_routing(),
        context_budget=ContextBudget(),
        recent_dialogue_max_message_chars=900,
        model_context_window_tokens=1000,
        context_warn_ratio=0.85,
    )

    result = await _run_generation_stage(stage, _generation_run_args())

    assert any(
        "context preflight: actual prompt_tokens 900" in warning for warning in result.warnings
    )


@pytest.mark.asyncio
async def test_generation_stage_actual_usage_warning_disabled_by_default() -> None:
    # Even a huge reported prompt_tokens must not warn when the window is disabled.
    provider = _UsageScriptedProvider(usage={"prompt_tokens": 999_999})
    stage = TurnGenerationStage(
        provider=provider,
        cloud_provider=None,
        routing_stage=_routing(),
        context_budget=ContextBudget(),
        recent_dialogue_max_message_chars=900,
    )

    result = await _run_generation_stage(stage, _generation_run_args())

    assert not any("context preflight" in warning for warning in result.warnings)


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
        route_provider=ModelProviderName.LOCAL,
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


def test_retrieval_stage_over_fetches_for_standing_facts_and_n3_dedup_margin() -> None:
    # docs/22 N3 review round: the over-fetch must cover both the C1 standing-facts
    # exclusion count AND a bounded worst-case margin for read-time normalized-text
    # dedup (retrieved_chunks - 1), or the prompt block can silently shrink below
    # budget when duplicate-text/distinct-id chunks land inside a fixed-size window.
    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> list[RetrievedChunk]:
        captured.update(kwargs)
        return []

    stage = TurnRetrievalStage(
        actor_context_retriever=SimpleNamespace(retrieve_for_actor=capture),
        context_budget=ContextBudget(retrieved_chunks=5),
    )
    context = _context()
    context = LoadedTurnContext(
        session=context.session,
        persona=context.persona,
        scene=context.scene,
        recent_turns=context.recent_turns,
        standing_facts=("fact one", "fact two"),
    )

    stage.run(
        turn_input=TurnInput(session_id="session", message="Look around."),
        context=context,
    )

    # 5 (budget) + 2 (standing facts) + 4 (N3 worst-case margin, retrieved_chunks - 1).
    assert captured["top_k"] == 11


def test_retrieval_stage_backfill_survives_duplicate_text_collisions_end_to_end() -> None:
    # Reproduces the reviewer's boundary case end to end: a ranked window sized only
    # for the pre-fix formula (retrieved_chunks + len(standing_facts)) contains two
    # duplicate-text pairs and one distinct chunk. Under the pre-fix top_k the fetch
    # window itself would be capped at 5, so the block would silently shrink to 3
    # selected chunks even though 5 distinct-text PLAYER chunks exist upstream. The
    # widened over-fetch must surface enough of them for the prompt step to still
    # reach the full budget.
    def make_chunk(chunk_id: str, text: str) -> RetrievedChunk:
        return RetrievedChunk(
            id=chunk_id,
            source="demo.md",
            source_type="lore",
            text=text,
            score=0.9,
            visibility=Visibility.PLAYER,
        )

    ranked_pool = [
        make_chunk("a1", "The regent distrusts the chancellor."),
        make_chunk("a2", "the regent distrusts THE chancellor."),
        make_chunk("b1", "A storm is expected by nightfall."),
        make_chunk("b2", "  A storm is expected by  NIGHTFALL."),
        make_chunk("c1", "The bridge was repaired last week."),
        make_chunk("c2", "A new merchant arrived in town."),
        make_chunk("c3", "The well ran dry this morning."),
        make_chunk("c4", "Soldiers were spotted near the pass."),
        make_chunk("c5", "The harvest festival begins tomorrow."),
    ]

    def fake_retrieve(*, top_k: int, **_: object) -> list[RetrievedChunk]:
        # Mirrors rerank_chunks: the ranked pool is truncated to exactly top_k.
        return ranked_pool[:top_k]

    stage = TurnRetrievalStage(
        actor_context_retriever=SimpleNamespace(retrieve_for_actor=fake_retrieve),
        context_budget=ContextBudget(retrieved_chunks=5),
    )

    result = stage.run(
        turn_input=TurnInput(session_id="session", message="Look around."),
        context=_context(),
    )

    from app.orchestration.context_budget import select_retrieved_chunks_for_prompt

    selected = select_retrieved_chunks_for_prompt(
        result.chunks, budget=ContextBudget(retrieved_chunks=5)
    )
    assert len(selected) == 5


def test_over_fetch_margin_does_not_cover_heavy_single_fact_duplicate_pileup() -> None:
    # docs/22 P6 honesty note (2026-07-11): the over-fetch margin (retrieved_chunks - 1)
    # is a *bounded* worst-case heuristic, not a universal guarantee that the block
    # always reaches budget.retrieved_chunks. This pins the documented residual gap: one
    # fact repeated more times than the margin can still push genuinely distinct chunks
    # entirely out of rerank_chunks' truncated top_k window, since that truncation runs
    # *before* select_retrieved_chunks_for_prompt's N3 dedup ever sees the rest of the
    # candidate pool. This is a real, deliberately-out-of-scope limitation of a
    # fixed-size fetch margin (would need write-side dedup or fetch-retry to close, per
    # the retrieval.py comment), not a regression to "fix" by widening the margin further.
    from app.rag.models import RagCollection
    from app.rag.ranking import RetrievalRankingContext, rerank_chunks

    def make(chunk_id: str, text: str, *, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            id=chunk_id,
            source="s",
            source_type="lore",
            text=text,
            score=score,
            visibility=Visibility.PLAYER,
        )

    retrieved_chunks = 5
    # One fact restated under 9 distinct ids, all ranked above 4 genuinely distinct
    # facts -- more duplicate copies than the (retrieved_chunks - 1) == 4 margin covers.
    candidates = [
        (
            RagCollection.CANON_LORE,
            make(f"dup-{i}", "The regent distrusts the chancellor.", score=0.9 - i * 0.001),
        )
        for i in range(9)
    ] + [
        (RagCollection.CANON_LORE, make(f"distinct-{i}", text, score=0.5))
        for i, text in enumerate(
            [
                "A storm is expected by nightfall.",
                "The bridge was repaired last week.",
                "A new merchant arrived in town.",
                "The well ran dry this morning.",
            ]
        )
    ]
    top_k = retrieved_chunks + max(retrieved_chunks - 1, 0)  # no standing facts here

    chunks, _ = rerank_chunks(
        context=RetrievalRankingContext(query="q", session_id="s", persona_id="p"),
        candidates=candidates,
        top_k=top_k,
    )
    from app.orchestration.context_budget import select_retrieved_chunks_for_prompt

    selected = select_retrieved_chunks_for_prompt(
        chunks, budget=ContextBudget(retrieved_chunks=retrieved_chunks)
    )

    # The 4 genuinely distinct facts exist in `candidates` but never reach the
    # truncated top_k window, so only the one duplicated fact survives -- documented,
    # expected under-fill, not a bug in the margin's sizing.
    assert len(selected) == 1


class RecordingMemoryStore:
    def __init__(self) -> None:
        self.persisted: list[Any] = []
        self.consolidated_ids: list[str] = []
        self.list_calls = 0

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
                source_turn_id=candidate.source_turn_id,
            )
            for index, candidate in enumerate(memories)
        ]
        self.persisted.extend(episodes)
        return episodes

    def list_memories_for_session(self, session_id: str) -> list[Any]:
        self.list_calls += 1
        return [episode for episode in self.persisted if episode.session_id == session_id]

    def mark_memories_consolidated(self, memory_ids: list[str]) -> None:
        # Mirrors SQLiteMemoryRepository.add_tag_to_memories: mutate the stored
        # episode's tags in place (not just record the call) so a *second*
        # list_memories_for_session -> select_consolidatable pass in the same test
        # correctly sees these as no-longer-eligible, same as the real store.
        self.consolidated_ids.extend(memory_ids)
        target_ids = set(memory_ids)
        for episode in self.persisted:
            if episode.id in target_ids and CONSOLIDATED_TAG not in episode.tags:
                episode.tags.append(CONSOLIDATED_TAG)


class _RecordingIndexer:
    def __init__(self, *, fail_index_memories: bool = False) -> None:
        self.indexed: list[list[Any]] = []
        self.unindexed: list[str] = []
        self.fail_index_memories = fail_index_memories
        self.index_memories_calls = 0

    def index_memories(self, memories: list[Any]) -> object:
        self.index_memories_calls += 1
        if self.fail_index_memories:
            raise RuntimeError("qdrant unavailable")
        self.indexed.append(list(memories))
        return None

    def unindex(self, memory_ids: Sequence[str]) -> None:
        self.unindexed.extend(memory_ids)


class _ConsolidatingCurator:
    def __init__(self, *, summary: str = "Consolidated summary.", fail: bool = False) -> None:
        self.summary = summary
        self.fail = fail

    async def curate(self, **_: object) -> Any:
        from app.domain import MemoryCuratorResult

        return MemoryCuratorResult(write_memory=False, memories=[], reason="nothing new")

    async def consolidate(self, **_: object) -> str:
        if self.fail:
            raise RuntimeError("consolidation provider failed")
        return self.summary


def _seed_filler(store: RecordingMemoryStore, session_id: str, count: int) -> None:
    from app.domain import MemoryCandidate, Visibility

    store.persist_memories(
        session_id=session_id,
        memories=[
            MemoryCandidate(
                summary=f"Filler observation {index}",
                visibility=Visibility.PLAYER,
                importance=1,
                tags=["mood"],
                scene_id="scene",
                actor_id="persona",
            )
            for index in range(count)
        ],
    )


@pytest.mark.asyncio
async def test_memory_stage_consolidates_when_threshold_reached() -> None:
    store = RecordingMemoryStore()
    context = _context()
    _seed_filler(store, context.session.id, 3)
    indexer = _RecordingIndexer()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=_ConsolidatingCurator(),
        memory_indexer=indexer,
        routing_stage=_routing(),
        consolidation_threshold=3,
        consolidation_importance_ceiling=3,
    )

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I look around the quiet gallery.",
        assistant_message="The mirrors are still.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    original_ids = [episode.id for episode in store.persisted[:3]]
    # A summary memory was persisted and indexed; originals marked + unindexed.
    assert len(store.persisted) == 4
    assert "consolidation_summary" in store.persisted[-1].tags
    assert store.persisted[-1].summary == "Consolidated summary."
    assert set(store.consolidated_ids) == set(original_ids)
    assert set(indexer.unindexed) == set(original_ids)
    assert any("rolled up 3 memories" in warning for warning in result.warnings)
    # docs/26 §3.1.1 (#76), all-NULL case: every folded original here came from
    # _seed_filler, which never sets source_turn_id -- so the non-null subset is
    # empty and the summary's source_turn_id must stay None, never a sentinel.
    # The source_ids audit trail is recorded regardless of provenance.
    assert store.persisted[-1].source_turn_id is None
    assert f"source_ids:{','.join(original_ids)}" in store.persisted[-1].tags


@pytest.mark.asyncio
async def test_memory_stage_consolidation_source_turn_id_min_when_all_provenanced() -> None:
    # docs/26 §3.1.1 (#76), all-provenanced case: the summary's source_turn_id is
    # the EARLIEST (min) of the folded originals' own source_turn_id values --
    # "when was this fact first established" -- regardless of fold order.
    from app.domain import MemoryCandidate

    store = RecordingMemoryStore()
    context = _context()
    store.persist_memories(
        session_id=context.session.id,
        memories=[
            MemoryCandidate(
                summary=f"Filler observation {index}",
                visibility=Visibility.PLAYER,
                importance=1,
                tags=["mood"],
                scene_id="scene",
                actor_id="persona",
                source_turn_id=source_turn_id,
            )
            for index, source_turn_id in enumerate((30, 10, 20))
        ],
    )
    indexer = _RecordingIndexer()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=_ConsolidatingCurator(),
        memory_indexer=indexer,
        routing_stage=_routing(),
        consolidation_threshold=3,
        consolidation_importance_ceiling=3,
    )

    await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I look around the quiet gallery.",
        assistant_message="The mirrors are still.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    original_ids = [episode.id for episode in store.persisted[:3]]
    summary = store.persisted[-1]
    assert summary.source_turn_id == 10
    assert f"source_ids:{','.join(original_ids)}" in summary.tags


@pytest.mark.asyncio
async def test_memory_stage_consolidation_source_turn_id_min_of_non_null_when_mixed() -> None:
    # docs/26 §3.1.1 (#76), mixed-NULL case: one legacy/unprovenanced original
    # (source_turn_id=None) folded together with one provenanced original in the
    # same pass. min() must be taken over the non-null subset only -- it must never
    # see None (Python's min() has no defined behavior comparing int and None) --
    # and the result here is the provenanced original's own value.
    from app.domain import MemoryCandidate

    store = RecordingMemoryStore()
    context = _context()
    store.persist_memories(
        session_id=context.session.id,
        memories=[
            MemoryCandidate(
                summary="Legacy filler with no provenance.",
                visibility=Visibility.PLAYER,
                importance=1,
                tags=["mood"],
                scene_id="scene",
                actor_id="persona",
                source_turn_id=None,
            ),
            MemoryCandidate(
                summary="Provenanced filler.",
                visibility=Visibility.PLAYER,
                importance=1,
                tags=["mood"],
                scene_id="scene",
                actor_id="persona",
                source_turn_id=15,
            ),
        ],
    )
    indexer = _RecordingIndexer()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=_ConsolidatingCurator(),
        memory_indexer=indexer,
        routing_stage=_routing(),
        consolidation_threshold=2,
        consolidation_importance_ceiling=3,
    )

    await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I look around the quiet gallery.",
        assistant_message="The mirrors are still.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    original_ids = [episode.id for episode in store.persisted[:2]]
    summary = store.persisted[-1]
    assert summary.source_turn_id == 15
    assert f"source_ids:{','.join(original_ids)}" in summary.tags


@pytest.mark.asyncio
async def test_memory_stage_consolidation_marks_originals_despite_index_failure_no_retrip() -> None:
    # docs/22 P3 (fixed 2026-07-11): SQLite-first ordering. Previously the summary was
    # indexed *before* the originals were marked consolidated, so an index failure (e.g.
    # a P1.4 embedding-model fingerprint mismatch) raised out of _persist_consolidation
    # between those two steps -- the summary row was already committed, but the
    # originals were never marked, so the same oldest-N backlog re-tripped the threshold
    # on the very next turn and minted a second summary of the same memories (one orphan
    # summary row per turn). With SQLite-first ordering, persist + mark both complete
    # before any Qdrant work runs, so an index failure degrades to a warning and the
    # next turn correctly sees a shrunk, below-threshold backlog.
    store = RecordingMemoryStore()
    context = _context()
    _seed_filler(store, context.session.id, 3)
    indexer = _RecordingIndexer(fail_index_memories=True)
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=_ConsolidatingCurator(),
        memory_indexer=indexer,
        routing_stage=_routing(),
        consolidation_threshold=3,
        consolidation_importance_ceiling=3,
    )

    first = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I look around the quiet gallery.",
        assistant_message="The mirrors are still.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    original_ids = [episode.id for episode in store.persisted[:3]]
    # The summary was still persisted and the originals still marked despite the
    # indexing failure -- the SQLite side is authoritative and complete.
    assert len(store.persisted) == 4
    assert set(store.consolidated_ids) == set(original_ids)
    assert indexer.unindexed == []
    assert any("indexing degraded" in warning for warning in first.warnings)
    assert any("rolled up 3 memories" in warning for warning in first.warnings)
    # Must not be reported as "skipped" -- that would misleadingly imply nothing was
    # persisted, when in fact the SQLite-side write fully completed.
    assert not any("consolidation skipped" in warning for warning in first.warnings)

    indexer.fail_index_memories = False  # e.g. Qdrant becomes reachable again
    second = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I wait by the mirrors.",
        assistant_message="Nothing changes.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    # No second summary: the originals are already marked consolidated in SQLite, so
    # the eligible backlog is back below threshold and this turn is a no-op.
    assert len(store.persisted) == 4
    assert second.warnings == ()


@pytest.mark.asyncio
async def test_memory_stage_consolidation_falls_back_to_deterministic_on_llm_failure() -> None:
    store = RecordingMemoryStore()
    context = _context()
    _seed_filler(store, context.session.id, 3)
    indexer = _RecordingIndexer()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=_ConsolidatingCurator(fail=True),
        memory_indexer=indexer,
        routing_stage=_routing(),
        consolidation_threshold=3,
    )

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I look around the quiet gallery.",
        assistant_message="The mirrors are still.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert len(store.persisted) == 4
    assert "Filler observation 0" in store.persisted[-1].summary
    assert any("deterministic roll-up" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_memory_stage_does_not_consolidate_below_threshold() -> None:
    store = RecordingMemoryStore()
    context = _context()
    _seed_filler(store, context.session.id, 2)
    indexer = _RecordingIndexer()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=_ConsolidatingCurator(),
        memory_indexer=indexer,
        routing_stage=_routing(),
        consolidation_threshold=3,
    )

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I look around the quiet gallery.",
        assistant_message="The mirrors are still.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert len(store.persisted) == 2
    assert store.consolidated_ids == []
    assert not any("consolidation" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_memory_stage_min_age_excludes_too_young_memories_from_threshold() -> None:
    # 3 filler memories reach the threshold=3 backlog, but min_age=1 holds the single
    # newest one out of the eligible pool, leaving only 2 -- below threshold, so
    # consolidation must not trigger even though the raw backlog count did.
    store = RecordingMemoryStore()
    context = _context()
    _seed_filler(store, context.session.id, 3)
    indexer = _RecordingIndexer()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=_ConsolidatingCurator(),
        memory_indexer=indexer,
        routing_stage=_routing(),
        consolidation_threshold=3,
        consolidation_importance_ceiling=3,
        consolidation_min_age=1,
    )

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I look around the quiet gallery.",
        assistant_message="The mirrors are still.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert len(store.persisted) == 3
    assert store.consolidated_ids == []
    assert not any("consolidation" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_memory_stage_batch_cap_folds_only_oldest_n_and_keeps_recent_window() -> None:
    # 5 filler memories clear threshold=3; batch_cap=2 folds only the oldest 2,
    # leaving the 3 newest untouched (the rolling recent window).
    store = RecordingMemoryStore()
    context = _context()
    _seed_filler(store, context.session.id, 5)
    indexer = _RecordingIndexer()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=_ConsolidatingCurator(),
        memory_indexer=indexer,
        routing_stage=_routing(),
        consolidation_threshold=3,
        consolidation_importance_ceiling=3,
        consolidation_batch_cap=2,
    )

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I look around the quiet gallery.",
        assistant_message="The mirrors are still.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    oldest_two_ids = [episode.id for episode in store.persisted[:2]]
    remaining_ids = [episode.id for episode in store.persisted[2:5]]
    assert len(store.persisted) == 6  # 5 filler - 2 consolidated + 1 summary
    assert set(store.consolidated_ids) == set(oldest_two_ids)
    assert set(indexer.unindexed) == set(oldest_two_ids)
    for remaining_id in remaining_ids:
        assert remaining_id not in store.consolidated_ids
    assert any("rolled up 2 memories" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_memory_stage_consolidation_defaults_match_pre_c2_behavior() -> None:
    # min_age=0, batch_cap=0 (the defaults, and what every pre-existing call site still
    # passes) must fold the whole eligible backlog in one shot, exactly as before C2.
    store = RecordingMemoryStore()
    context = _context()
    _seed_filler(store, context.session.id, 3)
    indexer = _RecordingIndexer()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=_ConsolidatingCurator(),
        memory_indexer=indexer,
        routing_stage=_routing(),
        consolidation_threshold=3,
        consolidation_importance_ceiling=3,
    )

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I look around the quiet gallery.",
        assistant_message="The mirrors are still.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    original_ids = [episode.id for episode in store.persisted[:3]]
    assert len(store.persisted) == 4
    assert set(store.consolidated_ids) == set(original_ids)
    assert any("rolled up 3 memories" in warning for warning in result.warnings)


class DecliningCurator:
    async def consolidate(self, **_: object) -> str:
        raise AssertionError("consolidate not expected in this test")

    async def curate(self, **_: object) -> Any:
        from app.domain import MemoryCuratorResult

        return MemoryCuratorResult(write_memory=False, memories=[], reason="nothing durable")


class CoveringCurator:
    async def consolidate(self, **_: object) -> str:
        raise AssertionError("consolidate not expected in this test")

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
    async def consolidate(self, **_: object) -> str:
        raise AssertionError("consolidate not expected in this test")

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
async def test_memory_stage_caches_session_summaries_across_turns() -> None:
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

    for _ in range(3):
        await run_turn()

    # The per-session cache means the store is loaded at most once across all
    # three turns, not once per turn.
    assert store.list_calls <= 1
    # First turn persists; later turns dedup against the cached summary.
    assert len(store.persisted) == 1
    assert stage._summary_cache._cache[context.session.id] == [
        "The player promised to return before dawn."
    ]


@pytest.mark.asyncio
async def test_memory_stage_summary_cache_is_per_session() -> None:
    store = RecordingMemoryStore()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=CoveringCurator(),
        memory_indexer=None,
        routing_stage=_routing(),
    )
    base = _context()
    other_session = base.session.model_copy(update={"id": "other-session"})

    async def run_turn(session: Any) -> Any:
        return await stage.run(
            session=session,
            scene=base.scene,
            persona=base.persona,
            user_message="I promise to return before dawn.",
            assistant_message="Iria nods once.",
            retrieval_confidence=None,
            scene_complexity=1,
        )

    first = await run_turn(base.session)
    second = await run_turn(other_session)

    # Each session writes its own memory; no cross-session dedup leakage.
    assert first.memory_written is True
    assert second.memory_written is True
    assert set(stage._summary_cache._cache) == {base.session.id, other_session.id}


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


class _ScriptedCurator:
    """Fake curator returning a fixed, pre-scripted MemoryCuratorResult. Unlike
    CoveringCurator/DecliningCurator (one hard-coded shape each), this lets a test
    control exactly what "the curator curated this turn" while the REAL deterministic
    extractor still runs for real against user_message -- docs/26 §3.2 (#77)'s
    best-match fold needs both sides genuine to be meaningfully tested."""

    def __init__(self, memories: list[MemoryCandidate]) -> None:
        self._memories = memories

    async def consolidate(self, **_: object) -> str:
        raise AssertionError("consolidate not expected in this test")

    async def curate(self, **_: object) -> MemoryCuratorResult:
        return MemoryCuratorResult(
            write_memory=bool(self._memories),
            memories=list(self._memories),
            reason="scripted",
        )


@pytest.mark.asyncio
async def test_memory_stage_folds_deterministic_tag_onto_covering_curated_summary() -> None:
    # docs/26 §3.2 (#77): reuses the EXACT #72 dawn-promise string pinned verbatim in
    # tests/unit/test_deterministic_extractor.py and app/evals/memory_write_lifecycle.py.
    # The real deterministic extractor turns an explicit entrust sentence into a
    # candidate that shares exactly the frame vocabulary {iria, vale, keep, return}
    # documented in deterministic_extractor.py's WRITE_DEDUP_COVERAGE_THRESHOLD
    # comment -- 0.50 coverage against a curated summary that never mentions the
    # compass at all. Pre-#77 that candidate (and its "entrusted" tag, importance=4)
    # would be silently discarded; the fold must land it on the curator's summary
    # instead of dropping it.
    dawn_promise_summary = (
        "The player promised to return to the archive before dawn, provided "
        "Iria Vale keeps the archive door unbarred."
    )
    store = RecordingMemoryStore()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=_ScriptedCurator(
            [
                MemoryCandidate(
                    summary=dawn_promise_summary,
                    visibility=Visibility.PLAYER,
                    importance=2,
                    tags=[],  # the curator forgot to tag this durable fact
                    scene_id="scene",
                    actor_id="persona",
                )
            ]
        ),
        memory_indexer=None,
        routing_stage=_routing(),
    )
    context = _context()

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I entrust my silver compass to Iria Vale, to keep until my return.",
        assistant_message="Iria nods once.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert result.memory_written is True
    # Folded, not appended: still exactly one persisted memory.
    assert len(store.persisted) == 1
    folded = store.persisted[0]
    assert folded.summary == dawn_promise_summary
    assert "entrusted" in folded.tags
    assert folded.importance == DETERMINISTIC_EVENT_IMPORTANCE  # raised from 2 to 4
    fold_warning = next(
        (w for w in result.warnings if w.startswith("deterministic candidate folded")), None
    )
    assert fold_warning is not None
    assert "coverage=0.50" in fold_warning
    assert "silver compass" in fold_warning


@pytest.mark.asyncio
async def test_memory_stage_folds_onto_argmax_not_first_listed_summary() -> None:
    # docs/26 §3.2 (#77), the judge-constructed wrong-donation scenario: TWO curated
    # summaries both clear the 0.5 coverage threshold for the SAME deterministic
    # candidate, with different scores. A first-match implementation would fold onto
    # curated[0] (the lower-scoring summary that merely mentions the event, checked
    # first, coverage=0.50); best-match must fold onto curated[1] instead (the
    # higher-scoring summary that is actually about the same event, coverage=0.83).
    # This test fails under a first-match implementation.
    low_match_summary = "The player mentioned guarding something before midnight, vaguely."
    high_match_summary = "The player swore to guard the amber vault before midnight without fail."
    store = RecordingMemoryStore()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=_ScriptedCurator(
            [
                MemoryCandidate(
                    summary=low_match_summary,
                    visibility=Visibility.PLAYER,
                    importance=2,
                    tags=[],
                    scene_id="scene",
                    actor_id="persona",
                ),
                MemoryCandidate(
                    summary=high_match_summary,
                    visibility=Visibility.PLAYER,
                    importance=3,
                    tags=["oath"],
                    scene_id="scene",
                    actor_id="persona",
                ),
            ]
        ),
        memory_indexer=None,
        routing_stage=_routing(),
    )
    context = _context()

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I promise to guard the amber vault before midnight.",
        assistant_message="Iria nods once.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert result.memory_written is True
    assert len(store.persisted) == 2
    low = next(m for m in store.persisted if m.summary == low_match_summary)
    high = next(m for m in store.persisted if m.summary == high_match_summary)
    # The lower-scoring, first-listed summary is untouched -- a first-match
    # implementation would have folded onto this one instead of curated[1].
    assert low.tags == []
    assert low.importance == 2
    # The higher-scoring summary received the fold; its own curator-assigned tag
    # ("oath") survives alongside the donated ones (ordered_union, no dupes).
    assert high.tags == ["oath", "promise", "deadline"]
    assert high.importance == DETERMINISTIC_EVENT_IMPORTANCE
    fold_warning = next(
        (w for w in result.warnings if w.startswith("deterministic candidate folded")), None
    )
    assert fold_warning is not None
    assert "coverage=0.83" in fold_warning


@pytest.mark.asyncio
async def test_memory_stage_extras_path_unchanged_when_nothing_covers_candidate() -> None:
    # docs/26 §3.2 (#77): when nothing in this turn's curated output clears the 0.5
    # coverage threshold, the deterministic candidate must still go to extras exactly
    # as pre-#77 -- byte-identical path, no fold warning.
    unrelated_summary = "The regent's banquet was postponed to next week."
    store = RecordingMemoryStore()
    stage = TurnMemoryStage(
        provider=UnusedProvider(),
        memory_store=cast(MemoryEpisodeStore, store),
        memory_curator=_ScriptedCurator(
            [
                MemoryCandidate(
                    summary=unrelated_summary,
                    visibility=Visibility.PLAYER,
                    importance=2,
                    tags=["lore"],
                    scene_id="scene",
                    actor_id="persona",
                )
            ]
        ),
        memory_indexer=None,
        routing_stage=_routing(),
    )
    context = _context()

    result = await stage.run(
        session=context.session,
        scene=context.scene,
        persona=context.persona,
        user_message="I promise to guard the amber vault before midnight.",
        assistant_message="Iria nods once.",
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert result.memory_written is True
    assert len(store.persisted) == 2
    unrelated = next(m for m in store.persisted if m.summary == unrelated_summary)
    assert unrelated.tags == ["lore"]
    assert unrelated.importance == 2
    extra = next(m for m in store.persisted if m.summary != unrelated_summary)
    assert "promise" in extra.tags
    assert "deadline" in extra.tags
    assert extra.importance == DETERMINISTIC_EVENT_IMPORTANCE
    assert any(
        w == "deterministic memory fallback added 1 explicit durable event(s)"
        for w in result.warnings
    )
    assert not any(w.startswith("deterministic candidate folded") for w in result.warnings)


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
    assert result.failed is False  # a deliberate gate is not a failure; the draft still serves
    assert result.warnings == ("critic gated: low-risk turn",)
    assert critic.calls == 0


class CapturingCriticAgent:
    """Records the retrieved_chunks it was actually handed by the stage."""

    def __init__(self, result: CriticResult) -> None:
        self.result = result
        self.received_chunks: list[RetrievedChunk] | None = None

    async def evaluate(
        self,
        *,
        retrieved_chunks: list[RetrievedChunk],
        **_: object,
    ) -> CriticResult:
        self.received_chunks = retrieved_chunks
        return self.result

    def build_local_repair_messages(self, **_: object) -> list[Any]:
        return []


def _mixed_visibility_chunks() -> tuple[RetrievedChunk, ...]:
    return (
        RetrievedChunk(
            id="player-1",
            source="lore.md",
            source_type="lore",
            text="Public lore.",
            score=0.5,
            visibility=Visibility.PLAYER,
        ),
        RetrievedChunk(
            id="gm-1",
            source="gm.md",
            source_type="lore",
            text="GM-only lore.",
            score=0.9,
            visibility=Visibility.GM,
        ),
        RetrievedChunk(
            id="private-1",
            source="private.md",
            source_type="memory",
            text="Character-private memory.",
            score=0.8,
            visibility=Visibility.CHARACTER_PRIVATE,
        ),
    )


@pytest.mark.asyncio
async def test_critique_stage_cloud_route_filters_non_player_chunks_and_warns() -> None:
    critic = CapturingCriticAgent(CriticResult(accepted=True))
    stage = TurnCritiqueStage(
        provider=UnusedProvider(),
        cloud_provider=UnusedProvider(),
        critic_agent=critic,
        routing_stage=_routing(),
    )
    context = _context()
    chunks = _mixed_visibility_chunks()

    result = await stage.run(
        persona=context.persona,
        scene=context.scene,
        user_message="What do I notice?",
        draft="...",
        retrieved_chunks=chunks,
        route_provider=ModelProviderName.CLOUD,
    )

    assert critic.received_chunks == [chunks[0]]
    assert result.warnings == (
        "critic context filtered: 2 non-player chunk(s) withheld from cloud critic",
    )
    assert result.critique == CriticResult(accepted=True)


@pytest.mark.asyncio
async def test_critique_stage_local_route_passes_all_chunks_through_unchanged() -> None:
    critic = CapturingCriticAgent(CriticResult(accepted=True))
    stage = TurnCritiqueStage(
        provider=UnusedProvider(),
        critic_agent=critic,
        routing_stage=_routing(),
    )
    context = _context()
    chunks = _mixed_visibility_chunks()

    result = await stage.run(
        persona=context.persona,
        scene=context.scene,
        user_message="What do I notice?",
        draft="...",
        retrieved_chunks=chunks,
        route_provider=ModelProviderName.LOCAL,
    )

    assert critic.received_chunks == list(chunks)
    assert result.warnings == ()


@pytest.mark.asyncio
async def test_critique_stage_cloud_route_with_no_chunks_emits_no_warning() -> None:
    critic = CapturingCriticAgent(CriticResult(accepted=True))
    stage = TurnCritiqueStage(
        provider=UnusedProvider(),
        cloud_provider=UnusedProvider(),
        critic_agent=critic,
        routing_stage=_routing(),
    )
    context = _context()

    result = await stage.run(
        persona=context.persona,
        scene=context.scene,
        user_message="What do I notice?",
        draft="...",
        retrieved_chunks=(),
        route_provider=ModelProviderName.CLOUD,
    )

    assert critic.received_chunks == []
    assert result.warnings == ()


@pytest.mark.asyncio
async def test_critique_stage_cloud_route_with_only_player_chunks_emits_no_warning() -> None:
    critic = CapturingCriticAgent(CriticResult(accepted=True))
    stage = TurnCritiqueStage(
        provider=UnusedProvider(),
        cloud_provider=UnusedProvider(),
        critic_agent=critic,
        routing_stage=_routing(),
    )
    context = _context()
    chunks = (_mixed_visibility_chunks()[0],)

    result = await stage.run(
        persona=context.persona,
        scene=context.scene,
        user_message="What do I notice?",
        draft="...",
        retrieved_chunks=chunks,
        route_provider=ModelProviderName.CLOUD,
    )

    assert critic.received_chunks == list(chunks)
    assert result.warnings == ()


def test_critique_stage_rejects_unknown_gating_mode() -> None:
    with pytest.raises(ValueError, match="gating must be one of"):
        TurnCritiqueStage(
            provider=UnusedProvider(),
            critic_agent=_EchoingCriticAgent(CriticResult(accepted=True)),
            routing_stage=_routing(),
            gating="sometimes",
        )


def test_memory_stage_rejects_unknown_gating_mode() -> None:
    with pytest.raises(ValueError, match="gating must be one of"):
        TurnMemoryStage(
            provider=UnusedProvider(),
            memory_store=None,
            memory_curator=None,
            memory_indexer=None,
            routing_stage=_routing(),
            gating="never",
        )


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
        # A cloud-routed risk signal (route_provider=CLOUD, below) now genuinely
        # dispatches to the cloud provider, so one must be configured even though
        # this stub critic ignores it.
        cloud_provider=UnusedProvider(),
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

    async def consolidate(self, **_: object) -> str:
        raise AssertionError("consolidate not expected in this test")

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


# --- Lane B lexical slice quotas at the stage level (docs/26 §3.4, #79) -----------


def _memory_episode(
    memory_id: str,
    summary: str,
    *,
    tags: list[str] | None = None,
) -> MemoryEpisode:
    return MemoryEpisode(
        id=memory_id,
        session_id="session",
        scene_id="scene",
        summary=summary,
        importance=2,
        visibility=Visibility.PLAYER,
        tags=tags or [],
    )


def _context_with_memories(memories: list[MemoryEpisode]) -> LoadedTurnContext:
    base = _context()
    return LoadedTurnContext(
        session=base.session,
        persona=base.persona,
        scene=base.scene,
        recent_turns=base.recent_turns,
        session_memories=tuple(memories),
    )


def _dense_result(chunks: list[RetrievedChunk]) -> object:
    from app.rag.diagnostics import (
        ChunkRetrievalDiagnostic,
        RetrievalDiagnostics,
        RetrievalResult,
    )
    from app.rag.models import RagCollection

    diagnostics = RetrievalDiagnostics(
        query="q",
        selected=[
            ChunkRetrievalDiagnostic(
                id=chunk.id,
                source=chunk.source,
                source_type=chunk.source_type,
                collection=RagCollection.SESSION_MEMORY,
                visibility=chunk.visibility,
                tags=list(chunk.tags),
                original_score=chunk.score,
                adjusted_score=chunk.score,
                applied_boosts={},
                selected_rank=index,
            )
            for index, chunk in enumerate(chunks, start=1)
        ],
        rejected=[],
    )
    return RetrievalResult(chunks=chunks, diagnostics=diagnostics)


def _dense_chunk(chunk_id: str, *, score: float, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        source=f"memory_episode:{chunk_id}",
        source_type="session_memory",
        text=text,
        score=score,
        visibility=Visibility.PLAYER,
        session_id="session",
    )


_BLUE_SEAL_QUERY = "what rule did we agree to trust"


def test_retrieval_stage_quota_off_ignores_memories_and_is_byte_identical() -> None:
    # Default quotas (lexical=0): the scorer never runs, session_memories are ignored,
    # and the stage output is exactly the retriever's -- no reorder, no slice labels,
    # no scorer warning.
    dense = [_dense_chunk("scene-1", score=0.6, text="iria stands by the window")]
    retriever = SimpleNamespace(
        retrieve_for_actor_with_diagnostics=lambda **_: _dense_result(dense)
    )
    stage = TurnRetrievalStage(
        actor_context_retriever=retriever,
        context_budget=ContextBudget(retrieved_chunks=5),
    )
    memories = [_memory_episode("blue", "the trust rule is a blue wax seal", tags=["rule"])]

    result = stage.run(
        turn_input=TurnInput(session_id="session", message=_BLUE_SEAL_QUERY),
        context=_context_with_memories(memories),
    )

    assert result.chunks == tuple(dense)
    assert result.confidence == 0.6
    assert result.warnings == ()
    assert result.diagnostics is not None
    assert all(entry.slice_guaranteed is False for entry in result.diagnostics.selected)


def test_retrieval_stage_injects_lexical_hit_not_in_dense_results() -> None:
    dense = [
        _dense_chunk("scene-1", score=0.58, text="iria stands quietly by the tall window"),
        _dense_chunk("scene-2", score=0.52, text="the palace market is crowded today"),
    ]
    retriever = SimpleNamespace(
        retrieve_for_actor_with_diagnostics=lambda **_: _dense_result(dense)
    )
    stage = TurnRetrievalStage(
        actor_context_retriever=retriever,
        context_budget=ContextBudget(retrieved_chunks=5),
        slice_quotas=SliceQuotas(lexical=2),
    )
    memories = [
        _memory_episode("blue", "the trust rule is a blue wax seal", tags=["rule"]),
        _memory_episode("chatter", "iria mentioned the market", tags=[]),
    ]

    result = stage.run(
        turn_input=TurnInput(session_id="session", message=_BLUE_SEAL_QUERY),
        context=_context_with_memories(memories),
    )

    ids = [chunk.id for chunk in result.chunks]
    assert ids[0] == "blue"  # the rare-term direct answer is promoted to the front
    assert ids.count("blue") == 1  # injected once, never duplicated
    # It reaches the actual actor prompt window through the UNCHANGED select walk.
    from app.orchestration.context_budget import select_retrieved_chunks_for_prompt

    selected = select_retrieved_chunks_for_prompt(
        result.chunks, budget=ContextBudget(retrieved_chunks=5)
    )
    assert "blue" in {chunk.id for chunk in selected}
    entry = next(e for e in result.diagnostics.selected if e.id == "blue")  # type: ignore[union-attr]
    assert entry.slice_guaranteed is True
    assert entry.slice_score is not None and entry.slice_score > 0
    assert "rule" in entry.slice_matched_terms and "trust" in entry.slice_matched_terms


def test_retrieval_stage_slice_only_rescue_is_not_low_confidence() -> None:
    # The flagged confidence interaction (docs/26 §3.4): dense results are all weak
    # (0.2), the turn is rescued entirely by an injected lexical hit (score 0.0), and
    # the confidence must NOT read as low (below the 0.45 default) on exactly that turn.
    from app.rag.ranking import SLICE_CONFIDENCE_EQUIVALENT

    dense = [_dense_chunk("scene-1", score=0.20, text="iria stands quietly by the window")]
    retriever = SimpleNamespace(
        retrieve_for_actor_with_diagnostics=lambda **_: _dense_result(dense)
    )
    stage = TurnRetrievalStage(
        actor_context_retriever=retriever,
        context_budget=ContextBudget(retrieved_chunks=5),
        slice_quotas=SliceQuotas(lexical=1),
    )
    memories = [_memory_episode("blue", "the trust rule is a blue wax seal", tags=["rule"])]

    result = stage.run(
        turn_input=TurnInput(session_id="session", message=_BLUE_SEAL_QUERY),
        context=_context_with_memories(memories),
    )

    assert result.confidence == SLICE_CONFIDENCE_EQUIVALENT
    assert result.confidence >= 0.45  # not low-confidence despite dense max being 0.20


def test_retrieval_stage_survives_retriever_failure_with_lexical_leg() -> None:
    # Fail-open direction 1 (fix 4): a dense/Qdrant exception must NOT take the
    # lexical leg down with it -- the stage degrades to a lexical-only prompt built
    # from the pre-retriever hits, strictly better than today's empty fallback.
    def boom(**_: object) -> object:
        raise RuntimeError("qdrant offline")

    stage = TurnRetrievalStage(
        actor_context_retriever=SimpleNamespace(retrieve_for_actor_with_diagnostics=boom),
        context_budget=ContextBudget(retrieved_chunks=5),
        slice_quotas=SliceQuotas(lexical=2),
    )
    memories = [_memory_episode("blue", "the trust rule is a blue wax seal", tags=["rule"])]

    result = stage.run(
        turn_input=TurnInput(session_id="session", message=_BLUE_SEAL_QUERY),
        context=_context_with_memories(memories),
    )

    assert [chunk.id for chunk in result.chunks] == ["blue"]
    assert result.confidence is not None and result.confidence >= 0.45
    assert any("retrieval degraded to lexical slice" in warning for warning in result.warnings)


def test_retrieval_stage_scorer_failure_degrades_to_dense_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fail-open direction 2: a lexical-scorer exception degrades Lane B to a no-op
    # (dense-only) with a warning, never failing the turn.
    def boom(**_: object) -> object:
        raise RuntimeError("scorer exploded")

    monkeypatch.setattr("app.orchestration.stages.retrieval.score_memories_lexical", boom)
    dense = [_dense_chunk("scene-1", score=0.6, text="iria stands by the window")]
    retriever = SimpleNamespace(
        retrieve_for_actor_with_diagnostics=lambda **_: _dense_result(dense)
    )
    stage = TurnRetrievalStage(
        actor_context_retriever=retriever,
        context_budget=ContextBudget(retrieved_chunks=5),
        slice_quotas=SliceQuotas(lexical=2),
    )
    memories = [_memory_episode("blue", "the trust rule is a blue wax seal", tags=["rule"])]

    result = stage.run(
        turn_input=TurnInput(session_id="session", message=_BLUE_SEAL_QUERY),
        context=_context_with_memories(memories),
    )

    assert result.chunks == tuple(dense)  # dense-only
    assert result.confidence == 0.6
    assert any("lexical slice scorer skipped" in warning for warning in result.warnings)
