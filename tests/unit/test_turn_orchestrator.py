from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from app.domain import (
    CriticResult,
    CriticStatus,
    MemoryCandidate,
    MemoryCuratorResult,
    PersonaCard,
    SceneState,
    SessionState,
    TurnInput,
    Visibility,
)
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from app.llm.router import ModelProviderName
from app.memory import MemoryEpisodeStore, RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator, TurnOrchestratorConfig
from app.persistence import (
    DemoWorldRecord,
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
    connect_sqlite,
    initialize_database,
)


class FakeProvider(LlmProvider):
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            text="I have heard enough to know the regent fears open daylight.",
            provider="fake",
            model=request.model,
            usage={"total_tokens": 15},
            finish_reason="stop",
        )


class StubMemoryCurator:
    def __init__(self, *, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0
        self.curate_calls: list[dict[str, object]] = []

    async def curate(self, **kwargs: object) -> Any:
        self.calls += 1
        self.curate_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result

    async def consolidate(self, **_: object) -> str:
        raise AssertionError("consolidate not expected in this test")


class StubCritic:
    def __init__(self, result: CriticResult | None = None) -> None:
        self.result = result or CriticResult(accepted=True)

    async def evaluate(self, **_: object) -> CriticResult:
        return self.result

    def build_local_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        rejected_draft: str,
        issues: list[str],
        repair_instruction: str | None,
    ) -> list[LlmMessage]:
        raise AssertionError("local repair should not be used in this test")

    def build_cloud_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        issues: list[str],
    ) -> list[LlmMessage]:
        raise AssertionError("cloud repair should not be used in this test")


class StubActorContextRetriever:
    def __init__(self, *, chunks: list[Any] | None = None, error: Exception | None = None) -> None:
        self.chunks = chunks or []
        self.error = error
        self.calls: list[dict[str, object]] = []

    def retrieve_for_actor(self, **kwargs: object) -> list[Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.chunks


class StubMemoryIndexer:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[list[Any]] = []

    def index_memories(self, memories: list[Any]) -> None:
        self.calls.append(memories)
        if self.error is not None:
            raise self.error

    def unindex(self, memory_ids: Sequence[str]) -> None:
        raise AssertionError("unindex not expected in this test")


class FakeLoader:
    def __init__(self, *, marker: str = "default") -> None:
        self.marker = marker

    def load_world(self, world_id: str) -> DemoWorldRecord:
        return DemoWorldRecord(
            id=world_id,
            name="Winter Palace Intrigue",
            default_scene_id="rose-gallery",
            persona_ids=["archivist", "warden"],
            scene_ids=["rose-gallery"],
        )

    def load_persona(self, persona_id: str) -> PersonaCard:
        if persona_id == "archivist":
            return PersonaCard(
                id="archivist",
                name=f"Iria Vale ({self.marker})",
                role="npc",
                public_description="A composed palace archivist.",
                private_description="She is quietly aiding the coup.",
                speaking_style="Precise and dry.",
            )
        if persona_id == "warden":
            return PersonaCard(
                id="warden",
                name=f"Corin Ashe ({self.marker})",
                role="npc",
                public_description="A gruff palace warden.",
                private_description="He owes the archivist a debt.",
                speaking_style="Blunt and clipped.",
            )
        raise ValueError(f"Unknown persona: {persona_id}")

    def load_scene(self, scene_id: str) -> SceneState:
        if scene_id != "rose-gallery":
            raise ValueError(f"Unknown scene: {scene_id}")
        return SceneState(
            id="rose-gallery",
            title=f"Rose Gallery ({self.marker})",
            location="Winter Palace",
            player_visible_summary="Courtiers drift between mirrors and roses.",
            gm_private_summary="The regent's spy is already in the room.",
        )


def _build_orchestrator(
    tmp_path: Path,
    provider: LlmProvider,
    *,
    critic: StubCritic | None = None,
    memory_curator: Any = None,
    memory_indexer: StubMemoryIndexer | None = None,
    actor_context_retriever: StubActorContextRetriever | None = None,
) -> TurnOrchestrator:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    session_repository.create_session(
        SessionState(
            id="demo-session",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    turn_repository = SQLiteTurnRepository(connection)
    memory_repository = SQLiteMemoryRepository(connection)
    return TurnOrchestrator(
        loader=FakeLoader(),
        provider=provider,
        critic_agent=critic or StubCritic(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=RecentDialogueStore(
            turn_repository=turn_repository,
            recent_turns=8,
        ),
        memory_store=MemoryEpisodeStore(memory_repository=memory_repository),
        memory_curator=memory_curator,
        memory_indexer=memory_indexer,
        actor_context_retriever=actor_context_retriever,
        config=TurnOrchestratorConfig(
            retrieval_top_k=3,
            max_retrieved_chunk_chars=800,
            local_model="local-model",
            cloud_model="cloud-model",
            local_max_tokens=700,
            cloud_max_tokens=1000,
            local_temperature=0.75,
            cloud_temperature=0.65,
            cloud_mode="ask",
        ),
    )


@pytest.mark.asyncio
async def test_turn_orchestrator_returns_turn_result(tmp_path: Path) -> None:
    provider = FakeProvider()
    orchestrator = _build_orchestrator(tmp_path, provider)
    turn_input = TurnInput(
        session_id="demo-session",
        message="What have you heard about the regent?",
    )

    result = await orchestrator.run_turn(turn_input=turn_input)

    assert result.text == "I have heard enough to know the regent fears open daylight."
    assert result.route.provider == ModelProviderName.LOCAL
    assert result.route.reason == "default local route"
    assert result.memory_written is False
    assert result.warnings == []
    dump = result.model_dump()
    assert set(dump.pop("stage_timings")) >= {
        "retrieval",
        "routing",
        "generation",
        "critique",
        "persistence",
        "memory",
    }
    assert dump == {
        "text": "I have heard enough to know the regent fears open daylight.",
        "route": {
            "provider": ModelProviderName.LOCAL,
            "model": "local-model",
            "max_tokens": 700,
            "temperature": 0.75,
            "reason": "default local route",
            "requires_user_confirmation": False,
        },
        "finish_reason": "stop",
        "memory_written": False,
        "critic_status": CriticStatus.ACCEPTED,
        "warnings": [],
        "retrieval": None,
    }
    assert len(provider.requests) == 1
    assert provider.requests[0].messages[1].content == "What have you heard about the regent?"


@pytest.mark.asyncio
async def test_persona_override_switches_the_session_persona(tmp_path: Path) -> None:
    # Success path: the persona switch is deferred (TurnSessionLoader.load no longer
    # writes through immediately) but must still land in the repository once the turn
    # actually persists (TurnOrchestrator.run_turn, right after the persistence stage).
    from app.domain import TurnOutcome

    provider = FakeProvider()
    orchestrator = _build_orchestrator(tmp_path, provider)

    result = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="Hello",
            active_persona_id="warden",
        )
    )

    assert result.outcome == TurnOutcome.SUCCESS
    reloaded = orchestrator.session_repository.get_session("demo-session")
    assert reloaded is not None
    assert reloaded.active_persona_id == "warden"


@pytest.mark.asyncio
async def test_persona_override_is_not_persisted_when_the_turn_fails(tmp_path: Path) -> None:
    # A turn that fails (here: repeated empty actor responses -> CONTROLLED_FAILURE)
    # returns before the persistence stage runs, so the persona switch must never reach
    # the repository -- the stored session persona stays exactly as it was before the
    # turn was attempted. This is the failure-path counterpart to
    # test_persona_override_switches_the_session_persona above.
    from app.domain import TurnOutcome
    from app.orchestration.turn_orchestrator import CONTROLLED_FAILURE_TEXT

    provider = SequencedProvider(["", ""])
    orchestrator = _build_orchestrator(tmp_path, FakeProvider())
    orchestrator.generation_stage.provider = provider

    result = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="Hello",
            active_persona_id="warden",
        )
    )

    assert result.text == CONTROLLED_FAILURE_TEXT
    assert result.outcome == TurnOutcome.CONTROLLED_FAILURE
    reloaded = orchestrator.session_repository.get_session("demo-session")
    assert reloaded is not None
    assert reloaded.active_persona_id == "archivist"


@pytest.mark.asyncio
async def test_containment_redacts_hidden_fact_before_persistence_and_memory(
    tmp_path: Path,
) -> None:
    # Pins the security ordering: the output-side containment backstop redacts a verbatim
    # hidden-fact echo BEFORE the reply is persisted or fed to memory extraction.
    secret = "The regent's spy is already in the room."  # FakeLoader scene gm_private_summary

    class SecretEchoProvider(LlmProvider):
        def __init__(self) -> None:
            self.requests: list[LlmRequest] = []

        async def generate(self, request: LlmRequest) -> LlmResponse:
            self.requests.append(request)
            return LlmResponse(
                text=f"Iria leans in. {secret}",
                provider="fake",
                model=request.model,
                usage={},
                finish_reason="stop",
            )

    class CapturingCurator:
        def __init__(self) -> None:
            self.assistant_message: str | None = None

        async def curate(self, *, assistant_message: str, **_: object) -> Any:
            from app.domain import MemoryCuratorResult

            self.assistant_message = assistant_message
            return MemoryCuratorResult(write_memory=False, memories=[], reason="none")

        async def consolidate(self, **_: object) -> str:
            raise AssertionError("consolidate not expected")

    curator = CapturingCurator()
    orchestrator = _build_orchestrator(tmp_path, SecretEchoProvider(), memory_curator=curator)

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="What do you know?")
    )

    assert secret not in result.text
    assert any("containment: redacted" in warning for warning in result.warnings)
    # Memory extraction received the redacted text, not the raw draft.
    assert curator.assistant_message is not None
    assert secret not in curator.assistant_message
    assert curator.assistant_message == result.text
    # And SQLite persisted the redacted text.
    connection = connect_sqlite(tmp_path / "sessions.db")
    stored = SQLiteTurnRepository(connection).list_all_turns("demo-session")
    connection.close()
    assert stored
    assert secret not in stored[-1].assistant_message


@pytest.mark.asyncio
async def test_turn_orchestrator_uses_stored_content_root_for_turns(tmp_path: Path) -> None:
    provider = FakeProvider()
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    session_repository.create_session(
        SessionState(
            id="custom-session",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
            content_root="packs/custom",
        )
    )
    turn_repository = SQLiteTurnRepository(connection)
    loader_roots: list[str] = []

    def build_loader(content_root: str) -> FakeLoader:
        loader_roots.append(content_root)
        return FakeLoader(marker=content_root)

    orchestrator = TurnOrchestrator(
        loader=FakeLoader(marker="entrypoint"),
        loader_factory=build_loader,
        provider=provider,
        critic_agent=StubCritic(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=RecentDialogueStore(
            turn_repository=turn_repository,
            recent_turns=8,
        ),
        config=TurnOrchestratorConfig(
            local_model="local-model",
            cloud_model="cloud-model",
            local_max_tokens=700,
            cloud_max_tokens=1000,
            local_temperature=0.75,
            cloud_temperature=0.65,
            cloud_mode="ask",
        ),
    )

    await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="custom-session",
            message="What have you heard about the regent?",
        )
    )

    assert loader_roots == ["packs/custom"]
    assert "packs/custom" in provider.requests[0].messages[0].content


@pytest.mark.asyncio
async def test_turn_orchestrator_raises_clear_error_for_missing_scene(tmp_path: Path) -> None:
    provider = FakeProvider()
    orchestrator = _build_orchestrator(tmp_path, provider)
    orchestrator.session_repository.create_session(
        SessionState(
            id="broken-session",
            world_id="demo_world",
            active_scene_id="missing-scene",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    turn_input = TurnInput(
        session_id="broken-session",
        message="What have you heard about the regent?",
    )

    with pytest.raises(ValueError) as exc_info:
        await orchestrator.run_turn(turn_input=turn_input)

    assert "missing-scene" in str(exc_info.value)


@pytest.mark.asyncio
async def test_turn_orchestrator_sets_memory_written_when_curator_persists_memory(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    orchestrator = _build_orchestrator(
        tmp_path,
        provider,
        memory_curator=StubMemoryCurator(
            result=type(
                "CuratorResult",
                (),
                {
                    "write_memory": True,
                    "memories": [
                        MemoryCandidate(
                            summary="The player promised to return before dawn.",
                            visibility=Visibility.PLAYER,
                            importance=4,
                            tags=["promise"],
                            scene_id="rose-gallery",
                            actor_id="archivist",
                        )
                    ],
                    "reason": "This should matter later.",
                },
            )()
        ),
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="I promise I will return before dawn to face the regent.",
        )
    )

    assert result.memory_written is True
    assert result.warnings == []


@pytest.mark.asyncio
async def test_turn_orchestrator_returns_warning_when_memory_curation_fails(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    orchestrator = _build_orchestrator(
        tmp_path,
        provider,
        memory_curator=StubMemoryCurator(error=ValueError("bad memory output")),
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="I promise I will return before dawn to face the regent.",
        )
    )

    assert result.text == "I have heard enough to know the regent fears open daylight."
    assert result.memory_written is True
    assert result.warnings == [
        "memory curation skipped: bad memory output",
        "deterministic memory fallback added 1 explicit durable event(s)",
    ]


@pytest.mark.asyncio
async def test_turn_orchestrator_returns_response_when_memory_indexing_fails(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    memory_indexer = StubMemoryIndexer(error=RuntimeError("qdrant offline"))
    orchestrator = _build_orchestrator(
        tmp_path,
        provider,
        memory_curator=StubMemoryCurator(
            result=type(
                "CuratorResult",
                (),
                {
                    "write_memory": True,
                    "memories": [
                        MemoryCandidate(
                            summary="The player promised to return before dawn.",
                            visibility=Visibility.PLAYER,
                            importance=4,
                        )
                    ],
                },
            )()
        ),
        memory_indexer=memory_indexer,
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="I promise I will return before dawn to face the regent.",
        )
    )

    assert result.text == "I have heard enough to know the regent fears open daylight."
    assert result.memory_written is True
    assert result.warnings == ["memory indexing skipped: qdrant offline"]
    assert len(memory_indexer.calls) == 1


@pytest.mark.asyncio
async def test_turn_orchestrator_includes_only_public_retrieved_context(tmp_path: Path) -> None:
    provider = FakeProvider()
    retriever = StubActorContextRetriever(
        chunks=[
            type(
                "Chunk",
                (),
                {
                    "id": "public",
                    "source": "lore.md",
                    "source_type": "lore",
                    "text": "Mirrors line the gallery.",
                    "score": 0.9,
                    "visibility": Visibility.PLAYER,
                    "tags": ["palace"],
                    "world_id": "demo_world",
                    "scene_id": None,
                    "persona_id": None,
                    "session_id": None,
                    "model_copy": lambda self, **_: self,
                },
            )(),
            type(
                "Chunk",
                (),
                {
                    "id": "gm",
                    "source": "lore.md",
                    "source_type": "lore",
                    "text": "The spy waits nearby.",
                    "score": 0.99,
                    "visibility": Visibility.GM,
                    "tags": [],
                    "world_id": "demo_world",
                    "scene_id": None,
                    "persona_id": None,
                    "session_id": None,
                    "model_copy": lambda self, **_: self,
                },
            )(),
        ]
    )
    orchestrator = _build_orchestrator(
        tmp_path,
        provider,
        actor_context_retriever=retriever,
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="What do I notice?")
    )

    prompt = provider.requests[0].messages[0].content
    assert "Mirrors line the gallery." in prompt
    assert "spy waits nearby" not in prompt
    assert result.warnings == []
    assert retriever.calls[0]["world_id"] == "demo_world"
    assert retriever.calls[0]["session_id"] == "demo-session"
    assert retriever.calls[0]["persona_id"] == "archivist"
    assert retriever.calls[0]["top_k"] == 3


@pytest.mark.asyncio
async def test_turn_orchestrator_continues_when_retrieval_fails(tmp_path: Path) -> None:
    provider = FakeProvider()
    orchestrator = _build_orchestrator(
        tmp_path,
        provider,
        actor_context_retriever=StubActorContextRetriever(error=RuntimeError("qdrant offline")),
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="What do I notice?")
    )

    assert result.text == "I have heard enough to know the regent fears open daylight."
    assert result.warnings == ["retrieval skipped: qdrant offline"]
    assert "Retrieved Context:\nNone." in provider.requests[0].messages[0].content


@pytest.mark.asyncio
async def test_turn_orchestrator_returns_retrieval_diagnostics(tmp_path: Path) -> None:
    from app.domain import RetrievedChunk
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

    class DiagnosticsRetriever:
        def retrieve_for_actor(self, **_: object) -> list[Any]:
            raise AssertionError("diagnostics variant should be preferred")

        def retrieve_for_actor_with_diagnostics(self, **kwargs: object) -> RetrievalResult:
            return RetrievalResult(
                chunks=[chunk],
                diagnostics=RetrievalDiagnostics(
                    query=str(kwargs["query"]),
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
                ),
            )

    provider = FakeProvider()
    orchestrator = _build_orchestrator(tmp_path, provider)
    orchestrator.actor_context_retriever = DiagnosticsRetriever()
    turn_input = TurnInput(
        session_id="demo-session",
        message="What did I promise?",
    )

    result = await orchestrator.run_turn(turn_input=turn_input)

    assert result.retrieval is not None
    assert [entry.id for entry in result.retrieval.selected] == ["memory-1"]
    assert result.retrieval.selected[0].collection == "session_memory"
    assert result.retrieval.selected[0].selected_rank == 1
    assert [entry.id for entry in result.retrieval.rejected] == ["lore-1"]
    assert result.retrieval.rejected[0].selected_rank is None


class SequencedProvider(LlmProvider):
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        index = len(self.requests)
        self.requests.append(request)
        return LlmResponse(
            text=self.texts[index],
            provider="fake",
            model=request.model,
            usage={"total_tokens": 15},
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_turn_orchestrator_retries_once_after_empty_actor_response(
    tmp_path: Path,
) -> None:
    provider = SequencedProvider(["", "The archive door stays unbarred."])
    orchestrator = _build_orchestrator(tmp_path, FakeProvider())
    orchestrator.generation_stage.provider = provider

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="What now?")
    )

    assert result.text == "The archive door stays unbarred."
    assert len(provider.requests) == 2
    assert any("empty actor response" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_turn_orchestrator_returns_controlled_failure_for_repeated_empty_responses(
    tmp_path: Path,
) -> None:
    from app.orchestration.turn_orchestrator import CONTROLLED_FAILURE_TEXT

    provider = SequencedProvider(["", ""])
    orchestrator = _build_orchestrator(tmp_path, FakeProvider())
    orchestrator.generation_stage.provider = provider

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="What now?")
    )

    assert result.text == CONTROLLED_FAILURE_TEXT
    assert result.memory_written is False
    assert len(provider.requests) == 2
    assert any("empty" in warning for warning in result.warnings)


class FinishReasonScriptedProvider(LlmProvider):
    def __init__(self, responses: list[tuple[str, str]]) -> None:
        self.responses = responses
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        index = min(len(self.requests), len(self.responses) - 1)
        self.requests.append(request)
        text, finish_reason = self.responses[index]
        return LlmResponse(
            text=text,
            provider="fake",
            model=request.model,
            usage={"total_tokens": 500},
            finish_reason=finish_reason,
        )


@pytest.mark.asyncio
async def test_turn_orchestrator_retries_truncated_actor_response_with_larger_budget(
    tmp_path: Path,
) -> None:
    provider = FinishReasonScriptedProvider(
        [
            ("The archivist begins a long story about", "length"),
            ("The archivist finishes the story about the regent.", "stop"),
        ]
    )
    orchestrator = _build_orchestrator(tmp_path, FakeProvider())
    orchestrator.generation_stage.provider = provider

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Tell me everything.")
    )

    assert result.text == "The archivist finishes the story about the regent."
    assert result.finish_reason == "stop"
    assert len(provider.requests) == 2
    assert provider.requests[1].max_tokens == provider.requests[0].max_tokens * 2
    assert any(
        "actor response truncated: finish_reason=length from local-model" in warning
        for warning in result.warnings
    )


class RepairFriendlyCritic(StubCritic):
    def build_local_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        rejected_draft: str,
        issues: list[str],
        repair_instruction: str | None,
    ) -> list[LlmMessage]:
        return [LlmMessage(role="user", content=f"repair: {'; '.join(issues)}")]


@pytest.mark.asyncio
async def test_turn_orchestrator_repairs_draft_with_unsupported_entity(
    tmp_path: Path,
) -> None:
    provider = SequencedProvider(
        [
            "Duke Erran handed me a silver map this morning.",
            "I have heard enough to know the regent fears open daylight.",
        ]
    )
    orchestrator = _build_orchestrator(tmp_path, FakeProvider(), critic=RepairFriendlyCritic())
    orchestrator.generation_stage.provider = provider

    result = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="What have you heard about the regent?",
        )
    )

    assert result.critic_status == CriticStatus.REPAIRED
    assert result.text == "I have heard enough to know the regent fears open daylight."
    assert any("unsupported entity Duke Erran" in warning for warning in result.warnings)
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_turn_orchestrator_does_not_repair_clean_draft(tmp_path: Path) -> None:
    provider = FakeProvider()
    orchestrator = _build_orchestrator(tmp_path, provider)

    result = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="What have you heard about the regent?",
        )
    )

    assert result.critic_status == CriticStatus.ACCEPTED
    assert all("draft validation" not in warning for warning in result.warnings)
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_turn_orchestrator_records_stage_timings_for_successful_turn(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    orchestrator = _build_orchestrator(tmp_path, provider)

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="What news?")
    )

    expected_stages = {
        "retrieval",
        "routing",
        "generation",
        "critique",
        "persistence",
        "memory",
    }
    assert expected_stages.issubset(result.stage_timings.keys())
    assert all(value >= 0.0 for value in result.stage_timings.values())


@pytest.mark.asyncio
async def test_turn_orchestrator_records_stage_timings_for_controlled_failure(
    tmp_path: Path,
) -> None:
    provider = FinishReasonScriptedProvider(
        [("The archivist begins a long story about", "length")]
    )
    orchestrator = _build_orchestrator(tmp_path, FakeProvider())
    orchestrator.generation_stage.provider = provider

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Tell me everything.")
    )

    assert {"retrieval", "routing", "generation"}.issubset(result.stage_timings.keys())
    assert all(value >= 0.0 for value in result.stage_timings.values())


@pytest.mark.asyncio
async def test_turn_orchestrator_returns_controlled_failure_for_repeated_truncation(
    tmp_path: Path,
) -> None:
    from app.domain import TurnOutcome
    from app.orchestration.turn_orchestrator import CONTROLLED_FAILURE_TEXT

    provider = FinishReasonScriptedProvider(
        [("The archivist begins a long story about", "length")]
    )
    orchestrator = _build_orchestrator(tmp_path, FakeProvider())
    orchestrator.generation_stage.provider = provider

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Tell me everything.")
    )

    assert result.text == CONTROLLED_FAILURE_TEXT
    assert result.outcome == TurnOutcome.CONTROLLED_FAILURE
    assert result.memory_written is False
    assert len(provider.requests) == 2
    assert any("truncated" in warning for warning in result.warnings)


class CountingCritic(StubCritic):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def evaluate(self, **_: object) -> CriticResult:
        self.calls += 1
        return self.result


@pytest.mark.asyncio
async def test_turn_orchestrator_auto_gating_skips_critic_and_curator_on_low_risk_turn(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    critic = CountingCritic()
    curator = StubMemoryCurator()
    orchestrator = _build_orchestrator(
        tmp_path,
        provider,
        critic=critic,
        memory_curator=curator,
    )
    orchestrator.critique_stage.gating = "auto"
    orchestrator.memory_stage.gating = "auto"

    result = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="What have you heard about the regent?",
        )
    )

    assert critic.calls == 1  # no retrieval confidence -> risky -> critic runs
    assert curator.calls == 0
    assert result.critic_status == CriticStatus.ACCEPTED
    assert "memory curation gated: no durable-event signals" in result.warnings


def test_turn_orchestrator_constructor_forwards_gating_modes(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "gating.db")
    initialize_database(connection)
    turn_repository = SQLiteTurnRepository(connection)
    orchestrator = TurnOrchestrator(
        loader=FakeLoader(),
        provider=FakeProvider(),
        critic_agent=StubCritic(),
        session_repository=SQLiteSessionRepository(connection),
        turn_repository=turn_repository,
        recent_dialogue_store=RecentDialogueStore(
            turn_repository=turn_repository,
            recent_turns=8,
        ),
        config=TurnOrchestratorConfig(
            local_model="local-model",
            cloud_model="cloud-model",
            local_max_tokens=700,
            cloud_max_tokens=1000,
            local_temperature=0.75,
            cloud_temperature=0.65,
            cloud_mode="off",
            critic_gating="auto",
            curator_gating="auto",
        ),
    )

    assert orchestrator.critique_stage.gating == "auto"
    assert orchestrator.memory_stage.gating == "auto"


class CloudCapableProvider(LlmProvider):
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            text="I have heard enough to know the regent fears open daylight.",
            provider="cloud",
            model=request.model,
            usage={"total_tokens": 20},
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_turn_orchestrator_ask_mode_returns_confirmation_required_without_generation(
    tmp_path: Path,
) -> None:
    from app.domain import TurnOutcome

    provider = FakeProvider()
    orchestrator = _build_orchestrator(tmp_path, provider)

    result = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="What have you heard about the regent?",
            user_requested_cloud=True,
        )
    )

    assert result.outcome == TurnOutcome.CONFIRMATION_REQUIRED
    assert result.text == ""
    assert result.route.provider == ModelProviderName.CLOUD
    assert result.route.requires_user_confirmation is True
    assert result.memory_written is False
    assert provider.requests == []
    assert orchestrator.turn_repository.count_turns("demo-session") == 0


@pytest.mark.asyncio
async def test_turn_orchestrator_cloud_confirmed_turn_routes_to_cloud(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    cloud_provider = CloudCapableProvider()
    orchestrator = _build_orchestrator(tmp_path, provider)
    orchestrator.cloud_provider = cloud_provider
    orchestrator.generation_stage.cloud_provider = cloud_provider

    result = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="What have you heard about the regent?",
            user_requested_cloud=True,
            cloud_confirmed=True,
        )
    )

    assert result.route.provider == ModelProviderName.CLOUD
    assert result.route.requires_user_confirmation is False
    assert len(cloud_provider.requests) == 1
    assert provider.requests == []


@pytest.mark.asyncio
async def test_turn_orchestrator_force_local_declines_cloud_route(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    orchestrator = _build_orchestrator(tmp_path, provider)

    result = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="What have you heard about the regent?",
            user_requested_cloud=True,
            force_local=True,
        )
    )

    assert result.route.provider == ModelProviderName.LOCAL
    assert result.route.reason == "user declined cloud"
    assert len(provider.requests) == 1


class FirstOnlyProvider(LlmProvider):
    """Returns one draft, then fails — exercises repair failure paths."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if len(self.requests) > 1:
            return LlmResponse(
                text="",
                provider="fake",
                model=request.model,
                usage={"total_tokens": 0},
                finish_reason="stop",
            )
        return LlmResponse(
            text=self.text,
            provider="fake",
            model=request.model,
            usage={"total_tokens": 15},
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_turn_orchestrator_keeps_original_draft_when_validator_only_repair_fails(
    tmp_path: Path,
) -> None:
    from app.domain import TurnOutcome

    provider = FirstOnlyProvider("Duke Erran handed me a silver map this morning.")
    orchestrator = _build_orchestrator(tmp_path, FakeProvider(), critic=RepairFriendlyCritic())
    orchestrator.generation_stage.provider = provider

    result = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="What have you heard about the regent?",
        )
    )

    assert result.outcome == TurnOutcome.SUCCESS
    assert result.text == "Duke Erran handed me a silver map this morning."
    assert result.critic_status == CriticStatus.ACCEPTED
    assert any("unsupported entity" in warning for warning in result.warnings)
    assert any("draft validation repair failed" in warning for warning in result.warnings)
    assert result.memory_written is False
    assert orchestrator.turn_repository.count_turns("demo-session") == 1


@pytest.mark.asyncio
async def test_run_turn_reports_stage_progression(tmp_path: Path) -> None:
    provider = FakeProvider()
    orchestrator = _build_orchestrator(tmp_path, provider)
    turn_input = TurnInput(
        session_id="demo-session",
        message="What have you heard about the regent?",
    )
    stages: list[str] = []

    await orchestrator.run_turn(turn_input=turn_input, on_stage=stages.append)

    assert stages[:4] == ["session", "retrieval", "routing", "generation"]
    assert stages[-2:] == ["persistence", "memory"]
    # Clean-draft happy path: no repair pass runs, so no "repair" stage frame should fire.
    assert "repair" not in stages


@pytest.mark.asyncio
async def test_run_turn_reports_repair_stage_when_critic_rejects(tmp_path: Path) -> None:
    provider = SequencedProvider(
        [
            "Duke Erran handed me a silver map this morning.",
            "I have heard enough to know the regent fears open daylight.",
        ]
    )
    orchestrator = _build_orchestrator(tmp_path, FakeProvider(), critic=RepairFriendlyCritic())
    orchestrator.generation_stage.provider = provider
    turn_input = TurnInput(
        session_id="demo-session",
        message="What have you heard about the regent?",
    )
    stages: list[str] = []

    result = await orchestrator.run_turn(turn_input=turn_input, on_stage=stages.append)

    assert result.critic_status == CriticStatus.REPAIRED
    assert "repair" in stages
    # "repair" must fire after critique (the stage that decides whether repair is needed)
    # and before persistence (which happens once the resolved draft is final).
    assert stages.index("critique") < stages.index("repair") < stages.index("persistence")


@pytest.mark.asyncio
async def test_run_turn_persists_diagnostics_matching_result(tmp_path: Path) -> None:
    provider = FakeProvider()
    curator = StubMemoryCurator(
        result=MemoryCuratorResult(
            write_memory=True,
            reason="durable promise",
            memories=[
                MemoryCandidate(
                    summary="The player promised to return before dawn.",
                    visibility=Visibility.PLAYER,
                    importance=4,
                    tags=["promise"],
                    scene_id="rose-gallery",
                    actor_id="archivist",
                )
            ],
        )
    )
    orchestrator = _build_orchestrator(tmp_path, provider, memory_curator=curator)
    turn_input = TurnInput(
        session_id="demo-session",
        message="What have you heard about the regent?",
    )

    result = await orchestrator.run_turn(turn_input=turn_input)

    stored = orchestrator.turn_repository.list_recent_turns("demo-session", limit=1)[0]
    assert stored.diagnostics is not None
    assert set(stored.diagnostics.stage_timings) == set(result.stage_timings)
    assert stored.diagnostics.critic_status == result.critic_status
    assert stored.diagnostics.retrieval == result.retrieval
    assert stored.diagnostics.warnings == result.warnings
    assert stored.diagnostics.memory_written == result.memory_written
    assert stored.diagnostics.finish_reason == result.finish_reason


@pytest.mark.asyncio
async def test_defer_memory_skips_curation_and_returns_a_job(tmp_path: Path) -> None:
    provider = FakeProvider()
    fake_curator = StubMemoryCurator(
        result=MemoryCuratorResult(write_memory=False, reason="should not be called")
    )
    orchestrator = _build_orchestrator(tmp_path, provider, memory_curator=fake_curator)
    turn_input = TurnInput(
        session_id="demo-session",
        message="What have you heard about the regent?",
    )

    result = await orchestrator.run_turn(turn_input=turn_input, defer_memory=True)

    assert result.memory_written is False
    assert result.deferred_memory is not None
    assert result.deferred_memory.assistant_message == result.text
    assert any("memory curation deferred" in w for w in result.warnings)
    assert fake_curator.calls == 0  # the fixture's curator was never invoked


@pytest.mark.asyncio
async def test_run_deferred_memory_writes_and_updates_diagnostics(tmp_path: Path) -> None:
    provider = FakeProvider()
    fake_curator = StubMemoryCurator(
        result=MemoryCuratorResult(
            write_memory=True,
            reason="durable promise",
            memories=[
                MemoryCandidate(
                    summary="The player promised to return before dawn.",
                    visibility=Visibility.PLAYER,
                    importance=4,
                    tags=["promise"],
                    scene_id="rose-gallery",
                    actor_id="archivist",
                )
            ],
        )
    )
    orchestrator = _build_orchestrator(tmp_path, provider, memory_curator=fake_curator)
    turn_input = TurnInput(
        session_id="demo-session",
        message="What have you heard about the regent?",
    )

    result = await orchestrator.run_turn(turn_input=turn_input, defer_memory=True)
    assert result.deferred_memory is not None
    await orchestrator.run_deferred_memory(result.deferred_memory)

    assert fake_curator.calls == 1
    stored = orchestrator.turn_repository.list_all_turns("demo-session")[-1]
    assert stored.diagnostics is not None
    assert stored.diagnostics.memory_written is True


@pytest.mark.asyncio
async def test_run_deferred_memory_uses_original_turns_scene_and_persona(
    tmp_path: Path,
) -> None:
    # Regression test: a scene/persona switch landing between the response and the
    # deferred job running must NOT cause the job to curate/attribute the turn's
    # memories under the NEW scene/persona. run_deferred_memory must use the job's
    # pinned scene_id/persona_id (captured from the turn at defer-time), not the
    # session's live (possibly since-switched) fields.
    provider = FakeProvider()
    fake_curator = StubMemoryCurator(
        result=MemoryCuratorResult(write_memory=False, reason="not exercised")
    )
    orchestrator = _build_orchestrator(tmp_path, provider, memory_curator=fake_curator)
    turn_input = TurnInput(
        session_id="demo-session",
        message="What have you heard about the regent?",
        active_persona_id="archivist",
    )

    result = await orchestrator.run_turn(turn_input=turn_input, defer_memory=True)
    assert result.deferred_memory is not None
    assert result.deferred_memory.scene_id == "rose-gallery"
    assert result.deferred_memory.persona_id == "archivist"

    # Switch the session's live persona AFTER the turn was generated/persisted but
    # BEFORE the deferred job runs -- exactly the race this fix closes.
    orchestrator.session_repository.update_active_persona("demo-session", "warden")

    await orchestrator.run_deferred_memory(result.deferred_memory)

    assert fake_curator.calls == 1
    curate_kwargs = fake_curator.curate_calls[0]
    persona = cast(PersonaCard, curate_kwargs["persona"])
    scene = cast(SceneState, curate_kwargs["scene"])
    # The curator must see the ORIGINAL turn's persona (archivist), never the
    # switched-to persona (warden) that is now live on the session.
    assert persona.id == "archivist"
    assert scene.id == "rose-gallery"
