from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.domain import (
    CriticResult,
    MemoryCandidate,
    PersonaCard,
    SceneState,
    SessionState,
    TurnInput,
    Visibility,
)
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from app.llm.router import ModelProviderName
from app.memory import MemoryEpisodeStore, RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator
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

    async def curate(self, **_: object) -> Any:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


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


class FakeLoader:
    def __init__(self, *, marker: str = "default") -> None:
        self.marker = marker

    def load_world(self, world_id: str) -> DemoWorldRecord:
        return DemoWorldRecord(
            id=world_id,
            name="Winter Palace Intrigue",
            default_scene_id="rose-gallery",
            persona_ids=["archivist"],
            scene_ids=["rose-gallery"],
        )

    def load_persona(self, persona_id: str) -> PersonaCard:
        if persona_id != "archivist":
            raise ValueError(f"Unknown persona: {persona_id}")
        return PersonaCard(
            id="archivist",
            name=f"Iria Vale ({self.marker})",
            role="npc",
            public_description="A composed palace archivist.",
            private_description="She is quietly aiding the coup.",
            speaking_style="Precise and dry.",
        )

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
    provider: FakeProvider,
    *,
    critic: StubCritic | None = None,
    memory_curator: StubMemoryCurator | None = None,
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
        retrieval_top_k=3,
        max_retrieved_chunk_chars=800,
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        cloud_mode="ask",
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
    assert result.model_dump() == {
        "text": "I have heard enough to know the regent fears open daylight.",
        "route": {
            "provider": ModelProviderName.LOCAL,
            "model": "local-model",
            "max_tokens": 700,
            "temperature": 0.75,
            "reason": "default local route",
            "requires_user_confirmation": False,
        },
        "memory_written": False,
        "warnings": [],
    }
    assert len(provider.requests) == 1
    assert provider.requests[0].messages[1].content == "What have you heard about the regent?"


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
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        cloud_mode="ask",
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
            message="I promise I will return before dawn.",
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
            message="I promise I will return before dawn.",
        )
    )

    assert result.text == "I have heard enough to know the regent fears open daylight."
    assert result.memory_written is False
    assert result.warnings == ["memory curation skipped: bad memory output"]


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
            message="I promise I will return before dawn.",
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
