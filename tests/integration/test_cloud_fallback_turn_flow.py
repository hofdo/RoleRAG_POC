from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.domain import (
    CriticResult,
    PersonaCard,
    RetrievedChunk,
    SceneState,
    SessionState,
    TurnInput,
    Visibility,
)
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from app.llm.router import CloudMode, ModelProviderName
from app.memory import RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator, TurnOrchestratorConfig
from app.persistence import DemoWorldRecord, SQLiteSessionRepository, SQLiteTurnRepository
from app.persistence.sqlite import connect_sqlite, initialize_database


class SequencedFakeProvider(LlmProvider):
    def __init__(self, responses: list[str], *, failure: Exception | None = None) -> None:
        self.responses = responses
        self.failure = failure
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return LlmResponse(
            text=self.responses[len(self.requests) - 1],
            provider="fake",
            model=request.model,
            usage={"total_tokens": 15},
            finish_reason="stop",
        )


class StubCritic:
    def __init__(self, results: list[CriticResult] | None = None) -> None:
        self.results = results or [CriticResult(accepted=True)]
        self.calls: list[dict[str, Any]] = []

    async def evaluate(self, **kwargs: Any) -> CriticResult:
        self.calls.append(kwargs)
        return self.results[len(self.calls) - 1]

    def build_local_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        rejected_draft: str,
        issues: list[str],
        repair_instruction: str | None,
    ) -> list[LlmMessage]:
        messages = list(actor_messages)
        messages.append(LlmMessage(role="assistant", content=rejected_draft))
        messages.append(LlmMessage(role="user", content=repair_instruction or ", ".join(issues)))
        return messages

    def build_cloud_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        issues: list[str],
    ) -> list[LlmMessage]:
        messages = list(actor_messages)
        messages.append(LlmMessage(role="user", content=", ".join(issues) or "repair"))
        return messages


class StubActorContextRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve_for_actor(self, **_: object) -> list[RetrievedChunk]:
        return self.chunks


class FakeLoader:
    def load_world(self, world_id: str) -> DemoWorldRecord:
        return DemoWorldRecord(
            id=world_id,
            name="Winter Palace Intrigue",
            default_scene_id="rose-gallery",
            persona_ids=["archivist"],
            scene_ids=["rose-gallery"],
        )

    def load_persona(self, persona_id: str) -> PersonaCard:
        return PersonaCard(
            id=persona_id,
            name="Iria Vale",
            role="npc",
            public_description="A composed palace archivist.",
            private_description="She is quietly aiding the coup.",
            speaking_style="Precise and dry.",
            secrets=["She hides a cipher key in the gallery clock."],
            forbidden_knowledge=["The regent ordered the poisoning."],
        )

    def load_scene(self, scene_id: str) -> SceneState:
        return SceneState(
            id=scene_id,
            title="Rose Gallery",
            location="Winter Palace",
            active_personas=["archivist"],
            player_visible_summary="Courtiers drift between mirrors and roses.",
            gm_private_summary="A spy waits behind the mirrored column.",
            open_conflicts=[],
            active_quests=[],
            recent_events=["The regent's envoy left in haste."],
        )


def _build_orchestrator(
    tmp_path: Path,
    *,
    provider: SequencedFakeProvider,
    cloud_provider: SequencedFakeProvider | None,
    critic: StubCritic | None = None,
    retriever: StubActorContextRetriever | None = None,
    cloud_mode: CloudMode = CloudMode.ASK,
) -> TurnOrchestrator:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    turn_repository = SQLiteTurnRepository(connection)
    session_repository.create_session(
        SessionState(
            id="demo-session",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    return TurnOrchestrator(
        loader=FakeLoader(),
        provider=provider,
        cloud_provider=cloud_provider,
        critic_agent=critic or StubCritic(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=RecentDialogueStore(turn_repository=turn_repository, recent_turns=8),
        actor_context_retriever=retriever,
        config=TurnOrchestratorConfig(
            local_model="local-model",
            cloud_model="cloud-model",
            local_max_tokens=700,
            cloud_max_tokens=1000,
            local_temperature=0.75,
            cloud_temperature=0.65,
            cloud_mode=cloud_mode,
        ),
    )


@pytest.mark.asyncio
async def test_ask_mode_never_silently_calls_cloud_for_initial_actor_route(tmp_path: Path) -> None:
    local_provider = SequencedFakeProvider(["Local answer"])
    cloud_provider = SequencedFakeProvider(["Cloud answer"])
    retriever = StubActorContextRetriever(
        [
            RetrievedChunk(
                id="lore-1",
                source="demo_lore.md",
                source_type="lore",
                text="The Rose Gallery has mirrored columns.",
                score=0.1,
                visibility=Visibility.PLAYER,
            )
        ]
    )
    orchestrator = _build_orchestrator(
        tmp_path,
        provider=local_provider,
        cloud_provider=cloud_provider,
        retriever=retriever,
        cloud_mode=CloudMode.ASK,
    )

    from app.domain import TurnOutcome

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="What do I notice?")
    )

    assert result.outcome == TurnOutcome.CONFIRMATION_REQUIRED
    assert result.text == ""
    assert result.route.provider == ModelProviderName.CLOUD
    assert result.route.requires_user_confirmation is True
    assert result.route.reason == "low retrieval confidence"
    assert len(local_provider.requests) == 0
    assert len(cloud_provider.requests) == 0

    declined = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="What do I notice?",
            force_local=True,
        )
    )

    assert declined.text == "Local answer"
    assert declined.route.provider == ModelProviderName.LOCAL
    assert declined.route.reason == "user declined cloud"
    assert len(cloud_provider.requests) == 0


@pytest.mark.asyncio
async def test_auto_mode_uses_cloud_actor_route_for_low_retrieval_confidence(
    tmp_path: Path,
) -> None:
    local_provider = SequencedFakeProvider(["Local answer"])
    cloud_provider = SequencedFakeProvider(["Cloud answer"])
    retriever = StubActorContextRetriever(
        [
            RetrievedChunk(
                id="lore-1",
                source="demo_lore.md",
                source_type="lore",
                text="The Rose Gallery has mirrored columns.",
                score=0.1,
                visibility=Visibility.PLAYER,
            )
        ]
    )
    orchestrator = _build_orchestrator(
        tmp_path,
        provider=local_provider,
        cloud_provider=cloud_provider,
        retriever=retriever,
        cloud_mode=CloudMode.AUTO,
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="What do I notice?")
    )

    assert result.text == "Cloud answer"
    assert result.route.provider == ModelProviderName.CLOUD
    assert result.route.reason == "low retrieval confidence"
    assert len(local_provider.requests) == 0
    assert len(cloud_provider.requests) == 1
    assert (
        "spy waits behind the mirrored column"
        not in cloud_provider.requests[0].messages[0].content
    )
    assert "cipher key" not in cloud_provider.requests[0].messages[0].content


@pytest.mark.asyncio
async def test_auto_mode_cloud_repair_uses_sanitized_context_only(tmp_path: Path) -> None:
    local_provider = SequencedFakeProvider(["Rejected draft", "Still rejected"])
    cloud_provider = SequencedFakeProvider(["Cloud repaired"])
    critic = StubCritic(
        [
            CriticResult(
                accepted=False,
                issues=["secret leakage"],
                repair_instruction="Remove hidden facts.",
            ),
            CriticResult(
                accepted=False,
                issues=["secret leakage"],
                repair_instruction="Remove hidden facts.",
            ),
            CriticResult(accepted=True),
        ]
    )
    orchestrator = _build_orchestrator(
        tmp_path,
        provider=local_provider,
        cloud_provider=cloud_provider,
        critic=critic,
        cloud_mode=CloudMode.AUTO,
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Tell me the truth.")
    )

    cloud_request = cloud_provider.requests[0]
    serialized_messages = "\n".join(message.content for message in cloud_request.messages)

    assert result.text == "Cloud repaired"
    assert result.route.provider == ModelProviderName.CLOUD
    assert result.route.reason == "local repair failed"
    assert "Rejected draft" not in serialized_messages
    assert "cipher key" not in serialized_messages
    assert "The regent ordered the poisoning." not in serialized_messages
    assert "spy waits behind the mirrored column" not in serialized_messages


@pytest.mark.asyncio
async def test_auto_mode_falls_back_to_cloud_when_local_provider_is_unavailable(
    tmp_path: Path,
) -> None:
    local_provider = SequencedFakeProvider([], failure=RuntimeError("local provider offline"))
    cloud_provider = SequencedFakeProvider(["Cloud fallback answer"])
    orchestrator = _build_orchestrator(
        tmp_path,
        provider=local_provider,
        cloud_provider=cloud_provider,
        cloud_mode=CloudMode.AUTO,
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="What do I notice?")
    )

    assert result.text == "Cloud fallback answer"
    assert result.route.provider == ModelProviderName.CLOUD
    assert result.route.reason == "local provider unavailable"
    assert result.warnings == ["local actor failed: local provider offline"]


@pytest.mark.asyncio
async def test_off_mode_explicit_cloud_request_stays_local_with_warning(tmp_path: Path) -> None:
    local_provider = SequencedFakeProvider(["Local answer"])
    cloud_provider = SequencedFakeProvider(["Cloud answer"])
    orchestrator = _build_orchestrator(
        tmp_path,
        provider=local_provider,
        cloud_provider=cloud_provider,
        cloud_mode=CloudMode.OFF,
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(
            session_id="demo-session",
            message="What do I notice?",
            user_requested_cloud=True,
        )
    )

    assert result.text == "Local answer"
    assert result.route.provider == ModelProviderName.LOCAL
    assert (
        result.route.reason
        == "cloud mode is off; cloud would have been used: user requested cloud"
    )
    assert result.warnings == ["cloud actor skipped: cloud mode is off (user requested cloud)"]
    assert len(cloud_provider.requests) == 0
