from __future__ import annotations

from pathlib import Path

import pytest

from app.domain import CriticResult, PersonaCard, SceneState, SessionState, TurnInput
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from app.memory import RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator, TurnOrchestratorConfig
from app.persistence import DemoWorldRecord, SQLiteSessionRepository, SQLiteTurnRepository
from app.persistence.sqlite import connect_sqlite, initialize_database


class SequencedFakeProvider(LlmProvider):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            text=self.responses[len(self.requests) - 1],
            provider="fake",
            model=request.model,
            usage={"total_tokens": 15},
            finish_reason="stop",
        )


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
            id="archivist",
            name="Iria Vale",
            role="npc",
            public_description="A composed palace archivist.",
            speaking_style="Precise and dry.",
        )

    def load_scene(self, scene_id: str) -> SceneState:
        return SceneState(
            id="rose-gallery",
            title="Rose Gallery",
            location="Winter Palace",
            player_visible_summary="Courtiers drift between mirrors and roses.",
        )


class FakeCritic:
    async def evaluate(self, **_: object) -> CriticResult:
        return CriticResult(accepted=True)

    def build_local_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        rejected_draft: str,
        issues: list[str],
        repair_instruction: str | None,
    ) -> list[LlmMessage]:
        raise AssertionError("repair should not be used in this test")


@pytest.mark.asyncio
async def test_turn_orchestrator_resumes_session_and_uses_recent_turn_window(
    tmp_path: Path,
) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    turn_repository = SQLiteTurnRepository(connection)
    session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )

    first_provider = SequencedFakeProvider(["First answer"])
    first = TurnOrchestrator(
        loader=FakeLoader(),
        provider=first_provider,
        critic_agent=FakeCritic(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=RecentDialogueStore(
            turn_repository=turn_repository,
            recent_turns=1,
        ),
        config=TurnOrchestratorConfig(
            local_model="local-model",
            cloud_model="cloud-model",
            local_max_tokens=700,
            cloud_max_tokens=1000,
            local_temperature=0.75,
            cloud_temperature=0.65,
        ),
    )

    await first.run_turn(turn_input=TurnInput(session_id="session-1", message="First question"))

    second_provider = SequencedFakeProvider(["Second answer"])
    second = TurnOrchestrator(
        loader=FakeLoader(),
        provider=second_provider,
        critic_agent=FakeCritic(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=RecentDialogueStore(
            turn_repository=turn_repository,
            recent_turns=1,
        ),
        config=TurnOrchestratorConfig(
            local_model="local-model",
            cloud_model="cloud-model",
            local_max_tokens=700,
            cloud_max_tokens=1000,
            local_temperature=0.75,
            cloud_temperature=0.65,
        ),
    )

    await second.run_turn(turn_input=TurnInput(session_id="session-1", message="Second question"))

    roles = [message.role for message in second_provider.requests[0].messages]
    contents = [message.content for message in second_provider.requests[0].messages]

    assert roles == ["system", "user", "assistant", "user"]
    assert contents[1:] == ["First question", "First answer", "Second question"]
