from __future__ import annotations

from pathlib import Path

import pytest

from app.domain import PersonaCard, SceneState, SessionState, TurnInput
from app.llm.provider import LlmProvider, LlmRequest, LlmResponse
from app.llm.router import ModelProviderName
from app.memory import RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator
from app.persistence import (
    DemoWorldRecord,
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
        if persona_id != "archivist":
            raise ValueError(f"Unknown persona: {persona_id}")
        return PersonaCard(
            id="archivist",
            name="Iria Vale",
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
            title="Rose Gallery",
            location="Winter Palace",
            player_visible_summary="Courtiers drift between mirrors and roses.",
            gm_private_summary="The regent's spy is already in the room.",
        )


def _build_orchestrator(tmp_path: Path, provider: FakeProvider) -> TurnOrchestrator:
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
    return TurnOrchestrator(
        loader=FakeLoader(),
        provider=provider,
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
    assert len(provider.requests) == 1
    assert provider.requests[0].messages[1].content == "What have you heard about the regent?"


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
