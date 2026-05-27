from __future__ import annotations

import pytest

from app.domain import PersonaCard, SceneState, TurnInput
from app.llm.provider import LlmProvider, LlmRequest, LlmResponse
from app.llm.router import ModelProviderName
from app.orchestration.turn_orchestrator import TurnOrchestrator
from app.persistence import DemoWorldRecord


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


@pytest.mark.asyncio
async def test_turn_orchestrator_returns_turn_result() -> None:
    provider = FakeProvider()
    orchestrator = TurnOrchestrator(
        loader=FakeLoader(),
        provider=provider,
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        cloud_mode="ask",
    )
    turn_input = TurnInput(
        session_id="demo-session",
        active_persona_id="archivist",
        message="What have you heard about the regent?",
    )

    result = await orchestrator.run_turn(
        turn_input=turn_input,
        world_id="demo_world",
        scene_id="rose-gallery",
    )

    assert result.text == "I have heard enough to know the regent fears open daylight."
    assert result.route.provider == ModelProviderName.LOCAL
    assert result.route.reason == "default local route"
    assert result.memory_written is False
    assert result.warnings == []
    assert len(provider.requests) == 1
    assert provider.requests[0].messages[1].content == "What have you heard about the regent?"


@pytest.mark.asyncio
async def test_turn_orchestrator_raises_clear_error_for_missing_scene() -> None:
    provider = FakeProvider()
    orchestrator = TurnOrchestrator(
        loader=FakeLoader(),
        provider=provider,
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        cloud_mode="ask",
    )
    turn_input = TurnInput(
        session_id="demo-session",
        active_persona_id="archivist",
        message="What have you heard about the regent?",
    )

    with pytest.raises(ValueError) as exc_info:
        await orchestrator.run_turn(
            turn_input=turn_input,
            world_id="demo_world",
            scene_id="missing-scene",
        )

    assert "missing-scene" in str(exc_info.value)
