from __future__ import annotations

import pytest

from app.agents.memory_curator import MemoryCurator, MemoryCuratorOutputError
from app.domain import PersonaCard, SceneState, SessionState, Visibility
from app.llm.provider import LlmProvider, LlmRequest, LlmResponse
from app.llm.router import ModelProviderName, ModelRoute


class FakeProvider(LlmProvider):
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            text=self.response_text,
            provider="fake",
            model=request.model,
            usage={"total_tokens": 12},
            finish_reason="stop",
        )


def _build_route() -> ModelRoute:
    return ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=700,
        temperature=0.0,
        reason="memory extraction stays local",
    )


def _build_session() -> SessionState:
    return SessionState(
        id="session-1",
        world_id="demo_world",
        active_scene_id="rose-gallery",
        active_persona_id="archivist",
        player_name="Avery",
    )


def _build_scene() -> SceneState:
    return SceneState(
        id="rose-gallery",
        title="Rose Gallery",
        location="Winter Palace",
        player_visible_summary="Courtiers drift between mirrors and roses.",
    )


def _build_persona() -> PersonaCard:
    return PersonaCard(
        id="archivist",
        name="Iria Vale",
        role="npc",
        public_description="A composed palace archivist.",
        speaking_style="Precise and dry.",
    )


@pytest.mark.asyncio
async def test_memory_curator_returns_structured_candidates_from_valid_json() -> None:
    provider = FakeProvider(
        """
        {
          "write_memory": true,
          "memories": [
            {
              "summary": "The player promised to return before dawn.",
              "visibility": "player",
              "importance": 4,
              "tags": ["promise", "deadline"],
              "scene_id": "rose-gallery",
              "actor_id": "archivist"
            }
          ],
          "reason": "The promise should matter later."
        }
        """
    )

    result = await MemoryCurator().curate(
        provider=provider,
        route=_build_route(),
        session=_build_session(),
        scene=_build_scene(),
        persona=_build_persona(),
        user_message="I promise I will return before dawn.",
        assistant_message="Then I will keep the archive door unbarred until first light.",
    )

    assert result.write_memory is True
    assert result.memories[0].visibility == Visibility.PLAYER
    assert provider.requests[0].response_format == "json"


@pytest.mark.asyncio
async def test_memory_curator_rejects_invalid_json() -> None:
    provider = FakeProvider("not json")

    with pytest.raises(MemoryCuratorOutputError):
        await MemoryCurator().curate(
            provider=provider,
            route=_build_route(),
            session=_build_session(),
            scene=_build_scene(),
            persona=_build_persona(),
            user_message="Hello there.",
            assistant_message="Good evening.",
        )
