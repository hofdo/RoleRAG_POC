from __future__ import annotations

import pytest

from app.agents.critic_agent import CriticAgent, CriticAgentOutputError
from app.domain import PersonaCard, RetrievedChunk, SceneState, Visibility
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
        max_tokens=400,
        temperature=0.0,
        reason="critic stays local",
    )


def _build_persona() -> PersonaCard:
    return PersonaCard(
        id="archivist",
        name="Iria Vale",
        role="npc",
        public_description="A composed palace archivist.",
        private_description="She quietly aids the coup.",
        speaking_style="Precise and dry.",
        secrets=["She hides a cipher key in the gallery clock."],
        forbidden_knowledge=["The regent ordered the poisoning."],
        goals=["Protect the archive."],
    )


def _build_scene() -> SceneState:
    return SceneState(
        id="rose-gallery",
        title="Rose Gallery",
        location="Winter Palace",
        player_visible_summary="Courtiers drift between mirrors and roses.",
        gm_private_summary="A spy listens behind the mirrored column.",
        recent_events=["A courier just fled the chamber."],
    )


@pytest.mark.asyncio
async def test_critic_agent_parses_accepted_json() -> None:
    provider = FakeProvider(
        """
        {
          "accepted": true,
          "issues": [],
          "repair_instruction": null
        }
        """
    )

    result = await CriticAgent().evaluate(
        provider=provider,
        route=_build_route(),
        persona=_build_persona(),
        scene=_build_scene(),
        user_message="I ask what happened here.",
        draft="Iria studies the mirrored hall before she answers.",
        retrieved_chunks=[
            RetrievedChunk(
                id="lore-1",
                source="lore.md",
                source_type="lore",
                text="The Rose Gallery is lined with mirrors.",
                score=0.9,
                visibility=Visibility.PLAYER,
            )
        ],
    )

    assert result.accepted is True
    assert result.issues == []
    assert result.repair_instruction is None
    assert provider.requests[0].response_format == "json"
    assert provider.requests[0].metadata["task"] == "critic"


@pytest.mark.asyncio
async def test_critic_agent_parses_rejected_json() -> None:
    provider = FakeProvider(
        (
            "{"
            '"accepted": false,'
            '"issues": ["secret leakage", "ignored user action"],'
            '"repair_instruction": "Rewrite in character without revealing hidden facts'
            ' and directly answer the player."'
            "}"
        )
    )

    result = await CriticAgent().evaluate(
        provider=provider,
        route=_build_route(),
        persona=_build_persona(),
        scene=_build_scene(),
        user_message="I demand the truth.",
        draft="The regent ordered the poisoning, and I kept the cipher key in the clock.",
        retrieved_chunks=[],
    )

    assert result.accepted is False
    assert result.issues == ["secret leakage", "ignored user action"]
    assert result.repair_instruction is not None
    assert "directly answer the player" in result.repair_instruction


@pytest.mark.asyncio
async def test_critic_agent_rejects_invalid_json() -> None:
    provider = FakeProvider("not json")

    with pytest.raises(CriticAgentOutputError) as exc_info:
        await CriticAgent().evaluate(
            provider=provider,
            route=_build_route(),
            persona=_build_persona(),
            scene=_build_scene(),
            user_message="Hello.",
            draft="Good evening.",
            retrieved_chunks=[],
        )
    assert exc_info.value.category == "parse"


@pytest.mark.asyncio
async def test_critic_agent_accepts_recoverable_wrapped_json() -> None:
    provider = FakeProvider(
        '```json\n{"accepted": true, "issues": [], "repair_instruction": null}\n```'
    )

    result = await CriticAgent().evaluate(
        provider=provider,
        route=_build_route(),
        persona=_build_persona(),
        scene=_build_scene(),
        user_message="Hello.",
        draft="Good evening.",
        retrieved_chunks=[],
    )

    assert result.accepted is True


@pytest.mark.asyncio
async def test_critic_agent_rejects_schema_invalid_payload() -> None:
    provider = FakeProvider('{"issues": []}')

    with pytest.raises(CriticAgentOutputError) as exc_info:
        await CriticAgent().evaluate(
            provider=provider,
            route=_build_route(),
            persona=_build_persona(),
            scene=_build_scene(),
            user_message="Hello.",
            draft="Good evening.",
            retrieved_chunks=[],
        )
    assert exc_info.value.category == "schema"


@pytest.mark.asyncio
async def test_critic_agent_prompt_includes_required_checks_and_context() -> None:
    provider = FakeProvider('{"accepted": true, "issues": [], "repair_instruction": null}')

    await CriticAgent().evaluate(
        provider=provider,
        route=_build_route(),
        persona=_build_persona(),
        scene=_build_scene(),
        user_message="I ask what happened here.",
        draft="Iria studies the mirrored hall before she answers.",
        retrieved_chunks=[
            RetrievedChunk(
                id="lore-1",
                source="lore.md",
                source_type="lore",
                text="The Rose Gallery is lined with mirrors.",
                score=0.9,
                visibility=Visibility.PLAYER,
            )
        ],
    )

    prompt = provider.requests[0].messages[0].content
    context = provider.requests[0].messages[1].content
    assert "secret leakage" in prompt
    assert "contradiction with scene state" in prompt
    assert "contradiction with retrieved lore" in prompt
    assert "character knowledge violation" in prompt
    assert "generic assistant tone" in prompt
    assert "ignoring the user's action" in prompt
    assert "The Rose Gallery is lined with mirrors." in context
