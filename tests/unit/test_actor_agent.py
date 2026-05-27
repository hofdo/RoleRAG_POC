from __future__ import annotations

import pytest

from app.agents.actor_agent import ActorAgent
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from app.llm.router import ModelProviderName, ModelRoute


class RecordingProvider(LlmProvider):
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            text="The archivist glances aside before answering.",
            provider="fake",
            model=request.model,
            usage={"total_tokens": 12},
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_actor_agent_calls_provider_with_built_messages() -> None:
    provider = RecordingProvider()
    route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=700,
        temperature=0.75,
        reason="default local route",
    )
    messages = [
        LlmMessage(role="system", content="Stay in character."),
        LlmMessage(role="user", content="Tell me what you know."),
    ]

    text = await ActorAgent().generate(provider=provider, route=route, messages=messages)

    assert text == "The archivist glances aside before answering."
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.messages == messages
    assert request.model == "local-model"
    assert request.max_tokens == 700
    assert request.temperature == 0.75
