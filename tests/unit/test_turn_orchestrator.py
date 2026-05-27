from __future__ import annotations

import pytest

from app.domain import TurnInput
from app.llm.provider import LlmProvider, LlmRequest, LlmResponse
from app.llm.router import ModelProviderName
from app.orchestration.turn_orchestrator import TurnOrchestrator


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


@pytest.mark.asyncio
async def test_turn_orchestrator_returns_turn_result() -> None:
    provider = FakeProvider()
    orchestrator = TurnOrchestrator(
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

    result = await orchestrator.run_turn(turn_input)

    assert result.text == "I have heard enough to know the regent fears open daylight."
    assert result.route.provider == ModelProviderName.LOCAL
    assert result.route.reason == "default local route"
    assert result.memory_written is False
    assert result.warnings == []
    assert len(provider.requests) == 1
    assert provider.requests[0].messages[1].content == "What have you heard about the regent?"

