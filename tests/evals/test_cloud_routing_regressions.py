from __future__ import annotations

import pytest

from app.domain import TurnInput
from app.evals.fixtures import build_eval_fixture
from app.llm.provider import LlmProvider, LlmRequest, LlmResponse
from app.llm.router import ModelProviderName, ModelTask, choose_route


def test_every_task_routes_to_the_session_provider() -> None:
    fixture = build_eval_fixture()
    for provider in (ModelProviderName.LOCAL, ModelProviderName.CLOUD):
        for task in (
            ModelTask.ACTOR_RESPONSE,
            ModelTask.REPAIR,
            ModelTask.CRITIC,
            ModelTask.MEMORY_EXTRACTION,
        ):
            route = choose_route(
                task=task,
                session_provider=provider,
                local_model=fixture.local_route.model,
                cloud_model=fixture.cloud_route.model,
                local_max_tokens=fixture.local_route.max_tokens,
                cloud_max_tokens=fixture.cloud_route.max_tokens,
                local_temperature=fixture.local_route.temperature,
                cloud_temperature=fixture.cloud_route.temperature,
            )
            assert route.provider == provider


def test_structured_tasks_are_greedy_on_both_providers() -> None:
    fixture = build_eval_fixture()
    for provider in (ModelProviderName.LOCAL, ModelProviderName.CLOUD):
        for task in (ModelTask.CRITIC, ModelTask.MEMORY_EXTRACTION):
            route = choose_route(
                task=task,
                session_provider=provider,
                local_model=fixture.local_route.model,
                cloud_model=fixture.cloud_route.model,
                local_max_tokens=fixture.local_route.max_tokens,
                cloud_max_tokens=fixture.cloud_route.max_tokens,
                local_temperature=fixture.local_route.temperature,
                cloud_temperature=fixture.cloud_route.temperature,
            )
            assert route.temperature == 0.0


@pytest.mark.asyncio
async def test_local_session_never_calls_the_cloud_provider() -> None:
    fixture = build_eval_fixture()

    class ExplodingProvider(LlmProvider):
        async def generate(self, request: LlmRequest) -> LlmResponse:
            raise AssertionError("cloud provider must never be called for a local session")

    orchestrator, local_provider, _ = fixture.build_orchestrator(
        session_provider=ModelProviderName.LOCAL,
        actor_response_text="Local answer",
    )
    orchestrator.cloud_provider = ExplodingProvider()
    orchestrator.generation_stage.cloud_provider = ExplodingProvider()

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id=fixture.session.id, message="What do I notice?")
    )

    assert result.route.provider == ModelProviderName.LOCAL
    assert len(local_provider.requests) == 1
