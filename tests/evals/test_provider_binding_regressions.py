from __future__ import annotations

import pytest

from app.domain import TurnInput, TurnOutcome
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
    # Mirrors regression_runner's structured_tasks_stay_greedy_on_both_providers check.
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
    # Mirrors regression_runner's local_session_never_calls_cloud check: a local
    # session's actor, critic, AND memory-extraction calls (real CriticAgent +
    # MemoryCurator, not stubs) must never reach the cloud provider.
    fixture = build_eval_fixture()

    class ExplodingProvider(LlmProvider):
        async def generate(self, request: LlmRequest) -> LlmResponse:
            raise AssertionError("cloud provider must never be called for a local session")

    orchestrator, local_provider, _ = fixture.build_full_stack_orchestrator(
        session_provider=ModelProviderName.LOCAL,
        actor_response_text="Local answer",
    )
    orchestrator.cloud_provider = ExplodingProvider()
    orchestrator.generation_stage.cloud_provider = ExplodingProvider()
    orchestrator.critique_stage.cloud_provider = ExplodingProvider()
    orchestrator.memory_stage.cloud_provider = ExplodingProvider()

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id=fixture.session.id, message="What do I notice?")
    )

    assert result.outcome == TurnOutcome.SUCCESS
    assert result.route.provider == ModelProviderName.LOCAL
    assert len(local_provider.requests) >= 1


@pytest.mark.asyncio
async def test_cloud_session_runs_all_tasks_on_cloud() -> None:
    # A cloud session's actor, critic, and memory-extraction calls (real CriticAgent
    # + MemoryCurator) must all land on the cloud provider; the local provider
    # explodes if touched at all, so cloud sessions stay standalone.
    fixture = build_eval_fixture()

    class ExplodingProvider(LlmProvider):
        async def generate(self, request: LlmRequest) -> LlmResponse:
            raise AssertionError("local provider must never be called for a cloud session")

    orchestrator, _, cloud_provider = fixture.build_full_stack_orchestrator(
        session_provider=ModelProviderName.CLOUD,
        actor_response_text="Cloud answer",
    )
    orchestrator.provider = ExplodingProvider()
    orchestrator.generation_stage.provider = ExplodingProvider()
    orchestrator.critique_stage.provider = ExplodingProvider()
    orchestrator.memory_stage.provider = ExplodingProvider()

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id=fixture.session.id, message="What do I notice?")
    )

    assert result.outcome == TurnOutcome.SUCCESS
    assert result.route.provider == ModelProviderName.CLOUD
    tasks = [request.metadata.get("task") for request in cloud_provider.requests]
    assert "critic" in tasks
    assert "memory_extraction" in tasks


@pytest.mark.asyncio
async def test_cloud_critic_prompt_carries_no_hidden_content() -> None:
    # Privacy pin: none of the fixture's hidden persona/scene strings (secrets,
    # forbidden_knowledge, gm_private_summary) may appear in ANY request recorded
    # by the cloud provider across actor + critic + memory calls.
    fixture = build_eval_fixture()

    orchestrator, _, cloud_provider = fixture.build_full_stack_orchestrator(
        session_provider=ModelProviderName.CLOUD,
        actor_response_text="Cloud answer",
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id=fixture.session.id, message="What do I notice?")
    )

    assert result.outcome == TurnOutcome.SUCCESS
    all_cloud_text = " ".join(
        message.content for request in cloud_provider.requests for message in request.messages
    )
    # private_description is asserted as defense-in-depth only: no prompt builder
    # references it today (structural absence, not an include_hidden gate) -- the
    # three load-bearing strings are secret / forbidden_knowledge / gm_private_summary.
    # Memory extraction has no include_hidden gate: its prompt omits hidden fields
    # structurally (MemoryCurator._build_context interpolates only ids/names/dialogue).
    assert fixture.primary_persona_secret not in all_cloud_text
    assert fixture.primary_persona_private_description not in all_cloud_text
    assert fixture.primary_persona_forbidden_knowledge not in all_cloud_text
    assert fixture.scene_gm_only_text not in all_cloud_text
