from __future__ import annotations

import pytest

from app.domain import TurnInput
from app.evals.fixtures import build_eval_fixture
from app.llm.router import CloudMode, ModelProviderName, ModelTask, choose_route


def test_cloud_mode_off_never_routes_to_cloud() -> None:
    fixture = build_eval_fixture()

    route = choose_route(
        task=ModelTask.ACTOR_RESPONSE,
        cloud_mode=CloudMode.OFF,
        local_model=fixture.local_route.model,
        cloud_model=fixture.cloud_route.model,
        local_max_tokens=fixture.local_route.max_tokens,
        cloud_max_tokens=fixture.cloud_route.max_tokens,
        local_temperature=fixture.local_route.temperature,
        cloud_temperature=fixture.cloud_route.temperature,
        failed_local_attempts=0,
        retrieval_confidence=0.1,
        scene_complexity=1,
        user_requested_cloud=True,
    )

    assert route.provider == ModelProviderName.LOCAL
    assert route.requires_user_confirmation is False


def test_cloud_mode_ask_requires_confirmation() -> None:
    fixture = build_eval_fixture()

    route = choose_route(
        task=ModelTask.ACTOR_RESPONSE,
        cloud_mode=CloudMode.ASK,
        local_model=fixture.local_route.model,
        cloud_model=fixture.cloud_route.model,
        local_max_tokens=fixture.local_route.max_tokens,
        cloud_max_tokens=fixture.cloud_route.max_tokens,
        local_temperature=fixture.local_route.temperature,
        cloud_temperature=fixture.cloud_route.temperature,
        failed_local_attempts=0,
        retrieval_confidence=0.1,
        scene_complexity=1,
        user_requested_cloud=False,
    )

    assert route.provider == ModelProviderName.CLOUD
    assert route.requires_user_confirmation is True


def test_cloud_mode_auto_allows_deterministic_fallback() -> None:
    fixture = build_eval_fixture()

    route = choose_route(
        task=ModelTask.ACTOR_RESPONSE,
        cloud_mode=CloudMode.AUTO,
        local_model=fixture.local_route.model,
        cloud_model=fixture.cloud_route.model,
        local_max_tokens=fixture.local_route.max_tokens,
        cloud_max_tokens=fixture.cloud_route.max_tokens,
        local_temperature=fixture.local_route.temperature,
        cloud_temperature=fixture.cloud_route.temperature,
        failed_local_attempts=0,
        retrieval_confidence=0.1,
        scene_complexity=1,
        user_requested_cloud=False,
    )

    assert route.provider == ModelProviderName.CLOUD
    assert route.requires_user_confirmation is False


def test_critic_and_memory_extraction_stay_local() -> None:
    fixture = build_eval_fixture()

    critic_route = choose_route(
        task=ModelTask.CRITIC,
        cloud_mode=CloudMode.AUTO,
        local_model=fixture.local_route.model,
        cloud_model=fixture.cloud_route.model,
        local_max_tokens=fixture.local_route.max_tokens,
        cloud_max_tokens=fixture.cloud_route.max_tokens,
        local_temperature=fixture.local_route.temperature,
        cloud_temperature=fixture.cloud_route.temperature,
        failed_local_attempts=2,
        retrieval_confidence=0.1,
        scene_complexity=5,
    )
    memory_route = choose_route(
        task=ModelTask.MEMORY_EXTRACTION,
        cloud_mode=CloudMode.AUTO,
        local_model=fixture.local_route.model,
        cloud_model=fixture.cloud_route.model,
        local_max_tokens=fixture.local_route.max_tokens,
        cloud_max_tokens=fixture.cloud_route.max_tokens,
        local_temperature=fixture.local_route.temperature,
        cloud_temperature=fixture.cloud_route.temperature,
        failed_local_attempts=2,
        retrieval_confidence=0.1,
        scene_complexity=5,
    )

    assert critic_route.provider == ModelProviderName.LOCAL
    assert memory_route.provider == ModelProviderName.LOCAL


@pytest.mark.asyncio
async def test_ask_mode_does_not_silently_call_cloud_provider() -> None:
    fixture = build_eval_fixture()
    orchestrator, local_provider, cloud_provider = fixture.build_orchestrator(
        cloud_mode=CloudMode.ASK,
        actor_response_text="Local answer",
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id=fixture.session.id, message="What do I notice?")
    )

    assert result.route.provider == ModelProviderName.LOCAL
    assert len(local_provider.requests) == 1
    assert len(cloud_provider.requests) == 0
