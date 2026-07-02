from __future__ import annotations

import pytest

from app.llm.router import ModelProviderName, ModelTask, choose_route


@pytest.mark.parametrize("provider", [ModelProviderName.LOCAL, ModelProviderName.CLOUD])
@pytest.mark.parametrize(
    "task",
    [
        ModelTask.ACTOR_RESPONSE,
        ModelTask.REPAIR,
        ModelTask.CRITIC,
        ModelTask.MEMORY_EXTRACTION,
    ],
)
def test_every_task_routes_to_the_session_provider(
    provider: ModelProviderName, task: ModelTask
) -> None:
    route = choose_route(
        task=task,
        session_provider=provider,
        local_model="local-m",
        cloud_model="cloud-m",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.7,
        cloud_temperature=0.65,
        local_structured_max_tokens=640,
    )
    assert route.provider == provider
    assert route.model == ("local-m" if provider == ModelProviderName.LOCAL else "cloud-m")
    assert route.reason == f"session provider: {provider.value}"


def test_structured_tasks_pin_zero_temperature_on_both_providers() -> None:
    for provider in (ModelProviderName.LOCAL, ModelProviderName.CLOUD):
        for task in (ModelTask.CRITIC, ModelTask.MEMORY_EXTRACTION):
            route = choose_route(
                task=task,
                session_provider=provider,
                local_model="l", cloud_model="c",
                local_max_tokens=700, cloud_max_tokens=1000,
                local_temperature=0.7, cloud_temperature=0.65,
                local_structured_max_tokens=640,
            )
            assert route.temperature == 0.0
            # Grammar-constrained budget applies to the local server only.
            assert route.max_tokens == (640 if provider == ModelProviderName.LOCAL else 1000)


def test_actor_and_repair_use_configured_temperature_and_tokens() -> None:
    for provider in (ModelProviderName.LOCAL, ModelProviderName.CLOUD):
        for task in (ModelTask.ACTOR_RESPONSE, ModelTask.REPAIR):
            route = choose_route(
                task=task,
                session_provider=provider,
                local_model="l", cloud_model="c",
                local_max_tokens=700, cloud_max_tokens=1000,
                local_temperature=0.7, cloud_temperature=0.65,
                local_structured_max_tokens=640,
            )
            if provider == ModelProviderName.LOCAL:
                assert route.temperature == 0.7
                assert route.max_tokens == 700
            else:
                assert route.temperature == 0.65
                assert route.max_tokens == 1000


def test_local_structured_max_tokens_defaults_to_local_max_tokens_when_unset() -> None:
    route = choose_route(
        task=ModelTask.CRITIC,
        session_provider=ModelProviderName.LOCAL,
        local_model="l", cloud_model="c",
        local_max_tokens=700, cloud_max_tokens=1000,
        local_temperature=0.7, cloud_temperature=0.65,
    )
    assert route.max_tokens == 700
