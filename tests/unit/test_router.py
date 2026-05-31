from app.llm.router import (
    CloudMode,
    ModelProviderName,
    ModelTask,
    choose_route,
)


def test_router_chooses_local_by_default() -> None:
    route = choose_route(
        task=ModelTask.ACTOR_RESPONSE,
        cloud_mode=CloudMode.ASK,
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        failed_local_attempts=0,
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert route.provider == ModelProviderName.LOCAL
    assert route.requires_user_confirmation is False


def test_router_never_chooses_cloud_when_cloud_mode_is_off() -> None:
    route = choose_route(
        task=ModelTask.REPAIR,
        cloud_mode=CloudMode.OFF,
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        failed_local_attempts=2,
        retrieval_confidence=0.2,
        scene_complexity=5,
    )

    assert route.provider == ModelProviderName.LOCAL
    assert route.reason == "cloud mode is off"
    assert route.requires_user_confirmation is False


def test_router_marks_cloud_usage_as_confirmation_required_in_ask_mode() -> None:
    route = choose_route(
        task=ModelTask.REPAIR,
        cloud_mode=CloudMode.ASK,
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        failed_local_attempts=2,
        retrieval_confidence=None,
        scene_complexity=1,
    )

    assert route.provider == ModelProviderName.CLOUD
    assert route.requires_user_confirmation is True


def test_router_allows_cloud_without_confirmation_in_auto_mode() -> None:
    route = choose_route(
        task=ModelTask.ACTOR_RESPONSE,
        cloud_mode=CloudMode.AUTO,
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        failed_local_attempts=0,
        retrieval_confidence=0.2,
        scene_complexity=1,
    )

    assert route.provider == ModelProviderName.CLOUD
    assert route.requires_user_confirmation is False


def test_router_keeps_memory_extraction_local() -> None:
    route = choose_route(
        task=ModelTask.MEMORY_EXTRACTION,
        cloud_mode=CloudMode.AUTO,
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        failed_local_attempts=2,
        retrieval_confidence=0.2,
        scene_complexity=5,
    )

    assert route.provider == ModelProviderName.LOCAL
    assert route.reason == "memory extraction stays local"


def test_router_keeps_critic_local_with_zero_temperature() -> None:
    route = choose_route(
        task=ModelTask.CRITIC,
        cloud_mode=CloudMode.AUTO,
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        failed_local_attempts=2,
        retrieval_confidence=0.1,
        scene_complexity=5,
    )

    assert route.provider == ModelProviderName.LOCAL
    assert route.temperature == 0.0
    assert route.reason == "critic stays local"
