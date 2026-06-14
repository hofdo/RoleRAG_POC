from app.llm.router import (
    HIGH_SCENE_COMPLEXITY,
    LOW_RETRIEVAL_CONFIDENCE,
    CloudMode,
    ModelProviderName,
    ModelRoute,
    ModelTask,
    choose_route,
)


def _auto_actor_route(
    *,
    scene_complexity: int = 1,
    retrieval_confidence: float | None = None,
    low_retrieval_confidence: float = LOW_RETRIEVAL_CONFIDENCE,
    high_scene_complexity: int = HIGH_SCENE_COMPLEXITY,
) -> ModelRoute:
    return choose_route(
        task=ModelTask.ACTOR_RESPONSE,
        cloud_mode=CloudMode.AUTO,
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        failed_local_attempts=0,
        retrieval_confidence=retrieval_confidence,
        scene_complexity=scene_complexity,
        low_retrieval_confidence=low_retrieval_confidence,
        high_scene_complexity=high_scene_complexity,
    )


def test_router_honors_high_scene_complexity_override() -> None:
    # Default threshold (4) escalates at complexity 4; raising it keeps it local.
    assert _auto_actor_route(scene_complexity=4).provider == ModelProviderName.CLOUD
    assert (
        _auto_actor_route(scene_complexity=4, high_scene_complexity=5).provider
        == ModelProviderName.LOCAL
    )


def test_router_honors_low_retrieval_confidence_override() -> None:
    # Default threshold (0.45) escalates at confidence 0.4; lowering it keeps it local.
    assert _auto_actor_route(retrieval_confidence=0.4).provider == ModelProviderName.CLOUD
    assert (
        _auto_actor_route(retrieval_confidence=0.4, low_retrieval_confidence=0.3).provider
        == ModelProviderName.LOCAL
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
    assert route.reason == "cloud mode is off; cloud would have been used: local repair failed"
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
        local_structured_max_tokens=350,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        failed_local_attempts=2,
        retrieval_confidence=0.2,
        scene_complexity=5,
    )

    assert route.provider == ModelProviderName.LOCAL
    assert route.max_tokens == 350
    assert route.reason == "memory extraction stays local"


def test_router_keeps_critic_local_with_zero_temperature() -> None:
    route = choose_route(
        task=ModelTask.CRITIC,
        cloud_mode=CloudMode.AUTO,
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        local_structured_max_tokens=350,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        failed_local_attempts=2,
        retrieval_confidence=0.1,
        scene_complexity=5,
    )

    assert route.provider == ModelProviderName.LOCAL
    assert route.max_tokens == 350
    assert route.temperature == 0.0
    assert route.reason == "critic stays local"
