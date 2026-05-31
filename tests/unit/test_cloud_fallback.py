from app.llm.router import CloudMode, ModelProviderName, ModelTask, choose_route


def test_router_preserves_skipped_cloud_reason_when_mode_is_off() -> None:
    route = choose_route(
        task=ModelTask.ACTOR_RESPONSE,
        cloud_mode=CloudMode.OFF,
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        failed_local_attempts=0,
        retrieval_confidence=0.2,
        scene_complexity=1,
        user_requested_cloud=False,
        local_provider_failed=False,
    )

    assert route.provider == ModelProviderName.LOCAL
    assert route.reason == "cloud mode is off; cloud would have been used: low retrieval confidence"
    assert route.requires_user_confirmation is False


def test_router_marks_explicit_cloud_request_as_confirmation_required_in_ask_mode() -> None:
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
        user_requested_cloud=True,
        local_provider_failed=False,
    )

    assert route.provider == ModelProviderName.CLOUD
    assert route.reason == "user requested cloud"
    assert route.requires_user_confirmation is True


def test_router_allows_cloud_when_local_provider_failed_in_auto_mode() -> None:
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
        retrieval_confidence=None,
        scene_complexity=1,
        user_requested_cloud=False,
        local_provider_failed=True,
    )

    assert route.provider == ModelProviderName.CLOUD
    assert route.reason == "local provider unavailable"
    assert route.requires_user_confirmation is False
