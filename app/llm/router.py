from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class CloudMode(str, Enum):
    OFF = "off"
    ASK = "ask"
    AUTO = "auto"


class ModelProviderName(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class ModelTask(str, Enum):
    INTENT_CLASSIFICATION = "intent_classification"
    ACTOR_RESPONSE = "actor_response"
    CRITIC = "critic"
    MEMORY_EXTRACTION = "memory_extraction"
    REPAIR = "repair"
    SUMMARIZATION = "summarization"


class ModelRoute(BaseModel):
    provider: ModelProviderName
    model: str
    max_tokens: int
    temperature: float
    reason: str
    requires_user_confirmation: bool = False


def choose_route(
    *,
    task: ModelTask,
    cloud_mode: CloudMode,
    local_model: str,
    cloud_model: str,
    local_max_tokens: int,
    cloud_max_tokens: int,
    local_temperature: float,
    cloud_temperature: float,
    failed_local_attempts: int,
    retrieval_confidence: float | None,
    scene_complexity: int,
    user_requested_cloud: bool = False,
    local_provider_failed: bool = False,
    local_structured_max_tokens: int | None = None,
) -> ModelRoute:
    if task == ModelTask.CRITIC:
        return ModelRoute(
            provider=ModelProviderName.LOCAL,
            model=local_model,
            max_tokens=local_structured_max_tokens or local_max_tokens,
            temperature=0.0,
            reason="critic stays local",
        )

    if task == ModelTask.MEMORY_EXTRACTION:
        return ModelRoute(
            provider=ModelProviderName.LOCAL,
            model=local_model,
            max_tokens=local_structured_max_tokens or local_max_tokens,
            temperature=0.0,
            reason="memory extraction stays local",
        )

    should_use_cloud = False
    reason = "default local route"

    if user_requested_cloud and task == ModelTask.ACTOR_RESPONSE:
        should_use_cloud = True
        reason = "user requested cloud"
    elif local_provider_failed and task in {ModelTask.ACTOR_RESPONSE, ModelTask.REPAIR}:
        should_use_cloud = True
        reason = "local provider unavailable"
    elif task == ModelTask.REPAIR and failed_local_attempts > 1:
        should_use_cloud = True
        reason = "local repair failed"
    elif task == ModelTask.ACTOR_RESPONSE and scene_complexity >= 4:
        should_use_cloud = True
        reason = "high scene complexity"
    elif (
        task == ModelTask.ACTOR_RESPONSE
        and retrieval_confidence is not None
        and retrieval_confidence < 0.45
    ):
        should_use_cloud = True
        reason = "low retrieval confidence"

    if cloud_mode == CloudMode.OFF:
        if should_use_cloud:
            reason = f"cloud mode is off; cloud would have been used: {reason}"
        else:
            reason = "cloud mode is off"
        return ModelRoute(
            provider=ModelProviderName.LOCAL,
            model=local_model,
            max_tokens=local_max_tokens,
            temperature=local_temperature,
            reason=reason,
        )

    if should_use_cloud:
        return ModelRoute(
            provider=ModelProviderName.CLOUD,
            model=cloud_model,
            max_tokens=cloud_max_tokens,
            temperature=cloud_temperature,
            reason=reason,
            requires_user_confirmation=cloud_mode == CloudMode.ASK,
        )

    return ModelRoute(
        provider=ModelProviderName.LOCAL,
        model=local_model,
        max_tokens=local_max_tokens,
        temperature=local_temperature,
        reason=reason,
    )
