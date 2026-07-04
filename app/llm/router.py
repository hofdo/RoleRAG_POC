"""Model routing: which provider and parameters serve each turn task.

Defines the routing vocabulary (``CloudMode``, ``ModelProviderName``,
``ModelTask``, ``ModelRoute``) and ``choose_route``, which maps a task onto the
session's bound provider. Since the 2026-07-02 session-bound-provider decision
there is deliberately no escalation, fallback, or per-turn override: every task
runs on the provider chosen at session creation, with structured tasks (critic,
memory extraction) pinned to greedy decoding. ``LOW_RETRIEVAL_CONFIDENCE`` /
``HIGH_SCENE_COMPLEXITY`` remain here only as default critic/curator auto-gating
thresholds; routing itself no longer reads them.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

# Still used as default gating thresholds for the critic/curator auto-risk heuristic
# (TurnCritiqueStage._is_risky_turn); routing itself no longer reads these.
LOW_RETRIEVAL_CONFIDENCE = 0.45
HIGH_SCENE_COMPLEXITY = 4


class CloudMode(str, Enum):
    OFF = "off"
    ASK = "ask"
    AUTO = "auto"


class ModelProviderName(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class ModelTask(str, Enum):
    ACTOR_RESPONSE = "actor_response"
    CRITIC = "critic"
    MEMORY_EXTRACTION = "memory_extraction"
    REPAIR = "repair"


class ModelRoute(BaseModel):
    provider: ModelProviderName
    model: str
    max_tokens: int
    temperature: float
    reason: str


def choose_route(
    *,
    task: ModelTask,
    session_provider: ModelProviderName,
    local_model: str,
    cloud_model: str,
    local_max_tokens: int,
    cloud_max_tokens: int,
    local_temperature: float,
    cloud_temperature: float,
    local_structured_max_tokens: int | None = None,
) -> ModelRoute:
    """Every task runs on the session's provider. There is deliberately no
    escalation, fallback, or per-turn override: cloud is a peer choice made at
    session creation, never a rescue mechanism (decision 2026-07-02)."""
    structured = task in {ModelTask.CRITIC, ModelTask.MEMORY_EXTRACTION}
    if session_provider == ModelProviderName.CLOUD:
        return ModelRoute(
            provider=ModelProviderName.CLOUD,
            model=cloud_model,
            max_tokens=cloud_max_tokens,
            # Structured tasks pin greedy decoding on both providers.
            temperature=0.0 if structured else cloud_temperature,
            reason="session provider: cloud",
        )
    return ModelRoute(
        provider=ModelProviderName.LOCAL,
        model=local_model,
        max_tokens=(
            (local_structured_max_tokens or local_max_tokens) if structured else local_max_tokens
        ),
        temperature=0.0 if structured else local_temperature,
        reason="session provider: local",
    )
