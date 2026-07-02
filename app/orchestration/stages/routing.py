from __future__ import annotations

from dataclasses import dataclass

from app.domain import SceneState
from app.llm.router import ModelProviderName, ModelRoute, ModelTask, choose_route


@dataclass(frozen=True)
class RoutingStageResult:
    route: ModelRoute
    scene_complexity: int
    warnings: tuple[str, ...]


class TurnRoutingStage:
    def __init__(
        self,
        *,
        local_model: str,
        cloud_model: str,
        local_max_tokens: int,
        local_structured_max_tokens: int,
        cloud_max_tokens: int,
        local_temperature: float,
        cloud_temperature: float,
    ) -> None:
        self.local_model = local_model
        self.cloud_model = cloud_model
        self.local_max_tokens = local_max_tokens
        self.local_structured_max_tokens = local_structured_max_tokens
        self.cloud_max_tokens = cloud_max_tokens
        self.local_temperature = local_temperature
        self.cloud_temperature = cloud_temperature

    def actor(self, *, provider: ModelProviderName, scene: SceneState) -> RoutingStageResult:
        return RoutingStageResult(
            route=self._choose(task=ModelTask.ACTOR_RESPONSE, provider=provider),
            scene_complexity=self.compute_scene_complexity(scene),
            warnings=(),
        )

    def critic(self, *, provider: ModelProviderName) -> ModelRoute:
        return self._choose(task=ModelTask.CRITIC, provider=provider)

    def repair(self, *, provider: ModelProviderName) -> ModelRoute:
        return self._choose(task=ModelTask.REPAIR, provider=provider)

    def memory(self, *, provider: ModelProviderName) -> ModelRoute:
        return self._choose(task=ModelTask.MEMORY_EXTRACTION, provider=provider)

    def _choose(self, *, task: ModelTask, provider: ModelProviderName) -> ModelRoute:
        return choose_route(
            task=task,
            session_provider=provider,
            local_model=self.local_model,
            cloud_model=self.cloud_model,
            local_max_tokens=self.local_max_tokens,
            cloud_max_tokens=self.cloud_max_tokens,
            local_temperature=self.local_temperature,
            cloud_temperature=self.cloud_temperature,
            local_structured_max_tokens=self.local_structured_max_tokens,
        )

    @staticmethod
    def compute_scene_complexity(scene: SceneState) -> int:
        complexity = 1
        if scene.open_conflicts:
            complexity += 1
        if scene.active_quests:
            complexity += 1
        if len(scene.active_personas) > 1:
            complexity += min(2, len(scene.active_personas) - 1)
        return min(complexity, 5)
