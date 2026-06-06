from __future__ import annotations

from dataclasses import dataclass

from app.domain import SceneState, TurnInput
from app.llm.router import CloudMode, ModelProviderName, ModelRoute, ModelTask, choose_route


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
        cloud_mode: CloudMode | str,
    ) -> None:
        self.local_model = local_model
        self.cloud_model = cloud_model
        self.local_max_tokens = local_max_tokens
        self.local_structured_max_tokens = local_structured_max_tokens
        self.cloud_max_tokens = cloud_max_tokens
        self.local_temperature = local_temperature
        self.cloud_temperature = cloud_temperature
        self.cloud_mode = CloudMode(cloud_mode)

    def actor(
        self,
        *,
        turn_input: TurnInput,
        scene: SceneState,
        retrieval_confidence: float | None,
    ) -> RoutingStageResult:
        scene_complexity = self.compute_scene_complexity(scene)
        route = self.choose(
            task=ModelTask.ACTOR_RESPONSE,
            failed_local_attempts=0,
            retrieval_confidence=retrieval_confidence,
            scene_complexity=scene_complexity,
            user_requested_cloud=turn_input.user_requested_cloud,
        )
        warnings: list[str] = []
        if route.provider == ModelProviderName.CLOUD and route.requires_user_confirmation:
            warnings.append(
                f"cloud actor skipped: confirmation required for {route.model} ({route.reason})"
            )
            route = self.build_local_route(
                reason=f"confirmation required before cloud route: {route.reason}"
            )
        elif route.provider == ModelProviderName.LOCAL and route.reason != "default local route":
            warnings.append(self.warning_for_skipped_cloud(route.reason))
        return RoutingStageResult(
            route=route,
            scene_complexity=scene_complexity,
            warnings=tuple(warnings),
        )

    def critic(self) -> ModelRoute:
        return self.choose(
            task=ModelTask.CRITIC,
            failed_local_attempts=0,
            retrieval_confidence=None,
            scene_complexity=1,
            use_structured_tokens=True,
        )

    def repair(
        self,
        *,
        failed_local_attempts: int,
        retrieval_confidence: float | None,
        scene_complexity: int,
    ) -> ModelRoute:
        return self.choose(
            task=ModelTask.REPAIR,
            failed_local_attempts=failed_local_attempts,
            retrieval_confidence=retrieval_confidence,
            scene_complexity=scene_complexity,
            use_structured_tokens=failed_local_attempts == 1,
        )

    def memory(
        self,
        *,
        retrieval_confidence: float | None,
        scene_complexity: int,
    ) -> ModelRoute:
        return self.choose(
            task=ModelTask.MEMORY_EXTRACTION,
            failed_local_attempts=0,
            retrieval_confidence=retrieval_confidence,
            scene_complexity=scene_complexity,
        )

    def provider_failure(
        self,
        *,
        task: ModelTask,
        retrieval_confidence: float | None,
        scene_complexity: int,
    ) -> ModelRoute:
        return self.choose(
            task=task,
            failed_local_attempts=0,
            retrieval_confidence=retrieval_confidence,
            scene_complexity=scene_complexity,
            local_provider_failed=True,
        )

    def choose(
        self,
        *,
        task: ModelTask,
        failed_local_attempts: int,
        retrieval_confidence: float | None,
        scene_complexity: int,
        user_requested_cloud: bool = False,
        local_provider_failed: bool = False,
        use_structured_tokens: bool = False,
    ) -> ModelRoute:
        return choose_route(
            task=task,
            cloud_mode=self.cloud_mode,
            local_model=self.local_model,
            cloud_model=self.cloud_model,
            local_max_tokens=self.local_max_tokens,
            local_structured_max_tokens=(
                self.local_structured_max_tokens if use_structured_tokens else None
            ),
            cloud_max_tokens=self.cloud_max_tokens,
            local_temperature=self.local_temperature,
            cloud_temperature=self.cloud_temperature,
            failed_local_attempts=failed_local_attempts,
            retrieval_confidence=retrieval_confidence,
            scene_complexity=scene_complexity,
            user_requested_cloud=user_requested_cloud,
            local_provider_failed=local_provider_failed,
        )

    def build_local_route(self, *, reason: str) -> ModelRoute:
        return ModelRoute(
            provider=ModelProviderName.LOCAL,
            model=self.local_model,
            max_tokens=self.local_max_tokens,
            temperature=self.local_temperature,
            reason=reason,
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

    @staticmethod
    def warning_for_skipped_cloud(route_reason: str) -> str:
        prefix = "cloud mode is off; cloud would have been used: "
        if route_reason.startswith(prefix):
            return (
                "cloud actor skipped: cloud mode is off "
                f"({route_reason.removeprefix(prefix)})"
            )
        return f"cloud actor skipped: {route_reason}"
