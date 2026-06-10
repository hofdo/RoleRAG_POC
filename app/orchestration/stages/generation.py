from __future__ import annotations

from dataclasses import dataclass

from app.agents import ActorAgent
from app.domain import TurnInput
from app.llm.provider import LlmMessage, LlmProvider, LlmResponse
from app.llm.router import ModelProviderName, ModelRoute, ModelTask
from app.orchestration.context_budget import ContextBudget
from app.orchestration.context_builder import build_actor_messages
from app.orchestration.stages.retrieval import RetrievalStageResult
from app.orchestration.stages.routing import RoutingStageResult, TurnRoutingStage
from app.orchestration.stages.session import LoadedTurnContext


class EmptyProviderResponseError(RuntimeError):
    def __init__(self, *, provider: str, model: str) -> None:
        super().__init__(f"{provider} provider returned empty text for {model} after one retry")
        self.provider = provider
        self.model = model


@dataclass(frozen=True)
class GenerationStageResult:
    text: str
    route: ModelRoute
    finish_reason: str | None
    messages: tuple[LlmMessage, ...]
    warnings: tuple[str, ...]


class TurnGenerationStage:
    def __init__(
        self,
        *,
        provider: LlmProvider,
        cloud_provider: LlmProvider | None,
        routing_stage: TurnRoutingStage,
        context_budget: ContextBudget,
        recent_dialogue_max_message_chars: int,
        actor_agent: ActorAgent | None = None,
    ) -> None:
        self.provider = provider
        self.cloud_provider = cloud_provider
        self.routing_stage = routing_stage
        self.context_budget = context_budget
        self.recent_dialogue_max_message_chars = recent_dialogue_max_message_chars
        self.actor_agent = actor_agent or ActorAgent()

    async def run(
        self,
        *,
        turn_input: TurnInput,
        context: LoadedTurnContext,
        retrieval: RetrievalStageResult,
        routing: RoutingStageResult,
    ) -> GenerationStageResult:
        messages = build_actor_messages(
            persona=context.persona,
            scene=context.scene,
            turn_input=turn_input,
            recent_turns=context.recent_turns,
            retrieved_chunks=retrieval.chunks,
            context_budget=self.context_budget,
            recent_dialogue_max_message_chars=self.recent_dialogue_max_message_chars,
        )
        text, route, finish_reason, warnings = await self.generate_messages(
            route=routing.route,
            messages=messages,
            task=ModelTask.ACTOR_RESPONSE,
            retrieval_confidence=retrieval.confidence,
            scene_complexity=routing.scene_complexity,
        )
        return GenerationStageResult(
            text=text,
            route=route,
            finish_reason=finish_reason,
            messages=tuple(messages),
            warnings=warnings,
        )

    async def generate_messages(
        self,
        *,
        route: ModelRoute,
        messages: list[LlmMessage] | tuple[LlmMessage, ...],
        task: ModelTask,
        retrieval_confidence: float | None,
        scene_complexity: int,
    ) -> tuple[str, ModelRoute, str | None, tuple[str, ...]]:
        if route.requires_user_confirmation:
            raise RuntimeError("confirmation-required route reached provider dispatch")
        try:
            response, empty_warnings = await self._generate_non_empty(
                route=route, messages=list(messages)
            )
            return (
                response.text,
                route,
                response.finish_reason,
                (*empty_warnings, *_truncation_warnings(response, route)),
            )
        except Exception as exc:
            if route.provider != ModelProviderName.LOCAL:
                raise
            warnings = [f"local actor failed: {exc}"]
            fallback_route = self.routing_stage.provider_failure(
                task=task,
                retrieval_confidence=retrieval_confidence,
                scene_complexity=scene_complexity,
            )
            if fallback_route.provider == ModelProviderName.LOCAL:
                raise
            if fallback_route.requires_user_confirmation:
                warnings.append(
                    f"cloud actor skipped: confirmation required for {fallback_route.model} "
                    f"({fallback_route.reason})"
                )
                raise
            response, empty_warnings = await self._generate_non_empty(
                route=fallback_route, messages=list(messages)
            )
            warnings.extend(empty_warnings)
            warnings.extend(_truncation_warnings(response, fallback_route))
            return response.text, fallback_route, response.finish_reason, tuple(warnings)

    async def _generate_non_empty(
        self,
        *,
        route: ModelRoute,
        messages: list[LlmMessage],
    ) -> tuple[LlmResponse, tuple[str, ...]]:
        response = await self._generate(route=route, messages=messages)
        if response.text.strip():
            return response, ()
        warnings = (f"empty actor response from {route.model}; retried once",)
        retry_response = await self._generate(route=route, messages=messages)
        if retry_response.text.strip():
            return retry_response, warnings
        raise EmptyProviderResponseError(provider=route.provider.value, model=route.model)

    async def _generate(
        self,
        *,
        route: ModelRoute,
        messages: list[LlmMessage],
    ) -> LlmResponse:
        provider = (
            self.provider
            if route.provider == ModelProviderName.LOCAL
            else self.cloud_provider
        )
        if provider is None:
            raise RuntimeError(f"Missing provider for route: {route.provider.value}")
        return await self.actor_agent.generate(provider=provider, route=route, messages=messages)


def _truncation_warnings(response: LlmResponse, route: ModelRoute) -> tuple[str, ...]:
    if response.finish_reason != "length":
        return ()
    return (f"actor response truncated: finish_reason=length from {route.model}",)
