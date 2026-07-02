from __future__ import annotations

from dataclasses import dataclass

from app.agents import ActorAgent
from app.domain import TurnInput
from app.llm.provider import LlmMessage, LlmProvider, LlmResponse
from app.llm.router import ModelProviderName, ModelRoute
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


class TruncatedProviderResponseError(RuntimeError):
    def __init__(self, *, provider: str, model: str) -> None:
        super().__init__(
            f"{provider} provider returned truncated text for {model} "
            "after one retry with a larger budget"
        )
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
        # Default mirrors Settings.truncation_retry_budget_multiplier; composition
        # passes the configured value (single source of truth).
        truncation_retry_budget_multiplier: int = 2,
    ) -> None:
        self.provider = provider
        self.cloud_provider = cloud_provider
        self.routing_stage = routing_stage
        self.context_budget = context_budget
        self.recent_dialogue_max_message_chars = recent_dialogue_max_message_chars
        self.actor_agent = actor_agent or ActorAgent()
        self.truncation_retry_budget_multiplier = truncation_retry_budget_multiplier

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
            standing_facts=context.standing_facts,
            context_budget=self.context_budget,
            recent_dialogue_max_message_chars=self.recent_dialogue_max_message_chars,
        )
        text, route, finish_reason, warnings = await self.generate_messages(
            route=routing.route,
            messages=messages,
        )
        return GenerationStageResult(
            text=text,
            route=route,
            finish_reason=finish_reason,
            messages=tuple(messages),
            warnings=(*self._context_truncation_warnings(context=context, retrieval=retrieval),
                      *warnings),
        )

    def _context_truncation_warnings(
        self,
        *,
        context: LoadedTurnContext,
        retrieval: RetrievalStageResult,
    ) -> tuple[str, ...]:
        """Surface otherwise-silent prompt clipping so long-session continuity
        loss is visible in TurnResult.warnings."""
        warnings: list[str] = []
        max_message = self.recent_dialogue_max_message_chars
        clipped_messages = sum(
            1
            for turn in context.recent_turns
            for message in (turn.user_message, turn.assistant_message)
            if len(message) > max_message
        )
        if clipped_messages:
            warnings.append(
                f"recent dialogue truncated: {clipped_messages} prior message(s) exceeded "
                f"{max_message} chars and were clipped for the prompt"
            )
        max_chunk = self.context_budget.max_retrieved_chunk_chars
        clipped_chunks = sum(1 for chunk in retrieval.chunks if len(chunk.text) > max_chunk)
        if clipped_chunks:
            warnings.append(
                f"retrieved context truncated: {clipped_chunks} chunk(s) exceeded {max_chunk} chars"
            )
        return tuple(warnings)

    async def generate_messages(
        self,
        *,
        route: ModelRoute,
        messages: list[LlmMessage] | tuple[LlmMessage, ...],
    ) -> tuple[str, ModelRoute, str | None, tuple[str, ...]]:
        response, used_route, complete_warnings = await self._generate_complete(
            route=route, messages=list(messages)
        )
        return response.text, used_route, response.finish_reason, complete_warnings

    async def _generate_complete(
        self,
        *,
        route: ModelRoute,
        messages: list[LlmMessage],
    ) -> tuple[LlmResponse, ModelRoute, tuple[str, ...]]:
        response, warnings = await self._generate_non_empty(route=route, messages=messages)
        if response.finish_reason != "length":
            return response, route, warnings
        retry_route = route.model_copy(
            update={"max_tokens": route.max_tokens * self.truncation_retry_budget_multiplier}
        )
        warnings = (
            *warnings,
            f"actor response truncated: finish_reason=length from {route.model}; "
            f"retried with max_tokens={retry_route.max_tokens}",
        )
        retry_response, retry_warnings = await self._generate_non_empty(
            route=retry_route, messages=messages
        )
        warnings = (*warnings, *retry_warnings)
        if retry_response.finish_reason == "length":
            raise TruncatedProviderResponseError(
                provider=route.provider.value, model=route.model
            )
        return retry_response, retry_route, warnings

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
