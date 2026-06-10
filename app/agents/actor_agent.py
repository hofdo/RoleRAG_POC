from __future__ import annotations

from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from app.llm.router import ModelRoute


class ActorAgent:
    async def generate(
        self,
        *,
        provider: LlmProvider,
        route: ModelRoute,
        messages: list[LlmMessage],
    ) -> LlmResponse:
        request = LlmRequest(
            messages=messages,
            model=route.model,
            max_tokens=route.max_tokens,
            temperature=route.temperature,
        )
        return await provider.generate(request)
