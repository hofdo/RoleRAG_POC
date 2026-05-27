from __future__ import annotations

from app.llm.provider import LlmMessage, LlmProvider, LlmRequest
from app.llm.router import ModelRoute


class ActorAgent:
    async def generate(
        self,
        *,
        provider: LlmProvider,
        route: ModelRoute,
        messages: list[LlmMessage],
    ) -> str:
        request = LlmRequest(
            messages=messages,
            model=route.model,
            max_tokens=route.max_tokens,
            temperature=route.temperature,
        )
        response = await provider.generate(request)
        return response.text
