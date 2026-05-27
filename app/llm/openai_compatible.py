from __future__ import annotations

from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.llm.provider import LlmProvider, LlmRequest, LlmResponse


class OpenAICompatibleProvider(LlmProvider):
    def __init__(self, *, provider_name: str, base_url: str, api_key: str) -> None:
        self.provider_name = provider_name
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def generate(self, request: LlmRequest) -> LlmResponse:
        messages = cast(
            list[ChatCompletionMessageParam],
            [message.model_dump() for message in request.messages],
        )
        response = await self.client.chat.completions.create(
            model=request.model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        return LlmResponse(
            text=content,
            provider=self.provider_name,
            model=request.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            finish_reason=choice.finish_reason,
        )
