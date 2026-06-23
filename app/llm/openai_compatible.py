from __future__ import annotations

from typing import cast

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.llm.provider import (
    LlmProvider,
    LlmRequest,
    LlmResponse,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.llm.structured_output import inline_schema_refs


class OpenAICompatibleProvider(LlmProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
        # Required, not defaulted: the real values live in Settings (local 300s/1
        # retry, cloud 120s/1). A constructor default here would silently diverge
        # from config and mislead any caller that copy-pastes it.
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        self.provider_name = provider_name
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
            # No keep-alive: a wedged pooled socket otherwise gets reused by
            # the SDK retry, so both attempts hang and a rare transient stall
            # still kills the turn (2026-06-12/13 live acceptance runs).
            http_client=httpx.AsyncClient(
                timeout=timeout_seconds,
                limits=httpx.Limits(max_keepalive_connections=0),
            ),
        )

    async def generate(self, request: LlmRequest) -> LlmResponse:
        messages = cast(
            list[ChatCompletionMessageParam],
            [message.model_dump() for message in request.messages],
        )
        try:
            if request.response_schema is not None:
                response = await self.client.chat.completions.create(
                    model=request.model,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    # Neutralize server-side repetition penalties for grammar-constrained
                    # JSON: aggressive presets (e.g. --presence-penalty 1.5) penalize the
                    # repeated braces/keys/quotes that valid JSON requires, pushing the
                    # model off-grammar. No-op when the server already runs penalty 0.
                    presence_penalty=0,
                    frequency_penalty=0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": request.metadata.get("task", "structured_output"),
                            "schema": inline_schema_refs(request.response_schema),
                        },
                    },
                )
            elif request.response_format == "json":
                response = await self.client.chat.completions.create(
                    model=request.model,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    presence_penalty=0,
                    frequency_penalty=0,
                    response_format={"type": "json_object"},
                )
            else:
                response = await self.client.chat.completions.create(
                    model=request.model,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
        except APITimeoutError as exc:
            # APITimeoutError subclasses APIConnectionError, so it must be caught first.
            raise ProviderTimeoutError(
                provider=self.provider_name,
                model=request.model,
                timeout_seconds=self.timeout_seconds,
            ) from exc
        except APIConnectionError as exc:
            # Server down / wrong URL / connection refused — distinct from a timeout.
            raise ProviderUnavailableError(
                provider=self.provider_name,
                model=request.model,
                base_url=self.base_url,
            ) from exc

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
