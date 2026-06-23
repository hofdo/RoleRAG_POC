from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LlmMessage(BaseModel):
    role: str
    content: str


class LlmRequest(BaseModel):
    messages: list[LlmMessage]
    model: str
    max_tokens: int
    temperature: float
    response_format: str | None = None
    response_schema: dict[str, Any] | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class LlmResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str | None = None


class ProviderTimeoutError(RuntimeError):
    def __init__(self, *, provider: str, model: str, timeout_seconds: float | None) -> None:
        suffix = f" after {timeout_seconds} seconds" if timeout_seconds is not None else ""
        super().__init__(f"{provider} provider request for {model} timed out{suffix}")
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds


class LlmProvider(ABC):
    @abstractmethod
    async def generate(self, request: LlmRequest) -> LlmResponse:
        raise NotImplementedError


async def generate_with_truncation_retry(
    provider: LlmProvider,
    request: LlmRequest,
    *,
    # Default mirrors Settings.truncation_retry_budget_multiplier; composition wires the
    # configured value in via the critic/curator agents (single source of truth).
    multiplier: int = 2,
) -> LlmResponse:
    """Generate, retrying once with a larger budget when the model stopped on length.

    Grammar-constrained JSON (critic, memory extraction) must close every brace; a
    verbose model that hits max_tokens truncates mid-string and fails the parse. The
    actor path already retries on finish_reason=length; this gives structured calls the
    same protection so a tight budget degrades to a retry, not a silent skip.

    Divergence from the actor path is deliberate: a still-truncated retry is returned
    here for the caller's Pydantic parse to reject, whereas TurnGenerationStage raises
    TruncatedProviderResponseError because plain actor text has no schema to fail on.
    """
    response = await provider.generate(request)
    if response.finish_reason != "length":
        return response
    retried = request.model_copy(update={"max_tokens": request.max_tokens * multiplier})
    return await provider.generate(retried)
