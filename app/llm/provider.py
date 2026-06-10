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
