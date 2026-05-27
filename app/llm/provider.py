from __future__ import annotations

from abc import ABC, abstractmethod

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
    metadata: dict[str, str] = Field(default_factory=dict)


class LlmResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str | None = None


class LlmProvider(ABC):
    @abstractmethod
    async def generate(self, request: LlmRequest) -> LlmResponse:
        raise NotImplementedError
