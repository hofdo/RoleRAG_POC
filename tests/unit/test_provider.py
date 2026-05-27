import pytest

from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse


class FakeProvider(LlmProvider):
    async def generate(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(
            text=request.messages[-1].content,
            provider="fake",
            model=request.model,
            usage={"total_tokens": 1},
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_provider_abstraction_uses_request_response_models() -> None:
    provider = FakeProvider()
    request = LlmRequest(
        messages=[LlmMessage(role="user", content="hello")],
        model="local-model",
        max_tokens=32,
        temperature=0.2,
    )

    response = await provider.generate(request)

    assert response.text == "hello"
    assert response.provider == "fake"
    assert response.model == "local-model"
