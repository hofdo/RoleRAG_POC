import pytest

from app.llm.provider import (
    LlmMessage,
    LlmProvider,
    LlmRequest,
    LlmResponse,
    ProviderUnavailableError,
    resolve_provider,
)
from app.llm.router import ModelProviderName, ModelRoute


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


def test_resolve_provider_raises_provider_unavailable_when_cloud_route_has_no_cloud_provider() -> (
    None
):
    """A cloud session's turn, on a machine whose cloud API key was removed after the
    session was created, must fail as a clean ProviderUnavailableError (API 503) --
    not a bare RuntimeError (API 500). The session-bound provider never falls back to
    local, so this is the only way the missing-cloud-provider case is reachable."""
    route = ModelRoute(
        provider=ModelProviderName.CLOUD,
        model="gpt-4.1-mini",
        max_tokens=1000,
        temperature=0.65,
        reason="session provider: cloud",
    )

    with pytest.raises(ProviderUnavailableError) as exc_info:
        resolve_provider(route, local=FakeProvider(), cloud=None)

    assert exc_info.value.provider == "cloud"
    assert exc_info.value.model == "gpt-4.1-mini"
