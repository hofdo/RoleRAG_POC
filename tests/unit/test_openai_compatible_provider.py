from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError

from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.provider import (
    LlmMessage,
    LlmRequest,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


def _request() -> LlmRequest:
    return LlmRequest(
        messages=[LlmMessage(role="user", content="Hello.")],
        model="local-model",
        max_tokens=100,
        temperature=0.5,
    )


def test_provider_configures_explicit_timeout_and_retries() -> None:
    provider = OpenAICompatibleProvider(
        provider_name="local",
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        timeout_seconds=42.5,
        max_retries=3,
    )

    assert provider.client.timeout == 42.5
    assert provider.client.max_retries == 3


@pytest.mark.asyncio
async def test_provider_translates_api_timeout_into_provider_timeout_error() -> None:
    provider = OpenAICompatibleProvider(
        provider_name="local",
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        timeout_seconds=0.5,
        max_retries=0,
    )

    async def raise_timeout(**_: object) -> object:
        raise APITimeoutError(request=httpx.Request("POST", "http://127.0.0.1:8080/v1"))

    completions = provider.client.chat.completions
    completions.create = raise_timeout  # type: ignore[method-assign, assignment]

    with pytest.raises(ProviderTimeoutError) as exc_info:
        await provider.generate(_request())

    assert exc_info.value.provider == "local"
    assert exc_info.value.model == "local-model"
    assert exc_info.value.timeout_seconds == 0.5


@pytest.mark.asyncio
async def test_provider_translates_connection_error_into_provider_unavailable_error() -> None:
    provider = OpenAICompatibleProvider(
        provider_name="local",
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        timeout_seconds=30.0,
        max_retries=0,
    )

    async def raise_connection(**_: object) -> object:
        raise APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:8080/v1"))

    completions = provider.client.chat.completions
    completions.create = raise_connection  # type: ignore[method-assign, assignment]

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.generate(_request())

    assert exc_info.value.provider == "local"
    assert exc_info.value.model == "local-model"
    assert exc_info.value.base_url == "http://127.0.0.1:8080/v1"


@pytest.mark.asyncio
async def test_provider_sends_json_schema_response_format_when_schema_present() -> None:
    provider = OpenAICompatibleProvider(
        provider_name="local",
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        timeout_seconds=30.0,
        max_retries=0,
    )
    captured: dict[str, object] = {}

    async def capture(**kwargs: object) -> object:
        captured.update(kwargs)

        class Choice:
            message = type("Message", (), {"content": '{"accepted": true}'})()
            finish_reason = "stop"

        class Response:
            choices = [Choice()]
            usage = None

        return Response()

    completions = provider.client.chat.completions
    completions.create = capture  # type: ignore[method-assign, assignment]

    schema = {"type": "object", "properties": {"accepted": {"type": "boolean"}}}
    request = LlmRequest(
        messages=[LlmMessage(role="user", content="Check.")],
        model="local-model",
        max_tokens=100,
        temperature=0.0,
        response_format="json",
        response_schema=schema,
        metadata={"task": "critic"},
    )

    response = await provider.generate(request)

    assert response.text == '{"accepted": true}'
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "critic", "schema": schema},
    }
    # Repetition penalties are neutralized so aggressive server presets can't corrupt JSON.
    assert captured["presence_penalty"] == 0
    assert captured["frequency_penalty"] == 0


@pytest.mark.asyncio
async def test_provider_leaves_penalties_to_server_for_unstructured_actor_calls() -> None:
    provider = OpenAICompatibleProvider(
        provider_name="local",
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        timeout_seconds=30.0,
        max_retries=0,
    )
    captured: dict[str, object] = {}

    async def capture(**kwargs: object) -> object:
        captured.update(kwargs)

        class Choice:
            message = type("Message", (), {"content": "A reply."})()
            finish_reason = "stop"

        class Response:
            choices = [Choice()]
            usage = None

        return Response()

    provider.client.chat.completions.create = capture  # type: ignore[method-assign, assignment]

    await provider.generate(_request())

    # Actor turns keep the server's intended sampling - no penalty override sent.
    assert "presence_penalty" not in captured
    assert "frequency_penalty" not in captured


@pytest.mark.asyncio
async def test_provider_inlines_schema_refs_before_sending() -> None:
    provider = OpenAICompatibleProvider(
        provider_name="local",
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        timeout_seconds=30.0,
        max_retries=0,
    )
    captured: dict[str, object] = {}

    async def capture(**kwargs: object) -> object:
        captured.update(kwargs)

        class Choice:
            message = type("Message", (), {"content": '{"write_memory": false}'})()
            finish_reason = "stop"

        class Response:
            choices = [Choice()]
            usage = None

        return Response()

    completions = provider.client.chat.completions
    completions.create = capture  # type: ignore[method-assign, assignment]

    schema = {
        "$defs": {"Visibility": {"enum": ["player"], "type": "string"}},
        "type": "object",
        "properties": {"visibility": {"$ref": "#/$defs/Visibility"}},
    }
    request = LlmRequest(
        messages=[LlmMessage(role="user", content="Curate.")],
        model="local-model",
        max_tokens=100,
        temperature=0.0,
        response_format="json",
        response_schema=schema,
        metadata={"task": "memory_extraction"},
    )

    await provider.generate(request)

    sent = captured["response_format"]
    assert sent == {
        "type": "json_schema",
        "json_schema": {
            "name": "memory_extraction",
            "schema": {
                "type": "object",
                "properties": {"visibility": {"enum": ["player"], "type": "string"}},
            },
        },
    }
