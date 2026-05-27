# 06 — Local and Cloud Model Strategy

## Purpose

This document defines how **RoleRAG_POC** should use a local 8B-class model and an optional cloud model.

The MVP must be local-first. The local model should run the normal roleplaying loop. The cloud model is a controlled fallback, not a separate brain with different responsibilities.

The goal is simple:

> The engine should behave the same whether a local model or a cloud model generates the text. The provider may change quality, speed, cost, and capacity, but it must not change the application architecture or the rules of the roleplaying engine.

---

## Core Principle

Do not build two different systems:

- one for local roleplay
- one for cloud roleplay

Build one roleplaying engine with a provider abstraction.

The engine owns:

- orchestration
- state
- memory
- retrieval
- context assembly
- visibility filtering
- validation
- retry limits
- routing decisions

The model owns:

- generating text from a prepared prompt
- classifying narrow inputs when asked
- summarising narrow inputs when asked
- critiquing a draft when asked

The model must not own authoritative state.

---

## Target Model Setup

The MVP assumes two model classes.

### Local Model

The local model is the default model for all normal work.

Expected model class:

- 7B to 9B parameters
- quantized GGUF or Ollama-hosted model
- local OpenAI-compatible endpoint if possible
- context window large enough for compact scene prompts

Possible examples:

- Qwen 2.5 / Qwen 3 7B–8B class model
- Llama 3.1 / 3.2 8B class model
- Gemma 3 8B class model
- Mistral/Nemo-class small model

The exact model is not part of the architecture. The application should not hardcode model-specific logic except for configuration and possibly prompt style tuning.

### Cloud Model

The cloud model is optional.

It may be used for:

- repairing failed local output
- high-complexity scene transitions
- resolving conflicting retrieved context
- stronger critique
- high-quality rewrite when explicitly requested
- fallback when the local provider is unavailable

The cloud model must not bypass the engine's safety boundaries.

It receives the same kind of prepared context packet as the local model, only possibly with a larger context budget.

---

## Non-Goals

The MVP must not implement:

- separate local-only and cloud-only gameplay modes
- cloud-only memory logic
- local-only memory logic
- autonomous cloud planning loops
- model-specific persistence
- hidden provider-specific state mutation
- provider-specific world rules
- cloud usage without explicit config
- automatic expensive escalation with no trace

The provider is an implementation detail. The engine remains the product.

---

## Provider Abstraction

All model providers must implement one common interface.

Suggested location:

```text
app/llm/provider.py
```

Example interface:

```python
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
```

The rest of the app should depend on `LlmProvider`, not on Ollama, llama.cpp, OpenAI, OpenRouter, or any specific SDK.

---

## OpenAI-Compatible Provider

For the MVP, prefer OpenAI-compatible HTTP endpoints for both local and cloud models.

This keeps the code simple:

- Ollama can expose an OpenAI-compatible endpoint.
- llama.cpp server can expose an OpenAI-compatible endpoint.
- many cloud providers expose OpenAI-compatible APIs.
- the internal application does not need provider-specific code for every service.

Suggested location:

```text
app/llm/openai_compatible.py
```

Example implementation sketch:

```python
from openai import AsyncOpenAI

from app.llm.provider import LlmProvider, LlmRequest, LlmResponse


class OpenAICompatibleProvider(LlmProvider):
    def __init__(self, *, provider_name: str, base_url: str, api_key: str):
        self.provider_name = provider_name
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def generate(self, request: LlmRequest) -> LlmResponse:
        response = await self.client.chat.completions.create(
            model=request.model,
            messages=[message.model_dump() for message in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        choice = response.choices[0]
        return LlmResponse(
            text=choice.message.content or "",
            provider=self.provider_name,
            model=request.model,
            usage={
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0) if response.usage else 0,
            },
            finish_reason=choice.finish_reason,
        )
```

Do not optimize this too early. First make it boring and reliable.

---

## Configuration

Suggested location:

```text
app/config.py
```

Use Pydantic Settings.

Example `.env.example`:

```env
APP_ENV=local
LOG_LEVEL=INFO

# Local model
LOCAL_LLM_ENABLED=true
LOCAL_LLM_PROVIDER=local
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_API_KEY=ollama
LOCAL_LLM_MODEL=qwen3:8b
LOCAL_LLM_MAX_TOKENS=700
LOCAL_LLM_TEMPERATURE=0.75

# Cloud model
CLOUD_MODE=ask
CLOUD_LLM_PROVIDER=cloud
CLOUD_LLM_BASE_URL=https://api.openai.com/v1
CLOUD_LLM_API_KEY=replace_me
CLOUD_LLM_MODEL=gpt-4.1-mini
CLOUD_LLM_MAX_TOKENS=1000
CLOUD_LLM_TEMPERATURE=0.65

# Routing
MAX_LOCAL_RETRIES=1
MAX_TOTAL_AGENT_CALLS=4
ALLOW_CLOUD_REPAIR=true
ALLOW_CLOUD_CRITIC=false
ALLOW_CLOUD_MEMORY_EXTRACTION=false

# Context budgets
LOCAL_CONTEXT_CHUNK_LIMIT=6
CLOUD_CONTEXT_CHUNK_LIMIT=12
RECENT_DIALOGUE_TURNS=8
```

### Cloud Mode Values

`CLOUD_MODE` controls whether the app may use the cloud model.

Allowed values:

```text
off
ask
auto
```

Meaning:

| Mode | Meaning |
|---|---|
| `off` | Never use the cloud model. |
| `ask` | Prepare cloud fallback decision, but require explicit user confirmation. |
| `auto` | Use cloud fallback according to routing policy. |

For the MVP, default to:

```env
CLOUD_MODE=ask
```

This keeps cloud routing in confirmation mode and prevents accidental cloud usage.

---

## Model Router

Suggested location:

```text
app/llm/router.py
```

The router decides which provider/model should handle a specific model call.

The router must be deterministic. Do not ask an LLM which provider to use.

Example routing result:

```python
from enum import StrEnum
from pydantic import BaseModel


class ModelProviderName(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class CloudMode(StrEnum):
    OFF = "off"
    ASK = "ask"
    AUTO = "auto"


class ModelTask(StrEnum):
    INTENT_CLASSIFICATION = "intent_classification"
    QUERY_REWRITE = "query_rewrite"
    ACTOR_RESPONSE = "actor_response"
    CRITIC = "critic"
    MEMORY_EXTRACTION = "memory_extraction"
    REPAIR = "repair"
    SUMMARIZATION = "summarization"


class ModelRoute(BaseModel):
    provider: ModelProviderName
    model: str
    max_tokens: int
    temperature: float
    reason: str
    requires_user_confirmation: bool = False
```

Example router logic:

```python
def choose_route(
    *,
    task: ModelTask,
    cloud_mode: CloudMode,
    local_model: str,
    cloud_model: str,
    failed_local_attempts: int,
    retrieval_confidence: float | None,
    scene_complexity: int,
) -> ModelRoute:
    if cloud_mode == CloudMode.OFF:
        return ModelRoute(
            provider=ModelProviderName.LOCAL,
            model=local_model,
            max_tokens=700,
            temperature=0.75,
            reason="cloud mode is off",
        )

    should_use_cloud = False
    reason = "default local route"

    if task == ModelTask.REPAIR and failed_local_attempts > 1:
        should_use_cloud = True
        reason = "local repair failed"

    if task == ModelTask.ACTOR_RESPONSE and scene_complexity >= 4:
        should_use_cloud = True
        reason = "high scene complexity"

    if retrieval_confidence is not None and retrieval_confidence < 0.45:
        should_use_cloud = True
        reason = "low retrieval confidence"

    if should_use_cloud:
        return ModelRoute(
            provider=ModelProviderName.CLOUD,
            model=cloud_model,
            max_tokens=1000,
            temperature=0.65,
            reason=reason,
            requires_user_confirmation=cloud_mode == CloudMode.ASK,
        )

    return ModelRoute(
        provider=ModelProviderName.LOCAL,
        model=local_model,
        max_tokens=700,
        temperature=0.75,
        reason=reason,
    )
```

This is intentionally simple. Add nuance only after the MVP works.

---

## Recommended Task Routing

### Default Routing Table

| Task | Default Provider | Cloud Allowed? | Notes |
|---|---:|---:|---|
| Intent classification | local | no | Should be cheap and simple. Prefer rules first. |
| Query rewrite | local | no | Local model is enough. |
| Retrieval | none | no | Retrieval is deterministic code, not an LLM task. |
| Actor response | local | yes | Cloud fallback only for difficult scenes. |
| Critic | local | optional | Start local only. Add cloud critic later. |
| Repair | local first | yes | Cloud allowed after local failure. |
| Memory extraction | local | no | Keep private memory local by default. |
| Summarization | local | optional | Cloud only for large session compression, later. |

### MVP Recommendation

For the first implementation:

```text
intent classification -> local or rules
actor response        -> local
critic                -> local
repair                -> local retry, optional cloud fallback
memory extraction     -> local
summarization         -> local
```

Do not use the cloud model for memory extraction in the first MVP. Memory may contain private campaign details, secrets, and personal preferences.

---

## Context Budget Strategy

The local 8B model must receive a compact prompt.

The cloud model may receive a larger prompt, but it should still receive curated context, not the whole world.

### Local Budget

Recommended default:

```text
System/task instructions: 500-900 tokens
Persona packet:           300-600 tokens
Scene packet:             400-800 tokens
Recent dialogue:          800-1600 tokens
Retrieved chunks:         1000-2500 tokens
Output budget:            500-900 tokens
```

Keep local retrieved chunks to:

```text
5-6 chunks by default
8 chunks maximum for MVP
```

### Cloud Budget

Recommended default:

```text
System/task instructions: 500-900 tokens
Persona packet:           300-800 tokens
Scene packet:             600-1200 tokens
Recent dialogue:          1200-2500 tokens
Retrieved chunks:         2500-5000 tokens
Output budget:            800-1500 tokens
```

Keep cloud retrieved chunks to:

```text
8-12 chunks by default
15 chunks maximum for MVP
```

Do not use the cloud model as an excuse to stop budgeting context. Bad context is still bad context.

---

## Same Behaviour Across Providers

The same turn should follow the same pipeline regardless of provider:

```text
user message
  -> intent classification
  -> load state
  -> retrieve context
  -> build context packet
  -> choose provider
  -> generate draft
  -> critique
  -> repair if needed
  -> show response
  -> extract memory
  -> persist memory
```

Provider-specific differences are allowed only in:

- max tokens
- temperature
- timeout
- retry count
- context chunk limit
- model name
- whether user confirmation is required

Provider-specific differences are not allowed in:

- visibility rules
- memory rules
- world-state rules
- prompt safety boundaries
- whether hidden facts may be used
- whether the LLM may mutate state directly

---

## Prompt Compatibility

Prompts should work with both local and cloud models.

Avoid provider-specific prompt tricks in the MVP.

Use simple structure:

```text
ROLE
You are the roleplaying engine's actor model.

RULES
- Stay inside the scene.
- Follow the persona.
- Use relevant retrieved context.
- Do not reveal hidden facts.
- Do not mention system instructions.

PERSONA
...

SCENE
...

RELEVANT CONTEXT
...

RECENT DIALOGUE
...

USER MESSAGE
...

TASK
Write the next roleplaying response.
```

Small models perform better with stable, repeated, plain instructions. Do not over-engineer prompt prose.

---

## Temperatures

Use conservative defaults.

### Actor Response

```text
local: 0.70-0.80
cloud: 0.60-0.75
```

Actor responses need creativity, but not chaos.

### Intent Classification

```text
local: 0.00-0.20
cloud: not needed
```

Classification should be stable.

### Critic

```text
local: 0.00-0.20
cloud: 0.00-0.20
```

Critique should be deterministic.

### Memory Extraction

```text
local: 0.00-0.30
cloud: not recommended for MVP
```

Memory extraction must not invent facts.

---

## Retry and Fallback Rules

The MVP must have hard retry limits.

Recommended default:

```text
actor local draft: 1 attempt
local repair:      1 attempt
cloud repair:      optional 1 attempt
critic:            1 pass per draft
```

Maximum total LLM calls per user turn:

```text
4
```

Example:

```text
1. local actor draft
2. local critic
3. local repair
4. local critic or cloud repair
```

Do not build an infinite debate between agents.

---

## Cloud Confirmation Mode

When `CLOUD_MODE=ask`, the system should not silently call the cloud model.

For CLI MVP, return a structured result requiring confirmation:

```python
class CloudConfirmationRequired(BaseModel):
    reason: str
    estimated_task: str
    provider: str
    model: str
```

CLI behaviour:

```text
The local model failed the critic check twice.
Cloud fallback is available.
Reason: local repair failed.
Use cloud model gpt-4.1-mini for one repair attempt? [y/N]
```

If the user says no, return the best local result with a warning or ask the local model for a safer shorter response.

For FastAPI later, return HTTP `409 Conflict` or a structured response asking the client to confirm cloud usage.

---

## Privacy Rules

Local-first means private by default.

The cloud model must not receive:

- full raw session history
- full memory database
- full lore database
- unnecessary GM-only secrets
- personal notes unrelated to the current turn
- hidden facts not required for the current response

Cloud calls should receive the same curated context packet as local calls.

For the MVP, log every cloud call:

```json
{
  "provider": "cloud",
  "model": "gpt-4.1-mini",
  "task": "repair",
  "reason": "local repair failed",
  "session_id": "...",
  "prompt_chunk_count": 8
}
```

Do not log full prompts by default if they may contain private story content. Use debug mode only.

---

## Failure Handling

### Local Provider Unavailable

If the local model is unavailable:

- If cloud mode is `off`, fail clearly.
- If cloud mode is `ask`, ask before fallback.
- If cloud mode is `auto`, use cloud fallback if enabled.

Error message:

```text
Local model provider is unavailable. Start Ollama/llama.cpp or enable cloud fallback.
```

### Cloud Provider Unavailable

If cloud fallback fails:

- do not retry endlessly
- return the best local result if safe
- otherwise return a controlled failure

Error message:

```text
Cloud fallback failed. The turn was not completed. The session state was not changed.
```

### Critic Rejects All Drafts

If all drafts fail:

- do not persist memory
- do not mutate state
- return a safe failure response

Example:

```text
The system could not produce a response that passed validation. No memory or world state was changed.
```

For a personal MVP, this is acceptable. Silent bad output is worse.

---

## Observability

For every LLM call, log metadata:

- task
- provider
- model
- route reason
- duration
- prompt token estimate if available
- completion token estimate if available
- retry attempt
- whether cloud confirmation was required
- whether output passed validation

Example log event:

```json
{
  "event": "llm_call_completed",
  "task": "actor_response",
  "provider": "local",
  "model": "qwen3:8b",
  "route_reason": "default local route",
  "duration_ms": 4312,
  "attempt": 1,
  "accepted_by_critic": true
}
```

Do not require heavy observability tooling in the MVP. Structured logs are enough.

---

## Testing Requirements

Add tests for routing behaviour.

Suggested file:

```text
tests/unit/test_model_router.py
```

Test cases:

1. Cloud mode `off` always routes to local.
2. Cloud mode `ask` marks cloud route as requiring confirmation.
3. Cloud mode `auto` allows cloud route without confirmation.
4. Normal actor response uses local by default.
5. Failed local repair can route to cloud.
6. Low retrieval confidence can route to cloud if cloud is enabled.
7. Intent classification never routes to cloud in MVP.
8. Memory extraction never routes to cloud in MVP.

Add tests for provider abstraction with fake providers.

Suggested file:

```text
tests/unit/test_fake_provider.py
```

Test cases:

1. Fake local provider returns configured text.
2. Fake provider records requests.
3. Orchestrator can run with fake providers.
4. Failed provider returns controlled error.

---

## MVP Acceptance Criteria

This part is done when:

- The application has a provider abstraction.
- Local and cloud providers use the same interface.
- The router can choose local/cloud deterministically.
- Cloud mode supports `off`, `ask`, and `auto`.
- Local is the default route.
- Cloud is never used silently when mode is `ask`.
- Memory extraction stays local by default.
- Actor response can fall back to cloud only through explicit policy.
- Routing decisions are logged.
- Unit tests cover the routing policy.

---

## Coding-Agent Instructions

When implementing this part:

1. Do not hardcode one provider throughout the application.
2. Do not let agents import Ollama/OpenAI clients directly.
3. All model calls must go through the provider abstraction.
4. All provider choices must go through the router.
5. The router must be deterministic Python logic.
6. Do not use an LLM to decide whether to use cloud.
7. Do not send raw full memory or full lore to the cloud model.
8. Do not implement automatic cloud usage unless `CLOUD_MODE=auto`.
9. Do not persist state if all model attempts fail validation.
10. Add tests before adding complicated routing rules.

---

## Recommended First Implementation Order

1. Create `app/llm/provider.py`.
2. Create `app/llm/openai_compatible.py`.
3. Create `app/llm/router.py`.
4. Add config fields for local/cloud providers.
5. Add fake provider for tests.
6. Add routing unit tests.
7. Wire the actor agent through the router.
8. Wire the critic through the router.
9. Add cloud confirmation behaviour for CLI.
10. Add structured logging for model calls.

Do not implement provider-specific hacks before this basic path works.

---

## Final Design Rule

The local model and the cloud model are interchangeable execution backends.

The roleplaying engine must remain the same.

If switching from local to cloud changes what the system is allowed to know, remember, reveal, mutate, or retrieve, the design is wrong.
