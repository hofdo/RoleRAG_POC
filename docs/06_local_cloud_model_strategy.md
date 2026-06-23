# 06 — Current Local and Cloud Model Strategy

## Purpose

This document records the local/cloud strategy that is implemented today.

## Core Rule

There is one roleplaying engine and one routing layer. Provider selection changes capacity and fallback behavior, not application rules.

## Provider Layer

The common request/response boundary lives in [app/llm/provider.py](../app/llm/provider.py).

The concrete implementation in the repository is [app/llm/openai_compatible.py](../app/llm/openai_compatible.py).

Both local and cloud model access go through that same provider shape.

## Settings

The settings are defined in [app/config.py](../app/config.py); the model-strategy-relevant ones
are below. See [`.env.example`](../.env.example) for the full surface with per-field commentary.

```env
LOCAL_LLM_BASE_URL=http://127.0.0.1:8080/v1
LOCAL_LLM_API_KEY=local
LOCAL_LLM_MODEL=chatgpt-onnechan
LOCAL_LLM_MAX_TOKENS=500
# Critic/memory JSON budget; verbose models truncate mid-JSON at 350, 640 fits.
LOCAL_STRUCTURED_MAX_TOKENS=640
LOCAL_LLM_TEMPERATURE=0.75
# Slow dense models can take minutes; one retry absorbs a transient request stall.
LOCAL_LLM_TIMEOUT_SECONDS=300
LOCAL_LLM_MAX_RETRIES=1

CLOUD_MODE=ask
CLOUD_LLM_BASE_URL=https://api.openai.com/v1
CLOUD_LLM_API_KEY=replace_me
CLOUD_LLM_MODEL=gpt-4.1-mini
CLOUD_LLM_MAX_TOKENS=1000
CLOUD_LLM_TEMPERATURE=0.65
CLOUD_LLM_TIMEOUT_SECONDS=120
CLOUD_LLM_MAX_RETRIES=1
```

If `CLOUD_LLM_API_KEY=replace_me`, [app/composition.py](../app/composition.py) does not build a cloud provider.

## Implemented Routing

Routing is implemented in [app/llm/router.py](../app/llm/router.py).

Implemented model tasks:

- `intent_classification`
- `actor_response`
- `critic`
- `memory_extraction`
- `repair`
- `summarization`

Only `actor_response` and `repair` can currently route to cloud.

### Local-only tasks

- critic stays local with temperature `0.0`
- memory extraction stays local with temperature `0.0`

### Cloud-eligible cases

For actor or repair tasks, cloud may be selected when:

- the user explicitly requests cloud for an actor turn
- the local provider is unavailable
- local repair already failed
- scene complexity is high
- retrieval confidence is low

## Cloud Modes

### `off`

- cloud is never used
- if cloud would have been selected, the route reason records that cloud was blocked by policy

### `ask`

- cloud routes may still be chosen by the router with `requires_user_confirmation=True`
- the runtime never executes a confirmation-required cloud call; it returns a
  `confirmation_required` turn result before any generation or persistence happens
- the API reports `status: "confirmation_required"` (JSON) or a
  `confirmation_required` SSE frame; the client resubmits the same message with
  `cloud_confirmed: true` (approve) or `force_local: true` (decline)
- the play UI shows a confirmation dialog and resubmits automatically
- the CLI prompts interactively, or accepts `--confirm-cloud` / `--force-local`
  for non-interactive use
- a declined turn is answered locally with route reason `user declined cloud`

### `auto`

- cloud may be used automatically when the router selects it and a cloud provider is configured

## Privacy Boundaries

- actor and repair prompts use curated context, not the full database
- cloud calls do not bypass visibility filtering
- critic and memory extraction remain local
- memory extraction is not sent to cloud in the current design

## Failure Handling

### Local provider unavailable

- if policy permits and cloud is configured, actor or repair work may fall back to cloud
- in `ask` mode, confirmation requirements still block silent cloud dispatch
- with no usable cloud fallback, an unreachable model server surfaces as a clean
  `ProviderUnavailableError` (API `503 provider_unavailable`, CLI exit 1) rather than a
  raw traceback; a request that exceeds the timeout surfaces as `ProviderTimeoutError`
  (API `504 provider_timeout`)

### Cloud unavailable

- the runtime does not loop indefinitely
- bounded local behavior still applies
- the orchestrator can return a controlled failure text when validation cannot be satisfied safely

## Known Limitations

- no structured cloud-call audit log exposed outside route metadata and warnings
- the repair escalation is fixed in code (local, then cloud); provider retries and the
  structured-truncation retry budget are configurable (`LOCAL_LLM_MAX_RETRIES`,
  `CLOUD_LLM_MAX_RETRIES`, `TRUNCATION_RETRY_BUDGET_MULTIPLIER`)

## Invariants to Preserve

- do not fork the gameplay engine into separate local and cloud architectures
- do not let cloud bypass prompt visibility boundaries
- do not move memory extraction to cloud in the MVP
- do not make routing probabilistic
