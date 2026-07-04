# 06 — Current Local and Cloud Model Strategy

> Reviewed: 2026-07-04 @ 571acc8

## Purpose

This document records the local/cloud strategy that is implemented today.

## Core Rule

There is one roleplaying engine and one routing layer. The provider (`local` or `cloud`)
is a **session-bound choice**: it is picked once at session creation and is immutable for
that session's lifetime. Provider selection changes capacity, not application rules —
cloud is a peer choice, never a rescue mechanism (decision 2026-07-02, see
[docs/BACKLOG.md](BACKLOG.md)).

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

- `actor_response`
- `critic`
- `memory_extraction`
- `repair`

Every task runs on the session's bound provider. `choose_route` does nothing but map
`session_provider` to a route: there is no escalation, no fallback, and no per-turn
override. The turn request body carries no provider or routing flags (no `request_cloud`,
`cloud_confirmed`, or `force_local` — see [docs/12_api_contract.md](12_api_contract.md)).

Structured tasks (critic, memory extraction) pin temperature `0.0` on both providers;
grammar-constrained JSON wants deterministic, greedy decoding.

## Cloud Modes

`CLOUD_MODE` gates **cloud session creation** only. It never affects turns, and existing
sessions are unaffected by changing it — a session's provider never changes after creation.

### `off`

- creating a `cloud` session is rejected with `400 cloud_unavailable`
- local sessions are unaffected

### `ask`

- creating a `cloud` session requires interactive confirmation **once, at creation**:
  the CLI prompts via `typer.confirm`, the SPA via `window.confirm`
- once confirmed, the whole session runs on cloud with no further prompts
- there is no per-turn confirmation flow; `CreateTurnResponse.status` is always
  `"completed"` on success, and no `confirmation_required` status or SSE frame exists

### `auto`

- cloud sessions are created silently, without an interactive prompt

Regardless of `CLOUD_MODE`, creating a `cloud` session also requires a configured cloud
API key.

## Privacy Boundaries

- actor and repair prompts use curated context, not the full database
- cloud calls do not bypass visibility filtering
- hidden authored content — persona `secrets` and `forbidden_knowledge` fields, and scene
  `gm_private_summary` — never enters a cloud request. This is enforced structurally by an
  `include_hidden` gate in [app/orchestration/stages/critique.py](../app/orchestration/stages/critique.py)
  (`include_hidden=route.provider == ModelProviderName.LOCAL`), which only ever allows
  hidden fields into a prompt when the route is local, plus a provider-binding eval test
  that pins the invariant
- critic evaluation and memory extraction follow the session's bound provider like every
  other task; on a cloud session they run on cloud, with hidden content stripped as above

## Failure Handling

### Provider unavailable

- there is no cross-provider fallback: a local session never escalates to cloud, and a
  cloud session never falls back to local
- an unreachable model server surfaces as a clean `ProviderUnavailableError`
  (API `503 provider_unavailable`, CLI exit 1) rather than a raw traceback; a request that
  exceeds the timeout surfaces as `ProviderTimeoutError` (API `504 provider_timeout`)

### Validation failures

- the runtime does not loop indefinitely
- repair runs on the session's bound provider, bounded by the configured retry budgets
- the orchestrator can return a controlled failure text when validation cannot be satisfied safely

## Known Limitations

- no structured cloud-call audit log exposed outside route metadata and warnings
- provider retries and the structured-truncation retry budget are configurable
  (`LOCAL_LLM_MAX_RETRIES`, `CLOUD_LLM_MAX_RETRIES`, `TRUNCATION_RETRY_BUDGET_MULTIPLIER`)

## Invariants to Preserve

- do not fork the gameplay engine into separate local and cloud architectures
- do not let cloud bypass prompt visibility boundaries
- do not let hidden authored content (persona secrets/forbidden knowledge, scene
  `gm_private_summary`) enter a cloud request
- do not reintroduce cross-provider escalation, fallback, or per-turn provider overrides
- do not make routing probabilistic
