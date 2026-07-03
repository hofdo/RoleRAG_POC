# 02 — Current MVP Architecture

## Purpose

This document explains the architecture that is implemented in the repository now. It is a current-state reference, not a speculative skeleton.

## Architectural Rule

The application owns state, routing, retrieval, visibility, persistence, and retry limits. LLM providers generate or evaluate text inside those boundaries.

## High-Level Flow

```text
CLI / web UI (/app SPA) / FastAPI
  -> composition
  -> TurnOrchestrator
      -> load session, world, scene, persona
      -> retrieve actor context when enabled (fail-open)
      -> choose route
      -> ActorAgent.generate
      -> validate draft
      -> CriticAgent.evaluate
      -> repair if rejected (on the session's bound provider)
      -> output-side secret containment
      -> persist turn
      -> MemoryCurator.curate (+ index, dedup, consolidate)
      -> persist memory episodes
```

See [docs/README.md](README.md) for the rendered component, turn-pipeline, and routing diagrams.

## Module Layout

### Entry points

- [app/cli.py](../app/cli.py): Typer commands for configuration, sessions, routing, ingestion, and turns.
- [app/main.py](../app/main.py): FastAPI application bootstrap.
- [app/api/routes.py](../app/api/routes.py): thin HTTP adapters over shared services.
- [frontend/](../frontend): Angular 19 SPA (play, RAG inspector, analytics, eval), built to
  `frontend/dist/frontend/browser` and mounted at `/app` by `app/main.py`.

### Wiring and settings

- [app/config.py](../app/config.py): `Settings` model loaded from `.env`.
- [app/composition.py](../app/composition.py): provider, repository, retriever, and orchestrator construction.

### Domain and orchestration

- [app/domain/models.py](../app/domain/models.py): typed state, turn, memory, and retrieval models.
- [app/domain/visibility.py](../app/domain/visibility.py): `player`, `gm`, and `character_private`.
- [app/orchestration/turn_orchestrator.py](../app/orchestration/turn_orchestrator.py): the core application service.
- [app/orchestration/context_builder.py](../app/orchestration/context_builder.py): actor prompt assembly.
- [app/orchestration/context_budget.py](../app/orchestration/context_budget.py): retrieved-context budget enforcement.

### Agents and provider layer

- [app/agents/actor_agent.py](../app/agents/actor_agent.py): generation dispatch only.
- [app/agents/critic_agent.py](../app/agents/critic_agent.py): structured critique and repair-message construction.
- [app/agents/memory_curator.py](../app/agents/memory_curator.py): structured memory extraction.
- [app/llm/provider.py](../app/llm/provider.py): common request/response models and provider protocol.
- [app/llm/openai_compatible.py](../app/llm/openai_compatible.py): OpenAI-compatible provider implementation.
- [app/llm/router.py](../app/llm/router.py): deterministic route selection.

### Persistence and retrieval

- [app/persistence/file_loader.py](../app/persistence/file_loader.py): JSON world, scene, and persona loading.
- [app/persistence/sqlite.py](../app/persistence/sqlite.py): SQLite connection and schema initialization.
- [app/persistence/repositories.py](../app/persistence/repositories.py): session, turn, and memory repositories.
- [app/memory/store.py](../app/memory/store.py): recent-dialogue and memory-episode store adapters.
- [app/rag/](../app/rag): chunking, embeddings, ingestion, retriever, and vector-store abstractions.

### Evals

- [app/evals/](../app/evals): deterministic fixture-based regression checks.

## Ownership Boundaries

### Authoritative data

- JSON files under `data/` own demo world, scene, and persona definitions.
- SQLite owns sessions, turns, and durable memory episodes.
- Qdrant stores retrievable chunk vectors and payloads.

### Non-authoritative data

- LLM outputs are suggestions until the application validates and persists them.
- Retrieval is a context source, not a state store.

## Runtime Boundaries

### CLI, API, and local UI

- accept user input
- build shared services
- delegate to the orchestrator
- do not build prompts or access the vector store directly

The browser UI calls the API rather than shared services directly. It sends documented request
fields only, renders API content as text, and does not own scenario-pack, persistence, retrieval,
validation, routing, or memory logic.

### TurnOrchestrator

- validates session and scene references
- builds retrieval query
- computes routing inputs
- dispatches actor generation
- enforces critique and repair bounds
- persists the final turn
- invokes memory curation after the final response

### ActorAgent

- sends prepared messages to the selected provider
- does not retrieve, persist, or choose routes

### CriticAgent

- evaluates drafts with structured JSON output
- can inspect hidden context for leak detection
- runs on the session's bound provider

### MemoryCurator

- extracts structured memory candidates from the completed turn
- runs on the session's bound provider
- does not persist directly

## Visibility Model

The only implemented visibility values are:

- `player`
- `gm`
- `character_private`

Actor prompts include only `player`-visible retrieved chunks. Hidden fields may still be used by critic evaluation to detect leaks, but they are not passed through as player-facing story context.

## Cloud Routing

The router in [app/llm/router.py](../app/llm/router.py) is deterministic. Provider is a
session-bound choice made once at session creation; `choose_route` only maps that bound
provider to a route — there is no escalation, no fallback, and no per-turn override.

- `off`: creating a `cloud` session is rejected (`400 cloud_unavailable`); local sessions unaffected
- `ask`: creating a `cloud` session requires interactive confirmation once, at creation; once
  confirmed, the whole session runs on cloud with no further prompts
- `auto`: cloud sessions are created without an interactive prompt

See [docs/06_local_cloud_model_strategy.md](06_local_cloud_model_strategy.md) for the full model.

## Retrieval Architecture

Runtime retrieval uses:

- `canon_lore`
- `session_memory`
- `persona_memory`

Current limitations:

- ingestion supports `.md` and `.txt` only
- retrieval failure is non-fatal for turn execution
- persisted SQLite memory episodes are indexed into `session_memory` after curation

## Testing Architecture

The repository relies on three test layers:

- unit tests for models, router, persistence, prompt assembly, and retrieval helpers
- integration tests for CLI, API, persistence, repair flow, and memory curation
- Node tests for browser request shaping, buffered SSE parsing, transcript state, and thin-client boundaries
- eval tests and a standalone regression runner for retrieval, visibility, memory, role consistency, and cloud-routing regressions

## Extension Rule

Any future change should preserve these invariants:

- LLMs do not own authoritative state
- actor prompts remain visibility-filtered
- route handlers remain thin
- retrieval stays optional at runtime failure points
- tests remain runnable without real providers

See [docs/09_current_architecture_map.md](09_current_architecture_map.md) for a module map and [docs/08_agent_handoff.md](08_agent_handoff.md) for contributor guidance.
