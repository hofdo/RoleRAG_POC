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

Top-level packages only. The maintained file-level map — including the orchestration stage
modules — lives in [docs/09_current_architecture_map.md](09_current_architecture_map.md).

- [app/cli.py](../app/cli.py), [app/main.py](../app/main.py), [app/api/](../app/api): Typer CLI,
  FastAPI bootstrap, and thin HTTP adapters over shared services.
- [frontend/](../frontend): Angular 19 SPA (play, RAG inspector, analytics, eval), built to
  `frontend/dist/frontend/browser` and mounted at `/app` by `app/main.py`.
- [app/config.py](../app/config.py), [app/composition.py](../app/composition.py): `Settings`
  loaded from `.env`; provider, repository, retriever, and orchestrator construction.
- [app/domain/](../app/domain): typed state, turn, memory, and retrieval models plus the
  visibility values.
- [app/orchestration/](../app/orchestration): the turn pipeline as an injectable stage package
  (`stages/`), plus prompt assembly, context budgeting, session-canon building, draft
  validation, and turn-error classification.
- [app/agents/](../app/agents): actor, critic, memory curator, and the deterministic secret
  guard.
- [app/llm/](../app/llm): provider protocol, OpenAI-compatible implementation, deterministic
  router, and structured-output handling.
- [app/persistence/](../app/persistence), [app/memory/](../app/memory): JSON content loading,
  SQLite schema and repositories, memory store adapters, dedup, and consolidation.
- [app/rag/](../app/rag): chunking, embeddings, ingestion, retriever, ranking, and vector-store
  abstractions.
- [app/evals/](../app/evals), [app/diagnostics/](../app/diagnostics): deterministic
  fixture-based regression checks and runtime verification tooling.

## Ownership Boundaries

### Authoritative data

- JSON files under `data/` own demo world, scene, and persona definitions.
- SQLite owns sessions, turns, durable memory episodes, and pinned canon facts.
- Qdrant stores retrievable chunk vectors and payloads.

The SQLite data model is four tables (`sessions`, `turns`, `memory_episodes`, `canon_facts`),
created on startup with additive-only auto-migration: missing columns are added via
`ALTER TABLE` in [app/persistence/sqlite.py](../app/persistence/sqlite.py), never dropped or
rewritten. The Qdrant collections (`canon_lore`, `session_memory`, `persona_memory`) are
derived indexes that can be rebuilt from SQLite and re-ingested lore.

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

The repository relies on five test layers:

- unit tests for models, router, persistence, prompt assembly, and retrieval helpers
- integration tests for CLI, API, persistence, repair flow, and memory curation
- Node tests for browser request shaping, buffered SSE parsing, transcript state, and thin-client boundaries
- eval tests and a standalone regression runner for retrieval, visibility, memory, role consistency, and provider-binding regressions
- a Playwright e2e spec (`tests/e2e/spa-play.spec.mjs`, run via `npm run test:e2e-spa`) that drives the built SPA through a live play-through; it needs the full stack and a model server, so it runs on demand rather than in the default `pytest` gate

## Extension Rule

Any future change should preserve these invariants:

- LLMs do not own authoritative state
- actor prompts remain visibility-filtered
- route handlers remain thin
- retrieval stays optional at runtime failure points
- tests remain runnable without real providers

See [docs/09_current_architecture_map.md](09_current_architecture_map.md) for a module map and [docs/08_agent_handoff.md](08_agent_handoff.md) for contributor guidance.
