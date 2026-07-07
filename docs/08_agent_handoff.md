# 08 — Agent Handoff Guide

> Reviewed: 2026-07-07 @ 7888ee7

## Purpose

This guide is the fastest safe onboarding path for a future coding agent or developer.

## Read This First

Coding agents: [CLAUDE.md](../CLAUDE.md) at the repo root is the condensed entry point
(commands, conventions, invariants) and is auto-loaded by Claude Code sessions.

Read in this order:

1. [README.md](../README.md)
2. [docs/09_current_architecture_map.md](09_current_architecture_map.md)
3. [docs/21_fable_handoff_reasoning.md](21_fable_handoff_reasoning.md) — predecessor-agent
   reasoning chains: why the architecture is shaped this way, and how to think about changes
4. [app/composition.py](../app/composition.py)
5. [app/orchestration/turn_orchestrator.py](../app/orchestration/turn_orchestrator.py)
6. [tests/](../tests)

## Mental Model

The repository is a bounded backend engine, not an autonomous-agent framework.

- CLI and API are entry points.
- `composition.py` wires dependencies.
- `TurnOrchestrator` owns lifecycle coordination.
- agents generate or evaluate text only.
- SQLite owns sessions, turns, and durable memory episodes.
- Qdrant is a vector index for retrieval, not the source of truth.

## Safe Working Rules

- Do not let the LLM own authoritative state.
- Do not move retrieval, persistence, or routing into the actor.
- Do not put orchestration logic in API routes.
- Do not bypass visibility filtering for player-facing actor prompts.
- Do not make tests depend on real providers or live Qdrant.
- The provider is session-bound: it is chosen once at session creation (`POST /sessions`) and is
  immutable for the session's lifetime; every task runs on the bound provider and no per-turn cloud
  flags exist. `CLOUD_MODE` gates cloud session creation only: `off` rejects cloud sessions with
  400 `cloud_unavailable`, `ask` requires one interactive confirmation at creation (`typer.confirm`
  in the CLI, `window.confirm` in the SPA), `auto` creates them silently. See
  [docs/06_local_cloud_model_strategy.md](06_local_cloud_model_strategy.md).

## Where To Change Things

### CLI or API surface

- CLI commands: [app/cli.py](../app/cli.py)
- API schemas and routes: [app/api/](../app/api)

### Turn lifecycle

- orchestration: [app/orchestration/turn_orchestrator.py](../app/orchestration/turn_orchestrator.py)
- actor prompt assembly: [app/orchestration/context_builder.py](../app/orchestration/context_builder.py)
- context limits: [app/orchestration/context_budget.py](../app/orchestration/context_budget.py)

### Routing and provider behavior

- settings: [app/config.py](../app/config.py)
- router: [app/llm/router.py](../app/llm/router.py)
- provider implementation: [app/llm/openai_compatible.py](../app/llm/openai_compatible.py)

### Retrieval and memory

- ingestion: [app/rag/ingestion.py](../app/rag/ingestion.py)
- retrieval: [app/rag/retriever.py](../app/rag/retriever.py)
- vector store: [app/rag/vector_store.py](../app/rag/vector_store.py)
- memory extraction: [app/agents/memory_curator.py](../app/agents/memory_curator.py)
- persistence: [app/persistence/repositories.py](../app/persistence/repositories.py)
- diagnostics and smoke runner: [app/diagnostics/](../app/diagnostics)

### Content authoring and validation

- validation and templates: [app/content/](../app/content)
- JSON loading for runtime use: [app/persistence/file_loader.py](../app/persistence/file_loader.py)
- demo data roots: [data/](../data)

## Known Danger Zones

- `TurnOrchestrator` is the highest-leverage file. Small changes there can affect routing, persistence, warnings, and repair flow.
- Retrieval is intentionally fail-open. Do not silently change that without updating tests and docs.
- Durable SQLite memories are indexed into `session_memory` after persistence. If indexing fails,
  the turn remains valid and `reindex-memories` can repair the derived Qdrant index.
- Retrieval ranking is intentionally deterministic and transparent. Keep boost policy in `app/rag`,
  preserve the original vector score, and avoid player-facing hidden-text diagnostics.
- Repair runs one bounded same-provider pass on the session's bound provider (same route family as
  actor/critic); a rejected repair goes straight to controlled failure — there is no
  local-then-cloud escalation and no cross-provider fallback. Provider retries and the
  structured-truncation budget remain configurable (`LOCAL_LLM_MAX_RETRIES`, `CLOUD_LLM_MAX_RETRIES`,
  `TRUNCATION_RETRY_BUDGET_MULTIPLIER`).

## Verification Before Claiming Success

Run:

```bash
ruff check .
mypy .
pytest
python -m app.evals.regression_runner
```

If you change command docs or settings docs, also verify:

```bash
python -m app.cli --help
python -m app.cli health
python -m app.cli doctor
python -m app.cli smoke-run
python -m app.cli validate-content
python -m app.cli config
python -m app.cli ingest --help
python -m app.cli reindex-memories --help
python -m app.cli retrieve-debug --help
rolerag --help
```

For optional live dependency checks:

```bash
python -m app.cli doctor --check-qdrant --check-local-provider
python -m app.cli smoke-run --real-runtime
```

Interpretation:

- `health` is config-only.
- `doctor` verifies temporary SQLite initialization, demo data loading, and optional external reachability checks.
- `smoke-run` exercises the real orchestrator, persistence, memory indexing, and retrieval path with fake provider responses and in-memory retrieval.
- `validate-content` is a read-only authoring check for worlds, scenes, personas, and optional lore manifests.
- live checks stay shallow and read-only; they do not run real completions or mutate Qdrant.

Authoring rules:

- keep content validation deterministic and conservative
- reuse existing Pydantic domain models rather than duplicating schema logic
- do not leak GM-private or character-private text in validation output
- treat standalone scenario packs as runtime content roots once validated; sessions persist the selected root

## Safe Next Work

The next reasonable implementation areas are documented in
[docs/10_next_steps_after_mvp.md](10_next_steps_after_mvp.md). For the RAG core
specifically (larger scenarios, ~27B local models), the verified roadmap is
[docs/22_rag_scaling_roadmap.md](22_rag_scaling_roadmap.md).

Stay conservative:

- preserve backend boundaries
- add tests with each behavior change
- prefer targeted improvements over architecture churn
