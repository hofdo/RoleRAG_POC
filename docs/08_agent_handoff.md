# 08 — Agent Handoff Guide

## Purpose

This guide is the fastest safe onboarding path for a future coding agent or developer.

## Read This First

Read in this order:

1. [README.md](../README.md)
2. [docs/09_current_architecture_map.md](09_current_architecture_map.md)
3. [app/composition.py](../app/composition.py)
4. [app/orchestration/turn_orchestrator.py](../app/orchestration/turn_orchestrator.py)
5. [tests/](../tests)

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
- Do not assume `CLOUD_MODE=ask` provides an interactive approval loop. It does not.

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

## Known Danger Zones

- `TurnOrchestrator` is the highest-leverage file. Small changes there can affect routing, persistence, warnings, and repair flow.
- Retrieval is intentionally fail-open. Do not silently change that without updating tests and docs.
- Durable SQLite memories are indexed into `session_memory` after persistence. If indexing fails,
  the turn remains valid and `reindex-memories` can repair the derived Qdrant index.
- Retrieval ranking is intentionally deterministic and transparent. Keep boost policy in `app/rag`,
  preserve the original vector score, and avoid player-facing hidden-text diagnostics.
- `MAX_LOCAL_RETRIES` is present in settings but is not the sole source of retry truth yet.

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
- live checks stay shallow and read-only; they do not run real completions or mutate Qdrant.

## Safe Next Work

The next reasonable implementation areas are documented in [docs/10_next_steps_after_mvp.md](10_next_steps_after_mvp.md).

Stay conservative:

- preserve backend boundaries
- add tests with each behavior change
- prefer targeted improvements over architecture churn
