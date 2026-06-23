# 04 — Current Agent Workflows

## Purpose

This document describes the bounded agent workflow that is implemented today.

The repository does not implement an autonomous multi-agent system. It uses a small number of explicit agent classes inside a deterministic orchestrator.

## Implemented Agent-Like Components

The implemented components are:

- `ActorAgent`
- `CriticAgent`
- `MemoryCurator`

There is no implemented `IntentClassifier`, `PersonaAssembler`, or standalone `RetrievalAgent` class in the current codebase. Their responsibilities are currently handled by deterministic orchestrator logic, context building, and retrieval helpers.

## Turn Workflow

```text
1. Receive user message
2. Load session from SQLite
3. Load world, scene, and persona from JSON
4. Load recent dialogue from SQLite
5. Build a retrieval query
6. Retrieve player-visible chunks when retrieval is enabled
7. Choose actor route
8. Build actor messages
9. Generate actor draft
10. Validate the draft (deterministic checks)
11. Critique draft locally
12. If needed, run one local repair
13. If still needed and policy allows, run one cloud repair
14. Apply output-side secret containment
15. Persist final turn
16. Curate durable memory locally (with a deterministic fallback extractor)
17. Persist, index, dedup, and optionally consolidate memory episodes
```

This flow is finite. No component recursively calls the orchestrator.

## ActorAgent

Implemented in [app/agents/actor_agent.py](../app/agents/actor_agent.py).

Responsibilities:

- send prepared `LlmRequest` objects to the chosen provider
- return text only

Non-responsibilities:

- retrieval
- persistence
- routing
- state mutation

## CriticAgent

Implemented in [app/agents/critic_agent.py](../app/agents/critic_agent.py).

Responsibilities:

- evaluate drafts with structured JSON output
- detect secret leakage, contradictions, character-knowledge violations, generic assistant tone, and ignored user action
- build local and cloud repair prompts

Current routing behavior:

- critic evaluation always stays local
- invalid critic output is treated as a skipped critic step with warnings

## MemoryCurator

Implemented in [app/agents/memory_curator.py](../app/agents/memory_curator.py).

Responsibilities:

- inspect the completed turn
- return structured memory candidates
- decide whether durable memory should be written

Current routing behavior:

- memory extraction always stays local
- invalid output is skipped with warnings

## Orchestrator Ownership

[app/orchestration/turn_orchestrator.py](../app/orchestration/turn_orchestrator.py) owns:

- ordering
- route selection inputs
- retry bounds
- cloud fallback behavior
- persistence
- warning accumulation

The orchestrator is the only place allowed to coordinate the full lifecycle.

## Retrieval Workflow

Retrieval is deterministic code, not an agent class.

- [app/rag/retriever.py](../app/rag/retriever.py) builds a retrieval query from visible context and recent turns.
- `ActorContextRetriever` queries `session_memory`, `persona_memory`, and `canon_lore`.
- [app/orchestration/context_budget.py](../app/orchestration/context_budget.py) filters to `player`-visible chunks and truncates them before prompt insertion.

## Failure Handling

### Retrieval failure

- turn execution continues
- warnings include retrieval failure details
- actor prompt is built without retrieved chunks

### Local provider failure

- the orchestrator may choose cloud when policy permits
- in `ask` mode, cloud is not called silently

### Critic or curator failure

- response generation does not retroactively fail
- warnings are recorded

After successful memory persistence, the orchestrator asks `MemoryIndexer` to embed and upsert the
persisted episodes into `session_memory`. SQLite remains authoritative. Indexing failures append a
warning and do not discard the completed turn or persisted memory.
- invalid structured output is treated as a skipped downstream step

## Safety Boundaries

- actor prompts only include player-visible retrieved chunks
- critic may inspect hidden context but produces non-player-facing output
- memory extraction stays local
- no agent persists directly
- no agent can choose arbitrary tools or routes

## Tests That Protect This Workflow

Key coverage exists in:

- `tests/unit/test_turn_orchestrator.py`
- `tests/unit/test_repair_loop.py`
- `tests/unit/test_critic_agent.py`
- `tests/unit/test_memory_curator.py`
- `tests/unit/test_retrieval_context_builder.py`
- `tests/integration/test_cloud_fallback_turn_flow.py`
- `tests/evals/`

Any future workflow change should extend those tests before changing runtime logic.
