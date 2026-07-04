# 04 — Current Agent Workflows

> Reviewed: 2026-07-04 @ 571acc8

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
7. Choose actor route (session's bound provider)
8. Build actor messages
9. Generate actor draft
10. Validate the draft (deterministic checks)
11. Critique draft on the session's bound provider
12. If needed, run one repair on the session's bound provider
13. Apply output-side secret containment
14. Persist final turn
15. Curate durable memory on the session's bound provider (with a deterministic fallback extractor)
16. Persist, index, dedup, and optionally consolidate memory episodes
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

- critic evaluation runs on the session's bound provider, like every other task
- any critic exception — invalid structured output or otherwise — fails the turn closed: the
  draft is withheld and the turn resolves to a controlled failure with `critic_status=rejected`
  ([app/orchestration/stages/critique.py](../app/orchestration/stages/critique.py))

Gating behavior:

- `CRITIC_GATING` defaults to `always`; the `auto` mode skips critique on low-risk turns
- in `auto` mode a turn counts as risky — and the critic still runs — when the deterministic
  draft validator flagged the draft, retrieval confidence is below `LOW_RETRIEVAL_CONFIDENCE`
  or missing entirely, scene complexity is at or above `HIGH_SCENE_COMPLEXITY`, or the session's
  bound provider is cloud
  ([app/orchestration/stages/critique.py](../app/orchestration/stages/critique.py))
- a gated skip is deliberate, not a failure: the draft is served with a warning and
  `critic_status=skipped`
- the `always` default is intentional: live acceptance found auto-gating regressed 50-turn
  recall (see [docs/05_rag_memory_design.md](05_rag_memory_design.md))

## MemoryCurator

Implemented in [app/agents/memory_curator.py](../app/agents/memory_curator.py).

Responsibilities:

- inspect the completed turn
- return structured memory candidates
- decide whether durable memory should be written

Current routing behavior:

- memory extraction runs on the session's bound provider, like every other task
- invalid output is skipped with warnings
- `CURATOR_GATING` defaults to `always`; in `auto` mode LLM extraction is skipped when the turn
  shows no durable-event signals (no deterministic-extractor hits and no durable-event terms in
  either message)

## Orchestrator Ownership

[app/orchestration/turn_orchestrator.py](../app/orchestration/turn_orchestrator.py) owns:

- ordering
- route selection inputs
- retry bounds
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

### Provider failure

- there is no cross-provider fallback; a session's provider never changes mid-session
- an unreachable provider surfaces as a controlled `ProviderUnavailableError`

### Critic failure

- any critic exception — invalid structured output or any other error — fails the turn closed:
  the draft is withheld and the turn is persisted as a controlled failure with
  `critic_status=rejected`
- this fail-closed rule also applies to the re-critique after a repair pass
- a deliberate auto-gating skip is not a failure: the draft is served with a warning and
  `critic_status=skipped`

### Curator failure

- response generation does not retroactively fail; the turn has already been persisted
- invalid structured output is treated as a skipped curation step with warnings (the
  deterministic fallback extractor still contributes its candidates)

### Memory indexing failure

After successful memory persistence, the orchestrator asks `MemoryIndexer` to embed and upsert the
persisted episodes into `session_memory`. SQLite remains authoritative. Indexing failures append a
warning and do not discard the completed turn or persisted memory.

## Safety Boundaries

- actor prompts only include player-visible retrieved chunks
- critic may inspect hidden context but produces non-player-facing output
- hidden authored content never enters a prompt when the session's bound provider is cloud
  (the `include_hidden` gate in [app/orchestration/stages/critique.py](../app/orchestration/stages/critique.py))
- no agent persists directly
- no agent can choose arbitrary tools or routes

## Tests That Protect This Workflow

Key coverage exists in:

- `tests/unit/test_turn_orchestrator.py`
- `tests/unit/test_repair_loop.py`
- `tests/unit/test_critic_agent.py`
- `tests/unit/test_memory_curator.py`
- `tests/unit/test_retrieval_context_builder.py`
- `tests/evals/`

Any future workflow change should extend those tests before changing runtime logic.
