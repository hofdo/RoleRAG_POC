# 10 — Next Steps After 1.0

> Reviewed: 2026-07-11 @ a06695a

> **Updated 2026-07-04** — two efforts shipped after this list was written:
> play-experience v1.2 (2026-07-01/02,
> [plan](superpowers/plans/2026-07-01-play-experience-v1.2.md)) and the session-bound
> provider model (2026-07-02,
> [plan](superpowers/plans/2026-07-02-session-bound-provider.md)). Items they closed are
> marked below; [docs/BACKLOG.md](BACKLOG.md) is the live source of next steps.

## Purpose

This document lists the safest next development steps after the 1.0 release
(see `docs/15_v1_acceptance_report.md`). It is intentionally limited to
follow-on work that fits the existing architecture.

1.0 closed the former top items: per-stage timings, conditional critic/curator
gating, deterministic draft validation, the `CLOUD_MODE=ask` confirmation flow
(since reshaped: the 2026-07-02 session-bound-provider change moved it to a
single confirmation at session creation), session management CLI
(list/delete/export/import/reset), the memory viewer, and one-command startup
(`scripts/dev-up.sh`). Remaining candidates below.

## Highest-Value Next Steps

### 1. Extend CI as needed

The repository CI already runs:

- `ruff check .`
- `mypy .`
- `pytest`
- SPA unit tests (`ng test`, headless)
- `python -m app.evals.regression_runner`

Extend the matrix only when another supported Python version or platform needs coverage.

### 2. Improve integration coverage

The repository now has deterministic runtime verification through `doctor` and `smoke-run`.
It also has deterministic authoring validation through `validate-content` and
standalone pack scaffolding/runtime launch through `create-scenario-template` and
`start-session --content-root`.
The three gaps named here (2026-07-04) are now covered:

- CLI plus retrieval wiring — already covered by
  `test_cli_turn_uses_in_memory_retrieval_and_excludes_hidden_or_isolated_chunks` and
  `test_cli_turn_uses_stored_scenario_pack_world_for_retrieval` in
  `tests/integration/test_cli.py` (real composition wiring, fakes only at the
  provider/embedding/loader/vector-store edges, chunks seeded into an
  `InMemoryVectorStore` and asserted present/absent in the actor prompt by visibility
  and world).
- API plus persistence plus retrieval — lore seed-through-retrieval-through-diagnostics
  was already covered by `test_post_turn_uses_in_memory_retrieval_without_live_qdrant`
  in `tests/integration/test_api_turns.py`; the remaining half (a turn's memory
  written and indexed, then retrieved by the *next* turn) is now covered by
  `test_api_turn_writes_and_indexes_memory_then_next_turn_retrieves_it` in the same
  file (real SQLite session/turn/memory repos, a real `InMemoryVectorStore` shared by
  lore and memory indexing, a fake provider; drives `routes._run_turn` and
  `orchestrator.run_deferred_memory` directly for determinism since the real
  fire-and-forget deferred-memory task intentionally builds its own `AppServices`
  from `Settings`).
- provider-unavailability edge cases on the session's bound provider — covered by
  `tests/integration/test_provider_unavailability.py` (LOCAL and CLOUD sessions,
  `ProviderTimeoutError`/`ProviderUnavailableError` at the generation seam, no
  cross-provider fallback, session resumable and the next turn succeeds once the
  provider recovers) plus a CLI-level counterpart in `test_cli.py`. This surfaced a
  real, pinned asymmetry rather than a bug fix: a provider error during actor
  generation is *not* caught by `TurnOrchestrator.run_turn` (only
  `EmptyProviderResponseError`/`TruncatedProviderResponseError` are), so it propagates
  uncaught to the API/CLI caller and no controlled-failure turn is persisted for that
  attempt; the same error raised from the critic stage instead fails closed into a
  persisted `CONTROLLED_FAILURE` turn (invariant #4), because `TurnCritiqueStage.run`
  catches any exception from the critic. See the module docstring in
  `test_provider_unavailability.py` for the full trace.

### 3. Tune retrieval quality from evidence

Safe candidates:

- better query construction heuristics
- stronger ranking signals from metadata and recency
- broader eval coverage for retrieval regressions
- richer operator diagnostics if another debug surface is justified

## Product-Level Follow-Ons

### Web UI evolution

The repository now includes an Angular SPA (play, RAG inspector, analytics, eval) over the
existing API; it replaced the earlier framework-free `/play` page in 1.1.0. Session resume
shipped in v1.2: the setup page offers a resume picker over recent sessions and reloads the
transcript through the bulk turn-details endpoint. Keep the browser thin: backend behavior
stays in the backend; remaining UI candidates live in [docs/BACKLOG.md](BACKLOG.md).

### Streaming evolution

The API now exposes buffered SSE as transport behavior over the existing orchestrator.
Validated fragmentation shipped in v1.2 as a config-gated capability (`SSE_TEXT_CHUNK_CHARS`,
default 0 = a single text frame) and preserved the standing rule: never emit provider tokens
or draft text before critic validation. The stream also gained live metadata-only
stage-progress frames.

### Richer authoring

Safe scope:

- additional demo content
- richer scenario-pack templates and documentation

## Conditional Future Work

### Authentication

Only relevant if the project becomes multi-user. Do not add auth while the repository remains explicitly personal-use.

### Production secrets handling

Relevant if deployment becomes serious. Current `.env`-based local configuration is acceptable for the MVP.

### Deployment hardening

Add only when there is an actual deployment target. That includes:

- production config separation
- service monitoring
- backup strategy (shipped in v1.2: WAL mode, `rolerag backup`, and auto-snapshots before
  destructive operations)
- tighter operational error handling

## Work To Avoid

- introducing LangChain or LangGraph
- autonomous planning loops
- database churn without a concrete need
- frontend-first refactors that move backend logic into the client
- provider-specific gameplay forks

## Decision Rule

Prefer changes that:

- preserve deterministic orchestration
- keep state ownership in application code
- strengthen visibility boundaries
- improve verification and maintenance

Avoid changes that increase abstraction, hidden behavior, or provider coupling without a measured benefit.
