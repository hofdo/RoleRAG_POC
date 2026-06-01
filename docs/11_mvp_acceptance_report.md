# Milestone 1 MVP Acceptance Report

This report freezes the Phase 18 MVP acceptance baseline for future agents. It records what is implemented, what remains out of scope, the architectural boundaries to preserve, and the verification commands used for the baseline.

## Implemented capabilities

- CLI and FastAPI entrypoints are present for local runtime access.
- CLI/API layers stay thin and delegate composition and runtime work to application services.
- `TurnOrchestrator` is the orchestration center for turn execution.
- SQLite owns sessions, turns, and memory as the authoritative state.
- Qdrant retrieval is a derived index that can be rebuilt or repaired from authoritative state.
- Actor context is filtered through visibility boundaries before player-facing generation.
- Critic validation uses a bounded repair loop.
- Memory extraction remains local-only.
- Deterministic evals cover regression-sensitive behavior.
- Runtime smoke verification exercises the MVP flow.
- Content validation and standalone scenario-template scaffolding are available.

## Explicitly not implemented

- Frontend.
- Auth or multi-user support.
- Streaming API.
- Production hardening.
- Autonomous planning or game-master loops.
- Runtime scenario-pack selection/loading support.

## Known limitations and risks

- Runtime content is still tied to the active `data/` tree.
- Standalone scenario packs can be scaffolded and validated but not selected at runtime.
- Retrieval depends on Qdrant availability, while authoritative state remains in SQLite.
- Actor prompts must stay player-visible only.
- The critic can inspect hidden context only for validation, not for player output.
- Local/cloud routing exists, but critic and memory extraction must remain local.

Future work should preserve SQLite as authoritative state ownership, Qdrant as a derived and repairable index, thin CLI/API entrypoints, orchestration in `TurnOrchestrator`, and the existing visibility and secrecy boundaries.

## Exact verification commands run

The acceptance commands are run from an activated Python 3.12 virtualenv:

```bash
ruff check .
mypy .
pytest
python -m app.evals.regression_runner
python -m app.cli smoke-run
python -m app.cli validate-content
```

## Verification results

- `ruff check .`: pass.
- `mypy .`: pass.
- `pytest`: pass, with 161 tests and one upstream Starlette/FastAPI testclient deprecation warning.
- `python -m app.evals.regression_runner`: pass, including 31 deterministic checks. The command emitted a Python runpy warning because the module was already present in `sys.modules`; the checks still completed successfully.
- `python -m app.cli smoke-run`: pass.
- `python -m app.cli validate-content`: pass for default `data/`.

## Recommended next milestone

The next milestone should be `runtime scenario-pack support`: enable runtime selection and loading of validated standalone scenario packs without changing the current state-ownership boundaries.
