# 03 — MVP Implementation and Operation Guide

> Reviewed: 2026-07-04 @ 571acc8

## Purpose

This document explains how to run, test, and safely extend the implemented MVP.

## Setup

### Fresh clone

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
docker compose up -d qdrant
python -m app.cli config
python -m app.cli health
```

Two faster entry points are documented in the root [README](../README.md): `docker compose up
--build` runs the app + Qdrant in containers (model server stays on the host), and `make`
(or `make help`) lists task shortcuts (`make up`, `make dev`, `make install`, `make check`).

### Local model

Configure an OpenAI-compatible local endpoint in `.env`.

Default:

```env
LOCAL_LLM_BASE_URL=http://127.0.0.1:8080/v1
LOCAL_LLM_API_KEY=local
LOCAL_LLM_MODEL=chatgpt-onnechan
```

Alternative Ollama-compatible setup:

```env
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_API_KEY=ollama
LOCAL_LLM_MODEL=qwen3:8b
```

### Qdrant

The runtime retrieval path expects a reachable Qdrant instance:

```bash
docker compose up -d qdrant
```

Without Qdrant or embeddings, turn execution continues without retrieved context and emits a warning.

## CLI Operations

Commands exposed by [app/cli.py](../app/cli.py) — run `python -m app.cli --help` for the full,
authoritative list. Grouped:

- diagnostics: `config`, `health`, `doctor`, `smoke-run`, `validate-content`, `embedding-ab`
- sessions / maintenance: `start-session`, `resume`, `turn`, `list-sessions`, `inspect-memories`,
  `turn-history`, `export-session`, `import-session`, `delete-session`, `backup`, `reset-db`
- retrieval / content: `ingest`, `ingest-scenario-lore`, `reindex-memories`, `reset-index`,
  `retrieve-debug`, `create-scenario-template`, `route`

Recommended local flow:

```bash
python -m app.cli start-session \
  --world-id demo_world \
  --scene-id rose-gallery \
  --active-persona-id archivist \
  --player-name Avery

python -m app.cli ingest data/documents/demo_lore.md \
  --visibility player \
  --source-type lore \
  --world-id demo_world

python -m app.cli turn \
  --session-id <session-id> \
  --message "What have you heard about the regent?"
```

If Qdrant was unavailable while durable memories were curated, backfill the derived index:

```bash
python -m app.cli reindex-memories --session-id <session-id>
```

`rolerag` is also installed as a console script and exposes the same commands.

`health` reports app metadata and redacted settings without probing SQLite, Qdrant, or model
providers.

## API Operations

Start the API:

```bash
uvicorn app.main:app --reload
```

The full HTTP surface — endpoints, request/response shapes, and error codes — is documented in
[docs/12_api_contract.md](12_api_contract.md). In brief: `GET /runtime/status`,
`GET /content/catalog`, session CRUD (`POST`/`GET /sessions`, `GET /sessions/{id}`),
mid-session scene switching (`POST /sessions/{id}/scene`), turns
(`POST /sessions/{id}/turns` and `/turns/stream`), last-turn deletion / reroll support
(`DELETE /sessions/{id}/turns/last`), per-turn and bulk diagnostics
(`GET /sessions/{id}/turns/{i}`, `GET /sessions/{id}/turn-details`), durable memories
(`GET /sessions/{id}/memories`), session canon
(`GET`/`POST`/`DELETE /sessions/{id}/canon`), and eval-run summaries and details
(`GET /diagnostics/eval-runs`, `GET /diagnostics/eval-runs/{run_id}`).

The API and CLI both call the same service wiring in [app/composition.py](../app/composition.py).
The public HTTP contract and exposure boundaries are documented in
[docs/12_api_contract.md](12_api_contract.md).
The SSE route is transport-only and buffered: it emits player-visible text after the shared
orchestration pipeline completes, not provider tokens or pre-validation drafts.

Open [http://127.0.0.1:8000/app/](http://127.0.0.1:8000/app/) for the Angular SPA (built by
`make dev`; see [frontend/README.md](../frontend/README.md)). Use `Start session` to begin from
the catalog selectors; turns run over buffered SSE. The UI is a thin same-origin client:
scenario packs remain a process-level backend choice through `CONTENT_ROOT`, and orchestration,
retrieval, validation, routing, persistence, memory, and authoritative session state stay in the
backend.

## Config Notes

All settings are defined in [app/config.py](../app/config.py) and mirrored — with defaults and
per-field commentary — in [`.env.example`](../.env.example). Those two are the single source of
truth; copy `.env.example` to `.env` and edit there. Beyond the base provider/Qdrant settings, the
surface covers provider timeouts and retries, critic/curator gating, the full RAG ranking weights
and boosts, durable-memory caps and consolidation, session canon, routing thresholds, and the
output-side containment threshold.

Repair runs on the session's bound provider; provider retries and the structured-truncation
retry budget are configurable (`LOCAL_LLM_MAX_RETRIES`, `CLOUD_LLM_MAX_RETRIES`,
`TRUNCATION_RETRY_BUDGET_MULTIPLIER`).

## Runtime Safety Rules

- The LLM never owns authoritative state.
- Actor generation uses only prepared messages.
- Retrieval is visibility-filtered before actor prompt construction.
- SQLite remains authoritative for durable memories; Qdrant indexing is fail-open during turns.
- Actor prompts include at most `RECENT_DIALOGUE_TURNS` prior persisted turns.
- Prior recent-dialogue messages are clipped to `RECENT_DIALOGUE_MAX_MESSAGE_CHARS` characters
  (default `900`) with an explicit omission marker during actor prompt construction; clipping is
  surfaced as a turn warning. The current incoming player message is not clipped.
- Older continuity returns through retrieved durable memory when relevant; use
  `reindex-memories` to repair the derived index from authoritative SQLite episodes.
- Memory extraction runs on the session's bound provider.
- API routes stay thin and do not duplicate orchestration logic.
- Buffered SSE output is emitted only after validation, persistence, and memory handling complete.
- Tests and evals should continue to avoid live providers.

## Tests and Verification

Primary repository checks:

```bash
ruff check .
mypy .
pytest
(cd frontend && npm test -- --watch=false --browsers=ChromeHeadless)
```

Test layout under `tests/`:

- `tests/unit/` — models, router, persistence, prompt assembly, and retrieval helpers
- `tests/integration/` — CLI, API, persistence, repair flow, memory curation, and documentation gates
- `tests/evals/` — deterministic fixture-based regression categories (also run standalone by
  `python -m app.evals.regression_runner`)
- `tests/e2e/` — Playwright browser spec (`spa-play.spec.mjs`) that plays through the built SPA
  against a live stack; config lives at the repo-root `playwright.config.mjs`, run with
  `npm run test:e2e-spa`

Eval-specific commands:

```bash
pytest tests/evals -q
python -m app.evals.regression_runner
```

Current known warnings that do not block the suite:

- FastAPI TestClient emits a Starlette/httpx deprecation warning in tests.

The deterministic 16-turn `memory_continuity` category uses SQLite, fake providers, keyword
embeddings, and `InMemoryVectorStore`. It verifies bounded prompt growth, durable-memory recall,
visibility isolation, and SQLite-backed reindex recovery. It does not prove live LLM behavior,
semantic embedding quality, or Qdrant quality.

## Extension Guidance

Safe extension means staying inside the current ownership boundaries:

- add behavior in the orchestrator only when it belongs to lifecycle coordination
- keep actor, critic, and curator responsibilities narrow
- do not push retrieval or persistence logic into route handlers or provider classes
- add tests using fake providers for any new branch in routing or orchestration
- do not introduce new frameworks or autonomous loops into the MVP backend

## Related Documents

- [docs/README.md](README.md) — documentation hub with architecture diagrams
- [docs/04_agent_workflows.md](04_agent_workflows.md)
- [docs/05_rag_memory_design.md](05_rag_memory_design.md)
- [docs/06_local_cloud_model_strategy.md](06_local_cloud_model_strategy.md)
- [docs/08_agent_handoff.md](08_agent_handoff.md)
- [docs/12_api_contract.md](12_api_contract.md)
