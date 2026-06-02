# 03 — MVP Implementation and Operation Guide

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

### Local model

Configure an OpenAI-compatible local endpoint in `.env`.

Default:

```env
LOCAL_LLM_BASE_URL=http://localhost:8080/v1
LOCAL_LLM_API_KEY=local
LOCAL_LLM_MODEL=local-model
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

Commands exposed by [app/cli.py](../app/cli.py):

- `python -m app.cli config`
- `python -m app.cli health`
- `python -m app.cli start-session`
- `python -m app.cli resume`
- `python -m app.cli route`
- `python -m app.cli ingest`
- `python -m app.cli reindex-memories`
- `python -m app.cli turn`

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

Implemented endpoints:

- `POST /sessions`
- `POST /sessions/{session_id}/turns`
- `GET /sessions/{session_id}`

The API and CLI both call the same service wiring in [app/composition.py](../app/composition.py).

## Config Notes

The active settings fields are defined in [app/config.py](../app/config.py). They currently include:

- `APP_ENV`
- `DATABASE_PATH`
- local LLM base URL, key, model, max tokens, and temperature
- `CLOUD_MODE`
- cloud LLM base URL, key, model, max tokens, and temperature
- `QDRANT_URL`
- `EMBEDDING_MODEL`
- `RAG_DEFAULT_TOP_K`
- `RAG_CHUNK_SIZE_CHARS`
- `RAG_CHUNK_OVERLAP_CHARS`
- `RAG_MAX_RETRIEVED_CHUNK_CHARS`
- `MAX_LOCAL_RETRIES`
- `RECENT_DIALOGUE_TURNS`

Current implementation note: `MAX_LOCAL_RETRIES` is part of settings, but the repair loop remains fixed and bounded in code.

## Runtime Safety Rules

- The LLM never owns authoritative state.
- Actor generation uses only prepared messages.
- Retrieval is visibility-filtered before actor prompt construction.
- SQLite remains authoritative for durable memories; Qdrant indexing is fail-open during turns.
- Actor prompts include at most `RECENT_DIALOGUE_TURNS` prior persisted turns.
- Individual recent-dialogue messages are not character-truncated during actor prompt construction.
- Older continuity returns through retrieved durable memory when relevant; use
  `reindex-memories` to repair the derived index from authoritative SQLite episodes.
- Memory extraction stays local.
- API routes stay thin and do not duplicate orchestration logic.
- Tests and evals should continue to avoid live providers.

## Tests and Verification

Primary repository checks:

```bash
ruff check .
mypy .
pytest
```

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

- [docs/04_agent_workflows.md](04_agent_workflows.md)
- [docs/05_rag_memory_design.md](05_rag_memory_design.md)
- [docs/06_local_cloud_model_strategy.md](06_local_cloud_model_strategy.md)
- [docs/08_agent_handoff.md](08_agent_handoff.md)
