# RoleRAG_POC

Minimal CLI-first RoleRAG proof of concept with SQLite-backed session persistence and basic local RAG ingestion infrastructure.

## Scope

This repository currently includes:

- Python project bootstrap
- typed settings
- local/cloud LLM provider abstraction
- deterministic model routing
- structured domain models and JSON demo data loading
- a small Typer CLI for local roleplay turns
- SQLite persistence for sessions and turns
- deterministic Markdown/text chunking
- local embedding abstraction
- vector store abstraction with Qdrant runtime support
- CLI document ingestion for local lore documents

This repository does not yet inject retrieved chunks into the actor prompt, add critic workflows, or expose API endpoints.

## Local model runtime

The default local configuration assumes an OpenAI-compatible server such as `llama.cpp`:

- `LOCAL_LLM_BASE_URL=http://localhost:8080/v1`
- `LOCAL_LLM_API_KEY=local`
- `LOCAL_LLM_MODEL=local-model`

You can swap in Ollama or another compatible runtime by changing environment variables.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Qdrant

```bash
docker compose up -d qdrant
```

## Commands

```bash
python -m app.cli --help
python -m app.cli config
python -m app.cli start-session --world-id demo_world --scene-id rose-gallery --active-persona-id archivist --player-name Avery
python -m app.cli resume --session-id <session-id>
python -m app.cli turn --session-id <session-id> --message "What have you heard about the regent?"
python -m app.cli route --task actor_response
python -m app.cli ingest data/documents/demo_lore.md --visibility player --source-type lore --world-id demo_world --tag palace
pytest
ruff check .
mypy app
```

All commands assume the virtualenv is activated.
