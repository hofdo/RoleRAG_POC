# RoleRAG_POC

Minimal CLI-first RoleRAG proof of concept with SQLite-backed session persistence.

## Scope

This repository currently includes:

- Python project bootstrap
- typed settings
- local/cloud LLM provider abstraction
- deterministic model routing
- structured domain models and JSON demo data loading
- a small Typer CLI for local roleplay turns
- SQLite persistence for sessions and turns

This repository does not yet include RAG, embeddings, vector storage, memory curation, critic workflows, or business endpoints.

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

## Commands

```bash
python -m app.cli --help
python -m app.cli config
python -m app.cli start-session --world-id demo_world --scene-id rose-gallery --active-persona-id archivist --player-name Avery
python -m app.cli resume --session-id <session-id>
python -m app.cli turn --session-id <session-id> --message "What have you heard about the regent?"
python -m app.cli route --task actor_response
pytest
ruff check .
mypy app
```

All commands assume the virtualenv is activated.
