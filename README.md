# RoleRAG_POC

Minimal Phase 0 and Phase 1 foundation for a personal RoleRAG proof of concept.

## Scope

This repository currently includes:

- Python project bootstrap
- typed settings
- local/cloud LLM provider abstraction
- deterministic model routing
- a small Typer CLI for smoke testing

This repository does not yet include RAG, memory, agent workflows, persistence, or business endpoints.

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
python -m app.cli route --task actor_response
pytest
ruff check .
mypy app
```

All commands assume the virtualenv is activated.
