# RoleRAG_POC

Personal-use RoleRAG MVP built around a CLI-first roleplay loop, a small FastAPI API, SQLite persistence, Qdrant-backed retrieval, and deterministic local/cloud routing.

## Current State

Implemented in this repository:

- typed settings with `pydantic-settings`
- local/cloud OpenAI-compatible provider abstraction
- deterministic routing with `CLOUD_MODE=off|ask|auto`
- structured JSON world, scene, and persona loading
- Typer CLI for session lifecycle, routing inspection, lore ingestion, and turns
- FastAPI endpoints for session creation, turn execution, and session lookup
- SQLite persistence for sessions, turns, and durable memory episodes
- Qdrant-backed vector storage for ingested lore and retrieval
- retrieval-aware actor prompt construction with visibility filtering
- bounded critic and repair flow
- local-only memory extraction
- deterministic eval harness using fake providers and in-memory retrieval fixtures

Not implemented:

- frontend
- authentication
- multi-user support
- streaming API
- production deployment hardening
- visible CI in this repository
- automatic indexing of persisted SQLite memories into Qdrant

## Quickstart

### Prerequisites

- Python `3.12+`
- Docker
- a local OpenAI-compatible model server such as `llama.cpp` or Ollama's compatibility layer

### Fresh Clone Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
docker compose up -d qdrant
python -m app.cli config
```

`python -m app.cli config` is the basic local configuration check for a fresh clone.

### Start a Session

```bash
python -m app.cli start-session \
  --world-id demo_world \
  --scene-id rose-gallery \
  --active-persona-id archivist \
  --player-name Avery
```

That command returns a JSON object containing the `session_id`.

### Run a Turn

```bash
python -m app.cli turn \
  --session-id <session-id> \
  --message "What have you heard about the regent?"
```

### Resume a Session

```bash
python -m app.cli resume --session-id <session-id>
```

### Ingest Demo Lore

```bash
python -m app.cli ingest data/documents/demo_lore.md \
  --visibility player \
  --source-type lore \
  --world-id demo_world \
  --tag palace
```

### Run the API

```bash
uvicorn app.main:app --reload
```

## Local Model Setup

The default local configuration expects an OpenAI-compatible endpoint:

```env
LOCAL_LLM_BASE_URL=http://localhost:8080/v1
LOCAL_LLM_API_KEY=local
LOCAL_LLM_MODEL=local-model
```

Example `llama.cpp` server flow:

```bash
./server -m /path/to/model.gguf --host 127.0.0.1 --port 8080
```

Example Ollama-compatible setup:

```env
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_API_KEY=ollama
LOCAL_LLM_MODEL=qwen3:8b
```

The app does not contain provider-specific gameplay logic. It relies on the configured endpoint speaking the OpenAI-compatible chat-completions shape used by [app/llm/openai_compatible.py](/Users/dominique/IdeaProjects/RoleRAG_POC/app/llm/openai_compatible.py).

## Environment and Routing

The settings model lives in [app/config.py](/Users/dominique/IdeaProjects/RoleRAG_POC/app/config.py). `.env.example` mirrors the actual fields used by `Settings`.

Key values:

```env
APP_ENV=local
DATABASE_PATH=data/rolerag.db

LOCAL_LLM_BASE_URL=http://localhost:8080/v1
LOCAL_LLM_API_KEY=local
LOCAL_LLM_MODEL=local-model
LOCAL_LLM_MAX_TOKENS=700
LOCAL_LLM_TEMPERATURE=0.75

CLOUD_MODE=ask
CLOUD_LLM_BASE_URL=https://api.openai.com/v1
CLOUD_LLM_API_KEY=replace_me
CLOUD_LLM_MODEL=gpt-4.1-mini
CLOUD_LLM_MAX_TOKENS=1000
CLOUD_LLM_TEMPERATURE=0.65

QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_DEFAULT_TOP_K=5
RAG_CHUNK_SIZE_CHARS=1000
RAG_CHUNK_OVERLAP_CHARS=120
RAG_MAX_RETRIEVED_CHUNK_CHARS=800
MAX_LOCAL_RETRIES=1
RECENT_DIALOGUE_TURNS=8
```

Cloud behavior:

- `CLOUD_MODE=off`: cloud is never used. If cloud would otherwise be chosen, the runtime stays local and records a warning.
- `CLOUD_MODE=ask`: cloud is never called silently. The runtime does not open an interactive confirmation prompt; it keeps actor turns local when possible and returns a controlled failure if cloud repair would require confirmation after local attempts are exhausted.
- `CLOUD_MODE=auto`: cloud may be used for explicit `--request-cloud`, low retrieval confidence, high scene complexity, failed local repair, or local-provider failure.

Critic evaluation and memory extraction stay local in all modes.

## Runtime Components

- `CLI`: [app/cli.py](/Users/dominique/IdeaProjects/RoleRAG_POC/app/cli.py) exposes `config`, `start-session`, `resume`, `route`, `ingest`, and `turn`.
- `FastAPI API`: [app/main.py](/Users/dominique/IdeaProjects/RoleRAG_POC/app/main.py) and [app/api/routes.py](/Users/dominique/IdeaProjects/RoleRAG_POC/app/api/routes.py) expose the same orchestrator through HTTP.
- `TurnOrchestrator`: [app/orchestration/turn_orchestrator.py](/Users/dominique/IdeaProjects/RoleRAG_POC/app/orchestration/turn_orchestrator.py) owns the bounded turn pipeline.
- `ActorAgent`: [app/agents/actor_agent.py](/Users/dominique/IdeaProjects/RoleRAG_POC/app/agents/actor_agent.py) performs text generation only.
- `CriticAgent`: [app/agents/critic_agent.py](/Users/dominique/IdeaProjects/RoleRAG_POC/app/agents/critic_agent.py) evaluates drafts and builds repair prompts.
- `MemoryCurator`: [app/agents/memory_curator.py](/Users/dominique/IdeaProjects/RoleRAG_POC/app/agents/memory_curator.py) extracts structured durable-memory candidates.
- `RAG`: [app/rag/](/Users/dominique/IdeaProjects/RoleRAG_POC/app/rag) handles chunking, embeddings, ingestion, retrieval, and vector-store access.
- `SQLite persistence`: [app/persistence/](/Users/dominique/IdeaProjects/RoleRAG_POC/app/persistence) stores sessions, turns, and memory episodes.
- `Composition`: [app/composition.py](/Users/dominique/IdeaProjects/RoleRAG_POC/app/composition.py) wires settings, providers, repositories, and retrieval.

More detail:

- [docs/08_agent_handoff.md](/Users/dominique/IdeaProjects/RoleRAG_POC/docs/08_agent_handoff.md)
- [docs/09_current_architecture_map.md](/Users/dominique/IdeaProjects/RoleRAG_POC/docs/09_current_architecture_map.md)

## Safety Boundaries

- The LLM does not own authoritative state.
- `ActorAgent` does not retrieve, persist, or mutate state directly.
- FastAPI route handlers stay thin and delegate to shared composition and orchestration code.
- Player-facing actor prompts only include `player`-visible retrieved chunks.
- `CriticAgent` may inspect hidden context to detect leakage, but that output is not player-facing.
- Memory extraction stays local even when actor or repair routing uses cloud.
- Tests and evals use fake/mock providers and in-memory vector stores rather than real external services.

## CLI Usage

Show commands:

```bash
python -m app.cli --help
rolerag --help
```

Inspect resolved settings:

```bash
python -m app.cli config
```

Inspect routing decisions:

```bash
python -m app.cli route --task actor_response
python -m app.cli route --task actor_response --request-cloud
python -m app.cli route --task repair --failed-local-attempts 2
```

Session lifecycle:

```bash
python -m app.cli start-session --world-id demo_world --scene-id rose-gallery --active-persona-id archivist --player-name Avery
python -m app.cli resume --session-id <session-id>
python -m app.cli turn --session-id <session-id> --message "Tell me what changed here."
python -m app.cli turn --session-id <session-id> --message "Give me the highest quality answer." --request-cloud
```

RAG ingestion:

```bash
python -m app.cli ingest --help
python -m app.cli ingest data/documents/demo_lore.md --visibility player --source-type lore --world-id demo_world
```

## API Usage

Start the server:

```bash
uvicorn app.main:app --reload
```

Implemented endpoints:

- `POST /sessions`
- `POST /sessions/{session_id}/turns`
- `GET /sessions/{session_id}`

Create a session:

```bash
curl -X POST http://127.0.0.1:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"world_id":"demo_world","scene_id":"rose-gallery","player_name":"Player","active_persona_id":"archivist"}'
```

Run a turn:

```bash
curl -X POST http://127.0.0.1:8000/sessions/<session-id>/turns \
  -H "Content-Type: application/json" \
  -d '{"message":"I ask what the locked door hides.","active_persona_id":"archivist","request_cloud":false}'
```

Read session state:

```bash
curl http://127.0.0.1:8000/sessions/<session-id>
```

## Qdrant and Retrieval

Qdrant is the runtime vector store for real retrieval:

```bash
docker compose up -d qdrant
```

Collections used by the runtime:

- `canon_lore`
- `session_memory`
- `persona_memory`

Important current behavior:

- document ingestion currently supports `.md` and `.txt`
- world, scene, and persona demo data are loaded from JSON files
- retrieval is fail-open; if Qdrant or embeddings are unavailable during a turn, generation continues without retrieved context
- persisted SQLite memory episodes are not automatically re-indexed into Qdrant yet

## Tests, Lint, and Evals

Run all checks:

```bash
ruff check .
mypy .
pytest
```

Run only eval tests:

```bash
pytest tests/evals -q
```

Run the standalone deterministic regression summary:

```bash
python -m app.evals.regression_runner
```

The eval harness covers retrieval quality, visibility boundaries, role consistency, memory curation behavior, and cloud-routing policy. It is regression coverage for engine behavior, not a prose-quality benchmark.

## Current Limitations

- personal-use MVP only
- no frontend
- no authentication or multi-user isolation
- no streaming API
- no production deployment hardening
- no visible CI in the repository
- local/cloud behavior depends on the configured providers actually being available
- Qdrant is required for real retrieval behavior
- durable SQLite memories are not automatically re-indexed into Qdrant
- `MAX_LOCAL_RETRIES` is a settings field, but the current repair loop still uses fixed bounded retry behavior in code

## Safe Next Steps

The next implementation candidates are tracked in [docs/10_next_steps_after_mvp.md](/Users/dominique/IdeaProjects/RoleRAG_POC/docs/10_next_steps_after_mvp.md). The short version:

- add CI for lint, typing, tests, and evals
- improve integration coverage
- add memory indexing and better retrieval ranking
- add optional reranking
- add a frontend and streaming only after the current backend boundaries stay intact
- add auth only if the project becomes multi-user

## Additional Documentation

- [docs/01_product_goal.md](/Users/dominique/IdeaProjects/RoleRAG_POC/docs/01_product_goal.md)
- [docs/02_architecture.md](/Users/dominique/IdeaProjects/RoleRAG_POC/docs/02_architecture.md)
- [docs/03_implementation_guide.md](/Users/dominique/IdeaProjects/RoleRAG_POC/docs/03_implementation_guide.md)
- [docs/04_agent_workflows.md](/Users/dominique/IdeaProjects/RoleRAG_POC/docs/04_agent_workflows.md)
- [docs/05_rag_memory_design.md](/Users/dominique/IdeaProjects/RoleRAG_POC/docs/05_rag_memory_design.md)
- [docs/06_local_cloud_model_strategy.md](/Users/dominique/IdeaProjects/RoleRAG_POC/docs/06_local_cloud_model_strategy.md)
- [docs/07_mvp_phases.md](/Users/dominique/IdeaProjects/RoleRAG_POC/docs/07_mvp_phases.md)
