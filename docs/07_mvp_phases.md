# 07 — MVP Phases

## Purpose

This document defines the staged MVP implementation plan for `RoleRAG_POC`.

The project is a personal-use Python proof of concept for a roleplaying RAG engine. It should support one user, one local 8B-class model as the default model, and one optional cloud model fallback.

The goal of this file is to give a coding agent a safe execution order. It must not try to build the whole system in one pass.

---

## Core Rule

Build the system in small vertical slices.

Each phase must leave the repository in a runnable state.

Do not merge half-implemented abstractions that cannot be exercised through either:

- a CLI command
- a FastAPI endpoint
- a unit test
- an integration test

A phase is not complete because files exist. A phase is complete when the behavior can be run and tested.

---

## MVP Definition

The MVP is complete when the system can:

1. Load one roleplaying world.
2. Load one scene.
3. Load one or more persona cards.
4. Accept a user message.
5. Build a compact context packet.
6. Route the request to the local model by default.
7. Optionally use a cloud model according to configuration.
8. Retrieve relevant lore or memory chunks.
9. Generate an in-character response.
10. Critique the response for obvious problems.
11. Persist durable memory when appropriate.
12. Continue the session after restart.
13. Run a small regression test suite.

The MVP does not need a frontend, multiplayer, image generation, combat automation, voice mode, or advanced autonomous agents.

---

## Phase 0 — Repository Bootstrap

### Goal

Create a clean Python repository foundation.

### Deliverables

```text
RoleRAG_POC/
  app/
    __init__.py
    main.py
    config.py
    cli.py

  tests/
    unit/
    integration/

  data/
    worlds/
    personas/
    sessions/
    documents/

  docs/

  pyproject.toml
  README.md
  .env.example
  .gitignore
```

### Required dependencies

Start small:

```toml
[project]
requires-python = ">=3.12"
dependencies = [
  "fastapi",
  "uvicorn[standard]",
  "pydantic",
  "pydantic-settings",
  "typer",
  "httpx",
  "openai",
  "python-dotenv",
  "rich",
]

[dependency-groups]
dev = [
  "pytest",
  "pytest-asyncio",
  "ruff",
  "mypy",
]
```

Do not add LangChain, LangGraph, Celery, Redis, or heavy orchestration frameworks in this phase.

### Acceptance Criteria

- `python -m app.cli --help` works.
- `pytest` runs.
- `ruff check .` runs.
- `README.md` explains how to install and run the skeleton.

---

## Phase 1 — Configuration and Provider Abstraction

### Goal

Make local and cloud model access interchangeable behind one interface.

### Deliverables

```text
app/
  config.py
  llm/
    __init__.py
    provider.py
    openai_compatible.py
    router.py
```

### Required behavior

The app must support:

- local OpenAI-compatible endpoint
- cloud OpenAI-compatible endpoint
- `cloud_mode = off | ask | auto`
- deterministic model routing

### Initial `.env.example`

```env
APP_ENV=local

LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_API_KEY=ollama
LOCAL_LLM_MODEL=qwen3:8b

CLOUD_MODE=ask
CLOUD_LLM_BASE_URL=https://api.openai.com/v1
CLOUD_LLM_API_KEY=replace_me
CLOUD_LLM_MODEL=gpt-4.1-mini

LOCAL_LLM_MAX_TOKENS=700
CLOUD_LLM_MAX_TOKENS=1000
LOCAL_LLM_TEMPERATURE=0.75
CLOUD_LLM_TEMPERATURE=0.65
```

### Acceptance Criteria

- A unit test can mock the provider and receive a generated response.
- The router chooses local by default.
- The router never chooses cloud when `CLOUD_MODE=off`.
- The router marks cloud usage as requiring confirmation when `CLOUD_MODE=ask`.
- No agent calls provider-specific code directly.

---

## Phase 2 — Domain Models

### Goal

Create explicit, typed models for roleplay state.

### Deliverables

```text
app/
  domain/
    __init__.py
    models.py
    visibility.py
```

### Required models

At minimum:

- `PersonaCard`
- `SceneState`
- `SessionState`
- `MemoryEpisode`
- `RetrievedChunk`
- `TurnInput`
- `TurnResult`
- `ModelRoute`
- `CriticResult`

### Visibility values

Use explicit visibility values:

```text
player
gm
character_private
```

### Hard Rule

Visibility is not optional.

Every memory and retrieved chunk must have a visibility.

### Acceptance Criteria

- Models validate correctly with Pydantic.
- Invalid visibility values are rejected.
- `PersonaCard` separates public description, private description, secrets, and forbidden knowledge.
- `SceneState` separates player-visible state and GM-private state.

---

## Phase 3 — CLI-First Local Roleplay Loop

### Goal

Make the project useful before adding RAG.

### Deliverables

```text
app/
  cli.py
  orchestration/
    __init__.py
    turn_orchestrator.py
    context_builder.py
  agents/
    __init__.py
    actor_agent.py
```

### Required behavior

The CLI should:

1. Load one test persona.
2. Load one test scene.
3. Accept a user message.
4. Build a compact prompt.
5. Send the prompt to the local model.
6. Print the response.

### Example command

```bash
python -m app.cli chat --world demo --scene tavern
```

### Acceptance Criteria

- A user can complete a basic local-model roleplay turn.
- Prompt construction is centralized in `context_builder.py`.
- The actor agent does not read files directly.
- The actor agent does not mutate state.
- No RAG is required yet.

---

## Phase 4 — Structured Data Loading

### Goal

Move hardcoded demo data into files.

### Deliverables

```text
data/
  worlds/
    demo_world.json
  personas/
    narrator.json
    innkeeper.json
  sessions/

app/
  persistence/
    __init__.py
    file_loader.py
```

### Required behavior

The app must load:

- world metadata
- scene state
- persona cards

The format can be JSON for the MVP. YAML is optional but not necessary.

### Acceptance Criteria

- CLI can load persona and scene from `data/`.
- Invalid files fail with clear validation errors.
- Private fields are not automatically inserted into player-facing prompts.
- Tests cover successful and failed loading.

---

## Phase 5 — Session Persistence

### Goal

Persist session history so roleplay can continue after restart.

### Deliverables

```text
app/
  persistence/
    sqlite.py
    repositories.py
  memory/
    __init__.py
    store.py
```

### Required storage

Use SQLite for MVP metadata and session history.

Tables:

```text
sessions
turns
memory_episodes
```

### Required behavior

Persist:

- session id
- world id
- active scene id
- user messages
- assistant messages
- model route used
- created memories

### Acceptance Criteria

- A session can be resumed after restarting the CLI.
- The last N turns can be loaded into context.
- The local model does not receive the full conversation forever.
- Tests verify persistence and resume behavior.

---

## Phase 6 — Memory Curator

### Goal

Extract durable memory from turns.

### Deliverables

```text
app/
  agents/
    memory_curator.py
  memory/
    policies.py
```

### Required behavior

After a turn, the memory curator decides whether to write memory.

It should write memory only for:

- player decisions
- relationship changes
- quest changes
- promises
- threats
- discovered secrets
- durable world facts
- strong player preferences

It should reject:

- greetings
- filler dialogue
- repeated facts
- short emotional reactions without consequence

### Output schema

```json
{
  "write_memory": true,
  "memories": [
    {
      "summary": "...",
      "visibility": "player",
      "importance": 4,
      "tags": ["quest", "innkeeper"]
    }
  ]
}
```

### Acceptance Criteria

- Memory extraction returns structured data.
- Memory visibility is required.
- Invalid memory output is rejected.
- Trivial turns do not create memory.
- Important turns create memory.
- Tests cover both cases.

---

## Phase 7 — Basic RAG Infrastructure

### Goal

Add local document ingestion and retrieval.

### Deliverables

```text
app/
  rag/
    __init__.py
    ingestion.py
    chunking.py
    embeddings.py
    vector_store.py
    retriever.py

docker-compose.yml
```

### Recommended vector store

Use Qdrant for the MVP.

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant:/qdrant/storage
```

### Required collections

Start with:

```text
canon_lore
session_memory
persona_memory
```

### Required metadata

Every chunk must have:

- `id`
- `source`
- `source_type`
- `visibility`
- `tags`
- optional `world_id`
- optional `scene_id`
- optional `persona_id`
- optional `session_id`

### Acceptance Criteria

- Markdown/text documents can be ingested.
- Chunks are embedded and stored.
- Retrieval returns top K chunks.
- Retrieval filters by visibility.
- Tests verify GM-only chunks are not returned for player-facing context.

---

## Phase 8 — Retrieval-Aware Context Builder

### Goal

Inject retrieved context into prompts safely.

### Deliverables

```text
app/
  orchestration/
    context_builder.py
    context_budget.py
```

### Required behavior

The context builder must assemble:

1. system/task instruction
2. active persona packet
3. player-visible scene state
4. recent dialogue window
5. selected memories
6. selected lore chunks
7. user message

### Context budget

The local 8B model must not receive excessive context.

Default limits:

```text
recent dialogue turns: 8
retrieved chunks: 5
max retrieved chunk length: 800 characters
local max output tokens: 700
```

### Acceptance Criteria

- Retrieved chunks are inserted in a clear section.
- GM-only chunks are excluded unless explicitly allowed.
- The context builder has unit tests.
- Prompt size is bounded.
- Actor agent still does not retrieve context directly.

---

## Phase 9 — Critic and Repair Loop

### Goal

Catch bad responses before showing them.

### Deliverables

```text
app/
  agents/
    critic_agent.py
```

### Required checks

The critic checks for:

- secret leakage
- contradiction with scene state
- contradiction with retrieved lore
- character knowledge violation
- generic assistant tone
- ignoring the user's action

### Loop limit

The loop must be bounded:

```text
local draft
local critique
one local repair attempt
optional cloud repair
final response
```

No infinite loops.

### Acceptance Criteria

- Critic returns structured JSON.
- Rejected drafts are repaired once.
- Cloud repair is only used according to router policy.
- Tests verify loop does not exceed configured attempts.
- Tests verify obvious secret leakage is rejected.

---

## Phase 10 — Cloud Fallback

### Goal

Allow cloud use without changing gameplay behavior.

### Required behavior

Cloud may be used when:

- local draft fails critique twice
- retrieval confidence is low
- scene transition is complex
- user explicitly requests cloud/high-quality mode
- configuration allows cloud

Cloud must not receive more private data than necessary.

### Acceptance Criteria

- Cloud is never used in `CLOUD_MODE=off`.
- In `CLOUD_MODE=ask`, cloud route is marked as requiring approval.
- In `CLOUD_MODE=auto`, cloud may be used according to policy.
- Route reason is recorded.
- Tests cover all cloud modes.

---

## Phase 11 — Evaluation Harness

### Goal

Prevent prompt and retrieval regressions.

### Deliverables

```text
app/
  evals/
    __init__.py
    fixtures.py
    regression_runner.py
    role_consistency.py
    retrieval_quality.py

tests/
  evals/
```

### Required fixtures

Create a small fixed test world with:

- one scene
- two NPCs
- one secret
- one player-visible lore fact
- one GM-only lore fact
- one memory that should be retrieved
- one memory that should not be retrieved

### Required evaluations

- retrieval top K includes expected public lore
- retrieval excludes GM-only lore for player-facing prompts
- actor response does not reveal secret
- memory curator writes important memory
- memory curator rejects trivial memory
- cloud router obeys config

### Acceptance Criteria

- Eval tests can run locally.
- Results are deterministic enough to catch obvious regressions.
- The evaluation harness is documented in `README.md`.

---

## Phase 12 — FastAPI MVP

### Goal

Expose the engine over HTTP after the CLI works.

### Deliverables

```text
app/
  main.py
  api/
    __init__.py
    routes.py
```

### Required endpoints

```http
POST /sessions
POST /sessions/{session_id}/turns
GET /sessions/{session_id}
```

### Example `POST /sessions`

```json
{
  "world_id": "demo_world",
  "scene_id": "tavern",
  "cloud_mode": "ask"
}
```

### Example `POST /sessions/{session_id}/turns`

```json
{
  "message": "I ask the innkeeper what happened last night."
}
```

### Acceptance Criteria

- API can start with `uvicorn app.main:app --reload`.
- Session creation works.
- Turn execution works.
- API uses the same orchestrator as the CLI.
- No duplicated orchestration logic exists in the routes.

---

## Phase 13 — Documentation and Agent Handoff

### Goal

Make the repository understandable for future coding agents.

### Required docs

At minimum:

```text
README.md
docs/01_product_goal.md
docs/02_architecture.md
docs/03_implementation_guide.md
docs/04_agent_workflows.md
docs/05_rag_memory_design.md
docs/06_local_cloud_model_strategy.md
docs/07_mvp_phases.md
```

### README must include

- project goal
- quickstart
- environment setup
- local model setup
- Qdrant setup
- run CLI
- run API
- run tests
- current limitations

### Acceptance Criteria

- A new coding agent can start from README and follow the docs.
- The phase plan is consistent with the actual repository structure.
- No docs claim completed behavior that does not exist.

---

## Coding Agent Execution Rules

A coding agent working on this repo must follow these rules:

1. Do one phase at a time.
2. Keep the project runnable after each phase.
3. Add tests for every non-trivial module.
4. Do not add a heavy framework without explicit approval.
5. Do not implement frontend code in the MVP.
6. Do not create autonomous long-running agent loops.
7. Do not let the LLM own authoritative state.
8. Do not send GM-only context to player-facing prompts.
9. Do not skip visibility filtering.
10. Do not use cloud unless the router allows it.
11. Do not store secrets in committed files.
12. Do not claim a phase is complete unless its acceptance criteria pass.

---

## Recommended First Coding Prompt

Use this prompt for the first coding-agent implementation pass:

```text
You are working in a new Python repository for a personal-use RoleRAG proof of concept.

Implement Phase 0 and Phase 1 only.

Goals:
- Create a clean Python 3.12 project skeleton.
- Add pyproject.toml with minimal dependencies.
- Add app/config.py using pydantic-settings.
- Add an LLM provider abstraction.
- Add an OpenAI-compatible provider implementation.
- Add a deterministic model router.
- Add a Typer CLI with a health/check command.
- Add .env.example.
- Add basic pytest tests.

Do not implement RAG yet.
Do not implement memory yet.
Do not add LangChain or LangGraph.
Do not add a frontend.
Keep the repository runnable and simple.
```

---

## Final MVP Success Criteria

The MVP is successful when this command sequence works:

```bash
cp .env.example .env
docker compose up -d qdrant
python -m app.cli ingest data/documents/demo_lore.md
python -m app.cli chat --world demo_world --scene tavern
```

And during chat:

- local model answers by default
- retrieved lore appears in the internal context
- GM-only facts are not leaked
- important events are remembered
- session can resume after restart
- cloud fallback can be enabled but is not required
- tests pass

That is the correct stopping point for the first real MVP.
