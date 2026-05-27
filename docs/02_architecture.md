# 02 — MVP Architecture

## Purpose

This document defines the initial architecture for `RoleRAG_POC`, a personal-use Python proof of concept for a roleplaying RAG system.

The MVP must prove that a local 8B-class model can run an interactive roleplaying loop when the application, not the model, owns orchestration, memory, retrieval, context assembly, and safety boundaries.

The architecture must stay boring, inspectable, and easy for one developer to modify.

---

## Architecture Goals

The system must support:

1. One active user.
2. One active roleplay session at a time for the MVP.
3. A local 8B model as the default LLM.
4. An optional cloud model fallback.
5. Structured scene and persona state.
6. Retrieval-augmented context from local documents and memories.
7. Durable memory across turns.
8. Basic validation before responses are shown.
9. Simple tests for core behaviour.

The system must not become a general autonomous-agent framework. It is a roleplaying engine with a small number of controlled agent steps.

---

## High-Level System View

```text
User
  |
  v
CLI / FastAPI Endpoint
  |
  v
Turn Orchestrator
  |
  +--> Intent Classifier
  +--> Session Loader
  +--> Scene Loader
  +--> Persona Assembler
  +--> Retrieval Agent
  +--> Context Builder
  +--> Model Router
  +--> Actor Agent
  +--> Critic Agent
  +--> Memory Curator
  |
  v
Response
  |
  v
Persistence + Vector Store
```

The user interacts with one interface. The orchestrator owns the workflow. The LLM only performs bounded tasks.

---

## Core Design Rule

The LLM is not the system.

The Python application is the system. The model is only used for:

- classifying intent when deterministic rules are not enough
- generating roleplay prose
- extracting memory candidates
- critiquing draft responses
- optionally repairing weak drafts

The model must not be trusted to:

- own authoritative state
- decide visibility rules
- remember facts without persistence
- mutate world state directly
- choose arbitrary tools
- receive the entire world state

---

## Main Runtime Flow

```text
1. User sends a message.
2. The app loads the current session.
3. The app classifies the turn intent.
4. The app loads the active scene and personas.
5. The app retrieves relevant lore and memory.
6. The app builds a compact context packet.
7. The model router chooses local or cloud model.
8. The actor agent generates a draft response.
9. The critic checks the draft.
10. If the draft is rejected, retry or repair.
11. The final response is returned to the user.
12. The memory curator extracts durable memory candidates.
13. The app persists accepted memory and session updates.
```

This flow is intentionally linear for the MVP. Do not introduce dynamic agent graphs yet.

---

## Recommended MVP Modules

```text
app/
  main.py
  config.py
  cli.py

  domain/
    models.py
    visibility.py
    errors.py

  llm/
    provider.py
    openai_compatible.py
    router.py

  agents/
    intent_classifier.py
    retrieval_agent.py
    actor_agent.py
    critic_agent.py
    memory_curator.py

  orchestration/
    turn_orchestrator.py
    context_builder.py
    context_budget.py

  rag/
    ingestion.py
    chunking.py
    embeddings.py
    vector_store.py
    retriever.py

  memory/
    store.py
    summarizer.py

  persistence/
    sqlite.py
    repositories.py

  world/
    loaders.py
    scene_store.py
    persona_store.py
```

Do not create all files immediately if they are empty. Add modules when the implementation reaches that phase.

---

## Component Responsibilities

## CLI / FastAPI Interface

The interface layer accepts user input and sends it to the turn orchestrator.

For the first MVP, a CLI is enough.

FastAPI can be added early if a frontend is planned, but it should remain thin.

Responsibilities:

- parse user input
- load configuration
- start or resume a session
- call the turn orchestrator
- display the response

It must not:

- build prompts
- access the vector store directly
- mutate memory directly
- contain roleplay logic

---

## Turn Orchestrator

The turn orchestrator is the central application service.

Responsibilities:

- coordinate the full turn lifecycle
- call agents in the correct order
- enforce retry limits
- enforce cloud routing policy
- persist final results
- produce a structured response object

It should be deterministic wherever possible.

The orchestrator should be easy to read from top to bottom. If it becomes hard to follow, the project is already drifting into agent-framework noise.

---

## Intent Classifier

The intent classifier determines what kind of user turn this is.

Example intents:

- `roleplay_continue`
- `player_action`
- `ask_lore`
- `ask_rules`
- `out_of_character_instruction`
- `memory_correction`
- `scene_transition`

For MVP, start with rule-based classification.

Only use the local LLM for classification after deterministic rules become insufficient.

---

## Session Loader

The session loader retrieves:

- current session id
- active world id
- active scene id
- recent dialogue
- session settings
- cloud mode

For MVP, use SQLite.

The session loader should return typed domain objects, not raw database rows.

---

## Scene Loader

The scene loader retrieves the active scene.

A scene includes:

- location
- visible summary
- private GM summary
- active personas
- open conflicts
- active quests
- recent events

The scene loader must preserve visibility boundaries.

Player-facing prompts should receive only player-visible scene information unless a specific agent is allowed to use private information.

---

## Persona Assembler

The persona assembler prepares a compact persona packet for the current turn.

It should not dump the whole character file into the prompt.

It selects:

- name
- role
- speaking style
- current goals
- relationships relevant to the scene
- known facts
- forbidden knowledge
- secrets that affect behaviour but should not be revealed directly

The actor agent receives this compact packet.

---

## Retrieval Agent

The retrieval agent finds relevant context for the current turn.

It uses:

- user message
- active scene
- active persona
- recent memory
- metadata filters

It retrieves from:

- canon lore
- session memory
- persona memory
- rules documents
- world events

For MVP, retrieval should return a small number of chunks:

```text
normal turn: 3–5 chunks
lore-heavy turn: 5–8 chunks
cloud fallback: up to 12 chunks
```

More chunks are not automatically better. An 8B model will degrade when overloaded.

---

## Context Builder

The context builder creates the final prompt packet.

It is one of the most important modules in the whole system.

Responsibilities:

- keep prompts compact
- apply visibility rules
- apply context budget rules
- include only relevant memory and lore
- include recent dialogue
- produce stable prompt structure

The context builder must be tested.

Especially test that GM-only facts are not included in player-facing actor prompts.

---

## Model Router

The model router chooses between local and cloud model.

Default route:

```text
local 8B model
```

Cloud route only when:

- cloud mode is `auto`
- local draft fails validation repeatedly
- retrieval confidence is too low for a complex turn
- user explicitly asks for higher quality
- scene transition or summarisation exceeds local model reliability

The model router must return a reason for the chosen route.

Example:

```json
{
  "provider": "local",
  "model": "qwen3:8b",
  "reason": "default local route"
}
```

---

## Actor Agent

The actor agent generates the actual roleplay response.

It receives:

- context packet
- selected model route
- generation settings

It returns:

- draft text
- model metadata
- token estimate if available

It must not:

- persist memory
- mutate scene state
- directly call retrieval
- bypass the context builder

---

## Critic Agent

The critic agent checks a draft before it is shown.

It should detect:

- secret leakage
- contradiction with scene state
- contradiction with retrieved lore
- character knowledge violation
- generic assistant tone
- failure to respond to the player action

For MVP, the critic can be simple.

It should return structured output:

```json
{
  "accepted": false,
  "issues": ["NPC reveals a secret the player has not discovered."],
  "repair_instruction": "Rewrite without revealing the source of the hidden cargo."
}
```

The orchestrator decides whether to retry locally or escalate to cloud.

---

## Memory Curator

The memory curator runs after a final response is chosen.

It extracts durable memory candidates from the turn.

It should remember:

- important player decisions
- promises
- discoveries
- relationship changes
- quest changes
- revealed secrets
- world-state changes

It should not remember:

- trivial banter
- every sentence
- temporary phrasing
- model mistakes

Every memory must have visibility metadata.

---

## Persistence Layer

For MVP, use SQLite.

Store:

- sessions
- turns
- memory records
- scene state snapshots
- configuration overrides

SQLite is enough for one user. Do not introduce PostgreSQL until there is a real need.

The persistence layer should expose repository classes/functions.

Application code should not spread raw SQL everywhere.

---

## Vector Store Layer

For MVP, use Qdrant as the vector store.

Qdrant stores:

- lore chunks
- memory chunks
- rule chunks
- world event chunks

Each vector record must include metadata:

```json
{
  "source_type": "canon_lore",
  "world_id": "default_world",
  "scene_id": "starting_room",
  "persona_id": "goddess",
  "visibility": "player",
  "tags": ["origin", "class_selection"]
}
```

Metadata filters are not optional. They are the safety boundary that prevents irrelevant or private context from entering prompts.

---

## Visibility Model

Visibility must be explicit from the beginning.

Recommended values:

```text
player
private
gm
character_private
system
```

Meaning:

| Visibility | Meaning |
|---|---|
| `player` | Safe to show or include in player-facing prompts |
| `private` | User/private configuration, not roleplay lore |
| `gm` | Hidden world facts, secrets, spoilers |
| `character_private` | Known only by specific NPCs or factions |
| `system` | Internal rules, never shown as fiction |

The context builder must filter by visibility for every prompt.

---

## Data Ownership

Authoritative state belongs to the application.

| Data | Owner |
|---|---|
| Session state | SQLite + application code |
| Persona definitions | JSON/YAML files loaded into domain models |
| Scene definitions | JSON/YAML files loaded into domain models |
| Lore documents | Files + vector store chunks |
| Embeddings | Vector store |
| Prompt text | Context builder |
| Generated prose | Actor agent output |
| Memory candidates | Memory curator output |
| Accepted memory | SQLite + vector store |

The LLM can suggest, but the application decides.

---

## Recommended External Services

For MVP:

```text
Ollama or llama.cpp server   local LLM
Qdrant                       vector store
SQLite                       structured persistence
Cloud OpenAI-compatible API  optional fallback
```

Avoid more services until the core loop works.

---

## Docker Compose Scope

Only Qdrant should be required in Docker for the first MVP.

Example:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant:/qdrant/storage
```

Run the local LLM separately through Ollama or llama.cpp on the host.

This avoids GPU/container complexity during early development.

---

## Configuration Model

The project should use environment variables loaded through Pydantic Settings.

Example settings:

```env
APP_ENV=local
DATABASE_URL=sqlite:///./data/role_rag.db
QDRANT_URL=http://localhost:6333

LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_API_KEY=ollama
LOCAL_LLM_MODEL=qwen3:8b
LOCAL_LLM_TEMPERATURE=0.75
LOCAL_LLM_MAX_TOKENS=700

CLOUD_MODE=ask
CLOUD_LLM_BASE_URL=https://api.openai.com/v1
CLOUD_LLM_API_KEY=replace_me
CLOUD_LLM_MODEL=gpt-4.1-mini
CLOUD_LLM_TEMPERATURE=0.65
CLOUD_LLM_MAX_TOKENS=1000

RETRIEVAL_TOP_K=6
RECENT_DIALOGUE_TURNS=8
MAX_LOCAL_RETRIES=1
MAX_TOTAL_AGENT_CALLS=4
```

Never commit real API keys.

---

## Prompt Boundary

There should be one module responsible for prompt assembly.

Do not let every agent invent its own prompt format independently.

Use stable prompt templates and pass explicit variables.

Recommended prompt categories:

```text
actor_prompt
critic_prompt
memory_curator_prompt
intent_classifier_prompt
query_rewrite_prompt
```

Each prompt should have tests for critical rules.

---

## Error Handling Strategy

For MVP, failures should be explicit and boring.

Examples:

| Failure | Behaviour |
|---|---|
| Local model unavailable | Return clear error or use cloud if allowed |
| Cloud unavailable | Fall back to local or return clear error |
| Qdrant unavailable | Continue without RAG only if configured |
| SQLite unavailable | Stop; persistence is required |
| Critic rejects twice | Use cloud if allowed, otherwise return best local response with warning |
| Memory extraction fails | Continue response, log memory failure |

Do not hide operational failures behind fake roleplay output.

---

## Testing Requirements

MVP architecture needs tests for:

1. Config loading.
2. Domain model validation.
3. Visibility filtering.
4. Context budget trimming.
5. Retrieval metadata filtering.
6. Model router decisions.
7. Orchestrator retry limits.
8. Memory curator parsing.
9. Prompt assembly.
10. End-to-end one-turn flow with mocked LLM.

The mocked LLM tests are mandatory. Without them, the project will become impossible to refactor safely.

---

## Minimal MVP Runtime Sequence

The first complete MVP should support this sequence:

```text
1. Start CLI.
2. Load default world and scene.
3. Load default narrator persona.
4. User enters a roleplay action.
5. Orchestrator builds context.
6. Local model generates response.
7. Critic accepts response.
8. Response is printed.
9. Memory curator writes one memory if relevant.
10. Next turn can use that memory.
```

RAG can initially be stubbed with in-memory chunks, then backed by Qdrant.

---

## Architecture Non-Goals

Do not implement these in the MVP:

- multiplayer
- authentication
- web UI
- autonomous tool-using agents
- fine-tuning
- voice
- image generation
- complex combat engine
- full timeline simulation
- distributed workers
- Kubernetes
- enterprise observability
- arbitrary plugin system

These are distractions until the core loop is stable.

---

## Coding Agent Instructions

A coding agent implementing this architecture should follow these rules:

```text
Implement incrementally.
Do not create empty architecture theater.
Do not introduce LangChain, LangGraph, or CrewAI in the MVP.
Do not create dynamic autonomous agent loops.
Do not hardcode provider-specific logic outside the provider layer.
Do not bypass visibility filtering.
Do not put prompt construction inside API routes.
Do not let generated text mutate state directly.
Write tests for every critical boundary.
Prefer simple Python modules over clever abstractions.
```

---

## Definition of Done for Architecture Phase

This architecture phase is complete when the repo contains:

- documented module layout
- clear component responsibilities
- local/cloud provider boundary
- explicit visibility model
- planned persistence boundary
- planned vector store boundary
- planned orchestration flow
- basic testing strategy

No full roleplay implementation is required yet.

The next document should specify the concrete implementation guide and repository skeleton.
