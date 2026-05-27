# 01 — Product Goal: Personal RoleRAG MVP

## Purpose

This document defines the product goal for the first MVP of **RoleRAG_POC**.

The project is a personal-use roleplaying RAG system built in Python. It is designed for one user, one local 8B-class language model, and one optional cloud model fallback.

The system should support long-running interactive roleplay by combining:

- structured scene state
- structured persona state
- short-term conversation memory
- long-term memory
- retrieval-augmented lore/context
- local-first model execution
- optional cloud fallback for difficult cases

The project must stay small, inspectable, and easy to run locally.

This is not an enterprise agent platform. This is not a multiplayer game engine. This is not a general chatbot wrapper. It is a focused proof of concept for roleplaying with controlled context, memory, and model routing.

---

## Product Vision

The user should be able to start a roleplaying session, speak or act as the player, and receive consistent narrative responses from the system.

The system should remember important events, retrieve relevant lore, keep NPCs from becoming omniscient, and avoid flooding the local model with too much context.

The local 8B model should be the default model. The cloud model should be used only when configured and justified by the orchestration layer.

The core idea is:

> The Python engine owns the state, memory, retrieval, routing, and safety boundaries. The LLM generates or evaluates text inside those boundaries.

---

## Target User

The MVP is for a single technical user running the system locally.

The user is expected to be comfortable with:

- Python
- local development environments
- Docker or local services
- environment variables
- running a local LLM server such as Ollama or llama.cpp
- editing Markdown, JSON, or YAML files

The MVP does not need authentication, multi-user sessions, account management, billing, permissions, or hosted deployment.

---

## Primary Use Case

The user wants to run a long-form roleplaying session with a local model.

Example flow:

1. The user starts a session.
2. The system loads a world, a scene, and active personas.
3. The user enters a message such as:

   ```text
   I approach Captain Maude and ask what happened to the missing cargo.
   ```

4. The engine classifies the turn.
5. The engine retrieves relevant scene memory and lore.
6. The engine builds a compact context packet.
7. The local model generates a response.
8. A critic checks for basic problems.
9. The system returns the response.
10. A memory curator extracts durable facts and stores them.
11. The next turn can use those memories.

---

## MVP Goal

The MVP is complete when the system can run a minimal but real roleplaying loop:

```text
User input
  -> turn orchestration
  -> persona/scene context assembly
  -> retrieval of relevant lore/memory
  -> local LLM response generation
  -> basic critique
  -> response returned to user
  -> durable memory extraction
  -> persisted session state
```

The MVP should prove the architecture, not polish the user experience.

---

## Core Product Requirements

### 1. Local-first roleplay

The system must work with a local 8B-class model as the default model.

The user should be able to configure a local OpenAI-compatible endpoint, for example:

```env
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_API_KEY=ollama
LOCAL_LLM_MODEL=qwen3:8b
```

The MVP should not require a cloud model to function.

---

### 2. Optional cloud fallback

The system may support a cloud model as fallback.

Cloud usage must be explicit and controlled by configuration:

```env
CLOUD_MODE=off|ask|auto
```

The cloud model should be used only for cases such as:

- failed local critique
- complex scene transition
- low retrieval confidence
- user explicitly requesting higher-quality output

The engine should behave the same regardless of whether the selected model is local or cloud. The provider changes capacity and quality, not the game rules.

---

### 3. Structured world and persona state

The system must represent scenes and personas as structured data, not only as prompt text.

The MVP should support at least:

- one world
- one active scene
- multiple personas
- one active narrator or NPC persona per turn
- public and private persona fields
- public and private scene fields

The system must distinguish between:

- player-visible state
- GM/private state
- character-private state

This is mandatory. Without visibility boundaries, long-running roleplay will leak secrets and collapse character knowledge.

---

### 4. Retrieval-augmented context

The system must retrieve relevant context instead of sending the whole world state to the model.

The MVP should retrieve from:

- session memory
- world lore documents
- persona notes
- scene notes

The retrieval result must be filtered by visibility before being added to the prompt.

The local 8B model should usually receive only a small number of chunks, for example 3–8 relevant pieces of context.

---

### 5. Durable memory

The system must remember important events across turns.

The MVP should persist memory items such as:

- promises
- decisions
- relationship changes
- quest changes
- discoveries
- revealed secrets
- important world events

The system should not remember every sentence. Trivial dialogue should not become permanent memory.

Every memory must have a visibility level.

---

### 6. Bounded orchestration

The system must avoid uncontrolled agent loops.

The MVP should use a fixed turn pipeline:

```text
classify -> retrieve -> build context -> generate -> critique -> optionally repair -> store memory
```

The maximum loop should be:

```text
local draft -> local critique -> one local retry -> optional cloud repair -> final response
```

No autonomous infinite multi-agent loops are allowed in the MVP.

---

### 7. Inspectable prompts and traces

The user should be able to inspect what happened during a turn.

The MVP should log or expose:

- selected model/provider
- routing reason
- retrieved context IDs
- context budget usage
- critic result
- memory writes

This does not need a complex UI. Structured logs are enough for the MVP.

---

## Non-Goals

The MVP must not attempt to solve everything.

The following are out of scope for the first MVP:

- multiplayer support
- user accounts
- authentication
- browser-based frontend
- voice input/output
- image generation
- combat engine
- rules automation
- autonomous NPC tool use
- fine-tuning
- hosted deployment
- Kubernetes
- message queues
- complicated agent frameworks
- full GraphRAG
- real-time streaming
- automatic world generation at large scale

These may be added later, but they should not contaminate the MVP.

---

## Design Constraints

### Small local model constraint

The system is designed around an 8B-class local model.

That means:

- do not send the full world state
- do not send all memories
- do not send every persona detail
- do not expect the model to manage state by itself
- do not ask the model to perform too many tasks in one call

The system must prepare a compact, relevant context packet for every turn.

---

### One-user constraint

Because this is for one user, the system can stay simple:

- SQLite is acceptable for metadata and session state.
- A local vector DB such as Qdrant is acceptable for retrieval.
- Local files are acceptable for worlds, personas, and lore documents.
- No access-control model is needed beyond internal visibility rules.

---

### Local-first privacy constraint

The local model should be the default for privacy, cost, and iteration speed.

If cloud fallback is enabled, the system should make it clear when cloud was used.

Sensitive/private lore should not be sent to cloud unless the user explicitly allows it through configuration.

---

## MVP Architecture Summary

The MVP should have these core components:

```text
CLI or FastAPI endpoint
  -> Turn Orchestrator
     -> Intent Classifier
     -> Scene/Persona Loader
     -> Retrieval Agent
     -> Context Builder
     -> Model Router
     -> Actor Agent
     -> Critic Agent
     -> Memory Curator
  -> Persistence Layer
  -> Vector Store
  -> LLM Providers
```

The orchestrator is the center of the system.

The LLM should not decide which memories exist, which facts are true, or which private information is allowed in the prompt. It may propose outputs, summaries, or memory candidates. The Python engine validates and stores them.

---

## Expected MVP User Interface

The first interface can be a CLI.

Example:

```bash
rolerag chat --world demo_world --scene harbor_intro
```

The user can then type messages interactively:

```text
> I ask the guard why the gates are closed.
```

The system responds:

```text
The guard shifts his weight, one hand resting near the horn at his belt...
```

A FastAPI endpoint can be added early or later, but the CLI is enough to prove the core engine.

---

## Data Inputs

The MVP should support simple local files.

Example structure:

```text
data/
  worlds/
    demo_world/
      world.yaml
      scenes/
        harbor_intro.yaml
      personas/
        narrator.yaml
        captain_maude.yaml
      lore/
        harbor_history.md
        factions.md
```

The system should be able to ingest lore Markdown files into the vector store.

---

## Acceptance Criteria

The MVP is acceptable when all of the following work:

1. A user can start a local roleplay session from the CLI.
2. The system loads a world, scene, and persona from local files.
3. The system sends a compact context packet to a local model.
4. The system retrieves at least three relevant chunks from memory or lore.
5. Retrieved chunks are filtered by visibility before prompt construction.
6. The local model generates an in-character or narrator-style response.
7. A critic checks the draft for basic issues.
8. The system stores at least one durable memory when appropriate.
9. The next turn can retrieve and use the stored memory.
10. The session can be stopped and resumed.
11. Cloud fallback can be disabled completely.
12. Cloud fallback can be enabled through config.
13. The system logs route, retrieval, critique, and memory decisions.
14. Unit tests cover visibility filtering and context assembly.
15. At least one integration test runs a full turn with mocked LLM providers.

---

## Quality Bar

The MVP does not need beautiful prose every time.

It does need architectural correctness:

- no secret leakage by default
- no uncontrolled agent loops
- no prompt blobs that contain the whole world
- no direct LLM mutation of authoritative state
- no cloud dependency for basic operation
- no hidden implicit memory rules
- no retrieval without metadata filtering

A boring reliable engine is better than an impressive chaotic demo.

---

## Key Risks

### Risk: The local model ignores instructions

Mitigation:

- keep prompts small
- use repetitive prompt structure
- use critic checks
- use structured outputs where possible
- keep state validation in Python

### Risk: Retrieval returns irrelevant chunks

Mitigation:

- metadata filters
- scene/persona-aware query construction
- small top-k
- tests with expected retrieved chunks

### Risk: Secret/private facts leak into narration

Mitigation:

- visibility flags on all memory and lore chunks
- deterministic filtering before prompt construction
- critic checks for leakage
- regression tests for private facts

### Risk: The architecture becomes too large

Mitigation:

- fixed MVP phases
- no frontend initially
- no heavy agent framework initially
- no enterprise infrastructure

### Risk: Memory becomes noisy

Mitigation:

- memory write policy
- importance score
- reject trivial memories
- periodic summaries later

---

## MVP Success Definition

The MVP succeeds if it proves this claim:

> A small Python engine can make a local 8B model usable for long-form roleplay by controlling context, memory, retrieval, visibility, and model routing outside the model.

The MVP fails if it becomes just another chatbot wrapper with a huge prompt.

---

## Guidance for Coding Agents

When implementing this MVP, follow these rules:

1. Start with documentation and a minimal runnable skeleton.
2. Do not implement the full architecture in one pass.
3. Keep each module small and testable.
4. Prefer explicit data models over unstructured dictionaries.
5. Keep visibility handling deterministic.
6. Mock LLM providers in tests.
7. Do not require real local or cloud models for unit tests.
8. Do not introduce frameworks unless they solve an immediate MVP problem.
9. Do not add frontend code in the MVP.
10. Preserve the local-first assumption.

---

## First Implementation Milestone

The first coding milestone after this document should create:

```text
pyproject.toml
README.md
.env.example
app/main.py
app/config.py
app/domain/models.py
app/llm/provider.py
app/llm/openai_compatible.py
app/orchestration/turn_orchestrator.py
tests/unit/test_config.py
tests/unit/test_domain_models.py
```

The first runnable behavior should be minimal:

```bash
python -m app.main
```

or:

```bash
rolerag chat
```

At this stage, fake/mock providers are acceptable. The first goal is the architecture seam, not model quality.
