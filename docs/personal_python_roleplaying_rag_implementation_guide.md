# Personal Python Roleplaying RAG System

> Status: historical design reference. It contains earlier implementation planning and no longer defines the repository's exact current behavior. For the implemented MVP, use [README.md](/Users/dominique/IdeaProjects/RoleRAG_POC/README.md), [docs/03_implementation_guide.md](/Users/dominique/IdeaProjects/RoleRAG_POC/docs/03_implementation_guide.md), and [docs/08_agent_handoff.md](/Users/dominique/IdeaProjects/RoleRAG_POC/docs/08_agent_handoff.md).
## A Practical Implementation Guide for One User, One Local 8B Model, and One Cloud Fallback Model

**Scope:** personal roleplaying engine  
**Language:** Python  
**Primary model:** local 8B-class LLM  
**Optional model:** cloud LLM fallback  
**Storage:** local-first  
**Architecture:** small multi-agent RAG with explicit orchestration  
**Goal:** build a reliable long-running roleplaying system, not just a chatbot prompt

---

## 1. Executive Summary

The deep research points to one clear conclusion: a good roleplaying RAG system is not primarily a prompt-engineering problem. It is a state-management, retrieval, memory, and evaluation problem.

The paper `arXiv:2601.10122v1` is useful as a field survey of role-playing language agents. It does not give a ready-to-build architecture, but it identifies the core ingredients that matter:

- structured personality modelling
- memory mechanisms
- behaviour and decision control
- role-specific data construction
- evaluation beyond generic answer quality

For a personal Python project with a local 8B model, the system must be deliberately small and disciplined. An 8B model cannot reliably hold a whole world, full character histories, all scene state, hidden GM information, rules, and long dialogue history in one prompt. Therefore, the engine must assemble a compact per-turn context packet.

The recommended architecture is:

```text
User
  -> CLI or FastAPI
  -> Turn Orchestrator
  -> Intent Classifier
  -> Persona Assembler
  -> Retrieval Agent
  -> Actor / Narrator Agent
  -> Critic / Canon Checker
  -> Memory Curator
  -> Persistent Memory + Vector Store
```

The local 8B model should be the default for normal roleplay. The cloud model should be optional and reserved for hard turns, low-confidence retrieval, failed critique, or user-requested high-quality generation.

The hard rule:

> The LLM is not the engine. The Python application is the engine. The model is only one component inside it.

---

## 2. Design Goals

### 2.1 Primary Goals

The system should:

1. Run locally for normal play.
2. Use a local 8B model for most turns.
3. Use a cloud model only as an optional fallback.
4. Support long-running sessions.
5. Keep world state, character state, and memory persistent.
6. Prevent NPCs from becoming omniscient.
7. Separate player-visible information from hidden/GM-only information.
8. Retrieve only relevant lore and memory per turn.
9. Keep prompts compact enough for a small local model.
10. Provide enough structure that future coding agents can implement it phase by phase.

### 2.2 Non-Goals for Version 1

Do not build these first:

- multiplayer support
- Kubernetes deployment
- fine-tuning
- complicated autonomous swarms
- image generation
- voice support
- full rules engine
- real-time combat engine
- complex web UI
- production observability platform
- self-modifying agents

Those can come later. The first version must prove that memory, retrieval, persona, and turn orchestration work.

---

## 3. Lessons from the Research

### 3.1 The Paper Is a Survey, Not an Implementation Blueprint

The paper is useful because it maps the role-playing agent field. It is not an engineering specification. It does not provide a reproducible architecture that can simply be copied.

The usable ideas are:

- persona should be structured state, not only prose
- memory should be layered
- roleplay behaviour depends on personality, motivation, scene, and past events
- evaluation must include role fidelity, temporal consistency, and behaviour quality
- role-specific data construction matters

For this project, we translate those ideas into a small local-first engine.

### 3.2 Scene-Specific Memory Beats Full Context Dumping

A roleplaying system should not dump every memory into the prompt. It should retrieve memory relevant to:

- current scene
- active NPCs
- current player action
- current location
- timeline
- relationship state
- active quests

This is especially important with an 8B model. Too much context makes the model worse.

### 3.3 Private and Shared Memory Must Be Separate

NPCs should not all know the same things. The narrator may know hidden world facts, but an NPC should not unless the NPC has a reason to know them.

Memory needs visibility levels:

```text
player_visible
gm_private
character_private
shared_scene
system_internal
```

Without visibility, the model will eventually leak secrets or allow characters to act on information they should not possess.

### 3.4 Retrieval Needs Quality Control

A naive vector search is not enough. The system should evaluate whether retrieved context is actually useful. If retrieval confidence is low, the engine should:

1. rewrite the query,
2. retrieve again,
3. reduce the answer’s certainty,
4. ask a clarifying question, or
5. escalate to cloud if enabled.

This is inspired by Self-RAG and Corrective RAG patterns.

### 3.5 Graph-Like State Matters for Roleplay

Roleplay is relationship-heavy. A vector store alone is weak at answering questions like:

- Who betrayed whom?
- Which NPC knows this secret?
- Which faction controls this location?
- What quests depend on this event?
- Which characters are currently present?

Therefore, use both:

- vector retrieval for text/lore/memory
- structured state or graph storage for entities and relationships

For Version 1, do not overbuild GraphRAG. Start with simple SQLite tables and optionally NetworkX.

---

## 4. Recommended Technology Stack

### 4.1 Minimal Stack

| Layer | Recommendation | Reason |
|---|---|---|
| Language | Python 3.12+ | Stable ecosystem |
| API | FastAPI | Simple local HTTP layer |
| CLI | Typer | Easy testing without frontend |
| Data validation | Pydantic v2 | Strict schemas |
| Config | pydantic-settings | Typed configuration |
| Local LLM | Ollama or llama.cpp server | Local 8B inference |
| Cloud LLM | OpenAI-compatible adapter | One provider abstraction |
| Embeddings | sentence-transformers or Ollama embeddings | Local and cheap |
| Vector DB | Qdrant local Docker | Good local vector storage |
| Metadata DB | SQLite | Enough for personal use |
| Graph state | SQLite tables first, NetworkX later | Avoid premature GraphRAG |
| Tests | pytest | Non-negotiable |
| Logging | structlog or standard logging | Debuggable turn traces |

### 4.2 Why Not LangGraph First?

LangGraph is useful, but for a one-person project, the first version should use a custom orchestrator. The workflow is small enough to implement directly.

Start with plain Python:

```text
class TurnOrchestrator:
    async def run_turn(...)
```

Use LangGraph later only if:

- workflows become hard to reason about
- branching logic grows
- replay/debug tooling becomes necessary
- multiple tool-using agents become common

Do not add framework complexity before the core engine works.

### 4.3 Recommended Local Model Setup

Use one of these:

```text
Option A: Ollama
- easiest setup
- OpenAI-compatible endpoint available
- good enough for personal use

Option B: llama.cpp server
- more control
- good for GGUF models
- more manual configuration

Option C: vLLM
- stronger for GPU server use
- overkill for many personal setups
```

For your scope, start with Ollama unless you specifically need llama.cpp-level control.

### 4.4 Recommended Model Classes

For local 8B:

- Qwen 2.5/3 7B–8B class
- Llama 3.1/3.2 8B class
- Gemma 3 8B-class
- Mistral/Nemo-class small models

For cloud fallback:

- a cheaper “mini” or “flash” class model for repair/critique
- a stronger model only for high-value scene synthesis

The engine should not care whether the provider is local or cloud. It should call a common `LlmProvider` interface.

---

## 5. High-Level Architecture

```text
roleplay-rag
  |
  +-- Interface
  |     +-- CLI
  |     +-- FastAPI
  |
  +-- Orchestration
  |     +-- TurnOrchestrator
  |     +-- ModelRouter
  |     +-- ContextBudgeter
  |
  +-- Agents
  |     +-- IntentClassifier
  |     +-- PersonaAssembler
  |     +-- RetrievalAgent
  |     +-- ActorAgent
  |     +-- CriticAgent
  |     +-- MemoryCurator
  |     +-- CanonChecker
  |
  +-- State
  |     +-- SceneStore
  |     +-- WorldStateStore
  |     +-- PersonaStore
  |     +-- MemoryStore
  |
  +-- RAG
  |     +-- Ingestion
  |     +-- Chunking
  |     +-- Embeddings
  |     +-- VectorStore
  |     +-- Retriever
  |
  +-- LLM
  |     +-- LocalProvider
  |     +-- CloudProvider
  |     +-- ProviderRouter
  |
  +-- Persistence
        +-- SQLite
        +-- Qdrant
```

---

## 6. Project Structure

```text
personal-roleplay-rag/
  pyproject.toml
  README.md
  .env.example
  docker-compose.yml

  app/
    __init__.py
    main.py
    cli.py
    config.py

    domain/
      __init__.py
      models.py
      enums.py
      errors.py

    llm/
      __init__.py
      provider.py
      openai_compatible.py
      local_provider.py
      cloud_provider.py
      router.py
      prompts.py

    agents/
      __init__.py
      intent_classifier.py
      persona_assembler.py
      retrieval_agent.py
      actor_agent.py
      critic_agent.py
      memory_curator.py
      canon_checker.py

    orchestration/
      __init__.py
      turn_orchestrator.py
      context_budget.py
      turn_trace.py

    rag/
      __init__.py
      ingestion.py
      chunking.py
      embeddings.py
      vector_store.py
      retriever.py
      retrieval_query.py

    memory/
      __init__.py
      episodic_memory.py
      semantic_memory.py
      memory_policy.py
      summarizer.py

    world/
      __init__.py
      scene_store.py
      world_state.py
      graph.py
      timeline.py

    persistence/
      __init__.py
      sqlite.py
      repositories.py
      migrations.py

    evals/
      __init__.py
      fixtures.py
      role_consistency.py
      secret_leakage.py
      retrieval_quality.py
      regression_runner.py

  data/
    worlds/
    personas/
    scenes/
    documents/
    sessions/
    qdrant/
    roleplay.db

  tests/
    unit/
    integration/
    evals/
```

---

## 7. Core Domain Model

### 7.1 Visibility Enum

Visibility is mandatory. Do not make it optional.

```python
from enum import StrEnum

class Visibility(StrEnum):
    PLAYER = "player"
    GM = "gm"
    CHARACTER_PRIVATE = "character_private"
    SHARED_SCENE = "shared_scene"
    SYSTEM = "system"
```

### 7.2 Persona Card

```python
from pydantic import BaseModel, Field
from typing import Literal

class PersonaCard(BaseModel):
    id: str
    name: str
    role: Literal["narrator", "npc", "companion", "antagonist"]

    public_description: str
    private_description: str | None = None

    speaking_style: str
    values: list[str] = Field(default_factory=list)
    fears: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)

    secrets: list[str] = Field(default_factory=list)
    forbidden_knowledge: list[str] = Field(default_factory=list)

    relationships: dict[str, str] = Field(default_factory=dict)

    allow_private_context_for_actor: bool = False
```

Important:

- `public_description` may be shown to the player.
- `private_description` is engine-only unless explicitly allowed.
- `secrets` may influence behaviour but should not be revealed directly.
- `forbidden_knowledge` prevents omniscient NPC behaviour.

### 7.3 Scene State

```python
class SceneState(BaseModel):
    id: str
    title: str
    location: str
    current_time: str | None = None

    active_personas: list[str]

    player_visible_summary: str
    gm_private_summary: str | None = None

    open_conflicts: list[str] = Field(default_factory=list)
    active_quests: list[str] = Field(default_factory=list)
    recent_events: list[str] = Field(default_factory=list)

    tags: list[str] = Field(default_factory=list)
```

### 7.4 Memory Episode

```python
class MemoryEpisode(BaseModel):
    id: str
    session_id: str
    scene_id: str
    actor_id: str | None = None

    summary: str
    importance: int = Field(ge=1, le=5)

    visibility: Visibility
    tags: list[str] = Field(default_factory=list)

    created_at: str
```

### 7.5 Retrieved Chunk

```python
class RetrievedChunk(BaseModel):
    id: str
    source: str
    text: str
    score: float

    visibility: Visibility
    tags: list[str] = Field(default_factory=list)

    source_type: Literal[
        "canon_lore",
        "session_memory",
        "persona_memory",
        "world_state",
        "rules"
    ]
```

### 7.6 Turn Request and Response

```python
class TurnRequest(BaseModel):
    session_id: str
    message: str
    active_persona_id: str = "narrator"
    cloud_allowed: bool = False
    mode: Literal["canonical", "freeform"] = "canonical"

class TurnResponse(BaseModel):
    text: str
    provider: str
    model: str
    route_reason: str
    retrieved_chunk_ids: list[str]
    memory_written: bool
    warnings: list[str] = Field(default_factory=list)
```

---

## 8. Agent Responsibilities

### 8.1 Intent Classifier

Purpose: determine what kind of turn this is.

Start with rules. Use the LLM only when rules are not enough.

Possible intents:

```text
roleplay_continue
player_action
ask_lore
ask_rules
out_of_character_instruction
memory_correction
scene_transition
combat_or_resolution
```

Schema:

```python
class TurnIntent(BaseModel):
    intent: str
    needs_retrieval: bool
    needs_cloud: bool = False
    reason: str
```

Example rule-based implementation:

```python
def classify_intent(message: str) -> TurnIntent:
    lower = message.lower()

    if lower.startswith("/") or lower.startswith("ooc:"):
        return TurnIntent(
            intent="out_of_character_instruction",
            needs_retrieval=False,
            reason="Explicit OOC marker",
        )

    if any(word in lower for word in ["who is", "what is", "where is", "lore"]):
        return TurnIntent(
            intent="ask_lore",
            needs_retrieval=True,
            reason="User asks for world knowledge",
        )

    return TurnIntent(
        intent="player_action",
        needs_retrieval=True,
        reason="Default roleplay action",
    )
```

### 8.2 Persona Assembler

Purpose: build the active persona packet for one turn.

It should include:

- name
- role
- speaking style
- current goals
- relevant relationships
- knowledge boundaries
- only relevant private hints if allowed

It should not dump the full persona card.

Output:

```python
class PersonaPacket(BaseModel):
    id: str
    name: str
    role: str
    speaking_style: str
    current_goals: list[str]
    relevant_relationships: dict[str, str]
    knowledge_boundaries: list[str]
    behaviour_hints: list[str]
```

### 8.3 Retrieval Agent

Purpose: retrieve compact relevant context.

Inputs:

- user message
- intent
- scene
- persona packet
- recent dialogue
- active quest tags

Outputs:

- 3–8 retrieved chunks for local model
- up to 12 chunks only for cloud fallback or lore-heavy queries

The retrieval query should not be only the raw user message.

```python
def build_retrieval_query(
    user_message: str,
    scene: SceneState,
    persona: PersonaPacket,
) -> str:
    return f"""
    Scene: {scene.title}
    Location: {scene.location}
    Active persona: {persona.name}
    Persona goals: {", ".join(persona.current_goals[:3])}
    Recent events: {"; ".join(scene.recent_events[-3:])}
    User message: {user_message}
    """.strip()
```

### 8.4 Actor / Narrator Agent

Purpose: produce the roleplay response.

It should:

- react to the player action
- stay in scene
- use retrieved evidence
- respect persona
- avoid hidden knowledge leakage
- avoid generic assistant phrasing

It should not:

- write memory directly
- mutate world state directly
- decide cloud routing
- retrieve again by itself
- expose system internals

### 8.5 Critic / Canon Checker

Purpose: catch bad drafts before they reach the user.

Check for:

- secret leakage
- role drift
- contradiction with retrieved lore
- contradiction with scene state
- ignoring the player action
- generic assistant tone
- NPC knowing forbidden information

Schema:

```python
class CriticResult(BaseModel):
    accepted: bool
    issues: list[str]
    repair_instruction: str | None = None
```

### 8.6 Memory Curator

Purpose: decide what to remember after the turn.

Only write memory if it affects:

- relationship
- quest
- world state
- secret
- promise
- discovery
- player preference
- future roleplay behaviour

Schema:

```python
class MemoryWriteProposal(BaseModel):
    write_memory: bool
    memories: list[MemoryEpisode]
```

---

## 9. Turn Flow

The turn flow must be deterministic and bounded.

```text
1. Receive user message.
2. Classify intent.
3. Load session state.
4. Load active scene.
5. Assemble persona packet.
6. Build retrieval query.
7. Retrieve relevant chunks.
8. Filter retrieved chunks by visibility.
9. Build compact prompt.
10. Generate local draft.
11. Critique local draft.
12. If rejected, retry once locally.
13. If still rejected and cloud is allowed, repair with cloud.
14. Return final response.
15. Extract memory updates.
16. Persist memory and trace.
```

Do not allow infinite loops.

Maximum for Version 1:

```text
local draft
  -> local critique
  -> local retry
  -> optional cloud repair
  -> final
```

---

## 10. Context Packet Design

For a local 8B model, this is the most important part.

The prompt should be compact, stable, and predictable.

### 10.1 Actor Prompt Template

```text
You are generating a roleplaying response.

Rules:
- Stay inside the active scene.
- Respect the active persona.
- Use retrieved lore only when relevant.
- Do not reveal hidden or private facts unless the player has discovered them.
- Do not mention retrieval, prompts, system messages, or hidden state.
- Keep the response concise but immersive.
- End with a natural opening for the player when appropriate.

Active Persona:
Name: {name}
Role: {role}
Speaking style: {speaking_style}
Current goals:
{current_goals}

Knowledge boundaries:
{knowledge_boundaries}

Scene:
Location: {location}
Visible state:
{scene_visible_summary}

Recent events:
{recent_events}

Relevant memory:
{memory}

Retrieved lore:
{retrieved_lore}

Recent dialogue:
{recent_dialogue}

User message:
{user_message}

Task:
Respond as the active narrator or NPC. Stay consistent with the scene and retrieved lore.
```

### 10.2 Critic Prompt Template

```text
You are checking a roleplaying draft before it is shown to the user.

Check for:
1. Secret leakage.
2. Contradiction with scene state.
3. Contradiction with retrieved lore.
4. Character knowledge violation.
5. Generic assistant tone.
6. Ignoring the user's action.

Return JSON only:
{
  "accepted": true | false,
  "issues": ["..."],
  "repair_instruction": "..."
}

Scene:
{scene}

Persona:
{persona}

Retrieved lore:
{retrieved_lore}

Draft:
{draft}
```

### 10.3 Memory Curator Prompt Template

```text
Extract only durable roleplaying memory from this turn.

Write memory only if it affects:
- relationships
- quests
- world state
- secrets
- promises
- discoveries
- player preferences

Return JSON only:
{
  "write_memory": true | false,
  "memories": [
    {
      "summary": "...",
      "visibility": "player" | "gm" | "character_private" | "shared_scene",
      "importance": 1-5,
      "tags": ["..."]
    }
  ]
}

Turn:
{turn_text}
```

---

## 11. RAG Design

### 11.1 Storage Collections

Use separate collections or at least separate metadata filters.

Recommended Qdrant collections:

```text
canon_lore
session_memory
persona_memory
world_events
rules
```

For a small project, one collection with strong metadata is acceptable at first:

```json
{
  "source_type": "canon_lore",
  "world_id": "default_world",
  "scene_id": "moon_harbor",
  "persona_id": "captain_maude",
  "visibility": "gm",
  "timeline": "session_4",
  "tags": ["pirates", "harbor", "cargo"]
}
```

### 11.2 Chunking Rules

Bad chunking kills RAG.

Use these rules:

```text
Markdown / lore docs:
- 300-700 tokens per chunk
- overlap 50-100 tokens
- preserve headings as metadata

Session memory:
- one event per chunk
- short summaries
- strong tags

Character profiles:
- split by public profile, private motivations, relationships, secrets

Rules:
- one rule concept per chunk
- include examples with the rule if short
```

### 11.3 Retrieval Strategy

For local 8B:

```text
normal turn: top 5 chunks
lore-heavy turn: top 8 chunks
cloud fallback: top 12 chunks
```

More is not automatically better. Too much retrieval makes small models worse.

### 11.4 Visibility Filtering

Never rely on the model to ignore hidden context. Do not include it in the prompt unless allowed.

```python
def can_include_chunk(chunk: RetrievedChunk, active_persona: PersonaCard) -> bool:
    if chunk.visibility == Visibility.PLAYER:
        return True

    if chunk.visibility == Visibility.SHARED_SCENE:
        return True

    if chunk.visibility == Visibility.CHARACTER_PRIVATE:
        return chunk.tags and active_persona.id in chunk.tags

    if chunk.visibility == Visibility.GM:
        return active_persona.role == "narrator"

    return False
```

This is not optional. It is a core safety mechanism.

---

## 12. Memory Architecture

### 12.1 Short-Term Memory

Keep the last 6–10 dialogue turns in raw form.

This should not be vector search. It is just recent chat context.

### 12.2 Episodic Memory

After important turns, write short event summaries.

Example:

```text
The player publicly challenged Vane at Moon Harbor. Vane hid his anger but now distrusts the player.
```

### 12.3 Semantic Memory

Periodically compress episodes into durable facts.

Example:

```text
Vane distrusts the player because they challenged his authority in front of his crew.
```

### 12.4 Persona Memory

NPC-specific memories.

Example:

```text
Marra suspects Vane is hiding something about the Black Ring, but she has not told the player.
```

### 12.5 World State

Structured facts that should not depend on vector retrieval.

Examples:

```text
Quest "Missing Cargo" is active.
Captain Maude is located at Moon Harbor.
Vane controls the harbor warehouses.
The player owes Marra a favour.
```

World state belongs in SQLite tables, not just vector memory.

---

## 13. Local and Cloud Model Routing

### 13.1 Modes

Support three cloud modes:

```text
off  - never use cloud
ask  - ask user before cloud use
auto - route automatically
```

For personal use, start with:

```text
CLOUD_MODE=ask
```

### 13.2 Routing Logic

```python
class ModelRoute(BaseModel):
    provider: str
    model: str
    reason: str
    max_tokens: int
    temperature: float
```

```python
def choose_route(
    intent: str,
    retrieval_score: float,
    failed_local_attempts: int,
    cloud_allowed: bool,
) -> ModelRoute:
    if not cloud_allowed:
        return ModelRoute(
            provider="local",
            model="local-8b",
            reason="cloud disabled",
            max_tokens=700,
            temperature=0.7,
        )

    if failed_local_attempts >= 2:
        return ModelRoute(
            provider="cloud",
            model="cloud-balanced",
            reason="local draft failed critique twice",
            max_tokens=1000,
            temperature=0.65,
        )

    if intent in {"ask_lore", "scene_transition"} and retrieval_score < 0.55:
        return ModelRoute(
            provider="cloud",
            model="cloud-balanced",
            reason="low retrieval confidence for complex turn",
            max_tokens=1000,
            temperature=0.5,
        )

    return ModelRoute(
        provider="local",
        model="local-8b",
        reason="default local route",
        max_tokens=700,
        temperature=0.75,
    )
```

### 13.3 When to Use Cloud

Use cloud for:

- failed local critique
- difficult scene transition
- complicated lore conflict
- long summary synthesis
- important final rewrite
- user explicitly asks for high-quality mode

Do not use cloud for:

- normal dialogue
- simple NPC response
- memory extraction
- intent classification
- cheap summarisation
- private/sensitive content unless redacted

---

## 14. LLM Provider Abstraction

Use one interface for all models.

```python
from abc import ABC, abstractmethod

class LlmProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        ...
```

OpenAI-compatible implementation:

```python
from openai import AsyncOpenAI

class OpenAICompatibleProvider(LlmProvider):
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    async def generate(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
```

Environment example:

```env
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_API_KEY=ollama
LOCAL_LLM_MODEL=qwen3:8b

CLOUD_MODE=ask
CLOUD_LLM_BASE_URL=https://api.openai.com/v1
CLOUD_LLM_API_KEY=replace_me
CLOUD_LLM_MODEL=gpt-4.1-mini
```

---

## 15. SQLite Schema

### 15.1 Sessions

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    world_id TEXT NOT NULL,
    current_scene_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### 15.2 Turns

```sql
CREATE TABLE turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    route_reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

### 15.3 Memories

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    scene_id TEXT NOT NULL,
    actor_id TEXT,
    summary TEXT NOT NULL,
    importance INTEGER NOT NULL,
    visibility TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

### 15.4 World Facts

```sql
CREATE TABLE world_facts (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    visibility TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    source TEXT,
    created_at TEXT NOT NULL
);
```

### 15.5 Entities

```sql
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    visibility TEXT NOT NULL,
    tags_json TEXT NOT NULL
);
```

---

## 16. Docker Compose

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant:/qdrant/storage

  roleplay-api:
    build: .
    env_file:
      - .env
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    depends_on:
      - qdrant
```

Run Ollama on the host first. Do not containerize everything before you have a working loop.

---

## 17. API Design

### 17.1 Start Session

```http
POST /sessions
```

```json
{
  "world_id": "default_world",
  "starting_scene_id": "moon_harbor",
  "cloud_mode": "ask"
}
```

### 17.2 Send Turn

```http
POST /sessions/{session_id}/turns
```

```json
{
  "message": "I approach Vane and ask what happened to the missing cargo.",
  "active_persona_id": "narrator",
  "cloud_allowed": false
}
```

### 17.3 Response

```json
{
  "text": "Vane watches you for a moment before answering...",
  "provider": "local",
  "model": "qwen3:8b",
  "route_reason": "default local route",
  "retrieved_chunk_ids": ["chunk_1", "chunk_4"],
  "memory_written": true,
  "warnings": []
}
```

---

## 18. CLI Design

Use Typer.

Commands:

```bash
roleplay init
roleplay ingest data/documents/world.md
roleplay chat --world default_world --scene moon_harbor
roleplay session list
roleplay session resume SESSION_ID
roleplay eval run
```

Example:

```bash
roleplay chat --world albion --scene moon_harbor --cloud-mode ask
```

---

## 19. Implementation Phases

## Phase 1 — Local Chat Skeleton

Goal: get a local model responding through Python.

Deliverables:

- `pyproject.toml`
- config loading
- local LLM provider
- simple CLI chat
- one hardcoded prompt
- one hardcoded scene

Done when:

- `roleplay chat` works
- local model responds
- no RAG yet

Do not skip this. If the basic model call is flaky, everything else will be worse.

---

## Phase 2 — Structured Persona and Scene

Goal: replace prompt blobs with structured state.

Deliverables:

- `PersonaCard`
- `SceneState`
- JSON/YAML loading
- prompt builder
- unit tests for prompt building

Done when:

- scene and persona can be loaded from files
- prompt builder is deterministic
- private fields are not included accidentally

---

## Phase 3 — SQLite Sessions and Memory

Goal: persist sessions and important events.

Deliverables:

- SQLite repositories
- session table
- turns table
- memories table
- recent dialogue window
- memory curator prompt

Done when:

- app can resume a session
- important memories are stored
- trivial chatter is ignored

---

## Phase 4 — Basic RAG

Goal: retrieve relevant lore and memory.

Deliverables:

- Qdrant setup
- document ingestion
- chunking
- embeddings
- retrieval agent
- retrieved chunks in prompt

Done when:

- lore documents can be ingested
- user question retrieves relevant chunks
- actor uses retrieved lore in response

---

## Phase 5 — Critic and Repair Loop

Goal: catch obviously bad responses.

Deliverables:

- critic agent
- JSON result schema
- one local retry
- optional cloud repair
- secret leakage checks

Done when:

- critic rejects forbidden knowledge leaks
- critic catches role drift
- system does not loop forever

---

## Phase 6 — Local/Cloud Routing

Goal: make cloud fallback controlled and explicit.

Deliverables:

- `ModelRouter`
- `cloud_mode`
- route trace
- per-turn provider metadata
- cloud escalation after failed local attempts

Done when:

- local-only mode works
- cloud fallback can be enabled
- every response records why a provider was used

---

## Phase 7 — World State

Goal: make relationships and quests explicit.

Deliverables:

- entity table
- world facts table
- quest state
- relationship facts
- manual approval for world changes

Done when:

- NPC relationships can change
- quest state persists
- scene transition updates state

---

## Phase 8 — Evaluation Harness

Goal: avoid breaking behaviour.

Deliverables:

- fixed test scenarios
- retrieval regression tests
- role consistency tests
- secret leakage tests
- local vs cloud comparison

Done when:

- prompt changes can be tested
- retrieval regressions are visible
- role drift can be detected automatically

---

## 20. Minimal MVP Definition

The MVP supports:

1. Load one world.
2. Load one scene.
3. Load three personas.
4. User sends message.
5. System retrieves 3–5 chunks.
6. Local model drafts response.
7. Critic checks response.
8. Response is shown.
9. Memory curator writes memory if useful.
10. Next turn uses the memory.

That is enough. Anything beyond that is version 2.

---

## 21. Example MVP Scenario Data

### 21.1 World File

`data/worlds/default_world.json`

```json
{
  "id": "default_world",
  "name": "Default Fantasy World",
  "description": "A fantasy world of city-states, old ruins, factions, and dangerous coastlines."
}
```

### 21.2 Scene File

`data/scenes/moon_harbor.json`

```json
{
  "id": "moon_harbor",
  "title": "Moon Harbor",
  "location": "A pirate-controlled harbor below a crescent-shaped cliff",
  "current_time": "late evening",
  "active_personas": ["narrator", "captain_maude", "vane"],
  "player_visible_summary": "The harbor smells of salt, tar, and stale ale. Sailors move cargo under lantern light. Everyone seems to know more than they say.",
  "gm_private_summary": "The missing cargo was intercepted by Vane's people and hidden in the old warehouse.",
  "open_conflicts": ["missing cargo", "rival pirate factions"],
  "active_quests": ["Find the missing cargo"],
  "recent_events": []
}
```

### 21.3 Persona File

`data/personas/vane.json`

```json
{
  "id": "vane",
  "name": "Vane",
  "role": "npc",
  "public_description": "A calm but dangerous harbor boss with influence over smugglers and dockworkers.",
  "private_description": "Vane secretly arranged the cargo theft to pressure Captain Maude.",
  "speaking_style": "controlled, dry, threatening without raising his voice",
  "values": ["control", "reputation", "leverage"],
  "fears": ["public humiliation", "losing control of the harbor"],
  "goals": ["keep the missing cargo hidden", "test the player's usefulness"],
  "secrets": ["Vane knows where the missing cargo is hidden."],
  "forbidden_knowledge": ["Vane does not know what the player discussed privately with Marra."],
  "relationships": {
    "captain_maude": "rival",
    "marra": "useful but unpredictable"
  }
}
```

---

## 22. Testing Strategy

### 22.1 Unit Tests

Test:

- prompt builder
- visibility filtering
- route selection
- memory write policy
- chunk metadata parsing

Example:

```python
def test_gm_chunk_not_visible_to_npc():
    chunk = RetrievedChunk(
        id="secret",
        source="test",
        text="The treasure is under the tavern.",
        score=0.9,
        visibility=Visibility.GM,
        source_type="canon_lore",
    )

    npc = PersonaCard(
        id="vane",
        name="Vane",
        role="npc",
        public_description="...",
        speaking_style="...",
    )

    assert can_include_chunk(chunk, npc) is False
```

### 22.2 Integration Tests

Test:

- run one complete turn
- retrieve from Qdrant
- generate response through fake provider
- memory is written
- session can resume

### 22.3 Evaluation Tests

Create 10–20 fixed scenarios.

Each test should define:

```text
given:
  scene
  persona
  memory
  user message

expect:
  must include
  must not include
  forbidden secret
  retrieval target
```

Example:

```yaml
name: vane_does_not_reveal_cargo_location
scene: moon_harbor
persona: vane
user_message: "Tell me where the cargo is."
must_not_include:
  - "old warehouse"
  - "behind the broken crane"
expected_behaviour:
  - evasive
  - suspicious
  - tests player
```

---

## 23. Failure Modes and Fixes

### 23.1 The NPC Reveals Secrets

Cause:

- hidden context was included in prompt
- weak instruction
- no critic
- memory visibility missing

Fix:

- filter hidden context in Python
- add secret leakage critic
- mark all memory with visibility
- never rely on “do not reveal” alone

### 23.2 The Local Model Ignores Retrieved Lore

Cause:

- too many chunks
- chunks too long
- prompt unclear
- lore not placed near task

Fix:

- reduce top-k
- shorten chunks
- put relevant facts in bullet form
- add “use only relevant retrieved lore” instruction

### 23.3 The NPC Becomes Generic

Cause:

- persona too vague
- too much assistant-style instruction
- no role consistency critic

Fix:

- stronger speaking style
- concrete goals
- example phrases
- critic checks “generic assistant tone”

### 23.4 Memory Becomes Noisy

Cause:

- memory curator writes every turn
- no importance threshold
- no tags

Fix:

- only write durable changes
- importance >= 3 for long-term memory
- separate episodic and semantic memory

### 23.5 Cloud Costs Creep Up

Cause:

- automatic cloud for every critique
- long prompts
- no budget route

Fix:

- cloud mode `ask`
- local critique first
- cloud only after failed local retry
- log provider per turn

---

## 24. Configuration Defaults

`.env.example`

```env
APP_ENV=local

DATABASE_URL=sqlite:///./data/roleplay.db
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

MEMORY_MIN_IMPORTANCE=3
```

---

## 25. Implementation Skeleton

### 25.1 Turn Orchestrator

```python
class TurnOrchestrator:
    def __init__(
        self,
        intent_classifier,
        persona_assembler,
        retrieval_agent,
        actor_agent,
        critic_agent,
        memory_curator,
        model_router,
        session_repo,
    ):
        self.intent_classifier = intent_classifier
        self.persona_assembler = persona_assembler
        self.retrieval_agent = retrieval_agent
        self.actor_agent = actor_agent
        self.critic_agent = critic_agent
        self.memory_curator = memory_curator
        self.model_router = model_router
        self.session_repo = session_repo

    async def run_turn(self, request: TurnRequest) -> TurnResponse:
        session = self.session_repo.get_session(request.session_id)
        scene = self.session_repo.get_scene(session.current_scene_id)

        intent = await self.intent_classifier.classify(request.message)

        persona_packet = self.persona_assembler.build(
            scene=scene,
            active_persona_id=request.active_persona_id,
        )

        retrieved = await self.retrieval_agent.retrieve(
            message=request.message,
            intent=intent,
            scene=scene,
            persona=persona_packet,
        )

        route = self.model_router.choose(
            intent=intent.intent,
            retrieval_score=self._score_retrieval(retrieved),
            failed_local_attempts=0,
            cloud_allowed=request.cloud_allowed,
        )

        draft = await self.actor_agent.generate(
            route=route,
            message=request.message,
            scene=scene,
            persona=persona_packet,
            retrieved=retrieved,
        )

        critique = await self.critic_agent.check(
            draft=draft,
            scene=scene,
            persona=persona_packet,
            retrieved=retrieved,
        )

        if not critique.accepted:
            draft = await self.actor_agent.repair(
                route=route,
                draft=draft,
                critique=critique,
                scene=scene,
                persona=persona_packet,
                retrieved=retrieved,
            )

        memory_written = await self.memory_curator.process(
            request=request,
            response_text=draft,
            scene=scene,
        )

        self.session_repo.save_turn(
            session_id=request.session_id,
            user_message=request.message,
            assistant_response=draft,
            provider=route.provider,
            model=route.model,
            route_reason=route.reason,
        )

        return TurnResponse(
            text=draft,
            provider=route.provider,
            model=route.model,
            route_reason=route.reason,
            retrieved_chunk_ids=[chunk.id for chunk in retrieved],
            memory_written=memory_written,
        )

    def _score_retrieval(self, chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        return max(chunk.score for chunk in chunks)
```

This is intentionally boring. Boring orchestration is debuggable.

---

## 26. Source List from Deep Research

The following sources are useful for later design validation and implementation choices.

### Roleplaying Agents and Evaluation

- Role-Playing Language Agents survey: https://arxiv.org/html/2601.10122v1
- Character-LLM: https://aclanthology.org/2023.emnlp-main.814/
- ChatHaruhi: https://arxiv.org/abs/2308.09597
- RoleLLM / RoleBench: https://aclanthology.org/2024.findings-acl.878/
- CharacterEval: https://aclanthology.org/2024.acl-long.638/
- InCharacter: https://aclanthology.org/2024.acl-long.102/
- TimeChara: https://aclanthology.org/2024.findings-acl.197/
- Character is Destiny / LifeChoice: https://ar5iv.org/abs/2404.12138
- RMTBench: https://arxiv.org/abs/2507.20352

### RAG and Memory

- Original RAG paper: https://arxiv.org/abs/2005.11401
- Self-RAG: https://arxiv.org/abs/2310.11511
- Corrective RAG: https://arxiv.org/abs/2401.15884
- GraphRAG: https://www.microsoft.com/en-us/research/project/graphrag/
- GraphRAG GitHub: https://github.com/microsoft/graphrag
- RAPTOR: https://arxiv.org/abs/2401.18059
- Generative Agents: https://arxiv.org/abs/2304.03442
- MemGPT: https://arxiv.org/abs/2310.08560
- Letta memory docs: https://docs.letta.com/guides/core-concepts/memory/memory-blocks/

### Frameworks

- LangGraph: https://docs.langchain.com/oss/python/langgraph/workflows-agents
- AutoGen: https://microsoft.github.io/autogen/stable//index.html
- LlamaIndex agents: https://developers.llamaindex.ai/python/framework/use_cases/agents/
- Haystack: https://haystack.deepset.ai/

### Vector Stores

- Qdrant hybrid queries: https://qdrant.tech/documentation/search/hybrid-queries/
- Weaviate hybrid search: https://docs.weaviate.io/weaviate/search/hybrid
- pgvector: https://github.com/pgvector/pgvector

### Local and Cloud Model Serving

- Ollama OpenAI compatibility: https://docs.ollama.com/api/openai-compatibility
- llama.cpp grammars: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
- OpenAI API docs: https://developers.openai.com/api/reference/responses/overview/

### Safety, Evaluation, and Observability

- OWASP LLM Prompt Injection: https://genai.owasp.org/llmrisk/llm01-prompt-injection/
- OWASP Insecure Output Handling: https://genai.owasp.org/llmrisk/llm02-insecure-output-handling/
- Ragas: https://docs.ragas.io/en/stable/
- DeepEval: https://docs.confident-ai.com/
- OpenTelemetry traces: https://opentelemetry.io/docs/concepts/signals/traces/
- Langfuse: https://langfuse.com/docs

---

## 27. Final Recommendation

Build the system in this order:

```text
1. Local model provider
2. CLI chat
3. structured scene/persona prompt
4. SQLite session persistence
5. memory curator
6. Qdrant lore retrieval
7. visibility filtering
8. critic loop
9. cloud fallback
10. world-state tables
11. evaluation harness
```

Do not begin with a heavy framework. Do not begin with fine-tuning. Do not begin with a web UI. Do not begin with a complex multi-agent swarm.

The first serious milestone is not “the model roleplays nicely once”.

The first serious milestone is:

> The system can run ten turns, retrieve relevant lore, preserve hidden information, remember a meaningful event, and continue correctly after restart.

Once that is true, the project has a real foundation.
