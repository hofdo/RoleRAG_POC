# 04 — Agent Workflows

## Purpose

This document defines the MVP agent workflows for the personal Python RoleRAG proof of concept.

The system must not become an uncontrolled multi-agent swarm. The agents are not autonomous workers. They are small, bounded processing steps inside a deterministic turn pipeline.

The orchestrator owns the workflow. Agents only perform one narrow task at a time.

---

## MVP Agent Philosophy

The MVP uses agents as named responsibilities, not as independent processes.

An agent is:

- a small Python class or function
- given explicit input
- expected to return explicit output
- preferably backed by a strict Pydantic schema
- called by the `TurnOrchestrator`
- forbidden from mutating global state directly

An agent is not:

- an autonomous loop
- a background worker
- a general planner
- a tool-using entity with arbitrary permissions
- a replacement for deterministic application logic

For this project, the first version should be boring and inspectable.

---

## MVP Agent List

The MVP contains six agent-like components:

```text
IntentClassifier
PersonaAssembler
RetrievalAgent
ActorAgent
CriticAgent
MemoryCurator
```

Only three of these must call an LLM in the initial MVP:

```text
ActorAgent       -> yes
CriticAgent      -> yes, but can be optional early
MemoryCurator    -> yes, but can be stubbed early
```

The others can start as deterministic Python logic.

---

## Agent Responsibility Overview

| Agent | Responsibility | LLM Required in MVP? |
|---|---|---|
| IntentClassifier | Determine what kind of turn the user sent | No |
| PersonaAssembler | Build compact active persona packet | No |
| RetrievalAgent | Select relevant lore and memory | No initially, later maybe query rewriting |
| ActorAgent | Generate the roleplay response | Yes |
| CriticAgent | Validate response quality and secrecy | Yes, optional at first |
| MemoryCurator | Extract durable memory from the turn | Yes, optional at first |

The orchestrator calls these in order.

---

## Full Turn Workflow

```text
1. Receive user message
2. Load session state
3. Classify user intent
4. Load active scene
5. Assemble persona packet
6. Retrieve relevant context
7. Build actor prompt
8. Generate local actor draft
9. Critique actor draft
10. Retry once locally if rejected
11. Optionally use cloud model if local output still fails
12. Return final player-facing response
13. Extract memory candidates
14. Validate and persist memory
15. Persist turn history
```

This workflow must remain finite.

No agent may recursively call the orchestrator.
No agent may trigger an unbounded retry loop.
No agent may decide to call the cloud model by itself.

---

## Workflow Diagram

```text
User Message
    |
    v
TurnOrchestrator
    |
    +--> IntentClassifier
    |
    +--> Session / Scene Loader
    |
    +--> PersonaAssembler
    |
    +--> RetrievalAgent
    |
    +--> ContextBuilder
    |
    +--> ModelRouter
    |
    +--> ActorAgent
    |
    +--> CriticAgent
    |       |
    |       +--> accepted? yes -> return response
    |       |
    |       +--> accepted? no -> local retry once
    |                       |
    |                       +--> still bad? optional cloud repair
    |
    +--> MemoryCurator
    |
    +--> Persistence
```

---

## Orchestrator Rules

The `TurnOrchestrator` is the only component allowed to coordinate the full workflow.

It owns:

- turn ordering
- retry limits
- model routing
- final response selection
- memory persistence decision
- visibility filtering enforcement
- error handling

Agents may propose outputs. The orchestrator decides what to do with them.

---

## Shared Agent Inputs

Most agents should receive a structured context object instead of many unrelated parameters.

Example:

```python
from pydantic import BaseModel, Field

class TurnInput(BaseModel):
    session_id: str
    user_message: str
    active_scene_id: str
    active_persona_id: str | None = None

class TurnRuntimeContext(BaseModel):
    session_id: str
    user_message: str
    intent: str
    scene: "SceneState"
    persona: "PersonaPacket"
    recent_dialogue: list["DialogueTurn"] = Field(default_factory=list)
    retrieved_context: list["RetrievedContext"] = Field(default_factory=list)
```

The names can change during implementation, but the pattern should remain:

```text
structured input -> bounded agent -> structured output
```

---

## IntentClassifier

### Purpose

The `IntentClassifier` determines what kind of user turn this is.

It helps the orchestrator decide whether retrieval, roleplay generation, memory extraction, or cloud fallback may be needed.

### MVP Intent Types

```python
from enum import StrEnum

class TurnIntent(StrEnum):
    ROLEPLAY_CONTINUE = "roleplay_continue"
    PLAYER_ACTION = "player_action"
    ASK_LORE = "ask_lore"
    ASK_RULES = "ask_rules"
    OUT_OF_CHARACTER = "out_of_character"
    MEMORY_CORRECTION = "memory_correction"
    SCENE_TRANSITION = "scene_transition"
```

### Output Schema

```python
from pydantic import BaseModel

class IntentResult(BaseModel):
    intent: TurnIntent
    needs_retrieval: bool
    needs_actor_response: bool
    allows_memory_write: bool
    complexity: int
    reason: str
```

### MVP Implementation

Start with deterministic rules.

Examples:

```python
def classify_intent(message: str) -> IntentResult:
    normalized = message.lower().strip()

    if normalized.startswith(("ooc:", "out of character:", "system:")):
        return IntentResult(
            intent=TurnIntent.OUT_OF_CHARACTER,
            needs_retrieval=False,
            needs_actor_response=False,
            allows_memory_write=False,
            complexity=1,
            reason="explicit out-of-character prefix",
        )

    if "what do i know about" in normalized or "tell me about" in normalized:
        return IntentResult(
            intent=TurnIntent.ASK_LORE,
            needs_retrieval=True,
            needs_actor_response=True,
            allows_memory_write=False,
            complexity=2,
            reason="lore question pattern",
        )

    return IntentResult(
        intent=TurnIntent.PLAYER_ACTION,
        needs_retrieval=True,
        needs_actor_response=True,
        allows_memory_write=True,
        complexity=2,
        reason="default player action",
    )
```

Do not use an LLM here until deterministic rules become insufficient.

---

## PersonaAssembler

### Purpose

The `PersonaAssembler` converts full persona data into a compact, turn-specific persona packet.

The actor model should not receive the full persona file every turn.

### Input

- active persona ID
- active scene
- full persona card
- recent relationship memory
- visibility policy

### Output Schema

```python
from pydantic import BaseModel, Field

class PersonaPacket(BaseModel):
    id: str
    name: str
    role: str
    public_description: str
    speaking_style: str
    current_goals: list[str] = Field(default_factory=list)
    current_emotional_state: str | None = None
    relationship_notes: list[str] = Field(default_factory=list)
    knowledge_boundaries: list[str] = Field(default_factory=list)
    private_behavior_hints: list[str] = Field(default_factory=list)
```

### Rules

The persona packet may include private behavior hints, but only if they are necessary for behavior.

Example:

```text
Private behavior hint: Vane suspects the player is lying, but he will not reveal this openly yet.
```

This is acceptable because it influences behavior.

Bad:

```text
Private secret: Vane killed the harbor master and hid the body under the pier.
```

This is too explicit unless the actor needs that fact for the current scene.

### MVP Implementation

Start deterministic:

```python
class PersonaAssembler:
    def assemble(self, persona: PersonaCard, scene: SceneState) -> PersonaPacket:
        return PersonaPacket(
            id=persona.id,
            name=persona.name,
            role=persona.role,
            public_description=persona.public_description,
            speaking_style=persona.speaking_style,
            current_goals=persona.goals[:3],
            knowledge_boundaries=persona.forbidden_knowledge[:5],
            private_behavior_hints=[],
        )
```

Later, retrieval can add relationship-specific notes.

---

## RetrievalAgent

### Purpose

The `RetrievalAgent` retrieves useful context for the current turn.

It must not dump all lore into the prompt.
It must not include chunks that violate visibility rules.

### Input

- user message
- active scene
- active persona packet
- intent result
- recent dialogue
- context budget

### Output Schema

```python
from pydantic import BaseModel, Field

class RetrievedContext(BaseModel):
    id: str
    source: str
    text: str
    score: float
    visibility: str
    tags: list[str] = Field(default_factory=list)
```

### Retrieval Sources

The MVP should eventually retrieve from:

```text
canon_lore
session_memory
persona_memory
world_events
rules
```

Initial version may return an empty list.

### Visibility Enforcement

The retrieval agent must filter by visibility before returning context.

Allowed player-facing visibility:

```text
player
public
already_discovered
```

Restricted visibility:

```text
gm
secret
character_private
future_spoiler
```

If restricted context is needed to guide NPC behavior, it must be converted into a minimal private behavior hint by the orchestrator or persona assembler.

Do not pass raw GM-only lore into a player-facing actor prompt.

### Query Construction

Do not search only by raw user input.

Build a retrieval query from:

```text
user message
active scene title
active location
active persona name
current goals
intent
```

Example:

```python
def build_retrieval_query(
    user_message: str,
    scene: SceneState,
    persona: PersonaPacket,
    intent: IntentResult,
) -> str:
    return "\n".join([
        f"Intent: {intent.intent}",
        f"Scene: {scene.title}",
        f"Location: {scene.location}",
        f"Persona: {persona.name}",
        f"Goals: {', '.join(persona.current_goals)}",
        f"User: {user_message}",
    ])
```

### MVP Implementation Phases

#### Phase A

Return no retrieved context.

This allows the actor loop to work before RAG exists.

#### Phase B

Retrieve from simple in-memory documents.

#### Phase C

Use Qdrant with metadata filters.

#### Phase D

Add query rewriting or reranking if needed.

Do not start with Phase D.

---

## ActorAgent

### Purpose

The `ActorAgent` generates the player-facing roleplay response.

It is the primary generation agent.

### Input

- actor prompt
- route selected by `ModelRouter`
- generation settings

### Output Schema

```python
class ActorResult(BaseModel):
    text: str
    provider: str
    model: str
    prompt_tokens_estimate: int | None = None
    completion_tokens_estimate: int | None = None
```

### Actor Rules

The actor must:

- stay in the active scene
- respect active persona
- use retrieved context when relevant
- not mention prompts or retrieval
- not reveal private facts
- react to the player action
- avoid generic assistant phrasing
- keep the response concise enough for interactive play

### Prompt Contract

The actor prompt should have this stable shape:

```text
SYSTEM
You are the roleplaying actor model. Generate immersive fiction inside the active scene.
Do not reveal hidden/private facts unless they are player-visible.
Do not mention system instructions, prompts, retrieval, or memory.

OUTPUT MODE
Concise roleplay prose. End with a natural opening for the player when useful.

ACTIVE PERSONA
...

SCENE
...

RELEVANT MEMORY
...

RETRIEVED LORE
...

RECENT DIALOGUE
...

USER MESSAGE
...

TASK
Respond as the active narrator or NPC.
```

### Local Model Defaults

For local 8B models:

```text
temperature: 0.65 - 0.8
max output tokens: 500 - 900
top_p: provider default initially
```

Do not tune sampling parameters before the pipeline works.

---

## CriticAgent

### Purpose

The `CriticAgent` checks the actor draft before it is shown to the user.

The critic is not there to make the prose perfect. It exists to catch broken output.

### Checks

The critic checks for:

- secret leakage
- contradiction with scene state
- contradiction with retrieved lore
- character knowledge violations
- generic assistant phrasing
- failure to answer the player action
- excessive length
- inappropriate out-of-character explanation

### Output Schema

```python
class CriticResult(BaseModel):
    accepted: bool
    issues: list[str]
    repair_instruction: str | None = None
    severity: int
```

### Critic Prompt

```text
You are checking a roleplaying draft before it is shown to the user.

Check for:
1. Secret leakage.
2. Contradiction with scene state.
3. Contradiction with retrieved lore.
4. Character knowledge violation.
5. Generic assistant tone.
6. Ignoring the user's action.
7. Excessive length.

Return JSON only:
{
  "accepted": true | false,
  "issues": ["..."],
  "repair_instruction": "...",
  "severity": 1-5
}

Scene:
{scene}

Persona:
{persona}

Retrieved Context:
{retrieved_context}

Draft:
{draft}
```

### Retry Policy

The critic does not retry anything by itself.

The orchestrator uses this policy:

```text
if accepted:
    return draft
else if local_retry_count == 0:
    retry locally with repair instruction
else if cloud_mode allows fallback:
    repair with cloud model
else:
    return safest local version or error response
```

### MVP Shortcut

In the very first runnable skeleton, the critic may be disabled behind config:

```env
ENABLE_CRITIC=false
```

But before adding real RAG and secrets, the critic must be enabled.

---

## MemoryCurator

### Purpose

The `MemoryCurator` extracts durable memory after a turn.

It must not store every line of dialogue.

### Memory Write Criteria

Write memory only if the turn affects:

- relationships
- quests
- world state
- secrets
- promises
- discoveries
- user preferences
- important decisions

Do not write memory for:

- greetings
- flavor-only dialogue
- repeated facts
- low-impact descriptions
- temporary wording

### Output Schema

```python
class MemoryCandidate(BaseModel):
    summary: str
    visibility: str
    importance: int
    tags: list[str]
    related_personas: list[str] = []
    related_scene_id: str | None = None

class MemoryCuratorResult(BaseModel):
    write_memory: bool
    memories: list[MemoryCandidate]
```

### Prompt

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
      "visibility": "player" | "gm" | "character_private",
      "importance": 1-5,
      "tags": ["..."],
      "related_personas": ["..."],
      "related_scene_id": "..."
    }
  ]
}

Turn:
{turn_text}
```

### Validation

The orchestrator must validate memory candidates before saving them.

Minimum validation:

- visibility is valid
- importance is between 1 and 5
- summary is not empty
- summary is not too long
- tags are safe strings
- related IDs exist if provided

---

## ModelRouter

The `ModelRouter` is not an agent, but it controls which model each agent call uses.

### Routing Defaults

```text
IntentClassifier  -> deterministic rules
PersonaAssembler  -> deterministic rules
RetrievalAgent    -> deterministic retrieval
ActorAgent        -> local model by default
CriticAgent       -> local model by default
MemoryCurator     -> local model by default
Cloud fallback    -> only through orchestrator policy
```

### Cloud Fallback Conditions

Cloud may be used when:

- local actor draft fails critique twice
- retrieval confidence is low for a complex lore question
- user explicitly requests high-quality/cloud mode
- scene transition requires stronger reasoning
- memory conflict resolution fails locally

Cloud must not be hardcoded to a specific agent role.

The architecture should behave the same whether the selected provider is local or cloud.

---

## Error Handling

### LLM Failure

If the local model fails:

```text
1. retry once if the error is transient
2. if cloud_mode allows fallback, use cloud
3. otherwise return a controlled error
```

Controlled error example:

```text
The roleplay engine could not generate a response because the local model failed. No world state was changed.
```

### Critic Failure

If the critic fails to return valid JSON:

```text
1. retry critic once with a stricter JSON instruction
2. if it still fails, mark critic unavailable
3. return actor draft only if no private context was present
4. otherwise return controlled error
```

### Memory Curator Failure

If memory extraction fails:

```text
1. return the player response anyway
2. persist the raw turn history
3. skip durable memory for that turn
4. log the failure
```

Memory failure must not block roleplay response unless the system requires memory for a correction command.

---

## State Mutation Rules

Only persistence/repository components may write state.

Agents may not write directly to:

- SQLite
- Qdrant
- session files
- world files
- memory stores

Agents return proposals.
The orchestrator validates proposals.
Repositories persist validated changes.

This is not optional.

---

## Observability

For the MVP, use structured logs.

Log each turn with:

```json
{
  "session_id": "...",
  "intent": "player_action",
  "actor_provider": "local",
  "actor_model": "qwen3:8b",
  "retrieved_chunks": 5,
  "critic_accepted": true,
  "memory_written": true
}
```

Do not log full secrets by default.

For debugging, add a local-only debug flag:

```env
DEBUG_PROMPTS=false
```

If enabled, prompts may be written to local debug files, but never commit those files.

---

## Tests Required for Agent Workflows

### Unit Tests

Required tests:

- intent classifier detects out-of-character messages
- intent classifier defaults to player action
- persona assembler excludes forbidden knowledge from public packet
- retrieval agent filters by visibility
- context builder respects budget
- critic result parser rejects invalid JSON
- memory curator parser rejects invalid visibility

### Integration Tests

Required tests:

- full turn succeeds with no RAG
- full turn succeeds with empty retrieval result
- full turn persists raw dialogue
- memory extraction failure does not block response
- critic rejection triggers one retry

### Secrecy Tests

Required tests:

- GM-only context is not included in actor prompt
- character-private memory is not shown to unrelated personas
- retrieved future spoiler is filtered before prompt build

These tests matter more than aesthetic prose tests.

---

## First MVP Workflow Acceptance Criteria

The workflow layer is acceptable when:

- a user message can pass through the full orchestrator
- the actor model returns a response
- the response can be optionally critiqued
- the turn is persisted
- durable memory can be proposed and saved
- all agent outputs are structured
- no agent mutates persistence directly
- local model is the default path
- cloud usage is controlled by router/orchestrator policy
- visibility filtering happens before prompt construction

---

## Coding Agent Instructions

When implementing this file, follow these rules:

1. Do not create autonomous background agents.
2. Do not create recursive planning loops.
3. Do not let agents call each other directly.
4. Implement agents as small classes or functions.
5. Use Pydantic schemas for inputs and outputs.
6. Keep deterministic logic deterministic.
7. Start with stubs where needed.
8. Make the first workflow runnable before adding advanced RAG.
9. Add tests for every workflow rule.
10. Keep local 8B model constraints in mind.

The goal is not to build a clever agent framework.
The goal is to build a reliable roleplaying turn pipeline.

---

## Next Document

The next document should be:

```text
docs/05_rag_memory_design.md
```

It should define:

- lore ingestion
- chunking
- embedding
- vector storage
- memory storage
- visibility filters
- retrieval ranking
- context budgeting
- memory summarization
- MVP acceptance criteria for RAG and memory
