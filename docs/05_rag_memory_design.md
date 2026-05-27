# 05 — RAG and Memory Design

## Purpose

This document defines the Retrieval-Augmented Generation and memory design for the RoleRAG proof of concept.

The MVP must support long-running personal roleplay with a local 8B model and an optional cloud model. The local model cannot receive the whole world, campaign history, all character memories, and all lore in every prompt. The engine must therefore retrieve and assemble only the context needed for the current turn.

This document is implementation guidance for a coding agent. It should be followed after the basic architecture and agent workflow documents are in place.

---

## Core Principle

RAG is not just document search.

For this project, RAG means controlled context selection across:

- world lore
- scene facts
- character memories
- session events
- player-visible history
- GM-only hidden state
- rules or mechanics notes

The retrieval system must protect visibility boundaries. It is better to retrieve too little than to accidentally leak private information into the player-facing prompt.

---

## MVP Goals

The MVP RAG and memory layer must provide:

1. Document ingestion for local Markdown/text files.
2. Chunking with metadata.
3. Local embeddings.
4. Vector search through Qdrant.
5. SQLite metadata storage.
6. Retrieval filtering by visibility.
7. Session memory persistence.
8. Memory extraction after each meaningful turn.
9. Context-budgeted prompt assembly.
10. Tests for retrieval, visibility, and memory writes.

---

## Non-Goals

Do not implement these in the MVP:

- autonomous long-horizon planning agents
- complex GraphRAG
- multi-user memory separation
- web search
- fine-tuning
- automatic canon rewriting
- real-time collaborative sessions
- image/audio memory
- cross-campaign global memory
- complex knowledge graph reasoning

These can come later. The MVP needs reliable retrieval and safe memory first.

---

## Storage Layers

Use two local storage layers.

```text
SQLite
  - sessions
  - turns
  - memory metadata
  - document metadata
  - scene state
  - persona metadata

Qdrant
  - embedded chunks
  - lore vectors
  - memory vectors
  - rule vectors
```

SQLite is the source of truth for metadata and structured state. Qdrant is the vector search index.

Do not treat Qdrant as the authoritative memory database. It is an index. If needed, the system should be able to rebuild Qdrant from SQLite and local documents.

---

## Recommended Collections

Create separate Qdrant collections for different retrieval domains:

```text
lore_chunks
session_memory_chunks
persona_memory_chunks
rules_chunks
```

This is clearer than one giant collection in the MVP.

A single collection with source metadata can work later, but separate collections make testing and filtering simpler.

---

## Visibility Model

Every retrievable item must have a visibility value.

```python
from enum import StrEnum

class Visibility(StrEnum):
    PLAYER = "player"
    GM = "gm"
    CHARACTER_PRIVATE = "character_private"
```

Meaning:

- `player`: safe to include in player-facing prompts.
- `gm`: only available to backend orchestration, critic, or narrator when hidden state is intentionally needed.
- `character_private`: known only to a specific character or persona.

Visibility must be checked in Python code before prompt construction.

Never rely on prompt instructions like “do not reveal this”. If a fact should not be used, do not put it into the actor prompt.

---

## Chunk Metadata

Every chunk stored in Qdrant must include enough metadata for filtering.

```python
from pydantic import BaseModel, Field

class ChunkMetadata(BaseModel):
    chunk_id: str
    source_id: str
    source_type: str
    world_id: str | None = None
    session_id: str | None = None
    scene_id: str | None = None
    persona_id: str | None = None
    visibility: Visibility
    tags: list[str] = Field(default_factory=list)
    created_at: str | None = None
```

Recommended `source_type` values:

```text
lore
session_memory
persona_memory
rules
scene
quest
```

---

## Document Ingestion

The MVP should ingest local Markdown and text files from:

```text
data/documents/
data/worlds/
data/personas/
```

The ingestion script should:

1. Read supported files.
2. Split into chunks.
3. Generate metadata.
4. Embed each chunk.
5. Store chunk text and metadata in SQLite.
6. Store vector and payload in Qdrant.

Initial supported file types:

```text
.md
.txt
.json
.yaml
.yml
```

PDF ingestion is not part of the MVP. Add it later only if needed.

---

## Chunking Rules

Keep chunks small enough for an 8B model.

Recommended defaults:

```text
chunk_size_chars = 800-1200
chunk_overlap_chars = 100-150
max_chunks_per_source = no hard limit, but log large files
```

Chunking should prefer semantic boundaries:

1. Markdown headings.
2. Paragraphs.
3. Bullets.
4. Character fallback.

Bad chunking will ruin retrieval. Do not split blindly every fixed number of characters if the file has clear headings.

---

## Embedding Strategy

Use a local embedding model first.

Good initial options:

```text
sentence-transformers/all-MiniLM-L6-v2
BAAI/bge-small-en-v1.5
BAAI/bge-base-en-v1.5
nomic-embed-text through Ollama
```

For the MVP, prefer a small fast local embedding model. The goal is not state-of-the-art retrieval yet. The goal is stable local retrieval with metadata filtering.

Define an abstraction:

```python
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
```

Do not call the embedding model directly from agents. Use the abstraction.

---

## Retrieval Query Construction

Do not retrieve using only the raw user message.

Build a retrieval query from:

- user message
- active scene title
- active location
- active persona name
- current goals
- recent turn summary

Example:

```python
def build_retrieval_query(
    user_message: str,
    scene_title: str,
    location: str,
    persona_name: str | None,
    recent_summary: str | None,
) -> str:
    parts = [
        f"Scene: {scene_title}",
        f"Location: {location}",
    ]

    if persona_name:
        parts.append(f"Active persona: {persona_name}")

    if recent_summary:
        parts.append(f"Recent events: {recent_summary}")

    parts.append(f"User message: {user_message}")
    return "\n".join(parts)
```

This improves retrieval because roleplay messages are often short or ambiguous.

---

## Retrieval Policy

The retriever should collect candidates from multiple sources, then rank and filter them.

MVP retrieval order:

```text
1. Session memory
2. Persona memory
3. Scene/world lore
4. Rules, only when relevant
```

Default limits:

```text
session_memory_top_k = 4
persona_memory_top_k = 3
lore_top_k = 5
rules_top_k = 3
final_context_top_k = 8
```

For the local 8B model, do not stuff 20 chunks into the prompt. More context usually makes small models worse.

---

## Retrieval Result Model

```python
from pydantic import BaseModel

class RetrievedContext(BaseModel):
    chunk_id: str
    source_id: str
    source_type: str
    text: str
    score: float
    visibility: Visibility
    tags: list[str]
```

The context builder should only accept `RetrievedContext` objects that have already passed visibility filtering.

---

## Visibility Filtering

Visibility filtering depends on output mode.

### Player-Facing Actor Prompt

Allowed:

- `player`
- selected `character_private` only if the active persona is allowed to know it and the prompt uses it as hidden behavioral context

Forbidden:

- unrelated `character_private`
- `gm` facts that would spoil hidden state

### Critic Prompt

Allowed:

- `player`
- `gm`
- relevant `character_private`

The critic may need hidden facts to detect leakage, but its output must never be shown directly to the player.

### Memory Curator Prompt

Allowed:

- full turn transcript
- current structured scene state
- relevant hidden state only if needed to assign visibility

The memory curator must produce structured memory writes with visibility values.

---

## Context Budgeting

The context builder owns the budget. Agents do not decide how much to include.

Initial budget:

```text
system + instructions: 700 tokens
scene/persona packet: 700 tokens
recent dialogue: 900 tokens
retrieved context: 1200 tokens
user message: 300 tokens
response budget: 700 tokens
```

For local 8B models, keep the prompt smaller than the model technically allows. A 32k context window does not mean the model can reason reliably over 32k tokens.

Recommended MVP total prompt target:

```text
2500-4000 tokens
```

---

## Memory Types

Use three memory types in the MVP.

### 1. Short-Term Dialogue Memory

Recent raw conversation turns.

Stored in SQLite as turns.

Used for immediate continuity.

Recommended window:

```text
last 6-10 turns
```

### 2. Episodic Memory

Short summaries of meaningful events.

Example:

```text
The player promised Marra Needlehand to keep her suspicion about Vane private.
```

Used for session continuity.

### 3. Persona Memory

Facts remembered by or about a specific character.

Example:

```text
Vane now distrusts the player because they questioned his authority in public.
```

Used to keep NPC behavior consistent.

---

## Memory Write Policy

Do not write memory for every turn.

Write memory only if the turn contains one of these:

- player decision
- promise
- threat
- discovery
- secret revealed
- relationship change
- quest update
- scene transition
- new durable fact
- correction from the user

Do not write memory for:

- generic description
- small talk with no consequences
- repeated facts already stored
- temporary mood that does not matter
- failed draft attempts

---

## Memory Model

```python
class MemoryRecord(BaseModel):
    id: str
    session_id: str
    world_id: str | None = None
    scene_id: str | None = None
    persona_id: str | None = None
    summary: str
    visibility: Visibility
    importance: int
    tags: list[str] = Field(default_factory=list)
    created_at: str
```

The `summary` is what gets embedded. Metadata is used for filtering.

Importance range:

```text
1 = minor but maybe useful
2 = small continuity detail
3 = normal durable event
4 = major relationship/quest/world update
5 = campaign-defining event
```

For MVP retrieval, prefer importance `>= 3` unless the scene/persona match is strong.

---

## Memory Curator Output Schema

The memory curator must return structured JSON.

```python
class MemoryCandidate(BaseModel):
    summary: str
    visibility: Visibility
    importance: int
    tags: list[str] = Field(default_factory=list)
    scene_id: str | None = None
    persona_id: str | None = None

class MemoryCuratorResult(BaseModel):
    write_memory: bool
    memories: list[MemoryCandidate] = Field(default_factory=list)
    reason: str
```

The engine validates this result before persisting anything.

---

## Memory Curator Prompt

```text
You extract durable roleplaying memory from a completed turn.

Only write memory if it will matter later.

Write memory for:
- player decisions
- promises
- discoveries
- secrets revealed
- relationship changes
- quest updates
- scene transitions
- durable world-state changes

Do not write memory for:
- generic narration
- temporary flavor
- facts already known
- failed drafts
- trivial dialogue

Return JSON only with this shape:
{
  "write_memory": true | false,
  "memories": [
    {
      "summary": "...",
      "visibility": "player" | "gm" | "character_private",
      "importance": 1,
      "tags": ["..."],
      "scene_id": "..." | null,
      "persona_id": "..." | null
    }
  ],
  "reason": "..."
}

Completed turn:
{turn_transcript}

Current scene:
{scene_summary}
```

---

## Deduplication

Before writing memory, check for near-duplicates.

MVP approach:

1. Embed the candidate memory.
2. Search existing memories for the same session/persona/scene.
3. If similarity is high, skip or update the old memory.

Simple rule:

```text
similarity >= 0.90 -> likely duplicate
similarity 0.80-0.90 -> keep only if the new memory adds important detail
similarity < 0.80 -> safe to add
```

Do not over-engineer this in the MVP. Log suspicious duplicates and improve later.

---

## Prompt Assembly with Retrieved Context

Retrieved context must be grouped by source type.

Example:

```text
RELEVANT MEMORY
- The player promised Marra to keep her suspicion about Vane private.
- Vane distrusts the player after being challenged publicly.

RELEVANT LORE
[1] Moon Harbor is controlled by Vane's crew and serves as the northern dock district.
[2] Marra Needlehand works from a lower-right building away from Vane's tower.

RULES NOTES
- None relevant.
```

Avoid dumping raw JSON into actor prompts. The actor should receive compact readable context.

---

## Database Tables

SQLite MVP tables:

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    world_id TEXT NOT NULL,
    active_scene_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    world_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT,
    source_type TEXT NOT NULL,
    text TEXT NOT NULL,
    visibility TEXT NOT NULL,
    world_id TEXT,
    session_id TEXT,
    scene_id TEXT,
    persona_id TEXT,
    tags_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    world_id TEXT,
    scene_id TEXT,
    persona_id TEXT,
    summary TEXT NOT NULL,
    visibility TEXT NOT NULL,
    importance INTEGER NOT NULL,
    tags_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

Keep the schema simple. Migrations can be added later with Alembic if needed.

---

## Qdrant Payload Example

```json
{
  "chunk_id": "chunk_123",
  "source_id": "doc_whaning_moon",
  "source_type": "lore",
  "world_id": "demo_world",
  "session_id": null,
  "scene_id": "moon_harbor",
  "persona_id": null,
  "visibility": "player",
  "tags": ["harbor", "pirates", "location"]
}
```

Store enough metadata to filter before final prompt construction.

---

## RAG Failure Handling

Retrieval can fail. The system should still produce a reasonable turn.

If retrieval returns no useful context:

1. Continue using scene state and recent dialogue.
2. Add a warning to logs.
3. Do not invent detailed lore.
4. Optionally ask the cloud model only if cloud mode allows it.

The actor prompt should include:

```text
No additional lore was retrieved. Stay conservative and do not invent major canon facts.
```

---

## Local vs Cloud Interaction

RAG should work the same for local and cloud models.

The same context packet should be usable by both providers. The cloud model may receive a slightly larger context budget, but it must not receive different hidden information unless the workflow explicitly allows it.

Recommended budgets:

```text
local actor context chunks: 5-8
cloud actor context chunks: 8-12
local critic context chunks: 5
cloud critic context chunks: 8
```

Cloud fallback should improve quality, not bypass architecture.

---

## Required Tests

### Chunking Tests

- Markdown headings are preserved where possible.
- Chunks do not exceed configured size by too much.
- Empty chunks are not stored.

### Ingestion Tests

- Supported files are ingested.
- Metadata is stored in SQLite.
- Vectors are stored in Qdrant or mocked vector store.
- Re-ingestion does not duplicate unchanged chunks.

### Retrieval Tests

- Relevant lore is retrieved for a scene query.
- Irrelevant lore is ranked lower.
- Retrieval respects `world_id`.
- Retrieval respects `session_id`.

### Visibility Tests

- Player-facing prompts never include GM-only chunks.
- NPC prompts never include another character's private memory.
- Critic can access hidden context when explicitly allowed.

### Memory Tests

- Important player decisions produce memory candidates.
- Trivial turns do not produce memory writes.
- Memory importance is validated.
- Duplicate memory is skipped or flagged.

---

## MVP Acceptance Criteria

This part is complete when:

- Local documents can be ingested.
- Chunks are stored with metadata.
- Embeddings are generated locally.
- Qdrant retrieval works.
- SQLite stores metadata and memories.
- Retrieved chunks are filtered by visibility.
- Actor prompts include only allowed retrieved context.
- Memory curator writes durable memories after meaningful turns.
- Session continuation uses previous memories.
- Tests cover retrieval and visibility boundaries.

---

## Coding-Agent Instructions

When implementing this document:

1. Keep the implementation small and explicit.
2. Do not introduce LangChain or LangGraph in the MVP.
3. Use small local embedding models first.
4. Treat SQLite as authoritative metadata storage.
5. Treat Qdrant as a rebuildable vector index.
6. Do not mix visibility filtering into prompt text only.
7. Do not retrieve unlimited chunks.
8. Do not let agents write directly to storage.
9. The orchestrator validates and persists memory writes.
10. Add tests before adding advanced retrieval behavior.

If a feature would require many assumptions, leave a clear TODO and implement the simpler version first.
