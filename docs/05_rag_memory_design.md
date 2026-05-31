# 05 — Current RAG and Memory Design

## Purpose

This document describes the retrieval and memory design that is implemented today, including the important gaps that still exist after the MVP.

## Implemented RAG Behavior

The repository currently supports:

1. document ingestion for `.md` and `.txt`
2. deterministic chunking
3. local embedding abstraction
4. Qdrant-backed vector storage
5. retrieval filtering by visibility and scope
6. actor-context retrieval across three collections
7. SQLite persistence for sessions, turns, and durable memory episodes

## Storage Model

### SQLite

SQLite stores:

- sessions
- turns
- memory episodes

SQLite does not currently store RAG document metadata tables, scene snapshots, or configuration overrides.

### Qdrant

Qdrant stores retrievable chunk payloads and vectors for:

- `canon_lore`
- `session_memory`
- `persona_memory`

Qdrant is an index, not the authoritative state store.

## Ingestion

Implemented in [app/rag/ingestion.py](../app/rag/ingestion.py).

Current behavior:

- accepts `.md` and `.txt`
- chunks text
- embeds the chunks
- ensures the target collection exists
- replaces all chunks for the same source path in the chosen collection

Current limitation:

- world, scene, and persona data are loaded from JSON files but are not ingested automatically into Qdrant

## Retrieval

Implemented in [app/rag/retriever.py](../app/rag/retriever.py).

Actor retrieval combines:

- `session_memory` filtered by `session_id`
- `persona_memory` filtered by `persona_id`
- `canon_lore` filtered by `world_id`

The retrieval query is built from:

- user message
- visible scene title and location
- active persona name
- persona goals when present
- the latest two dialogue turns

Private persona fields and GM scene summaries are excluded from the retrieval query.

## Visibility Rules

The only implemented visibility values are:

- `player`
- `gm`
- `character_private`

Actor prompt construction only accepts `player`-visible chunks. This is enforced before retrieved content is inserted into the prompt.

The critic may inspect hidden fields for leak detection. The actor does not.

## Context Budgeting

Implemented in [app/orchestration/context_budget.py](../app/orchestration/context_budget.py).

Current settings:

- retrieved chunk count defaults to `5`
- max retrieved chunk chars defaults to `800`

The budget layer:

- drops non-player-visible chunks
- de-duplicates by chunk id
- truncates oversized chunk text

## Durable Memory

Implemented through:

- [app/agents/memory_curator.py](../app/agents/memory_curator.py)
- [app/memory/store.py](../app/memory/store.py)
- [app/persistence/repositories.py](../app/persistence/repositories.py)

Current behavior:

- the curator returns structured memory candidates
- only the application decides whether to persist them
- persisted memory episodes are stored in SQLite with visibility, importance, and tags

Important current limitation:

- persisted SQLite memories are not automatically embedded and written back into Qdrant collections
- retrieval logic supports `session_memory` and `persona_memory`, but automatic indexing into those collections is future work

## Failure Handling

- If Qdrant or embeddings are unavailable during a turn, retrieval is skipped and the turn continues.
- If ingestion cannot embed or write chunks, the CLI command fails immediately.
- If memory curation returns invalid structured output, memory persistence is skipped and the turn still completes.

## Tests Covering This Design

- `tests/unit/test_chunking.py`
- `tests/unit/test_ingestion.py`
- `tests/unit/test_retriever.py`
- `tests/unit/test_retrieval_context_builder.py`
- `tests/evals/test_retrieval_quality.py`
- `tests/evals/test_visibility_regressions.py`
- `tests/evals/test_memory_regressions.py`

## Boundaries to Preserve

- Qdrant remains a replaceable vector-store layer
- SQLite remains authoritative for sessions, turns, and durable memories
- visibility filtering stays in application code, not prompt wording
- actor prompts never receive `gm` or unrelated `character_private` chunks

## Deferred Work

Deferred but not implemented in the MVP:

- automatic indexing of durable memories into retrieval collections
- richer ranking or reranking
- ingestion of additional source formats
- production retrieval observability

Those items are tracked in [docs/10_next_steps_after_mvp.md](10_next_steps_after_mvp.md).
