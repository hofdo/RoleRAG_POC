# 05 — Current RAG and Memory Design

> Reviewed: 2026-07-04 @ 571acc8

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
- canon facts (author-pinned session canon; see "Session Canon" below)

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

Authoring note:

- standalone scenario packs can include `documents/manifest.json` metadata
- manifest metadata is used by validation and by explicit scenario-lore ingestion
- lore documents are ingested explicitly through `python -m app.cli ingest` or
  `python -m app.cli ingest-scenario-lore --content-root <pack>`
- scenario startup does not automatically mutate Qdrant

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

Retrieval runs as a dual-query pass: the context blob above anchors scene/lore relevance, and a
second pass with the bare player message keeps indirect callbacks ("what rule did we agree?")
retrievable when the framed blob would otherwise bury them. The union is deduplicated by chunk id
before reranking ([app/rag/retriever.py](../app/rag/retriever.py)).

Current ranking behavior:

- retrieve a bounded candidate pool from each collection
- preserve the original vector score on every returned chunk
- apply deterministic post-retrieval boosts for collection, matching session/scene/persona
  metadata, memory importance, lexical overlap with the player message, and (optionally) recency
- every weight and boost is tunable via `RAG_*` settings (e.g. `RAG_SESSION_MEMORY_WEIGHT`,
  `RAG_LEXICAL_MATCH_STEP_BOOST`, `RAG_RECENCY_WEIGHT`); defaults mirror the constants in
  [app/rag/ranking.py](../app/rag/ranking.py), and recency is off (`0.0`) by default
- keep ranking policy in application code rather than inside Qdrant
- expose metadata-only diagnostics for selected chunks through the CLI

Retrieval guarantees:

- actor retrieval requests only `player`-visible chunks
- canon lore is scoped to the stored session `world_id`
- session memory is scoped to the stored `session_id`
- persona memory is scoped to the active `persona_id`
- deterministic reranking is applied after bounded per-collection candidate retrieval
- diagnostics serialize selected chunk metadata and scores, never chunk text

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

Recent dialogue is bounded separately, in two ways: by turn count (`RECENT_DIALOGUE_TURNS`,
default `8` — the orchestrator loads only that many prior persisted turns into each actor
prompt) and by per-message clipping (`RECENT_DIALOGUE_MAX_MESSAGE_CHARS`, default `900` — prior
messages over the cap are clipped with an explicit omission marker, and any clipping is surfaced
as a `TurnResult` warning). The current turn's incoming message is not clipped. Once an older
exchange leaves the recent window, continuity depends on relevant durable memory being retrieved.

## Durable Memory

Implemented through:

- [app/agents/memory_curator.py](../app/agents/memory_curator.py)
- [app/memory/store.py](../app/memory/store.py)
- [app/persistence/repositories.py](../app/persistence/repositories.py)

Current behavior:

- the curator returns structured memory candidates
- a deterministic extractor independently captures explicit player promises and agreements as a
  fallback when the LLM curator fails or misses them
  ([app/memory/deterministic_extractor.py](../app/memory/deterministic_extractor.py))
- only the application decides whether to persist them
- persisted memory episodes are stored in SQLite with visibility, importance, and tags
- memory extraction reads the turn text **after** output-side secret containment has redacted it
  (containment runs post-validation, pre-persistence in the orchestrator). This is deliberate:
  redacted secrets must not re-enter circulation through memory. The accepted trade-off is that a
  redacted phrase is also absent from extracted memories, so the engine will not "remember"
  content it refused to say

Current indexing behavior:

- persisted SQLite memories are embedded and upserted into `session_memory`
- indexed payloads preserve memory id, session id, scene id, optional actor id, summary, visibility,
  importance, and tags
- `python -m app.cli reindex-memories --session-id <session-id>` backfills or repairs the derived index

SQLite is authoritative for durable memory episodes. The vector store is derived and replaceable:
reindexing a session reads the persisted SQLite episodes and repairs retrieval state after an index
loss or outage.

Write-time dedup, consolidation, and index bounds:

- before persistence, a candidate is dropped if an existing session memory already covers it
  lexically; an optional semantic pass (cosine, `RAG_WRITE_DEDUP_COSINE_THRESHOLD`, off at `1.0`)
  additionally drops paraphrased near-duplicates
- consolidation ("sleep cycle") can roll up old low-importance memories into a single summary once
  a backlog threshold is reached (`MEMORY_CONSOLIDATION_THRESHOLD`, off at `0`;
  `MEMORY_CONSOLIDATION_MAX_IMPORTANCE` bounds eligibility), implemented in
  [app/memory/consolidation.py](../app/memory/consolidation.py) and
  [app/orchestration/stages/memory.py](../app/orchestration/stages/memory.py)
- a session-memory index cap (`SESSION_MEMORY_MAX_EPISODES`, off at `0`) can evict the lowest
  importance-then-recency episodes from the retrievable index; SQLite keeps every memory
- an index importance floor (`RAG_INDEX_IMPORTANCE_FLOOR`, off at `1`) keeps memories below the
  floor in SQLite without indexing them for retrieval
- all of these ship OFF by default: live acceptance found the index cap and auto-gating regressed
  50-turn recall, so the always-on, index-everything defaults are intentional

## Session Canon (Standing Facts)

Implemented in [app/orchestration/canon_builder.py](../app/orchestration/canon_builder.py),
backed by the SQLite `canon_facts` table.

Whenever canon exists, the actor prompt carries a pinned "Standing facts" block that is built
independently of vector retrieval:

- author-pinned canon facts (managed through the `/sessions/{id}/canon` API endpoints; see
  [docs/12_api_contract.md](12_api_contract.md)) come first, verbatim
- derived facts follow: a session memory qualifies only if it is `player`-visible, its importance
  is at or above `CANON_IMPORTANCE_FLOOR` (default `4`), **and** it carries at least one durable
  `CANON_TAGS` tag (promise, rule, agreement, oath, ...)
- eligible memories are ordered by importance then recency, the combined list is deduplicated by
  text, and the block is bounded by `CANON_MAX_ITEMS` (default `8`) and `CANON_MAX_CHARS`
  (default `900`)

Because the block is deterministic and pinned, the most load-bearing durable facts stay in the
prompt even when vector retrieval would miss them.

## Failure Handling

- If Qdrant or embeddings are unavailable during a turn, retrieval is skipped and the turn continues.
- If ingestion cannot embed or write chunks, the CLI command fails immediately.
- If memory curation returns invalid structured output, LLM-curated persistence is skipped (the
  deterministic fallback extractor may still persist explicit durable events) and the turn still
  completes.
- If memory indexing fails after SQLite persistence, a warning is recorded and the turn still completes.

## Retrieval Checks

Run the deterministic retrieval and policy eval tests:

```bash
pytest tests/evals -q
python -m app.evals.regression_runner
```

These checks use fake providers, keyword embeddings, SQLite, and `InMemoryVectorStore`. The
16-turn `memory_continuity` regression also proves that recent dialogue stays bounded, an old
player-visible promise is recalled after it leaves actor history, hidden memories stay out of actor
prompts, and a fresh scoped reindex recovers SQLite-backed memory. These checks do not prove live
LLM behavior, semantic embedding quality, Qdrant quality, or generated prose quality.

## Tests Covering This Design

- `tests/unit/test_chunking.py`
- `tests/unit/test_ingestion.py`
- `tests/unit/test_retriever.py`
- `tests/unit/test_retrieval_ranking.py`
- `tests/unit/test_retrieval_diagnostics.py`
- `tests/unit/test_retrieval_context_builder.py`
- `tests/evals/test_memory_recall_regressions.py`
- `tests/evals/test_retrieval_quality.py`
- `tests/evals/test_visibility_regressions.py`
- `tests/evals/test_memory_regressions.py`
- `tests/evals/test_memory_continuity.py`

## Boundaries to Preserve

- Qdrant remains a replaceable vector-store layer
- SQLite remains authoritative for sessions, turns, and durable memories
- visibility filtering stays in application code, not prompt wording
- actor prompts never receive `gm` or unrelated `character_private` chunks

## Current Limitations

- Qdrant remains a derived runtime index rather than an authoritative content store
- scenario startup does not auto-ingest lore into Qdrant
- production vector quality is not measured by the deterministic keyword eval fixtures

## Deferred Work

Deferred but not implemented:

- ingestion of additional source formats beyond `.md`/`.txt`
- broader production retrieval observability beyond CLI inspection

Recency-aware ranking and memory consolidation are implemented but ship OFF by default
(`RAG_RECENCY_WEIGHT=0.0`, `MEMORY_CONSOLIDATION_THRESHOLD=0`); see the ranking and durable-memory
sections above. Remaining items are tracked in
[docs/10_next_steps_after_mvp.md](10_next_steps_after_mvp.md).
