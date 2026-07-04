# 01 — Product Goal: Personal RoleRAG MVP

## Purpose

This document describes the current MVP goal for `RoleRAG_POC` as it exists in the repository today.

The project is a personal-use roleplaying engine for one technical user. It combines structured scene and persona data, a bounded turn pipeline, a session-bound local-or-cloud LLM provider, SQLite persistence, retrieval over ingested lore, and a local Angular web UI.

The product goal is not to build a general autonomous-agent framework. The goal is to keep a small, inspectable engine that can run locally and be extended safely.

## Current MVP Outcome

The repository currently proves this loop:

```text
user message
  -> load session, scene, persona
  -> retrieve player-visible lore context when available
  -> build bounded actor prompt
  -> route to the session's bound provider (local or cloud, fixed at session creation)
  -> generate draft
  -> critique draft
  -> retry once with repair if needed
  -> apply output-side secret containment
  -> persist turn
  -> curate durable memory
```

This is the MVP. It remains backend-owned, personal-use, and intentionally narrow.

## Target User

The target user is one developer-operator who is comfortable with:

- Python virtual environments
- environment-variable based configuration
- Docker for Qdrant
- running a local OpenAI-compatible model server
- reading and editing JSON and Markdown

The project does not target hosted end users, teams, or production tenants.

## Required Product Properties

### Local-first roleplay

- The default actor path uses a local OpenAI-compatible endpoint.
- Cloud is a peer provider choice, bound once at session creation, not a rescue mechanism.
- The engine should remain usable when cloud is disabled (`CLOUD_MODE=off`).

### Structured state

- World, scene, and persona definitions are structured data loaded from JSON.
- Visibility boundaries matter from the beginning.
- The LLM does not own scene or session state.

### Retrieval with boundaries

- Retrieval is limited to curated chunks rather than whole-world prompt stuffing.
- Actor prompts only receive `player`-visible retrieved context.
- Hidden scene and persona information stay outside player-facing actor prompts.

### Durable continuity

- Sessions and turns survive restart through SQLite persistence.
- Important memories can be extracted and stored as durable memory episodes.
- Memory extraction runs on the session's bound provider.

### Bounded orchestration

- The turn pipeline is finite and explicit.
- The runtime does not contain autonomous loops or free-form tool use.
- Provider is session-bound: chosen once at session creation, immutable for the session's lifetime.

## What the MVP Includes

- Typer CLI for sessions, turns, routing, lore ingestion, diagnostics, content validation, and
  session management (list/resume/export/import/delete/inspect)
- FastAPI API for the content catalog, session CRUD, turns (JSON + buffered SSE), durable
  memories, and session canon
- Angular SPA (play, RAG inspector, analytics, eval) over the same-origin API
- deterministic, session-bound local/cloud router
- actor generation, critic validation, and memory curation
- Qdrant-backed runtime retrieval
- deterministic eval harness using fake providers

## What the MVP Does Not Include

- authentication
- multi-user support
- provider token streaming
- production deployment hardening
- autonomous agent planning loops

## Safety Boundaries

- The LLM is never authoritative for state.
- The actor does not retrieve or persist directly.
- Route handlers remain thin.
- GM/private knowledge must not be passed into player-facing actor prompts.
- Tests and evals do not call real providers or live Qdrant.

## Success Definition

The MVP is successful when a future contributor can:

1. clone the repository
2. configure a local model endpoint
3. start Qdrant
4. create a session
5. ingest demo lore
6. run turns through the CLI or API
7. run tests and evals without real providers
8. extend the backend without breaking the state, routing, or visibility boundaries

## Current Gaps That Are Explicitly Accepted

- no production end-user interface
- no auth or account system
- no provider token streaming
- no production-grade observability or deployment story
- no learned reranking model; retrieval ranking is deterministic rule-based boosts (collection,
  session/scene/persona metadata, importance, lexical overlap, optional recency)

Those gaps are acceptable for this MVP because the repository is proving architecture and safety boundaries, not product polish.
