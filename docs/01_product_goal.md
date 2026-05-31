# 01 — Product Goal: Personal RoleRAG MVP

## Purpose

This document describes the current MVP goal for `RoleRAG_POC` as it exists in the repository today.

The project is a personal-use roleplaying engine for one technical user. It combines structured scene and persona data, a bounded turn pipeline, local-first LLM execution, optional cloud fallback, SQLite persistence, and retrieval over ingested lore.

The product goal is not to build a general autonomous-agent framework. The goal is to keep a small, inspectable engine that can run locally and be extended safely.

## Current MVP Outcome

The repository currently proves this loop:

```text
user message
  -> load session, scene, persona
  -> retrieve player-visible lore context when available
  -> build bounded actor prompt
  -> route to local or cloud provider deterministically
  -> generate draft
  -> critique draft
  -> retry once locally if needed
  -> optionally repair with cloud when policy allows
  -> persist turn
  -> curate durable memory locally
```

This is the MVP. It is backend-first, personal-use, and intentionally narrow.

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
- The cloud model is optional.
- The engine should remain usable when cloud is disabled.

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
- Memory extraction stays local.

### Bounded orchestration

- The turn pipeline is finite and explicit.
- The runtime does not contain autonomous loops or free-form tool use.
- Cloud fallback is deterministic and policy-bound.

## What the MVP Includes

- Typer CLI for session creation, resume, route inspection, lore ingestion, and turns
- FastAPI API for session creation, turn execution, and session lookup
- deterministic local/cloud router
- actor generation, critic validation, and local memory curation
- Qdrant-backed runtime retrieval
- deterministic eval harness using fake providers

## What the MVP Does Not Include

- frontend
- authentication
- multi-user support
- streaming responses
- production deployment hardening
- autonomous agent planning loops
- automatic durable-memory indexing back into Qdrant

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

- no end-user interface
- no auth or account system
- no streaming transport
- no production-grade observability or deployment story
- no automatic indexing of curated SQLite memories into retrieval collections

Those gaps are acceptable for this MVP because the repository is proving architecture and safety boundaries, not product polish.
