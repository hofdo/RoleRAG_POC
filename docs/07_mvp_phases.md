# 07 — MVP Phases Status

> **Historical record** — the phase-by-phase build log of the MVP. Point-in-time; not kept in
> sync with the code. For current state see [docs/README.md](README.md).

## Purpose

This document records the phase-by-phase status of the repository after Phase 13.

## Status Summary

### Implemented phases

- Phase 0: repository bootstrap
- Phase 1: configuration and provider abstraction
- Phase 2: domain models
- Phase 3: CLI-first local roleplay loop
- Phase 4: structured JSON data loading
- Phase 5: SQLite session and turn persistence
- Phase 6: durable memory curation and persistence
- Phase 7: basic RAG infrastructure
- Phase 8: retrieval-aware actor context construction
- Phase 9: critic agent and bounded repair loop
- Phase 10: explicit cloud fallback behavior
- Phase 11: deterministic evaluation harness
- Phase 12: FastAPI MVP
- Phase 13: documentation, consistency review, and agent handoff

## Notes on Implemented Scope

### Phase 3

The implemented CLI is session-based. The repository does not contain the older `chat` command examples from earlier planning drafts.

### Phase 4

Structured data loading is JSON-based for worlds, scenes, and personas.

### Phase 5

SQLite currently stores sessions, turns, and memory episodes. It does not currently store RAG document metadata tables, scene snapshots, or configuration overrides.

### Phase 7

Qdrant collections in the current runtime are:

- `canon_lore`
- `session_memory`
- `persona_memory`

Ingestion currently supports `.md` and `.txt`.

### Phase 10

`CLOUD_MODE=ask` marks routes as requiring confirmation, but the runtime currently does not expose an interactive confirmation loop or dedicated confirmation endpoint.

### Phase 11

The eval harness is deterministic regression coverage for retrieval, visibility, memory, role consistency, and cloud routing. It is not prose-quality scoring.

## Phase 13 Deliverables

Phase 13 completes:

- current-state README rewrite
- setup and command verification documentation
- agent handoff guide
- architecture map
- post-MVP next-steps guide
- consistency cleanup across the primary docs
- `.env.example` synchronization with the actual settings model

## Start-Here Documents After Phase 13

- [README.md](../README.md)
- [docs/08_agent_handoff.md](08_agent_handoff.md)
- [docs/09_current_architecture_map.md](09_current_architecture_map.md)
- [docs/10_next_steps_after_mvp.md](10_next_steps_after_mvp.md)

## Explicit Non-Goals Still In Force

- no new API endpoints for this phase
- no frontend
- no LangChain or LangGraph
- no autonomous loops
- no database schema changes
- no RAG behavior changes
- no real provider or Qdrant calls in tests

## Verification Standard

Phase 13 is considered complete when:

- docs reflect the implemented system accurately
- `.env.example` matches `Settings`
- command examples match the actual CLI and API
- `ruff check .`, `mypy .`, and `pytest` pass

That verification is the correct stopping point for this phase.
