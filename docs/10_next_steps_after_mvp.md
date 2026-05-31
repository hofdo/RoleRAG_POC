# 10 — Next Steps After MVP

## Purpose

This document lists the safest next development steps after the current MVP. It is intentionally limited to follow-on work that fits the existing architecture.

## Highest-Value Next Steps

### 1. Extend CI as needed

The repository CI already runs:

- `ruff check .`
- `mypy .`
- `pytest`
- `python -m app.evals.regression_runner`

Extend the matrix only when another supported Python version or platform needs coverage.

### 2. Improve integration coverage

The repository already has solid unit and eval coverage. The next gap is broader integration coverage around:

- CLI plus retrieval wiring
- API plus persistence plus retrieval
- cloud fallback edge cases

### 3. Index durable memories into retrieval collections

The current MVP persists durable memory episodes in SQLite but does not automatically embed and store them into Qdrant. Closing that gap would improve continuity without changing the high-level architecture.

### 4. Improve retrieval ranking

Safe candidates:

- better query construction heuristics
- stronger ranking signals from metadata and recency
- optional reranking as a bounded post-retrieval step

## Product-Level Follow-Ons

### Frontend

Add a frontend only after preserving the current API and orchestration boundaries. The frontend should consume existing backend behavior rather than pulling logic out of the backend.

### Streaming responses

Add streaming only after the current non-streaming API remains stable. Streaming should be transport behavior, not a new orchestration mode.

### Richer authoring

Safe scope:

- better world/scenario authoring workflows
- additional demo content
- more structured authoring helpers

## Conditional Future Work

### Authentication

Only relevant if the project becomes multi-user. Do not add auth while the repository remains explicitly personal-use.

### Production secrets handling

Relevant if deployment becomes serious. Current `.env`-based local configuration is acceptable for the MVP.

### Deployment hardening

Add only when there is an actual deployment target. That includes:

- production config separation
- service monitoring
- backup strategy
- tighter operational error handling

## Work To Avoid

- introducing LangChain or LangGraph
- autonomous planning loops
- database churn without a concrete need
- frontend-first refactors that move backend logic into the client
- provider-specific gameplay forks

## Decision Rule

Prefer changes that:

- preserve deterministic orchestration
- keep state ownership in application code
- strengthen visibility boundaries
- improve verification and maintenance

Avoid changes that increase abstraction, hidden behavior, or provider coupling without a measured benefit.
