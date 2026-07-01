# Changelog

Notable changes per release. The dated acceptance/report docs under `docs/` remain the deep
records; this file is the quick delta between versions.

## 1.1.0 — 2026-07-01

### Added

- **Angular 19 SPA** served at `/app` (root `/` redirects to it) with four pages: Play (catalog
  session setup, buffered-SSE turn loop, `CLOUD_MODE=ask` confirmation, memory + canon panels),
  RAG Inspector (per-turn retrieval drill-down), Analytics (stage-timing statistics), and Eval
  (eval-run trends with per-run drill-down). Signal store, fetch+SSE client, "Grimoire Console"
  design system, Karma unit tests, and a Playwright e2e spec.
- **API**: `GET /sessions/{id}/turn-details` (bulk turn diagnostics, so the SPA doesn't fan out
  N requests) and `GET /diagnostics/eval-runs` + `GET /diagnostics/eval-runs/{run_id}`
  (read-only eval-run summaries and drill-down).
- **API**: structured `errors` (category/stage/message/suggestion) alongside free-form
  `warnings` on turn responses (#19).
- **CLI**: auto-ingest scenario lore on `start-session` (#16); colored errors/warnings/success
  (#27).
- **RAG**: opt-in importance-aware recency boost, `RAG_RECENCY_WEIGHT`, default off (#6).
- **Live harness**: `26b-mtp` local-model profile (speculative MTP draft, ~10–14% lossless
  speedup) and a 100-turn extended checkpoint scenario (was 50).
- **Dev**: `make dev` / `dev-up.sh` builds the SPA and restarts a stale API so `/app` mounts;
  the Docker image builds the SPA in a frontend stage.

### Changed

- The web UI is the SPA only. Side-panel (memory/canon) failures surface as visible errors
  instead of silently showing stale state.
- CI runs the SPA's Karma tests headless (replacing the removed vanilla-JS module tests).
- Version is now sourced from `pyproject.toml`/`app.__version__` only.

### Removed

- The framework-free `/play` UI (`app/web/`, its vanilla-JS client, and tests). The SPA
  replaced it; keeping both doubled maintenance surface with no canonical owner.
- `setup.py` / `setup.cfg` (stale duplicate packaging metadata pinned at 0.1.0);
  `pyproject.toml` is the single packaging source.

### Fixed

- Fail closed when the critic errors instead of serving unvalidated text (#17).
- Gating-mode strings validated at stage construction (#24).
- Hidden facts redacted from structured-failure `raw_text` logging.

## 1.0.0 — 2026-06-12

First accepted baseline: bounded turn pipeline (retrieval → routing → generation → validation →
critique → repair → persistence → memory), deterministic local/cloud routing with
`CLOUD_MODE=off|ask|auto`, SQLite-authoritative persistence with Qdrant-derived retrieval,
durable memory with dedup/consolidation, secret containment, CLI + FastAPI surfaces, and the
deterministic eval harness. See [docs/15_v1_acceptance_report.md](docs/15_v1_acceptance_report.md).
