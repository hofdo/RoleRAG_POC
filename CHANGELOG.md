# Changelog

Notable changes per release. The dated acceptance/report docs under `docs/` remain the deep
records; this file is the quick delta between versions.

## Unreleased

### Added

- **Deterministic frontend↔backend contract test** (#60): `app/diagnostics/contract_app.py`
  serves the real FastAPI app with fake-provider overrides; a Playwright spec
  (`tests/e2e/spa-contract.spec.mjs`) drives the built SPA over real HTTP in CI (9097877).
- **Vector-store parity harness** (#50): one fixture set through both `InMemoryVectorStore`
  and embedded-Qdrant `.search()` paths across every filter dimension; Qdrant tags filter
  fixed to AND semantics.
- **Behavioral WAL concurrency tests** (#64), **coverage measurement** (#61, report-only,
  91% baseline), **frontend lint in CI** (#62), **SPA build in CI** (#52), and a
  **non-blocking dependency-audit job** (#63).

### Changed

- **Angular 19 → 21** via `ng update` (f0e4b8a, ae7f96f), clearing the `@angular/core`
  hydration advisory GHSA-rgjc-h3x7-9mwg surfaced by the #63 audit.
- **CLI/API config parity** (#48): the CLI now builds its orchestrator config through
  `composition.build_orchestrator_config`; a parity test pins both roots to the same config.
- **docker-compose**: Qdrant `/readyz` healthcheck + `service_healthy` readiness gate and
  `restart: unless-stopped` on both services (#58); `.dockerignore` excludes backups and
  WAL sidecars (#53); upper bounds on high-risk Python deps (#59).

### Fixed

- **RAG C1** — standing-facts double-spend no longer evicts distinct retrieved chunks
  (64db602); **RAG N1** — extractor framing stripped before write-dedup coverage math
  (0c11c29). Both validated on a live 26B + Qdrant run with zero recall regression.
- Removed the dead cloud-repair path (#56) and deduplicated controlled-failure /
  diagnostics assembly in the orchestrator (#54).

## 1.2.0 — 2026-07-04

### Changed (breaking)

- **Session-bound provider.** A session's provider is chosen once at `POST /sessions` and is
  immutable for the session's life; every task (actor, critic, repair, memory) runs on that bound
  provider. All automatic cloud paths are gone — no cross-provider fallback, no local-then-cloud
  repair ladder, no escalation, and no per-turn `request_cloud`/confirmation flow. `CLOUD_MODE`
  now gates cloud-session **creation** only (`off` = 400 `cloud_unavailable`, `ask` = one
  interactive confirm at creation enforced by the CLI/SPA clients, `auto` = silent) (a514f9c,
  87064f8). This supersedes 1.1.0's Play description of a `CLOUD_MODE=ask` confirmation inside the
  per-turn loop — that confirmation now happens once, at session creation.

### Added

- **Reroll**: `DELETE /sessions/{id}/turns/last` drops the last turn with its indexed memories,
  and the SPA exposes it as a one-click reroll (c69d741, 2c03f6b).
- **Scene switching and per-turn persona override**: `POST /sessions/{id}/scene` re-anchors the
  active scene mid-session, and a turn may name a different `active_persona_id` for a single
  exchange (3e93e4b).
- **Cross-session persona memory**: persona memories dual-write to a shared `persona_memory`
  store so a persona carries learned context across sessions (8f517e2).
- **Durable persistence**: SQLite WAL mode with a busy timeout, a `rolerag backup` command, and
  automatic snapshots before destructive operations (502f80c).
- **SSE stage frames**: `event: stage` frames report live pipeline progress during a streaming
  turn, and `SSE_TEXT_CHUNK_CHARS` (default `0` = single text frame) tunes text-frame
  chunking (dc3803a).
- **Failed-turn persistence**: controlled-failure turns are now persisted with an `outcome` flag
  (`"success"` / `"controlled_failure"`), so a failed turn is a recorded, inspectable turn rather
  than a dropped one (41db80d).
- **SPA resume picker**: the setup screen lists prior sessions and restores the full transcript;
  an in-progress draft survives a failed turn (5b24926).

### Docs

- Documentation overhaul: new content-authoring, security/backup, verification/eval-tooling, and
  player-guide references (docs 17–20) plus a project glossary, and `> Reviewed:` freshness
  headers across the living docs.

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
