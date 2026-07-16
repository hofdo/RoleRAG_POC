# RoleRAG POC — Side Projects

> Reviewed: 2026-07-16 @ 7f112ba

Ideas that build *on top of* the engine without changing its core. Source: the 10-agent deep
analysis. Four original entries shipped in 1.1.0 and are marked **BUILT** below; Transcript
Exporter shipped afterward (Unreleased) as a fifth. The remaining ideas are unbuilt. Engine
backlog lives in [BACKLOG.md](BACKLOG.md); this file is the durable record of the "what could we
build next" list.

## Scope guardrail

RoleRAG is a **personal-use, local-first POC**. Anything needing real multi-user, auth, tenancy, or
always-on hosting is out of scope unless that decision is deliberately revisited. That rules
**party mode** out by default and keeps the others single-user.

## What the engine already exposes (what these reuse)

- **HTTP API** (`app/api/routes.py`): `POST /sessions`, `GET /sessions`, `GET /sessions/{id}`,
  `POST /sessions/{id}/turns`, `POST /sessions/{id}/turns/stream` (SSE), `GET /sessions/{id}/turns/{i}`
  (turn detail + retrieval diagnostics + `errors`, #8/#19), `GET /sessions/{id}/turn-details` (bulk
  transcript + per-turn diagnostics), `DELETE /sessions/{id}/turns/last` (reroll),
  `POST /sessions/{id}/scene` (scene switch), `GET /sessions/{id}/memories`, canon endpoints,
  `GET /diagnostics/eval-runs` (+ `/{run_id}`), `GET /runtime/status`, `GET /content/catalog`.
- **Web UI** (`/app`): Angular SPA — play loop, RAG inspector, analytics, and eval pages.
- **CLI** (`app/cli.py`): `start-session`, `turn`, `resume`, `turn-history`, `inspect-memories`,
  `export-session` / `import-session` (JSON), `export-transcript` (markdown/HTML),
  `ingest-scenario-lore`, `retrieve-debug`, `smoke-run`,
  `list-sessions` / `delete-session`, `backup`, `reindex-memories`, `reset-index`, `embedding-ab`.
- **Data**: SQLite (authoritative sessions/turns/memories/canon, turns carry `diagnostics_json`);
  Qdrant (derived `canon_lore` / `session_memory` / `persona_memory` collections).

---

## Tier A — weekend, low risk, high payoff

### ~~★ Transcript Exporter~~ — **BUILT (Unreleased)**
Shipped as the `export-transcript` CLI command: renders a session's turns into a shareable
markdown or HTML transcript (player messages, served replies, turn/persona attribution,
controlled-failure turns marked distinctly) via a pure read of SQLite through the existing
repositories — zero new backend, zero schema change. Rendering lives in
`app/diagnostics/transcript_export.py` (pure functions, unit-tested independent of the CLI).
PDF was left out on purpose (the spec's "optionally PDF" would need a new dependency).
- **Command**: `rolerag export-transcript --session-id <id> --output transcript.md [--format html]`

### ~~RAG / Memory Inspector~~ — **BUILT (1.1.0)**
Shipped as the SPA's RAG Inspector page: per-session turn timeline with retrieval drill-down,
fed by the bulk `GET /sessions/{id}/turn-details` endpoint.

---

## Tier B — real feature, multi-day

### Discord bot
Drive turns from a Discord channel; one channel ↔ one session keeps it single-user-ish.
- **Builds on**: the turn API (`/turns` or `/turns/stream`). Thin wrapper + a Discord lib.
- **Scope note**: resist multi-player-per-channel (that's party mode / multi-user).

### Voice I/O
Speech-to-text in, text-to-speech out, wrapped around the turn loop.
- **Builds on**: the turn API unchanged; all work is client-side glue + a speech library.

### ~~Analytics dashboard~~ — **BUILT (1.1.0)**
Shipped as the SPA's Analytics page (per-stage latency and turn statistics). Recall trends and
memory-growth views remain open extensions.

---

## Tier C — large / deferred

- ~~**Polished React/Angular SPA**~~ — **BUILT (1.1.0, since upgraded to Angular 21)**: SPA at
  `/app`; the vanilla-JS `/play` UI was removed.
- ~~**Eval dashboard**~~ — **BUILT (1.1.0)**: SPA Eval page over `GET /diagnostics/eval-runs`.
- **Authoring studio** — UI to create/validate worlds, personas, scenes, lore. Wraps
  `validate-content` + `create-scenario-template` + `ingest-scenario-lore`.
- **Branching / replay** — fork a session at a turn into alternate timelines. Needs schema work
  (turn lineage) — the biggest data change here.
- **Mobile client** — phone-friendly play UI over the same API.

## Out of scope (for now)

- **Party mode** (multiple players, one shared session) — needs real multi-user state isolation +
  auth. Revisit only if the project's personal-use scope is deliberately changed.

---

**Recommendation:** Tier A is now fully built (Transcript Exporter + RAG/Memory Inspector). Of the
remaining open ideas, start with **Voice I/O** next — the turn API stays unchanged, all new work
is client-side glue plus a speech library, and (unlike the Discord bot) there's no external
bot/auth process to stand up and keep running.
