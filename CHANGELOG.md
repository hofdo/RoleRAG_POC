# Changelog

Notable changes per release. The dated acceptance/report docs under `docs/` remain the deep
records; this file is the quick delta between versions.

## Unreleased

### Added

- **docs/26 Stage 0 instrumentation** (#75, closes #74's harness half): new
  `memory_write_lifecycle` regression category drives the real `TurnMemoryStage` (real
  deterministic extractor + real write-dedup, faked curator LLM call only) and replays the
  #72 compass/dawn adversarial pair at the integration level; the live checkpoint now fails
  fast, naming the cause, when a probe's definition turn ends in `controlled_failure`
  instead of surfacing a misleading recall miss ten turns later at the callback; new
  read-only `app.diagnostics.replay_selection` offline replay script reports canon/
  standing-facts eligibility for a preserved live-run artifact with no model or Qdrant
  (run against the D3 artifact: 47 player-visible memories, 3 tag-eligible, 0 eligible at
  the importance floor of 4, confirming docs/26 §3.3). No runtime engine behavior changed.
- **docs/26 Stage 1 provenance substrate** (#76): `memory_episodes.source_turn_id` (nullable,
  fifth `_ensure_column` migration) threads an optional `turn_id` through
  `TurnMemoryStage.run` → `_run_extraction` → `_persist_and_index` from both orchestrator
  call sites (inline turn persistence and the deferred-memory job), stamping every
  persisted memory candidate with the turn that produced it. Consolidation carries
  provenance forward (earliest non-null `source_turn_id` of the folded originals, `None`
  if all legacy — never a sentinel) plus a `source_ids:` audit tag reusing `tags_json`. The
  live checkpoint's probe attribution is now provenance-OR-phrasing instead of
  phrasing-only, closing the paraphrase-fragility measurement gap for facts that survive
  extraction under a reworded summary — never provenance-only, and the multi-memory
  definition-turn over-attribution this implies is documented and tested, not silently
  tightened away. Attribution-only: not a fact-identity or dedup key. No runtime engine,
  ranking, or retrieval behavior changed.
- **docs/26 Stage 2 best-match tag/importance fold** (#77): at the curator-coverage-drop
  site, a deterministic fallback candidate (promise/entrusted/agreement/deadline,
  importance=4) covered by one of this turn's own curated summaries is no longer silently
  discarded — it now folds its tag(s) and importance onto the **best-matching** (argmax
  coverage, not first-match; deterministic lowest-index tie-break) curated summary, with an
  audited `"deterministic candidate folded (best-match, coverage=X.XX): ..."` warning. A
  candidate covered by no curated summary still goes to `extras` exactly as before
  (byte-identical). New pure helpers `best_covering_summary`/`ordered_union`
  (`app/orchestration/stages/memory_dedup.py`); `is_covered_by_summaries`, the 0.75/0.5
  thresholds, reversal markers, and framing-strip are unmodified. This is a real, narrow
  runtime behavior change (a previously-silent tag/importance loss is now recovered and
  audited); write-dedup, ranking, and retrieval are untouched.
- **docs/26 Stage 4 Lane B lexical slice quotas** (#79, opt-in, byte-identical default): new
  pure `app/rag/lexical.py` scorer ranks the session's own memory pool by summed
  session-pool IDF of the terms it shares with the player's message (reuses `content_terms`;
  IDF self-calibrates so frame vocabulary is free and rare terms expensive; no new
  tokenizer). `app/rag/ranking.py` gains `SliceQuotas` + `apply_lexical_slice_quotas`, which
  reorders/injects the top-quota lexical hits to the FRONT of the selection list so the
  UNCHANGED `select_retrieved_chunks_for_prompt` walk lands them in the final prompt window
  (a member already in the window is only reordered — no dense eviction; a deeper or
  non-dense-fetched hit is pulled in / injected from the memory). All four judge fixes ship:
  `CONSOLIDATED_TAG` pool exclusion (fix 1); `RAG_SLICE_MIN_SCORE` floor mechanism present
  but shipped UNSET / no-floor pending Stage 5 measurement (fix 2); a dedicated `slice_score`
  diagnostic field, not folded into `applied_boosts`, so `adjusted == original + sum(boosts)`
  holds for every chunk including injected `original=0.0` members (fix 3); lexical hits
  computed from `context.session_memories` BEFORE the retriever call, so a dense/Qdrant
  failure degrades to a lexical-only prompt instead of empty (fix 4). `retrieval_confidence`
  is slice-aware — a chunk holding a guaranteed slot floors at `SLICE_CONFIDENCE_EQUIVALENT`
  (0.5, clears the 0.45 `low_retrieval_confidence` default) so a slice-only rescue does not
  read as low-confidence — and is byte-identical with quotas off. New `RAG_SLICE_LEXICAL_QUOTA`
  (default 0; live preset 2) / `RAG_SLICE_MIN_SCORE` (default unset) Settings, mirrored in
  `.env.example` and wired through `build_orchestrator_config` (both composition roots).
  Retrieval diagnostics + the API inspection payload gain labeled slice info (matched terms,
  score, guaranteed flag). Offline D3 replay (`replay_selection --lexical-query`) ranks the
  blue-seal rule memory `354b8d98` #1/15 for the #73 callback query — guaranteed into the
  prompt at quota 2. `context_budget.py` and the additive-boost rerank math are byte-identical
  (proven by a quotas=0 golden test); no new table, store, LLM call site, or vector-store
  feature, so no `InMemoryVectorStore` parity work (Lane B runs outside the store).
- **docs/26 Stage 3 Lane A tag-eligible canon pinning** (#78, opt-in, byte-identical default):
  `CANON_TAG_PINNING=false` widens `build_standing_facts` eligibility (`app/orchestration/
  canon_builder.py`) to `visibility == PLAYER AND CANON_TAGS-intersects-tags AND
  (importance >= floor OR canon_tag_pinning)` — only sub-floor, curator-tagged memories (the
  blue-seal shape) are newly eligible; already-floor-eligible facts are unaffected. German tag
  aliases (`regel`/`versprechen`/`abmachung`/`frist`/`schwur`/`eid`/`anvertraut`) join the
  matched-tag set only when the flag is on (`CANON_TAGS`/`effective_canon_tags`), mirrored
  independently into `app/memory/consolidation.py`'s preserve-tags so a German-tagged durable
  fact is never folded either. Ships with the mandatory (flag-on only) §3.3.1 stale-fact
  safeguard: drops an older pinned entry when a newer entry sharing a canon tag has strictly
  more content terms (the amendment case) and a later `created_at`; every other case — partial
  overlap, equal term sets, cross-family, missing timestamps — keeps both pinned (Q3:
  over-inclusion, never silent loss), and every drop is audited via a warning that reaches turn
  diagnostics through a new `LoadedTurnContext.warnings` field. The flag threads through
  `build_orchestrator_config` (both composition roots) to two independent consumers —
  `TurnSessionLoader` and `MemoryConsolidator` (via `TurnMemoryStage`) — pinned by a dedicated
  parity test (the #48/#67 lesson). New additive `TurnDiagnostics`/`TurnResult` fields
  `standing_facts_count`/`standing_facts_chars`, populated every turn (both flag states) and
  mirrored wherever `token_usage` already flows (API schemas, routes, SSE); old
  `diagnostics_json` rows without the keys deserialize as `None`. Offline D3 replay
  (`replay_selection --pinning`, read-only): flag off reproduces the pre-#78 baseline (0/47
  eligible); flag on makes all 3 tag-eligible memories eligible, incl. the blue-seal rule
  memory `354b8d98` — pinned block measures 352/900 chars, 0 safeguard warnings (no amendment
  pairs in this transcript). `select_retrieved_chunks_for_prompt`, write-dedup, and the
  ranking/boost math are untouched.
- **docs/26 Stage 5 session-side wiring** (#80, harness-local, byte-identical default): new
  `--definition-retries`/`LIVE_DEFINITION_RETRIES` knob (default 0) re-sends a probe's
  DEFINITION turn message up to N times when it ends in a non-success outcome, before
  `run_checkpoint` falls back to #75's fail-fast — modeling a real player's retry on an errored
  turn (docs/26 §8 Q4), never an app.config.Settings field. A consumed retry inserts an extra
  persisted DB turn, so scripted step no longer equals DB `turn_index`; every turn-number-keyed
  computation is made offset-aware: `event_by_callback`/`event_by_definition_turn` stay
  step-keyed, a new step-keyed `turn_by_step` map replaces positional indexing into the raw
  `turns` log (which itself gains one extra, clearly labeled `is_definition_retry`/
  `retry_attempt` entry per consumed retry), and a new `definition_turn_db_index` map tracks
  the ACTUAL persisted ordinal at the moment each event's definition turn succeeds, threaded
  into `inspect_story_event`'s new `definition_turn_index` parameter so #76's provenance
  attribution resolves to the successful retried turn's DB id, never the failed original's or
  a naively-assumed step==index value. The persisted-turn-count assertion is offset-adjusted;
  `quality_metrics` gains `definition_retries_allowed`/`definition_retries_used` (aggregate and
  per-event) so the actual survival delta can be measured, not assumed, across the docs/25
  Phase E live runs — the rejected `0.067² ≈ 0.45%`/`~96.5%` independence math is not computed
  here. Also closes a pre-existing gap: `standing_facts_count`/`standing_facts_chars` (#78) were
  already returned by `CreateTurnResponse` but never surfaced in the checkpoint's own turn
  records — now captured so docs/25 Phase E can read pinning evidence per turn.
  `scripts/live-smoke.sh` plumbs `LIVE_DEFINITION_RETRIES` through the same four points as
  `LIVE_FAIL_ON_STRUCTURED_WARNINGS`. New docs/25 Phase E runbook section. +10 unit tests
  pinning the offset/retry mechanics. The live 100-turn validation runs and any resulting
  default flip remain on the owner's machine (#80 stays open in docs/BACKLOG.md).

## 1.3.0 — 2026-07-12

### Fixed (found live)

- **Lexical write-dedup false-dropped terse distinct facts** (#72): the always-on coverage
  dedup dropped any candidate with ≥50% of its content terms contained in one existing
  summary — terse phrasings of distinct facts crossed that bar on frame vocabulary alone
  (found by the docs/25 Phase D 100-turn runs; run-dependent, since extractor phrasing
  length varies). Dedicated `WRITE_DEDUP_COVERAGE_THRESHOLD = 0.75` for the dedup call site
  (deterministic-fallback coverage unchanged at 0.5); both dedup warnings now name every
  dropped summary, so drops are auditable from turn diagnostics (88cf2b5).
- **Live-checkpoint late-recall probes were paraphrase-brittle**: `amber_ring_token` /
  `north_stair_rendezvous` synonym groups widened to the extractor paraphrases a real run
  produced ("symbol/promise/pledge/pact", "rendezvous"); assertions unchanged (1db74e7).

### Validation (docs/24 + docs/25, all on `26b-mtp` + Qdrant)

- **First real semantic-benchmark run** (docs/22 P0.4): `all-MiniLM-L6-v2` reranked overall
  recall@10 **0.824** / nDCG@10 **0.761**, German subset 0.630; opt-in floors calibrated
  0.3/0.2/0.30 → 0.75/0.70/0.55; the P1 measurement gate is open (75388f4).
- **Live validation phases A–C green**: #48/#67 composition parity (14/14 PASS, 0 misses),
  #69 context accounting (warnings fire at a 1024 ceiling, silent at the real 16384, no
  llama.cpp context shift), #6 recency 0.02/0.04 no-regression at 8 turns — shipped default
  stays 0.0 (c5a2a91, ccde1b3, a30d192).
- **Phase D (P2.2 long-campaign preset) is blocked on P1.1, not on the preset** (#73):
  consolidation fired and held SQLite/Qdrant parity across three 100-turn attempts, but
  dense-only retrieval stops selecting direct-answer memories at ~50-memory pools — filed
  with an offline-reproducible acceptance case for P1.1 hybrid retrieval / P2.5 rerank
  (f7101d8).

### Decisions

- **#68**: paraphrase flags stay warn-and-serve (flags mark confabulated overlap with
  authored secrets, not prompt leakage). **#71**: actor-stage transport failures keep the
  503/504 contract with no persisted turn row (f430261).

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
- **Cloud critic visibility projection** (#65): `TurnCritiqueStage.run` now projects
  `retrieved_chunks` to the route's allowed visibility before calling the critic — cloud routes
  see player-visible chunks only, closing a gap where a misbehaving or custom
  `actor_context_retriever` could otherwise deliver a GM/private chunk into a cloud critic prompt.
  A new `provider_binding` regression check
  (`malicious_retriever_gm_chunk_never_reaches_cloud`) pins it, taking the regression runner from
  84 to 85 checks (fcfd304).
- **Token usage & context-ceiling preflight** (#69): `token_usage` (prompt/completion/total) is
  now persisted into turn diagnostics and exposed on `CreateTurnResponse`/`TurnDetailResponse`/SSE;
  opt-in `MODEL_CONTEXT_WINDOW_TOKENS` + `CONTEXT_WARN_RATIO` add a pre-generation estimate warning
  and a post-generation actual-usage warning when a turn's prompt approaches the configured context
  window; retrieved-chunk and recent-dialogue trimming now cuts at word boundaries instead of
  mid-word (f977ba8).
- **API auto-ingest scenario lore on `POST /sessions`** (#16 follow-up): the API (and therefore
  the SPA) now mirrors the CLI's `start-session` auto-ingest instead of requiring a separate
  `ingest-scenario-lore` call. Both surfaces share one `app.composition.auto_ingest_scenario_lore`
  helper — idempotent, fail-open; a failed ingest degrades to a new
  `CreateSessionResponse.warnings: list[str]` instead of failing session creation. New additive
  request field `skip_lore_ingest: bool = false` (CLI parity: `--skip-lore-ingest`).
- **Resilient `semantic-benchmark` CLI + one-command runner + runbook** (docs/22 P0.4):
  `rolerag semantic-benchmark` now tries each `--model`/`--keyword` provider independently
  instead of aborting the whole run on the first failure — a bad model name or a blocked
  download is logged to stderr (`[failed] <label>: <ExceptionType>: <message>`) and skipped,
  stdout stays machine-pure JSON/table for whichever providers succeeded, and the command still
  exits non-zero if any provider failed. New `scripts/semantic-benchmark.sh` wraps the CLI
  end to end (env-configured: `MODELS`/`TOP_K`/`INCLUDE_KEYWORD`/`RUN_PYTEST`/`ARTIFACT_PATH`),
  writing the `--json` artifact, printing a suggested-floor summary against it via new
  `scripts/lib/suggest_floors.py`, and optionally running the opt-in `-m semantic` pytest tier.
  New runbook [docs/24](docs/24_semantic_benchmark_runbook.md) walks the first real-model run
  end to end and the floor-calibration procedure for
  `tests/evals/test_semantic_benchmark_opt_in.py` (ea31def).
- **Live-checkpoint evidence for consolidation + context accounting** (prep for the docs/22
  P2.2 long-campaign preset and #69's live validation): the live checkpoint JSON now reports
  `persisted.consolidated_memory_count` / `persisted.consolidation_summary_count` (SQLite rows
  tagged `CONSOLIDATED_TAG`/`SUMMARY_TAG`, proving a long run actually rolled memories up) and
  `quality_metrics.context_preflight_warning_count` / `context_actual_warning_count`
  (aggregated #69 context-accounting warnings) — all report-only, zero on any run with
  consolidation or the context ceiling left off (b854814). New runbook
  [docs/25](docs/25_live_validation_runbook.md) chains this evidence with the three other
  pending live validations (#48/#67, #69, #6) into one guided sitting.

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
- **Reroll persona restore** (#66): deleting the last turn (`DELETE /sessions/{id}/turns/last`)
  now restores `sessions.active_persona_id` to the nearest surviving `SUCCESS` turn's persona when
  the deleted turn had committed a persona switch, instead of stranding the session on it; scene
  switches are unaffected since they only move via the explicit scene endpoint, never as a
  per-turn side effect (856c73b).
- **CLI service assembly** (#67): `cli._build_services` now delegates to
  `composition.build_services` instead of a hand-rolled ~55-line assembly, so CLI turns honor
  author-pinned canon facts, record structured-output failures, and can reach semantic
  write-dedup — collaborators (`canon_repository`, `structured_failure_sink`,
  `memory_embedding_provider`) the API composition root already wired but the CLI previously
  omitted (1c36821).
- **Atomic turn_index assignment** (#70): `SQLiteTurnRepository.append_turn` now assigns
  `turn_index` inside the `INSERT` itself via a correlated `MAX(turn_index)+1` subselect plus
  `RETURNING`, instead of a separate read-then-write, so concurrent writers on one session
  serialize on SQLite's write lock instead of racing to compute the same index; a true
  cross-process concurrency test replaces the prior single-process WAL test (d50fc7c).

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
