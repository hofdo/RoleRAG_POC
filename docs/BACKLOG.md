# RoleRAG POC — Working Backlog

> Reviewed: 2026-07-08 @ 5293417

Source: 10-agent deep analysis (47 improvements + side projects). This file is the durable
record — git commit subjects tag shipped items as `(#N)`. Keep it in sync as items land.

Numbers not listed here (#3–5, #23, #31–47) landed in the "Not doing" categories or the
side-project list rather than the engine backlog; only the items acted on carry a row below.
Items **#48+** come from the [2026-07-08 review](#review-2026-07-08--engine-quality-testing-ops)
(a fresh four-lens, code-grounded sweep) and continue the numbering without reusing an ID.

## Done

#1 #2 (early) · #7 redaction-ordering assertion · #8 persist+read turn diagnostics ·
#10 embedding-ab harness **ships** (`rolerag embedding-ab`, offline model A/B over the retrieval
fixtures); only the model **swap** was declined (candidates tied, so the default model stays) ·
#11 retrieval-miss eval ·
#12 TurnOrchestratorConfig · #13 redact raw_text in failure logging · #14 embedding-provider
failure tests · #20 repair decision → TurnRepairStage · #21 split TurnMemoryStage ·
#25 containment_overlap_threshold doc+tests · #30 skip semantic-dedup embed when threshold==1.0 ·
QE follow-ups (503 OpenAPI, C2/C3 named cap tests).

## Done — B tier (was loop: feat/b-tier)

All items below are resolved ([x] shipped, [~] dropped/deferred); the `feat/b-tier` loop branch
no longer exists. Order was value + independence, decisions last; each item was gate-verified
(`ruff && mypy && pytest && regression_runner && smoke-run`) and committed before the next.

- [x] **#17** critic error → fail closed (CONTROLLED_FAILURE) instead of serving unvalidated text
- [~] **#9** ~~memory-extraction cloud retry~~ — **DROPPED**: conflicts with the deliberate
  `memory_extraction_stays_local` / `critic_stays_local` invariant; deterministic fallback already covers it
- [x] **#16** auto-ingest scenario lore on `start-session` (CLI; graceful + `--skip-lore-ingest`;
  API create_session left as a follow-up)
- [~] **#19** structured `TurnResult.errors` — **DEFERRED to decision-batch**: `warnings: list[str]`
  spans 78 sites + 4 API schemas + persistence; contract-breaking + taxonomy decision; YAGNI for a
  single-user POC until a UI consumer needs it
- [x] **#15** web/SSE robustness tests + client malformed-frame hardening (404/MIME/traversal;
  parseFrame → ApiError; mid-retrieval already covered)
- [x] **#22** web `/play` retrieval-inspection modal (current turn; query + selected/rejected +
  scores + boosts). Historical per-turn via #8's endpoint = future extension
- [x] **#6** importance-aware recency boost — **opt-in** (`RAG_RECENCY_WEIGHT`, default 0.0,
  byte-identical); recency scaled by importance so it can't displace older high-importance memories;
  offline sweep 85/85, validate via live-smoke before enabling
- [x] **#19** structured turn errors — additive `errors` (category/stage/message/suggestion) on the
  API/SSE turn responses, derived from the warning strings via `classify_warnings()`; non-breaking
  (warnings preserved, TurnResult unchanged, no 78-site churn). Frontend display = future extension
- [x] **#29** documented why `session_memory_max_episodes` default is 0 (a hard cap regressed
  50-turn recall); no code change

## A tier (quick wins)

- [x] **#24** validate gating strings at stage construction (bad value → ValueError, not silent no-op)
- [x] **#27** CLI color: errors red / warnings yellow / success green (`typer.secho`, auto-strips in tests)
- [~] **#28** ~~extract test-scenarios module~~ — **SKIP (misread)**: `ROSE_GALLERY_MESSAGES`/`STORY_EVENTS`
  are production live-checkpoint data in `app/diagnostics/live_checkpoint.py`, already centralized +
  importable by tests. Moving them would be wrong (not test-only) + pure churn
- [~] **#26** ~~roleplay-aware stopwords~~ — **SKIP**: reverses a documented choice (framing words like
  ask/tell deliberately excluded so "I ask whether she…" doesn't boost unrelated chunks); the stemmer
  splits ask/tell/say variants anyway, so re-including them matches inconsistently. Benefit unprovable
  offline, risk is live-only
- [~] **#18** ~~narrow broad `except Exception`~~ — **SKIP**: the ~20 handlers are intentional
  resilience boundaries (memory/retrieval/critique/diagnostics best-effort degrade-to-warning).
  Narrowing them would let an unexpected type crash a turn — harms robustness for no benefit

## Decisions (2026-07-02: session-bound provider)

- **Cloud is a peer session-bound choice, not an escalation target.** `local`/`cloud` is
  chosen once at session creation and is immutable for that session's lifetime; every task
  (actor, repair, critic, memory extraction) runs on the session's bound provider. All
  automatic cloud escalation/fallback/rescue paths (low retrieval confidence, high scene
  complexity, failed local repair, local-provider failure) were removed — the router
  (`app/llm/router.py::choose_route`) does nothing but map `session_provider` to a route.
  The old per-turn `request_cloud` request field and the two-phase `confirmation_required`
  turn status were dropped entirely; `CLOUD_MODE=off|ask|auto` now only gates cloud
  **session creation** (reject / confirm-once / silently allow).
- **Secrets never reach cloud, enforced structurally.** Persona hidden fields
  (`secrets`, `forbidden_knowledge`) and scene `gm_private_summary` never enter a cloud
  request, on any provider, via an `include_hidden` gate
  (`app/orchestration/stages/critique.py`: `include_hidden=route.provider ==
  ModelProviderName.LOCAL`) plus a provider-binding eval test that pins the invariant.

## Shipped 2026-07-01/02 (play-experience v1.2)

The play-experience batch that landed on main alongside the decisions above. The SPA resume
picker (see "Follow-ups (SPA)") and the session-bound provider change (see the 2026-07-02
decision) are recorded elsewhere in this file and are not repeated here.

- SQLite WAL mode + `rolerag backup` command + automatic pre-write snapshots (`502f80c`).
- Reversal/dedup fix so a rerolled turn's memories are cleaned up correctly (`4e907c9`).
- Cached embedding-provider singletons — reuse instead of rebuilding per request (`698407a`).
- Live stage-progress SSE frames (`event: stage`) during the turn pipeline (`dc3803a`, `cd10719`).
- Reroll: `DELETE /sessions/{id}/turns/last` plus the SPA reroll control (`c69d741`, `2c03f6b`).
- Scene switching (`POST /sessions/{id}/scene`) and per-turn persona override (`3e93e4b`).
- Cross-session persona memory — persona episodes carried across a persona's sessions (`8f517e2`).
- Deferred memory curation — memory writes moved to a background job on API turns (`4a5c928`).

## Decisions (2026-07-01 audit)

- **Legacy `/play` UI: deleted.** The Angular SPA at `/app` is the only web UI; `app/web/`,
  its vanilla-JS client/tests, and the `/play` routes were removed in 1.1.0. Rationale: two
  coexisting UIs doubled maintenance and testing surface with no canonical owner.
- **Graduated to 1.1.0.** v1 acceptance passed and the SPA/endpoints shipped since 1.0.0;
  version bumped, `CHANGELOG.md` added, `setup.py`/`setup.cfg` (stale 0.1.0) deleted in favor of
  `pyproject.toml`.
- **Milestone 4 (shared world state): deferred with rationale.** The
  `world_facts`/`WorldFact` layer from the original plan remains unbuilt — deliberately. Durable
  memory + retrieval proved sufficient at live scale: recall was verified at 8/30 turns, and the
  100-turn run demonstrated stability and speed with recall assumed lossless (per the docs/16
  caveat — the extended checkpoint reports rather than hard-asserts recall). First-class mutable
  world state gets built when live evidence shows recall/consistency degrading because facts live
  only in memory episodes, not before.

## Follow-ups (SPA)

- [x] SPA session resume: wired into the setup picker (resume select + Resume button, backed by
  `GET /sessions`). `resume()` now loads the full transcript via `GET /sessions/{id}/turn-details`
  instead of only the 8 recent turns, and the composer keeps the draft when a turn fails
  (`sendMessage` returns `Promise<boolean>`; the per-turn cloud-confirm methods `confirmCloud`/`forceLocal`
  were later removed with the 2026-07-02 session-bound-provider change).

## Open follow-ups (workflow)

- [ ] Raise the `live-smoke.yml` `turn_count` validation cap (currently 5–50) — or add a separate
  `long_turn_count` input — so CI can drive the 100-turn runs that `scripts/live-smoke.sh` already
  supports. Deferred here because changing the range is a workflow behavior change, out of scope
  for the docs sweep.

## Review 2026-07-08 — engine quality, testing, ops

A fresh four-lens, code-grounded sweep (code-quality · testing/verification · RAG core ·
ops/DX) on top of `5293417`, deterministic gate green (`ruff` / `mypy --strict` /
`pytest`). Nothing below duplicates a done/dropped/skipped item above or a
[docs/21 rejected idea](21_fable_handoff_reasoning.md#ideas-already-tried-rejected-or-deliberately-deferred);
each was checked against the decision record first. **RAG-core recall findings live in
[docs/22](22_rag_scaling_roadmap.md#2026-07-08-review-confirmations--new-findings)** (that doc
owns retrieval/memory) — this section is engine-quality, test, and ops work. House style still
applies: additive/opt-in, byte-identical defaults for risky knobs, gate + (where the change is
invisible to the deterministic suite) live-smoke before claiming success, docs move with code.

Ordered by value. Effort S/M/L.

### Correctness — do first

- [x] **#48** *(bug)* **CLI is a second composition root that has drifted from the API.**
  `app/cli.py::_build_services` hand-rolled `TurnOrchestratorConfig(...)` with 9 of 25 fields
  instead of calling `composition.build_orchestrator_config`, so CLI turns fell back to dataclass
  defaults for ~16 `.env`-backed keys (`critic_gating`, `curator_gating`, `canon_*`,
  `rag_write_dedup_cosine_threshold`, `memory_consolidation_*`, `containment_overlap_threshold`,
  `recent_dialogue_max_message_chars`, …). **Live divergence:** `local_structured_max_tokens`
  defaults to **640** in `Settings`/API but **350** in the dataclass the CLI never set, so every
  CLI turn ran the local critic and memory extractor at a smaller structured budget than the API.
  **Shipped:** the CLI now delegates to `build_orchestrator_config` and passes the two
  `MemoryIndexer` kwargs (`importance_floor`, `session_memory_max_episodes`);
  `tests/unit/test_composition_config_parity.py` pins that both roots produce the same config.
  Gate green. Still worth a live-smoke pass (it changes the CLI structured-token budget).

- [x] **#49** **Auto-snapshot before destructive CLI ops is mocked but never asserted-called.**
  `tests/integration/test_cli.py` patched `app.cli._backup_database` for `delete-session` /
  `reset-db` with no `assert_called_once()`, so the v1.2 pre-write snapshot (`502f80c`) — the only
  guard against an irreversible wipe of authoritative SQLite state — could be refactored away with
  CI green. **Shipped:** both tests now capture the mock and `assert_called_once()`. (`import-session`
  has no snapshot — only 4 `_backup_database` call sites, none in import — so nothing to add there.)

### Testing & verification

- [x] **#50** **No paired InMemory↔Qdrant vector-store parity harness** (guards the visibility
  boundary, invariant #2). The two hand-written filter impls (`_chunk_matches_filters`,
  `_build_qdrant_filter`) diverged on tags — `issubset`/AND in-memory vs `MatchAny`/OR in Qdrant.
  **Shipped:** `tests/unit/test_vector_store_parity.py` runs one fixture set through *both*
  `.search()` paths (Qdrant via an embedded `QdrantClient(":memory:")` local-mode client, so the
  real matcher runs) across every filter dimension — visibility, world, scene, persona, session,
  single tag, two-tag AND, absent tag, combined, and match-nothing — asserting identical id sets.
  The Qdrant tags filter is now AND (one `MatchValue` per tag) to match the in-memory intent.
  This subsumes the docs/22 "Verified small fixes" tags item. Gate green (+12 tests).

- [ ] **#51** **`SessionSummaryCache` invalidate/copy semantics only indirectly covered.**
  `grep invalidate tests/` is empty. Nothing asserts (a) `load()` returns a copy (a caller
  mutating the dedup list can't corrupt the cached mirror) or (b) consolidation's
  `invalidate()` (`memory_consolidation.py:121`) forces a reload. If either broke, write-dedup
  would compare against a stale summary set and silently drop legitimate memories or keep
  duplicates — a recall-affecting path. Fix: a focused unit test on the three methods incl. the
  consolidation→invalidate→reload sequence. Effort S.

- [ ] **#60** **No deterministic frontend↔backend contract test.** The only browser test
  (`tests/e2e/spa-play.spec.mjs`) needs a real model + full stack and runs *only* in
  `live-smoke.yml` (self-hosted). `ci.yml` runs `ng test` with mocked HTTP. A backend schema
  rename (turn-detail payload, `DeleteLastTurnResponse`, SSE frame shape) that breaks the Angular
  client compiles clean, passes `ng test`, and only surfaces in a manual/weekly GPU run. Fix: a
  Playwright (or lighter schema-contract) run against FastAPI wired with fake providers +
  `InMemoryVectorStore` (the `smoke-run` stack — no model needed). Effort M.

- [ ] **#61** **No coverage measurement.** No `pytest-cov`/`coverage` in `pyproject.toml` or CI;
  which branches the ~69 test files exercise is invisible. #49/#51 are exactly the blind spots a
  report surfaces (mocked-but-unasserted safety net; uncovered `invalidate`), as are the
  intentional `except Exception` fail-open seams that could lose their one test unnoticed. Fix:
  `pytest-cov` + `--cov=app`, **report-only** (no hard gate — the per-branch report is the value,
  not a threshold number). Effort S.

- [ ] **#64** **WAL cross-process concurrency asserted only by pragma value.**
  `tests/unit/test_sqlite.py:178` checks `journal_mode=WAL` + `busy_timeout=5000` are *set*, but
  no test opens two connections and writes concurrently to prove the documented "CLI running
  while the API serves, no database-is-locked" claim. Effort S; lower priority (timing-flaky,
  single-user scope) — do only after the above.

### Code quality — maintainability

- [ ] **#54** **`run_turn` builds `TurnResult`/`TurnDiagnostics`/controlled-failure three ways.**
  `turn_orchestrator.py:291-522` repeats the same ~7-field `TurnResult(...)` (×4) and
  `TurnDiagnostics(...)` (×3) constructions that must be kept in lockstep by hand; the two
  controlled-failure paths near-duplicate persist+return. A `_build_controlled_failure_result(...)`
  helper and one `_diagnostics_from(...)` factory collapse it. Effort M. **Risk: medium** — this
  is the highest-leverage code and the [docs/21 danger zone](21_fable_handoff_reasoning.md#danger-zones-restated-with-reasoning);
  the persona-switch commit / deferred-memory reload ordering is *deliberate* and must be left
  alone. Behavior-preserving + gate-verified only; decline if the explicitness is preferred.

- [ ] **#55** **`AppServices` optional-repo fields force unreachable guards in the API routes.**
  `turn_repository`/`memory_repository`/`canon_repository`/`memory_indexer` are typed `| None`
  (`composition.py:51-60`) only because the CLI builds a leaner bundle, but `build_services`
  *always* populates them for the API path — so `routes.py:534-535,573-574,614-615`'s
  `raise RuntimeError(...)` branches are dead and untested, muddying the thin-routes invariant.
  Fix: a narrower `ReadServices`/`TurnServices` type with non-optional repos (M), or minimally
  drop the dead guards (S). Ties to #48 (both are the one-bundle-two-consumers seam).

- [ ] **#56** **Dead cloud-repair path taxes ~10 test fixtures.**
  `build_cloud_repair_messages` (`critique.py:66-71` Protocol; `critic_agent.py:107-126`,
  self-labeled "Currently uncalled") is dead since the 2026-07-02 session-bound-provider change
  removed cloud escalation, yet every critic double must stub it because it's in the Protocol.
  Fix: drop from Protocol + agent + fixtures + stubs (S). Decline if a future cloud-repair prompt
  is genuinely anticipated — but then at least pull it out of the Protocol.

- [ ] **#57** **Small dead / duplicated code on hot classes.** `_loader_for_content_root`
  (`turn_orchestrator.py:602-603`) exactly duplicates the public `loader_for_session` and is
  never called; `_build_local_route` (`:605-612`) is production code used only by one test
  (`test_api_sessions.py:773`). And `SessionState` row-mapping is inlined twice
  (`repositories.py:166-176`, `199-214`) instead of a `_row_to_session` helper mirroring the
  existing `_row_to_turn`. Trivial, very low risk. Effort S.

### Ops / DX / packaging

- [ ] **#53** **`.dockerignore` keeps the live DB out of the image but not backups or WAL
  sidecars.** It excludes `data/rolerag.db` and `data/qdrant` (clear intent: no personal data in
  image layers) but not `data/backups/` (each snapshot is a *full* DB copy) nor
  `data/rolerag.db-wal` / `-shm` (committed-but-uncheckpointed content). A `docker build` on a
  machine that has run the app or `rolerag backup` bakes full roleplay history into a layer. Fix:
  add `data/backups` + `data/*.db-wal` `data/*.db-shm` (or `data/*.db*`). Effort S; one line,
  same privacy intent the existing exclusion already encodes.

- [ ] **#52** **Frontend is unit-tested in CI but never type-checked or built there.** `ci.yml`
  runs `ng test` only; strict template type-checking and the 1 MB bundle budget fire only at
  `ng build`. A template type error or broken `@Input` passes CI green and surfaces later as
  `dev-up.sh` "API runs without a UI" (`dev-up.sh:96-100`). Fix: add `npx ng build` (or at least
  `tsc -p tsconfig.app.json --noEmit`) to CI — already known-green via Docker/dev-up, ~30–60 s.
  Not a matrix expansion, so not in tension with the docs/10 CI-scope guardrail. Effort S.

- [ ] **#58** **docker-compose: no Qdrant healthcheck, no readiness gate, no restart policy.**
  `app` has a healthcheck but `qdrant` has none and `app.depends_on: [qdrant]` waits only for
  *start*, not *ready* — so the app can take first turns before Qdrant accepts connections and
  (retrieval being fail-open) degrade silently to no-retrieval on first-boot ingest. Neither
  service sets `restart:`. `dev-up.sh` already polls `/readyz` (`:57`), so Compose is the weaker
  entry point. Fix: Qdrant `/readyz` healthcheck + `depends_on: {qdrant: {condition:
  service_healthy}}` + `restart: unless-stopped`. Effort S.

- [ ] **#59** **High-risk Python deps are unbounded `>=` with no lockfile** (inconsistent with the
  deliberately-capped `qdrant-client>=1.18,<2` and pinned `qdrant/qdrant:v1.18.1`). `openai>=1.40`
  in particular could resolve to a 2.x with schema changes to the `chat.completions` call the whole
  engine depends on; `pip install .` in the Dockerfile makes two builds months apart non-reproducible.
  Fix (light): upper-bound at least `openai`, `fastapi`, `pydantic`/`pydantic-settings` — the
  `openai` cap is worth it regardless. A full lockfile is likely YAGNI for a POC. Effort S (caps).

- [ ] **#62** **No frontend `lint` script.** `frontend/package.json` has `test` but no `lint`, and
  no ESLint config exists — so nothing lints the Angular/TS the way `ruff` guards the Python.
  Fix: add `ng lint` (Angular ESLint) + a `lint` script; optionally wire into CI (#52) and
  `make check`. Effort S. Low priority — thin-client by design.

- [ ] **#63** **No dependency-vulnerability surface** (no `dependabot.yml`, no `pip-audit`/`npm
  audit` step). The app binds `0.0.0.0` with no auth by design (docs/18), so a transitive CVE in
  the HTTP stack is the realistic threat. Fix: a **non-blocking** `pip-audit` + `npm audit`
  advisory step. Effort S. Most likely declined under personal-use scope + advisory noise from
  dev-deps; listed for completeness. Lowest value of the set.

*Minor notes (no ID):* Angular 19 is one major behind (thin client, no urgency — flag on next
dep sweep); `make check` runs Python-only so it's a narrower gate than CI (ties to #52);
`data/sessions/` is a git-tracked empty legacy dir (sessions live in SQLite now) — harmless.

## Not doing (personal-use scope)

StageGraph/DAG/plugin extensibility · hard memory-episode cap default (regressed recall) ·
auth/multi-user/streaming/tracing · corpus-scale micro-opts.

## Side projects

Full tiered list with effort + dependencies: [SIDE_PROJECTS.md](SIDE_PROJECTS.md) — the SPA,
RAG inspector, analytics, and eval dashboard entries shipped in 1.1.0.
Best next: ★ Transcript Exporter (weekend, zero backend).
