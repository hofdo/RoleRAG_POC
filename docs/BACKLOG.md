# RoleRAG POC — Working Backlog

> Reviewed: 2026-07-10 @ 61e45b6

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

- [x] **CI-cap** Raised the `live-smoke.yml` `turn_count` validation cap from 5–50 to 5–100,
  matching `scripts/live-smoke.sh`'s own `LIVE_TURN_COUNT` range, so CI can drive the 100-turn
  runs the script already supports; input description and `docs/19` updated to match.

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

**Status (worked through 2026-07-08).** Shipped and gate-verified: **#48, #49, #50, #51, #52,
#53, #54, #56, #57, #58, #59, #61, #62, #63, #64** (15, **#58** completed 2026-07-08: Qdrant
`/readyz` healthcheck over bash `/dev/tcp` + `service_healthy` gate, validated on a live daemon).
Deferred with rationale:
**#55** (its "dead" guards are load-bearing for mypy; clean fix is an unjustified type split).
**#60** since **shipped 2026-07-08** (fake-provider ASGI contract app + Playwright spec in CI).
The **RAG-core items** in
[docs/22 § 2026-07-08 review](22_rag_scaling_roadmap.md#2026-07-08-review-confirmations--new-findings):
the two highest-value, **C1** (standing-facts double-spend) and **N1** (write-dedup framing
inflation), are now **shipped** (`64db602`, `0c11c29`) — validated on a live 26B + Qdrant
live-smoke (zero recall/extraction/retrieval-selection regression) plus byte-level unit tests,
honoring the measure-first invariant. **N3** (read-time normalized-text dedup) and **C2**
(consolidation min-age floor + batch-size cap) also **shipped** (2026-07-09) — byte-tested,
defaults byte-identical to pre-change selection; both are pure deterministic logic so no
live-smoke was required, except live validation of non-zero C2 knobs, which still rides along
with long-campaign P2.2. **N2** remains deferred (waits for German play) and carries its named
validation gate in docs/22.

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

- [x] **#51** **`SessionSummaryCache` invalidate/copy semantics only indirectly covered.**
  Nothing asserted (a) `load()` returns a copy or (b) consolidation's `invalidate()` forces a
  reload — either breaking would make write-dedup compare against a stale/mutated summary set and
  silently drop memories or keep duplicates. **Shipped:**
  `tests/unit/orchestration/stages/test_session_summary_cache.py` covers load-once-then-cache,
  copy-on-load (caller mutation can't corrupt the mirror), append growth (incl. the not-cached
  no-op), and the consolidation→invalidate→reload sequence. Gate green (+5 tests).

- [x] **#60** **No deterministic frontend↔backend contract test.** **Shipped.** Added
  `app/diagnostics/contract_app.py`: it serves the *real* `app.main:app` (real routes, error
  handlers, SPA mount) but overrides the `get_read_services`/`get_turn_services` dependencies with
  an in-process fake stack — the `smoke_runner` `SmokeTestProvider` over a throwaway SQLite file,
  no model, no Qdrant, no retriever (the turn pipeline is fail-open on retrieval and the fake
  returns canned actor text regardless). `dependency_overrides` drives fakes over *real* HTTP, which
  the in-process `TestClient` approach couldn't do for Playwright. A new `tests/e2e/spa-contract.spec.mjs`
  (npm `test:e2e-contract`) loads the real SPA, creates a session, runs a turn, and asserts the
  fake provider's fixed line renders verbatim — the deterministic contract, ~0.5s vs the live
  spec's ~18s. Wired into the `deterministic` CI job as a background-service step (Playwright's
  bundled Chromium, no `--no-sandbox` friction on the non-root runner). Verified locally: spec green
  against `uvicorn app.diagnostics.contract_app:app`.

- [x] **#61** **No coverage measurement.** **Shipped:** added `pytest-cov` (dev extra) +
  `[tool.coverage.run]` config, a `make coverage` target, and `--cov=app --cov-report=term-missing`
  on CI's Python test step — **report-only**, no threshold gate. Baseline is 91% line / branch.
  The config pins `core = "sysmon"` (Python 3.12 sys.monitoring) because coverage's default C
  tracer collides with numpy's C extension under fastembed ("cannot load module more than once").

- [x] **#64** **WAL cross-process concurrency asserted only by pragma value.** Only the pragma
  *values* were checked. **Shipped:** two behavioral tests — a fully-deterministic one proving a
  WAL reader keeps reading the last committed snapshot while another connection holds an
  uncommitted write, and a coordinated-thread one proving a second writer *waits* on
  `busy_timeout` (then succeeds) instead of failing with "database is locked". Stable across
  repeated runs (0.2s held lock vs the 5s timeout gives wide margin). Gate green (+2 tests).

### Code quality — maintainability

- [x] **#54** **`run_turn` built `TurnResult`/`TurnDiagnostics`/controlled-failure three+ ways.**
  **Shipped:** extracted `_controlled_failure_result(...)` (pairs persist + the
  `CONTROLLED_FAILURE` `TurnResult` return so the two failure exits can't drift) and a
  `_turn_diagnostics(...)` factory (the 3 identical `TurnDiagnostics` builds on the deferred,
  memory, and failure paths). Pure extraction — the persona-switch commit / deferred-memory reload
  ordering ([docs/21 danger zone](21_fable_handoff_reasoning.md#danger-zones-restated-with-reasoning))
  was left untouched. Verified with the full gate **plus** `smoke-run` (no warnings); the
  repair/controlled-failure unit tests pass unchanged.

- [~] **#55** **`AppServices` optional-repo fields force unreachable guards in the API routes.**
  **Deferred with rationale after investigation.** The `raise RuntimeError(...)` branches are not
  purely dead — they're the **mypy narrowing** mechanism for the `| None` fields (drop them and
  `mypy --strict` errors on the Optional access), so the "minimal" fix isn't viable. And the
  Optional fields aren't only a CLI artifact: the API test doubles deliberately build *partial*
  bundles (`test_api_sessions.py:99` passes none of the repos; the `test_api_turns.py` doubles pass
  different subsets), so making the three always-populated repos required would churn 4 test
  constructions plus need the repos built where they aren't used. The clean fix is a
  `ReadServices`/`TurnServices` type split (composition + api deps + routes) — real work for a
  cosmetic leak. Consistent with the repo's "no churn-for-cleanliness without measured benefit"
  stance (cf. the #18 skip), this is left as-is rather than refactored.

- [x] **#56** **Dead cloud-repair path taxed ~10 test fixtures.**
  `build_cloud_repair_messages` (Protocol + `CriticAgent` impl, self-labeled "Currently
  uncalled") was dead since the 2026-07-02 session-bound-provider change removed cloud
  escalation, yet every critic double had to stub it because it was on the Protocol. **Shipped:**
  removed from `CriticEvaluatingAgent` Protocol, `CriticAgent`, the eval fixture, and all 9 test
  stubs. `TurnRepairStage` already uses `build_local_repair_messages` on both providers. Gate green.

- [x] **#57** **Small dead / duplicated code on hot classes.** **Shipped:** removed the never-called
  `_loader_for_content_root` (a duplicate of the public `loader_for_session`); removed
  `_build_local_route` from `TurnOrchestrator` and inlined its `ModelRoute` construction into the
  one test that used it (`test_api_sessions.py`); extracted a `_row_to_session` static method on
  `SQLiteSessionRepository` (mirroring the existing `_row_to_turn`) so `get_session` and
  `list_recent_sessions` no longer inline the same row-mapping twice. Gate green.

### Ops / DX / packaging

- [x] **#53** **`.dockerignore` kept the live DB out of the image but not backups or WAL
  sidecars.** It excluded `data/rolerag.db` and `data/qdrant` but not `data/backups/` (each
  snapshot is a *full* DB copy) nor `data/rolerag.db-wal` / `-shm`. **Shipped:** added
  `data/rolerag.db-wal`, `data/rolerag.db-shm`, and `data/backups` so a `docker build` on a
  machine that has run the app or `rolerag backup` no longer bakes roleplay history into a layer.

- [x] **#52** **Frontend was unit-tested in CI but never type-checked or built there.** `ci.yml`
  ran `ng test` only; strict template type-checking and the bundle budget fire only at `ng build`.
  **Shipped:** added a `Build SPA` (`npx ng build`) step to `ci.yml`; verified the build is green
  locally (Angular 19, ~8 s). Refreshed the stale `frontend/README.md` note that said CI doesn't
  build.

- [x] **#58** **docker-compose: no Qdrant healthcheck, no readiness gate, no restart policy.**
  `app` had a healthcheck but `qdrant` had none and `app.depends_on: [qdrant]` waited only for
  *start*, not *ready*; neither service set `restart:`. **Shipped:** `restart: unless-stopped` on
  both services, plus a Qdrant `/readyz` healthcheck and the `depends_on: {qdrant: {condition:
  service_healthy}}` readiness gate. The `qdrant/qdrant:v1.18.1` image ships no `curl`/`wget`/`nc`
  but *does* ship `bash`, so the probe hits `/readyz` over bash's `/dev/tcp` (dash `/bin/sh` lacks
  it, hence the explicit `bash`); a closed/unready port makes the connect fail non-zero, so the
  check reports unhealthy rather than deadlocking the gate. Validated against a live daemon:
  `docker compose config` is valid and `docker compose up -d qdrant` reaches `healthy` in ~8s.

- [x] **#59** **High-risk Python deps were unbounded `>=`** (inconsistent with the deliberately-capped
  `qdrant-client>=1.18,<2`). **Shipped:** upper-bounded `fastapi<1`, `openai<3`, `pydantic<3`,
  `pydantic-settings<3`. Note: `openai` already resolves to **2.44** and the gate is green, so `<2`
  would have wrongly forced a downgrade — `<3` allows the tested 2.x and guards the next major.
  Verified with `pip install --dry-run` (resolves against installed versions) + full gate. A full
  lockfile stays YAGNI for a single-user POC.

- [x] **#62** **No frontend `lint` script.** Nothing linted the Angular/TS the way `ruff` guards
  the Python. **Shipped:** `ng add angular-eslint` (flat `eslint.config.js`, `npm run lint`
  target) + a `Lint SPA` CI step. Fixed the 9 findings it surfaced — a genuinely unused import and
  a no-op test expression (both real), an `Array<T>`→`T[]` style nit, and the SSE payload typed
  `any`→`unknown` with per-branch casts; configured `no-unused-vars` to honor the `^_`
  intentionally-unused convention. Verified `ng lint` clean, `ng build` clean, and all 70 SPA unit
  tests green.

- [x] **#63** **No dependency-vulnerability surface.** **Shipped:** a **non-blocking** `audit` CI
  job (`continue-on-error: true`) running `pip-audit` + `npm audit --omit=dev` — advisory only,
  never gates a merge. It already earned its keep: `pip-audit` is clean, but `npm audit` surfaced
  a **high-severity Angular advisory** (`GHSA-rgjc-h3x7-9mwg`, hydration DOM-clobbering /
  response-cache poisoning, affecting `@angular/core ≤19.2.25` — this SPA is on 19.2). Its fix is
  the Angular 21 major (breaking); the app's LAN-only, no-auth, client-render-only posture
  (docs/18) mitigates the realistic exposure. Tracked under the Angular-upgrade note below.

*Minor notes (no ID):* **Angular 19 → 21 upgrade — DONE (2026-07-08).** `ng update` 19→20→21,
clearing `GHSA-rgjc-h3x7-9mwg` (the #63 audit's finding; hydration DOM-clobbering against
`@angular/core ≤19.2.25`) — confirmed gone from `npm audit`. Node 20.19.6 satisfies the Angular 21
engines floor (`^20.19.0`); TypeScript 5.7 → 5.9.3 (CLI-driven); `angular-eslint` stayed 21.0.1; all
`ng update` code migrations were no-ops. Verified: `ng lint` + `ng build` + 70/70 karma + a
live-smoke UI pass (Playwright green against the real 26B + Qdrant stack). Remaining `npm audit`
highs are `http-proxy-middleware`, a dev-server-only transitive of `@angular-devkit/build-angular`
(never in the shipped static bundle); no non-breaking fix, left as-is.
`make check` runs Python-only so it's a narrower gate than CI; `data/sessions/` is a git-tracked
empty legacy dir (sessions live in SQLite now) — harmless.

## Not doing (personal-use scope)

StageGraph/DAG/plugin extensibility · hard memory-episode cap default (regressed recall) ·
auth/multi-user/streaming/tracing · corpus-scale micro-opts.

## Side projects

Full tiered list with effort + dependencies: [SIDE_PROJECTS.md](SIDE_PROJECTS.md) — the SPA,
RAG inspector, analytics, and eval dashboard entries shipped in 1.1.0.
Best next: ★ Transcript Exporter (weekend, zero backend).
