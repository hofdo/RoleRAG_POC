# RoleRAG POC — Working Backlog

> Reviewed: 2026-07-14 @ 240a9bf

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
- [x] **#16** auto-ingest scenario lore on `start-session` (CLI; graceful + `--skip-lore-ingest`).
  **Follow-up shipped**: `POST /sessions` now mirrors it via the shared
  `app.composition.auto_ingest_scenario_lore` helper (both `app.cli._auto_ingest_scenario_lore`
  and `app.api.routes.create_session` call it) -- request field `skip_lore_ingest: bool = false`
  (additive), failures degrade to `CreateSessionResponse.warnings: list[str]` (additive) instead
  of failing session creation, idempotent (`ingest_document` replaces a source's chunks by path).
  Known/acceptable for personal use: like the CLI's `start-session`, API session creation loads
  the embedding model and embeds scenario lore inline on first use, so the first `POST /sessions`
  on a cold cache can be slow.
- [~] **#19** structured `TurnResult.errors` — **DEFERRED to decision-batch**: `warnings: list[str]`
  spans 78 sites + 4 API schemas + persistence; contract-breaking + taxonomy decision; YAGNI for a
  single-user POC until a UI consumer needs it
- [x] **#15** web/SSE robustness tests + client malformed-frame hardening (404/MIME/traversal;
  parseFrame → ApiError; mid-retrieval already covered)
- [x] **#22** web `/play` retrieval-inspection modal (current turn; query + selected/rejected +
  scores + boosts). Historical per-turn via #8's endpoint = future extension
- [x] **#6** importance-aware recency boost — **opt-in** (`RAG_RECENCY_WEIGHT`, default 0.0,
  byte-identical); recency scaled by importance so it can't displace older high-importance memories;
  offline sweep 85/85. Live-validated 2026-07-12 (docs/25 Phase C, `26b-mtp`, 8-turn runs at
  `RAG_RECENCY_WEIGHT=0.02` and `0.04`): both pass with 0 callback-recall / 0
  retrieval-selection misses, `retrieval_miss_ranks=[]` — identical to the Phase A baseline
  (which is already miss-free at 8 turns, so no measurable *benefit* either at this length;
  the 100-turn Phase D run carries `0.02` and is the meaningful long-session signal).
  Decision: shipped default stays `0.0`; `0.02` is validated safe to enable per-campaign in
  a personal `.env` (not `.env.example`)
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

## Decisions (2026-07-12: paraphrase flags + actor transport failures)

- **#68 — Paraphrase flags stay warn-and-serve; recorded as accepted risk.** `secret_guard`
  keeps redacting verbatim hidden-fact echoes and only *flagging* likely paraphrases; flagged
  replies are persisted and served with the warning attached. Rationale: hidden fields never
  enter the actor prompt (visibility invariant), so a paraphrase flag marks *confabulated*
  overlap with authored secrets, not an actual leak of prompt content — the containment
  regression checks pin exactly this split (`verbatim_echo_redacted`,
  `paraphrased_confabulation_flagged`, `in_character_deflection_not_flagged`). Escalating a
  flag to the repair pass would feed the hidden text into another generation to scrub a
  maybe-overlap (a larger real exposure than the flag itself), and escalating to controlled
  failure would raise the fail-closed rate for false-positive overlap on a personal-use app.
  Revisit only if a real session produces a flagged paraphrase that reads as a genuine
  disclosure — that concrete transcript reopens (b).
- **#71 — Actor-stage transport failures keep the 503/504 contract; no turn row.** A
  `ProviderTimeoutError`/`ProviderUnavailableError` before/during actor generation leaves
  nothing worth persisting: no draft text exists, the player's message is not consumed, and
  the correct client behavior is retry-same-message (the SPA's SSE client already aborts hung
  streams). Persisting a controlled-failure turn there would write history/memory rows for
  infrastructure blips and break clean retry semantics. The critic-stage asymmetry is
  principled, not accidental: by critic time real actor text exists, and fail-closed
  (persisted controlled failure) beats both silently dropping it and serving unvalidated text
  (invariant #4). `tests/integration/test_provider_unavailability.py` continues to pin the
  contract as intended behavior rather than an accident.

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
  Gate green. Live-smoke validated 2026-07-12 (docs/25 Phase A, `26b-mtp`): CLI
  structured-token budget change confirmed under a real model + Qdrant — 14/14 report steps
  PASS, conversation checkpoint `status: pass`, 0 callback-recall / retrieval-selection misses.

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

## Review 2026-07-10 — independent current-state analysis

A second, independent code-grounded review at `9097877` (cross-model verification against an
external GPT-5.6 analysis brief; both reviews reached the same overall verdict). Verified before
filing: deterministic gate green locally (ruff / `mypy --strict` / 614 pytest / 84-check
regression runner) and GitHub Actions CI green at this commit. Everything below was confirmed
against source, not inherited from either review. RAG-core findings were re-confirmed but add
nothing new — the P0.4 measurement gate and P2.2 long-campaign validation in
[docs/22](22_rag_scaling_roadmap.md) remain the authoritative retrieval roadmap (this review
independently endorses P0.4 as the highest-value RAG work; note the live checkpoint's recall
probes all land at or before turn 50, so 100-turn runs assert nothing about late recall — already
recorded there).

Ordered by value. Effort S/M/L.

### Correctness / safety — do first

- [x] **#65** *(safety, M)* **Critic prompt lacks a final visibility projection for retrieved
  chunks.** The actor path independently drops non-player chunks at prompt build
  (`context_budget.select_retrieved_chunks_for_prompt` skips `visibility != PLAYER`), but the
  critic path formats whatever chunks it is handed
  (`critic_agent._format_retrieved_chunks` prints the visibility label without filtering on it);
  the only chunk-level gates are upstream in the retriever. `include_hidden=False` covers
  authored persona/scene fields only. In the shipped configuration nothing leaks — but a
  misbehaving or future custom `actor_context_retriever` (settable via a public property on the
  orchestrator) would deliver a GM/private chunk straight into a **cloud** critic prompt,
  violating invariant #2 at the trust boundary. Fix: project `retrieved_chunks` to the route's
  allowed visibility inside `TurnCritiqueStage.run` (player-only when the route is cloud) and add
  a malicious-retriever case to the `provider_binding` regression category. Acceptance: a
  retriever that returns GM/character-private chunks cannot get one into any cloud request.
  **Shipped:** `TurnCritiqueStage.run` now projects `retrieved_chunks` to the route's allowed
  visibility *before* calling `critic_agent.evaluate` (`_project_chunks_to_route_visibility` in
  `app/orchestration/stages/critique.py`) — cloud routes keep PLAYER-visible chunks only and the
  stage warns `"critic context filtered: N non-player chunk(s) withheld from cloud critic"`; local
  routes are unaffected. `tests/unit/orchestration/stages/test_core_stages.py` covers the cloud
  filter+warning, local pass-through, and empty/player-only no-warning paths. The
  `provider_binding` eval gained a `malicious_retriever_gm_chunk_never_reaches_cloud` check
  (`app/evals/regression_runner.py`, mirrored in
  `tests/evals/test_provider_binding_regressions.py`) that wires a stub retriever
  (`MaliciousActorContextRetriever` in `app/evals/fixtures.py`) returning GM + character_private
  chunks into a full-stack CLOUD-session turn and asserts neither string reaches any recorded
  cloud request (actor, critic, or memory). Gate green.

- [x] **#66** *(bug, S)* **Reroll leaves a deleted turn's persona/scene switch committed.**
  `DELETE /sessions/{id}/turns/last` reverses the turn row, its memories (timestamp provenance),
  and its vectors — but not `sessions.active_persona_id`/`active_scene_id` committed by that
  turn's post-persistence switch (`turn_orchestrator.run_turn` persona-switch commit). Deleting
  the turn strands the session on a persona/scene the surviving history never switched to. Fix in
  the reroll path + integration test; while there, consider wrapping the three separately-committed
  delete steps in one transaction (a crash between them currently orphans memory rows).
  **Shipped:** scoped to persona only — scenes never change as a per-turn side effect
  (`TurnInput` has no scene field; `active_scene_id` only moves via the explicit
  `POST /sessions/{id}/scene` endpoint), so a scene switch made after the deleted turn is a
  deliberate, independent action and correctly survives a reroll; a test now pins that. Added
  `restore_persona_after_turn_delete` (`app/persistence/repositories.py`), called from
  `delete_last_turn` in `app/api/routes.py`: a CONTROLLED_FAILURE deleted turn never committed a
  persona change (no-op); a SUCCESS deleted turn restores the persona of the nearest remaining
  SUCCESS turn (CONTROLLED_FAILURE turns never move the pointer, so they're skipped); if no
  SUCCESS turn remains (the deleted turn was the first to ever switch persona since session
  creation), the pre-switch value isn't recoverable — `sessions` only stores the *current*
  `active_persona_id`, not the creation-time one, so it's left as-is (documented limitation).
  Deferred the three-way-transaction idea: `SQLiteTurnRepository`/`SQLiteMemoryRepository` each
  auto-commit per call on a shared connection, so wrapping turn-delete + memory-delete atomically
  would mean adding no-commit variants (or a cross-repository transaction helper) to both
  Protocols — more invasive than this fix warrants; left for a follow-up if the crash-window risk
  is ever judged worth it.

- [x] **#67** *(bug, S)* **CLI orchestrator wiring omits three collaborators the API wires.**
  #48 fixed *config* parity, but `cli._build_services` still constructed the `TurnOrchestrator`
  without `canon_repository` (CLI turns silently ignored author-pinned canon facts that API turns
  honor), `structured_failure_sink` (CLI structured-output failures were never recorded), and
  `memory_embedding_provider` (semantic write-dedup/consolidation could never activate on CLI even
  when configured); the returned `AppServices` also omitted `turn_repository`, `memory_repository`,
  `canon_repository`, and `memory_indexer`. **Shipped:** `cli._build_services` now delegates
  outright to `composition.build_services` — the ~55-line hand-rolled assembly is gone, replaced by
  a one-line call, so the two composition roots can no longer drift on *either* config or
  collaborators. The five CLI-local builder aliases that only the deleted assembly used
  (`_build_local_provider`, `_build_cloud_provider`, `_build_critic_agent`, `_build_file_loader`,
  `_build_memory_curator`) were removed as dead code; the three aliases other CLI commands still
  call directly (`ingest`, `ingest-scenario-lore`, `reindex-memories`, `reset-index`,
  `delete-session`, `reset-db`, `retrieve-debug`'s manual retriever, lore auto-ingest) —
  `_build_embedding_provider`, `_build_vector_store`, `_build_actor_context_retriever` — were kept.
  `tests/integration/test_cli.py` patches that targeted the removed CLI aliases for
  `_build_services`-routed commands (`start-session`, `resume`, `turn`, `retrieve-debug`'s loader)
  now target `app.composition.build_*` instead, since that's where those calls are resolved after
  delegation; patches for the still-direct CLI aliases were left as `app.cli._build_*`.
  `tests/unit/test_composition_config_parity.py` gained
  `test_cli_and_api_build_services_wire_the_same_collaborators`, which builds both roots with fake
  providers and asserts `canon_repository`, `structured_failure_sink`, and
  `memory_embedding_provider` all reach the built orchestrator on both surfaces — the concrete #67
  regression tripwire — plus `test_cli_build_services_delegates_to_composition_build_services`
  pinning the delegation itself. Gate green (85 regression checks). Live-smoke validated
  2026-07-12 (docs/25 Phase A, `26b-mtp`): canon injection, structured-failure logging, and
  semantic-dedup reachability confirmed under a real model + Qdrant — 14/14 report steps PASS,
  conversation checkpoint `status: pass`, 0 callback-recall / retrieval-selection misses.

### Decision

- [x] **#68** *(decision, S)* **Resolved 2026-07-12 — (a) accepted and recorded** under
  "Decisions (2026-07-12)" above: warn-and-serve stays; paraphrase flags mark confabulated
  overlap, not prompt leakage. Original item:
  **Paraphrase-flag policy is undocumented risk acceptance.**
  `secret_guard.scan_reply` redacts verbatim hidden-fact echoes but only *flags* likely
  paraphrases; the orchestrator appends a warning and still persists and returns the flagged
  reply to the player. No layer repairs or withholds it, and no decision record says this is
  intentional. Decide: (a) accept and record under Decisions (cheap, honest), or (b) escalate a
  paraphrase flag to the bounded repair pass / controlled failure (dearer; changes turn-failure
  rates — validate via live-smoke). Either outcome closes the gap between the code and a strict
  reading of the secrecy invariant.

### Observability

- [x] **#69** *(observability, M)* **Token usage is captured but dead; context budget is
  character-based and unverified against real context windows.** `LlmResponse.usage`
  (prompt/completion/total tokens, `openai_compatible.py`) has zero consumers; there is no
  preflight size estimate, no configured model-context ceiling, no overflow warning; retrieved
  chunks and recent dialogue are clipped mid-word by character count. Ship: persist usage into
  turn diagnostics (additive), a configured context ceiling + warning threshold, and
  word/sentence-boundary trimming. Acceptance: an oversized scenario is observable *before*
  generation, validated against llama.cpp logs (docs/22 measure-first: offline evals cannot see
  this). **Shipped:** three additive, opt-in pieces, all byte-identical by default.
  (1) *Usage persistence:* `TurnDiagnostics.token_usage` and `TurnResult.token_usage`
  (`app/domain/models.py`) — optional `dict[str, int] | None`, so old `diagnostics_json` rows
  without the key deserialize with `token_usage=None`. Usage now threads through
  `GenerationStageResult.usage` and `RepairStageResult`/`RepairResolution.usage`
  (`app/orchestration/stages/generation.py`, `repair.py`) up to
  `TurnOrchestrator._turn_diagnostics`: the reported usage is always the generation that produced
  the *served* text — the repair generation's when a repair ran, otherwise the initial actor
  generation's (mirrors how `finish_reason` was already tracked) — and `None` when no generation
  completed at all (actor failed before any usable response). Exposed as an optional
  `token_usage` field on `CreateTurnResponse`, `TurnDetailResponse`, and `StreamFinalPayload`/
  `StreamFailurePayload` (`app/api/schemas.py`, wired in `app/api/routes.py` +`app/api/sse.py`);
  SPA untouched. (2) *Context-ceiling preflight:* new `Settings.model_context_window_tokens`
  (default `0` = disabled) and `Settings.context_warn_ratio` (default `0.85`), mirrored in
  `.env.example`. When enabled, `app/orchestration/context_budget.py::context_preflight_warning`
  estimates prompt+completion tokens from the built actor messages via a chars/4 heuristic
  (`estimate_prompt_tokens`, `CHARS_PER_TOKEN_ESTIMATE`) and appends a warning
  (`"context preflight: estimated N tokens vs window W (warn ratio R)"`) to the turn's warnings
  when it crosses `warn_ratio * window` — before generation runs, warn-only, never blocking. Once
  a real usage report arrives, `context_usage_warning` checks actual `prompt_tokens` against the
  same threshold and appends a second, independent warning
  (`"context preflight: actual prompt_tokens N exceeded R * window W"`) if the real count crosses
  it — actual beats estimate, and both can fire independently (e.g. the estimate under-shot).
  (3) *Boundary-aware trimming:* `_truncate_text` (`context_budget.py`) and
  `_truncate_recent_dialogue_message` (`context_builder.py`) now cut at the last word boundary
  within budget instead of mid-word, falling back to the old hard cut only when no boundary exists
  within budget (a single unbroken run of characters at least as long as the budget); never
  exceeds `max_chars` either way. Existing regression fixtures use single-run test strings with no
  embedded spaces near the cut point, so their pinned truncated output is unaffected (verified by
  running the full offline gate). Tests: `tests/unit/test_context_budget.py` (boundary
  cut/exact-fit/no-space-fallback/unchanged-when-short, estimator math, preflight/actual-usage
  warning threshold and disabled-by-default cases), `tests/unit/test_context_builder.py` (same
  boundary cases for dialogue trimming), `tests/unit/orchestration/stages/test_core_stages.py`
  (usage threading, preflight/actual-usage warnings enabled vs. disabled),
  `tests/unit/test_turn_orchestrator.py` (token_usage reflects the repair generation when a repair
  ran, `None` when the actor fails before any generation, diagnostics-round-trip parity),
  `tests/unit/test_repositories.py` (a hand-written pre-#69 `diagnostics_json` payload with no
  `token_usage` key still deserializes), `tests/unit/test_config.py` (new-key defaults and
  overrides). **Live-validated 2026-07-12** (docs/25 Phase B, `26b-mtp`, two 8-turn live-smoke runs):
  sanity run at `MODEL_CONTEXT_WINDOW_TOKENS=1024` → `context_preflight_warning_count=7`,
  `context_actual_warning_count=5` (warning wiring fires against a real model); real run at
  `MODEL_CONTEXT_WINDOW_TOKENS=16384` (matching the profile's `-c 16384`) → 0/0 (prompts
  legitimately under the 85% warn ratio). `grep -i "context shift" raw/llama-server.log`
  empty on both runs — no eviction. Both checkpoints `status: pass`, 0 recall misses.

### Testing

- [x] **#70** *(testing, S)* **The #64 WAL test is single-process; the real CLI+API race is
  untested.** Both #64 tests run two connections in one interpreter — no second OS process
  despite the commit title. The actual exposure is `append_turn`'s read-then-write
  `turn_index = count_turns + 1`, which two concurrent writers can both compute; the
  `UNIQUE(session_id, turn_index)` constraint fails safe (IntegrityError) rather than corrupting,
  but nothing tests or documents that. Either add a true cross-process test + an atomic
  `INSERT … SELECT MAX(turn_index)+1`, or record the single-writer-per-session assumption as a
  documented limit. Low urgency for single-user scope; filed so the "proven cross-process" claim
  isn't over-read. **Shipped:** `SQLiteTurnRepository.append_turn`
  (`app/persistence/repositories.py`) now assigns `turn_index` inside the `INSERT` itself, via a
  correlated subselect (`SELECT COALESCE(MAX(turn_index), 0) + 1 FROM turns WHERE session_id = ?`)
  plus `RETURNING turn_index, id`, instead of a separate `count_turns()` read before the write —
  concurrent writers on one session now serialize on SQLite's write lock and each gets a distinct,
  contiguous index instead of racing to compute the same one. `count_turns` itself is unchanged
  (still used by the CLI, diagnostics, and evals). Added
  `tests/unit/test_sqlite.py::test_append_turn_assigns_contiguous_unique_indices_across_real_processes`,
  which spawns a genuine second OS process (`subprocess` running a `python -c` worker script) that
  races the pytest process to append turns to the same session in the same on-disk database file,
  synchronized via a filesystem ready/go handshake (no fixed sleeps); asserts zero errors on either
  side and a contiguous, duplicate-free `1..N` index sequence afterward. The existing #64 tests
  (`test_wal_reader_is_not_blocked_by_an_open_writer`,
  `test_second_writer_waits_on_busy_timeout_instead_of_locking`) remain as-is — they still validate
  WAL/busy_timeout behavior correctly, just within one process; they were not renamed or reframed.

*Fixed directly with this review (no ID):* stale "Angular 19" references swept to Angular 21
(README, docs/02, docs/09, docs/SIDE_PROJECTS, frontend/README); CHANGELOG gained an
`Unreleased` section covering the post-1.2.0 batch (#48–#64, Angular 21, RAG C1/N1, #60).

### Found while closing the docs/10 coverage gaps (2026-07-11)

- [x] **#71** *(decision, S)* **Resolved 2026-07-12 — (a) 503/504 contract kept and recorded**
  under "Decisions (2026-07-12)" above: transport failure = transient, retryable, no turn row;
  the pinning tests now document intended behavior. Original item:
  **Actor-stage provider-transport failures bypass controlled failure.** `TurnOrchestrator.run_turn` catches only `EmptyProviderResponseError` /
  `TruncatedProviderResponseError` around actor generation; a raw transport error
  (`ProviderTimeoutError`/`ProviderUnavailableError`) propagates uncaught to the API/CLI caller —
  the API maps it to 503/504, but **no CONTROLLED_FAILURE turn is persisted** and the player's
  message is not recorded for that attempt. The same exception from the *critic* stage fails
  closed into a persisted controlled-failure turn (invariant #4), because `TurnCritiqueStage.run`
  catches broadly. The asymmetry is pinned (not fixed) by
  `tests/integration/test_provider_unavailability.py`. Decide: (a) keep the 503/504 contract and
  record it here as intended (transport failure = transient infrastructure error, retryable, so
  no turn row belongs in history), or (b) catch transport errors in `run_turn` and persist a
  controlled-failure turn like the critic path — which changes the API envelope for provider
  outages and needs the SPA's error handling re-checked. Either way the tests document today's
  behavior; flip them with the decision.

### Found by the docs/25 Phase D live runs (2026-07-12)

- [x] **#72** *(correctness, S)* **Lexical write-dedup false-dropped terse distinct facts —
  found live, fixed.** The always-on coverage dedup
  (`app/orchestration/stages/memory_dedup.py` → `is_covered_by_summaries`) dropped any
  candidate whose content terms were ≥50% contained in ONE existing summary. Terse
  extractor phrasings of *distinct* facts cross that bar on frame vocabulary alone: the
  docs/25 Phase D re-run's turn-21 compass gift ("player gave Iria Vale a silver compass to
  keep until his return") hit 0.56 coverage against the compass-free dawn-promise memory
  (shared terms: {player, iria, vale, keep, return}) and was silently dropped —
  `memory_written: False`, later `CheckpointError` at the turn-31 callback. Run-dependent:
  the first Phase D run's longer phrasing (9+ distinct terms) survived the same check.
  **Shipped:** dedicated `WRITE_DEDUP_COVERAGE_THRESHOLD = 0.75` for the write-dedup call
  site (near-verbatim restatements still drop at ~1.0 coverage); the deterministic-fallback
  coverage check keeps the old `COVERAGE_THRESHOLD = 0.5` byte-identical. Both dedup
  warnings now carry snippets of every dropped summary (the drop was previously
  unrecoverable from diagnostics — it took a DB diff against the prior run to find).
  Regression tests pin the compass/dawn pair verbatim. Live validation rides with the
  Phase D re-run (docs/22 P2.2).

- [ ] **#73** *(retrieval, M — this is P1.1/P2.5's live acceptance case)* **Dense-only
  retrieval stops surfacing direct answers at ~50-memory pools.** docs/25 Phase D run 3:
  the blue-seal trust-rule memory ranked 12/17 (top-5 selected) for the callback question
  it directly answers — dense score 0.3221 vs 0.45–0.58 for scene-vocabulary chatter;
  lexical-leg boost (+0.15) insufficient; recency exonerated (identical at 0.0/0.02).
  Reproducible offline from the run's SQLite via `reindex-memories` + `inspect_story_event`
  (docs/22 § P2.2 third-run note has the method and numbers; the run's database is preserved
  at `docs/artifacts/live-validation-D3-2026-07-12.db`, session
  `1a3f65da-73f5-4729-a58d-f058d1d01aac`, target memory `354b8d98`). Candidate model swaps do NOT
  fix it (multilingual-L12 ranks it worse; jina-de fails to load under current onnxruntime —
  also a P1.2 viability note). Acceptance for P1.1 hybrid / P2.5 rerank: this query against
  this pool must place the blue-seal memory in the selected set; then re-run the full
  docs/25 Phase D preset, which is blocked on this item.
  **Update 2026-07-14 ([docs/26](26_memory_retrieval_redesign.md)):** fix path revised — the
  blue-seal instance is a dead `canon_importance_floor` predicate on validated-good
  `build_standing_facts` machinery, not a retrieval ceiling (verified against the preserved D3
  artifact); Lane A canon pinning (**#78**) fixes this instance retrieval-free, Lane B lexical
  slice quotas (**#79**) cover the general non-canon class, and P1.1/P2.5 are demoted to the
  conditional escalation rung ([docs/26 §6 Stage 6](26_memory_retrieval_redesign.md#6-staged-migration-plan)),
  pulled by a post-Stage-5 live miss neither lane covers — not built speculatively. The
  acceptance bar (blue-seal reaches the actor prompt every turn) is expected to be satisfied
  via Lane A's pinning path; that reinterpretation must be recorded explicitly at Stage 5 (#80).

- [x] **#74** *(diagnostics, S)* **Live checkpoint reports a controlled-failure definition
  turn as a recall miss 10 turns later.** docs/25 Phase D run 4: turn 55 (amber_ring_token's
  definition turn) ended `outcome: controlled_failure` (critic rejection, invariant #4 —
  correctly no memory written), and the checkpoint marched on, then failed at the turn-65
  callback with "no persisted matching memory" — a misleading message that cost a forensic
  round to trace back. Fix: when a probe's `definition_turn` ends in `controlled_failure`,
  fail fast at that turn with the real cause named (or record the probe as vacuous and fail
  the run with a distinct message). This is a *precision* fix, not a loosening — the run
  still fails; it just says why. Note the systemic implication recorded in docs/22 § P2.2:
  at the known ~6.7% local fail-closed rate, one of the 8 probe-definition turns is hit in
  a substantial fraction of 100-turn runs, so green P2.2 runs also depend on this
  distinction being visible.
  **Update 2026-07-14 ([docs/26](26_memory_retrieval_redesign.md)):** scoped as Stage 0
  (**#75**, harness fail-fast at the definition turn) plus Stage 1's provenance-OR-phrasing
  probe attribution (**#76**) and a labeled single definition-turn retry whose survival delta
  is *measured* at Stage 5 (**#80**), not assumed from an independence multiplication.
  **Shipped as part of #75's Stage 0** (2026-07-14): the harness fail-fast lives in
  `app/diagnostics/live_checkpoint.py::_validate_definition_turn_outcome`, wired into
  `run_checkpoint` right after each turn is recorded — see **#75** below for the concrete
  mechanism, gating, and tests. The single definition-turn retry
  (`LIVE_DEFINITION_RETRIES`) is out of scope here; it stays Stage 5 harness scope,
  tracked as **#80**.

## Planned 2026-07-14 — docs/26 memory/retrieval redesign (from the Phase D findings)

The four docs/25 Phase D 100-turn runs (2026-07-12/13) produced the failure dossier behind
#72–#74; [docs/26](26_memory_retrieval_redesign.md) synthesizes four independently-designed,
adversarially-judged redesigns into one staged plan: **two guarantee lanes over the existing
pipeline** (no new table, store, LLM call site, or vector-store feature), a shared provenance
substrate, and a corrected measurement layer. Write-dedup (0.75/0.5 + reversal markers) and the
additive boost rerank stay untouched by decision — every proposed rider on them was
judge-broken (docs/26 §3.5, §7). Ship stages in order; each is additive, default-off where it
changes runtime behavior, and closes with the full deterministic gate before any live run.
Owner answers recorded 2026-07-14 in
[docs/26 §8](26_memory_retrieval_redesign.md#8-open-questions-for-the-owner): Q2 wait /
Q3 keep-both / Q4 yes-both / Q5 English-first (N2 stays deferred). **Q1 (continuity-contract
framing) is still open — separate discussion requested; it gates #78.** Q6 was resolved
2026-07-14 by the [docs/27](27_world_chronicle_design.md) design conversation — see the
world-chronicle section below (#81–#83).

- [x] **#75** *(instrumentation, M)* **Stage 0 — instruments first.** New
  `memory_write_lifecycle` regression category (scripted transcript through the real extractor
  + dedup + task-aware fake curator), the #74 fail-fast (a probe definition turn ending in
  `controlled_failure` fails immediately, named, instead of surfacing as a recall miss 10 turns
  later), and the offline D3-artifact replay script (**read-only / copy-first** — untracked
  `-shm`/`-wal` sidecars already exist from a prior read-write open). Closes #74's harness
  half; lands the offline write-path benchmark docs/22's gap analysis calls for.
  [docs/26 §6 Stage 0](26_memory_retrieval_redesign.md#6-staged-migration-plan). ~1.5–2 days.
  **Shipped:** three pieces, zero runtime-engine behavior changed.
  (1) `app/evals/memory_write_lifecycle.py` (mirrored in
  `tests/evals/test_memory_write_lifecycle.py`, wired into `run_regressions()`) drives the
  REAL `TurnMemoryStage` — real deterministic extractor, real write-dedup — with only the
  curator's LLM call faked (task-aware, marker-keyed, mirrors
  `memory_continuity._TaskAwareFakeProvider`). Asserts a player-stated promise/deadline fact
  survives extraction+dedup into a persisted candidate set, and replays the #72 compass/dawn
  pair verbatim at the `TurnMemoryStage` integration level (not just the pure-function level
  `test_memory_dedup.py`/`test_deterministic_extractor.py` already cover): the distinct terse
  compass-gift candidate survives at the 0.75 write-dedup threshold, a near-verbatim
  restatement drops, and the drop warning names the dropped summary. Stage 2/3 assertions
  (tag-presence post-fold, standing-facts inclusion) are left as marked, unimplemented
  comments. (2) `app/diagnostics/live_checkpoint.py::_validate_definition_turn_outcome`:
  `run_checkpoint` now checks, right after each turn is recorded, whether that turn was a
  probe's `definition_turn` and its outcome was not `success`; if so it raises immediately —
  `"definition turn N for event <key> ended in controlled_failure: fact never entered the
  world — probe vacuous"` — instead of deferring to the callback turn's generic "no persisted
  matching memory". Gated on `fail_on_structured_warnings` (strict mode only): the check it
  preempts (`_validate_attribution`'s `matching_memory_ids` requirement) only ever raised in
  strict mode, so this relocates and renames an existing failure rather than introducing one
  in report-only runs — precision, not a loosening. 4 new tests in
  `tests/unit/test_live_checkpoint.py` (named-cause message, non-strict no-op, non-definition
  turn no-op, early-termination/auditability via progress snapshots) using the existing fake
  `event_inspector`/`inspector`/`client_factory` seam. (3) `app/diagnostics/replay_selection.py`
  (113 lines incl. docstrings/CLI — over the ~30–60 line guidance; a documented
  `immutable=1` finding plus a conventional argparse CLI earned the extra length): opens a
  preserved artifact via a `mode=ro&immutable=1`
  SQLite URI — `mode=ro` alone still lets SQLite create `-shm`/`-wal` sidecars for a WAL-mode
  DB on open, verified empirically against the real artifact before landing on `immutable=1`
  — and reports each PLAYER-visible memory's importance/tags/`CANON_TAGS` intersection/
  eligibility under today's `build_standing_facts` predicate, plus a summary line. Run against
  the real D3 artifact (`docs/artifacts/live-validation-D3-2026-07-12.db`, session
  `1a3f65da…`): **47** player-visible memories (48 total rows in that session; the 48th is
  `gm`-visibility, not `player` — docs/26 §3.3's "48-memory pool" figure is the unfiltered row
  count, not the player-visible one), importance distribution `8×1/38×2/1×3`, 3 tag-eligible
  (`354b8d98` blue-seal among them, `tags=["rule", ...]` as documented), **0** eligible at
  `importance_floor=4` — confirms docs/26 §3.3 exactly, modulo that one-row headline-count
  precision correction. `tests/unit/test_replay_selection.py` covers the eligibility predicate
  against a synthetic temp DB (never opens the real artifact) plus a dedicated test proving no
  sidecar files appear. `git status` clean under `docs/artifacts/` after every run. Full gate
  green (`ruff`/`mypy --strict`/`pytest`/`regression_runner`); regression runner now 91 checks
  (up from 85).
- [x] **#76** *(provenance, M)* **Stage 1 — `memory_episodes.source_turn_id`.** Fifth
  idempotent `_ensure_column` migration + optional `turn_id` threaded through the
  three-signature / two-call-site chain (inline + deferred) + consolidation carry-forward
  (`min()` over the non-null subset of folded originals; `None` if all legacy — never a
  sentinel) + provenance-OR-phrasing probe attribution. Attribution-only: explicitly NOT a
  fact-identity or dedup key (the quote/`fact_key` variant was judge-broken).
  [docs/26 §3.1](26_memory_retrieval_redesign.md#31-provenance-substrate--memory_episodessource_turn_id). ~1.5–2 days.
  **Shipped:** `memory_episodes.source_turn_id INTEGER` (nullable, fifth `_ensure_column`
  instance); `MemoryEpisode`/`MemoryCandidate` gain `source_turn_id: int | None = None` (old
  rows and pre-stamp candidates deserialize as `None`). `TurnMemoryStage.run` →
  `_run_extraction` → `_persist_and_index` thread an optional `turn_id` kwarg; the stamp is
  applied once, centrally, in `_persist_and_index` — the single choke point all three
  candidate producers (curator, deterministic extractor, curator-failure fallback) funnel
  through before `persist_memories` — so every producer is covered without repeating the
  assignment at each origin. Both orchestrator call sites pass it: the inline site
  (`persistence.turn.id`, `turn_orchestrator.py`) and `run_deferred_memory`
  (`DeferredMemoryJob.turn_id`). `MemoryConsolidator._persist_consolidation` carries
  provenance forward per §3.1.1 (`min()` over the non-null subset of folded originals' own
  `source_turn_id`; stays `None`, never a sentinel, when every original is legacy) and
  appends a `source_ids:<comma-joined-original-episode-ids>` audit tag (reuses `tags_json`,
  no new schema). Live-checkpoint attribution
  (`app/diagnostics/live_checkpoint.py::build_event_attribution`) is now provenance-OR-
  phrasing, gated on a `definition_turn_id` resolved via a direct SQLite lookup
  (`SQLiteTurnRepository.list_all_turns` inside `inspect_story_event`, matched on
  `turn_index` — `turns.id` is an internal PK the HTTP API never exposes, so this stays a
  DB-side lookup rather than adding a new API surface); the multi-memory-definition-turn
  over-attribution risk §4 flags is implemented as specified (never provenance-only) and
  pinned by a dedicated test, not silently tightened. Tests: repository round-trip +
  legacy-row-reads-`None` (`tests/unit/test_repositories.py`); inline/deferred call-site
  provenance extended onto the existing orchestrator memory tests
  (`tests/unit/test_turn_orchestrator.py`); three consolidation carry-forward cases —
  all-provenanced, mixed-NULL, all-NULL — the last folded into the pre-existing threshold
  test alongside a new audit-tag assertion
  (`tests/unit/orchestration/stages/test_core_stages.py`); provenance-OR-phrasing incl. the
  multi-memory-turn residual (`tests/unit/test_live_checkpoint.py`). No Settings/
  `.env.example`, ranking, retrieval, or write-dedup changes. Gate green (ruff, mypy
  --strict, 801 pytest, 91-check regression runner); `docs/artifacts/` untouched.
- [x] **#77** *(correctness, S)* **Stage 2 — best-match tag/importance fold.** At the
  curator-coverage-drop site (`stages/memory.py:169-177`), fold a covered deterministic
  candidate's tags/importance onto the **best-matching** (argmax coverage, not first-match)
  curated summary instead of silently discarding them — any player-first-person-stated durable
  event stays canon-taggable and clears the importance floor even when the curator forgets the
  tag. [docs/26 §3.2](26_memory_retrieval_redesign.md#32-deterministic-tagimportance-fold-best-match-not-first-match).
  **Shipped:** two pure helpers in `app/orchestration/stages/memory_dedup.py` —
  `best_covering_summary` (argmax-coverage variant of `is_covered_by_summaries`, itself
  unmodified: returns the index/score of the highest-coverage curated summary among those
  clearing the existing 0.5 `COVERAGE_THRESHOLD`, deterministic tie-break to the **lowest
  index** on an exact-score tie) and `ordered_union` (order-preserving, dedup'd tag union) —
  wired into the coverage-drop loop in `memory.py::_run_extraction`, which now folds
  (`curated[idx].model_copy(update={"tags": ordered_union(...), "importance":
  max(...)})`) instead of dropping, appending an audited `"deterministic candidate folded
  (best-match, coverage=X.XX): <summary[:80]>"` warning per fold. No coverage → unchanged
  `extras.append(candidate)` path, byte-identical. The fold runs before `_persist_and_index`,
  so #76's `source_turn_id` stamping composes untouched; `is_covered_by_summaries` itself, the
  0.75/0.5 thresholds, reversal markers, and framing-strip are byte-identical (the new helper
  reuses `deterministic_extractor`'s private `_strip_framing`/`_reversal_markers`, not a
  reimplementation, so the "does X cover Y" decision cannot silently drift from
  `is_covered_by_summaries`'s own — pinned by an explicit agreement test). Tests: 6
  pure-function unit tests (`tests/unit/orchestration/stages/test_memory_dedup.py`) covering
  argmax-vs-first-match, the lowest-index tie-break, the no-coverage/None case, agreement with
  `is_covered_by_summaries`, and the reversal-marker escape; 3 `TurnMemoryStage`-level tests
  with a scripted curator (`tests/unit/orchestration/stages/test_core_stages.py`) — the pinned
  #72 dawn-promise string folding a real entrust-triggered candidate (coverage=0.50), the
  judge-constructed argmax-vs-first-match scenario (two curated summaries both clear 0.5 for
  one candidate at different scores, listed low-score-first; **verified** by temporarily
  reverting `best_covering_summary` to first-match that this exact test fails, then reverting
  — diff-clean), and the no-coverage extras-path-unchanged case. Extended
  `app/evals/memory_write_lifecycle.py`'s marked Stage 2 extension point with a fifth scripted
  turn (a real deterministic trigger plus a curator paraphrase that forgets the tag and
  under-scores importance) and two new checks
  (`deterministic_tag_and_importance_fold_onto_curated_summary`, `fold_is_auditable`), pinned
  in `tests/evals/test_memory_write_lifecycle.py`. Also fixed 3 pre-existing
  `tests/unit/test_turn_orchestrator.py` tests whose shared fixture (a "return before dawn"
  promise message plus a curator summary already about returning before dawn) incidentally
  cleared the coverage threshold (0.67): their exact `warnings ==` assertions now include the
  new fold-audit warning that this previously-silent drop now emits — a real, expected
  behavior change (a tag that used to vanish with zero trace is now visible), not a test
  weakening. No Settings/`.env.example`, ranking, retrieval, threshold, or write-dedup
  changes. Gate green (ruff, mypy --strict, 810 pytest, 93-check regression runner);
  `docs/artifacts/` untouched.
- [ ] **#78** *(retrieval, M — fixes #73's canon-tagged instance)* **Stage 3 / Lane A —
  tag-eligible canon pinning.** `CANON_TAG_PINNING=false` (byte-identical default) widens
  `build_standing_facts` eligibility to CANON_TAG-carrying sub-floor memories — the blue-seal
  case is a dead importance-floor predicate (D3 curator distribution: 9×1 / 38×2 / 1×3, zero
  at the floor of 4), not a retrieval ceiling. Ships with the flag-gated stale-fact
  supersession safeguard (§3.3.1), German tag aliases (mirrored into `_PRESERVE_TAGS`), and
  `standing_facts_count`/`_chars` diagnostics. Offline-checkable today via #75's replay script
  against D3. Q3 confirmed 2026-07-14 (keep-both); still gated on Q1.
  [docs/26 §3.3](26_memory_retrieval_redesign.md#33-lane-a--tag-eligible-canon-pinning-retrieval-free-guarantee). ~1–1.5 days.
- [ ] **#79** *(retrieval, M/L — the general #73 fix)* **Stage 4 / Lane B — lexical slice
  quotas.** Session-pool-IDF lexical scorer (`app/rag/lexical.py`, pure, reuses
  `content_terms()`) + reserved slots reordered on the **final** top-5 selection window (not
  the ~13-item rerank window — the load-bearing placement detail), `RAG_SLICE_LEXICAL_QUOTA=0`
  default with a quotas=0 byte-identity golden test. All four judge fixes baked in from the
  start: `CONSOLIDATED_TAG` exclusion, `min_slice_score` shipped unset (measured at Stage 5,
  not guessed), dedicated `slice_score` field (preserves the boost-identity invariant),
  pre-retriever fail-open ordering; plus the named confidence-gating test. Offline-falsifiable
  against D3 (blue-seal query terms hit 4/48 pool memories) before any live run.
  [docs/26 §3.4](26_memory_retrieval_redesign.md#34-lane-b--lexical-slice-quotas-retrieval-time-guarantee-general-case). ~2–3 days.
- [ ] **#80** *(validation, M — owner's machine; closes docs/22 P2.2)* **Stage 5 — live
  validation + default flip.** At least **two** full docs/25 Phase D 100-turn runs (MTP
  non-determinism makes one green run weak evidence) with `CANON_TAG_PINNING=on`,
  `RAG_SLICE_LEXICAL_QUOTA=2`, `LIVE_DEFINITION_RETRIES=1` (harness-local scenario semantics,
  not a Settings field; Q4 confirmed 2026-07-14). Measure — don't assume — the definition-retry survival delta; derive
  `min_slice_score` from observed IDF distributions; watch pinned-block context-preflight
  pressure. Flip defaults only if both runs are clean, and record the #73 acceptance
  reinterpretation explicitly in docs/22. P1.1 hybrid stays the conditional escalation rung
  ([docs/26 §6 Stage 6](26_memory_retrieval_redesign.md#6-staged-migration-plan)) — pulled by
  a live miss neither lane covers, never pushed. 1–2 elapsed days.

## Planned 2026-07-14 — docs/27 world chronicle (Q6 design conversation)

Resolves docs/26 §8 Q6 and engages the deferred Milestone 4 decision.
[docs/27](27_world_chronicle_design.md) records the decided game model (owner design
conversation, two structured rounds): a persistent world — new hero, same world — with an
**automatic boundary chronicle** (batch promotion at session end, ordered supersession, no
live supersession), **world-scoped persona memory**, **tag-based carry-over** (commitments +
defining moments, replacing the dead importance floor), **NPC-held default visibility**
(public only via explicit consequence tags), automatic-plus-editable authorial control, and
**no session-start recap** (discover in play; recap explicitly rejected). A same-day
addendum ([docs/27 §3.5](27_world_chronicle_design.md#35-living-world-layer-rumors-same-day-addendum-decisions-912))
adds the **living-world rumor layer**: derived-with-distortion hearsay, generated
mid-session on the bound provider (background, flag-gated), boundary-reconciled — the
dedicated-small-model variant assessed and parked (§3.5.3). Build is evidence-gated:
strictly after docs/26 Stages 0–5 (#75–#80), pulled by a real second-campaign need,
probe first.

- [ ] **#81** *(instrumentation, S — chronicle C0)* **P2.4 two-session world probe.**
  Unchanged from [docs/22 § P2.4](22_rag_scaling_roadmap.md#p24-world-scoped-durable-memory-engages-the-deferred-milestone-4-decision):
  session A establishes facts, session B starts in the same world, measure what B recalls.
  Expected today: only importance-4 commitments via a shared persona — that baseline is what
  #83 is later measured against. [docs/27 §5](27_world_chronicle_design.md#5-staging-and-gates-backlog-8183).
- [ ] **#82** *(correctness, S — chronicle C1)* **World-scoped persona memory.** Add
  `world_id` to persona-memory chunk payloads + the retrieval filter (today's persona-only
  filter leaks NPC memories across worlds); rebuildable via `reindex-memories`;
  InMemoryVectorStore parity test per convention. Independent of #83 — fixes the leak
  regardless. [docs/27 §2](27_world_chronicle_design.md#2-current-state-verified-2026-07-14-against-source).
- [ ] **#83** *(feature, M/L — chronicle C2)* **Chronicle v1.** `world_facts` SQLite table
  (authoritative, Milestone 4's home) + boundary promotion pass (deterministic selection
  over tagged rows, source provenance, no new LLM call site) + tag-based persona selection
  replacing `PERSONA_MEMORY_IMPORTANCE_FLOOR` + audit CLI/API (list/edit/delete) +
  diagnostics. Live acceptance: the #81 probe re-run shows session B recalling the promoted
  set and nothing it shouldn't. [docs/27 §3](27_world_chronicle_design.md#3-target-design).
- [ ] **#84** *(feature, M — chronicle C3, depends on #83)* **Living-world rumor layer.**
  Player-visible hearsay rows derived (distortion allowed) from real memories/facts,
  generated mid-session by a cadence-driven background pass on the **session's bound
  provider** (flag-gated, default off, deferred-job path — no turn-latency cost), with a
  deterministic anchor-validation check (fail-open: unanchored rumor → skipped + warning),
  hearsay-labeled prompt framing via `CANON_LORE`/`source_type="world_rumor"`, and
  status reconciliation (confirmed/debunked/faded) at chronicle boundaries only — no
  spread simulation. Dedicated small rumor-model idea parked with a named trigger.
  [docs/27 §3.5](27_world_chronicle_design.md#35-living-world-layer-rumors-same-day-addendum-decisions-912).

## Not doing (personal-use scope)

StageGraph/DAG/plugin extensibility · hard memory-episode cap default (regressed recall) ·
auth/multi-user/streaming/tracing · corpus-scale micro-opts.

## Side projects

Full tiered list with effort + dependencies: [SIDE_PROJECTS.md](SIDE_PROJECTS.md) — the SPA,
RAG inspector, analytics, and eval dashboard entries shipped in 1.1.0.
Best next: ★ Transcript Exporter (weekend, zero backend).
