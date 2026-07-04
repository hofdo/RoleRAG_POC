# RoleRAG POC — Working Backlog

Source: 10-agent deep analysis (47 improvements + side projects). This file is the durable
record — git commit subjects tag shipped items as `(#N)`. Keep it in sync as items land.

Numbers not listed here (#3–5, #23, #31–47) landed in the "Not doing" categories or the
side-project list rather than the engine backlog; only the items acted on carry a row below.

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

## Not doing (personal-use scope)

StageGraph/DAG/plugin extensibility · hard memory-episode cap default (regressed recall) ·
auth/multi-user/streaming/tracing · corpus-scale micro-opts.

## Side projects

Full tiered list with effort + dependencies: [SIDE_PROJECTS.md](SIDE_PROJECTS.md) — the SPA,
RAG inspector, analytics, and eval dashboard entries shipped in 1.1.0.
Best next: ★ Transcript Exporter (weekend, zero backend).
