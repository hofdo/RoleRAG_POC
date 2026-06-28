# RoleRAG POC — Working Backlog

Source: 10-agent deep analysis (47 improvements + side projects). This file is the durable
record — git commit subjects tag shipped items as `(#N)`. Keep it in sync as items land.

## Done

#1 #2 (early) · #7 redaction-ordering assertion · #8 persist+read turn diagnostics ·
#10 embedding-ab harness → **declined** (models tie, no swap) · #11 retrieval-miss eval ·
#12 TurnOrchestratorConfig · #13 redact raw_text in failure logging · #14 embedding-provider
failure tests · #20 repair decision → TurnRepairStage · #21 split TurnMemoryStage ·
#25 containment_overlap_threshold doc+tests · #30 skip semantic-dedup embed when threshold==1.0 ·
QE follow-ups (503 OpenAPI, C2/C3 named cap tests).

## In progress — B tier (loop: feat/b-tier)

Order: value + independence, decisions last. Each item gate-verified
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

## Not doing (personal-use scope)

StageGraph/DAG/plugin extensibility · hard memory-episode cap default (regressed recall) ·
auth/multi-user/streaming/tracing · corpus-scale micro-opts.

## Side projects (none built)

★ Transcript Exporter (weekend, zero backend) → Memory Graph / RAG Inspector (needs #8) →
voice I/O → Discord bot → analytics dashboard. Deferred: authoring studio, branching/replay, SPA.
