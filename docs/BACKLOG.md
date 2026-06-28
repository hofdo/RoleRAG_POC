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
- [ ] **#19** structured `TurnResult.errors` — open (contract-breaking + YAGNI; user left open)
- [ ] **#29** `session_memory_max_episodes` default — open (data says a hard cap regresses recall;
  recommend doc-only, user left open)

## Open — A tier (quick wins, not in this loop)

#18 narrow broad `except Exception` · #24 validate gating strings at construction ·
#26 roleplay-aware stopwords · #27 CLI color/icons · #28 extract test-scenarios module.

## Not doing (personal-use scope)

StageGraph/DAG/plugin extensibility · hard memory-episode cap default (regressed recall) ·
auth/multi-user/streaming/tracing · corpus-scale micro-opts.

## Side projects (none built)

★ Transcript Exporter (weekend, zero backend) → Memory Graph / RAG Inspector (needs #8) →
voice I/O → Discord bot → analytics dashboard. Deferred: authoring studio, branching/replay, SPA.
