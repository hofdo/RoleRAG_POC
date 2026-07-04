# 15 — 1.0 Acceptance Report

> **Historical record** — the 1.0 acceptance baseline. Point-in-time; not kept in sync with the
> code. For current state see [docs/README.md](README.md).

Date: 2026-06-12

## Scope

This report freezes the 1.0 baseline. It follows the 0.1.0 MVP acceptance baseline in
`docs/11_mvp_acceptance_report.md` and closes the roadmap defined after the June 8 live
model quality assessment (`docs/13_live_model_quality_assessment.md`).

## Delivered for 1.0

1. **Per-stage timing capture** — every turn response carries `stage_timings`
   (session, retrieval, routing, generation, validation, critique, repair when run,
   persistence, memory). Exposed in the JSON API, SSE frames, the play-UI developer
   panel, and live checkpoint reports (`stage_latency_means`).
2. **Deterministic draft validation** — `app/orchestration/draft_validator.py` flags
   unsupported named entities (vs the player-visible vocabulary) and unaddressed direct
   player actions. Flags are report-only warnings that also force the existing repair
   loop; if a validator-only repair fails, the original draft is returned instead of a
   controlled failure.
3. **Conditional critic/curator gating** — `CRITIC_GATING` / `CURATOR_GATING`
   (`always|auto`). In `auto`, the critic runs only on risk signals (validator flags,
   retrieval confidence below 0.45, scene complexity >= 4, cloud route) and the LLM
   curator runs only when durable-event signals exist. The deterministic promise
   extractor always runs.
4. **`CLOUD_MODE=ask` confirmation flow** — two-phase turns. A confirmation-required
   cloud route returns `status: "confirmation_required"` (or a `confirmation_required`
   SSE frame) before any generation or persistence. Approve with `cloud_confirmed`,
   decline with `force_local` (route reason `user declined cloud`). Play UI prompts via
   dialog; CLI prompts interactively or accepts `--confirm-cloud` / `--force-local`.
   *Removed 2026-07-02: the provider is now bound at session creation and per-turn
   confirmation no longer exists; `CLOUD_MODE=ask` confirms once when a cloud session
   is created. See [docs/README.md](README.md) "Session-bound provider routing" and
   [docs/06](06_local_cloud_model_strategy.md).*
5. **Session management CLI** — `list-sessions`, `delete-session` (SQLite cascade +
   Qdrant session points), `export-session` / `import-session` (format_version 1
   envelope, `--new-id` remap), `inspect-memories`, `reset-db`.
6. **Memory viewer** — `GET /sessions/{id}/memories`, a read-only Memories panel in
   `/play`, and `inspect-memories` in the CLI.
7. **One-command startup** — `scripts/dev-up.sh` (Qdrant + llama-server + uvicorn with
   health gates, reuses already-running services) and `scripts/dev-down.sh` (stops only
   what dev-up started).

## Structured-output reliability fixes (found during 1.0 live validation)

- The bundled small-model chat template hardcoded `enable_thinking = true`; thought-channel
  output silently consumed the entire structured token budget, producing empty
  grammar-constrained responses. Fixed with a patched template
  (`scripts/templates/small-model.jinja`) wired into the small model profile.
- llama.cpp's json_schema grammar conversion does not resolve `$ref`/`$defs`; nested
  curator candidates were unconstrained (wrong keys, invalid enums). Fixed by inlining
  refs before dispatch (`inline_schema_refs` in `app/llm/structured_output.py`).
- A grammar-constrained model can claim `write_memory=true` with no candidates; this is
  now coerced to a decline instead of failing the whole curation.

## Live verification

Strict mode (`LIVE_FAIL_ON_STRUCTURED_WARNINGS=1`) on Apple Silicon with the small model
profile (gemma-4-E4B Q8_0):

### 8-turn baseline (gating `always`, 2026-06-11)

- checkpoint: **pass**, zero structured-output failures (critic, curation, indexing)
- latency: p50 47.6 s, p95 89.7 s, total 442.3 s (mean ~55 s/turn)
- stage means: generation 19.3 s, memory 16.2 s, critique 4.7 s, repair 23.7 s (when run)
- all finish reasons `stop`

### 12-turn acceptance (gating `auto`)

**Deferred.** Three attempts on 2026-06-12 failed before the checkpoint with provider
timeouts caused by host memory exhaustion (13 GB of 14 GB swap in use; local decode
collapsed from 15.6 to 0.13 tokens/s). This is a machine-state blocker, not an
application defect — the 8-turn strict baseline passed cleanly hours earlier on the
same build lineage.

Consequences:

- gating defaults ship as `always` (current behavior); flip `CRITIC_GATING` /
  `CURATOR_GATING` to `auto` after a clean 12-turn strict run shows the latency win
  with no lost durable events.
  *Resolved later: auto-gating regressed 50-turn recall in live acceptance; the
  `always` defaults are intentional (see [docs/05_rag_memory_design.md](05_rag_memory_design.md))
  and the addendum below.*
- rerun on a host with free RAM:
  `CRITIC_GATING=auto CURATOR_GATING=auto LIVE_TURN_COUNT=12 bash scripts/live-smoke.sh`
- expected from the 8-turn stage means: gating `auto` skips ~4.7 s critique + ~16.2 s
  curation on low-risk turns, projecting well past the 30 % latency target

## Verification suite at 1.0

- pytest: 370 passed
- frontend (`node --test`): 36 passed
- regression runner: 60 checks pass (includes new `draft_validation` category)
- ruff, mypy: clean

## Known limitations carried into 1.0

- personal-use single-user only; no auth
- buffered SSE only; no provider token streaming
- Qdrant remains a derived, repairable index; `reindex-memories` after restores
- gating thresholds are first-pass; tune from live `stage_timings` evidence
- live throughput depends on host memory pressure: heavy swap can collapse local
  decode speed by two orders of magnitude (observed 15.6 → 0.13 tokens/s); run
  `scripts/dev-up.sh` on a machine with free RAM for the model

## Version

`pyproject.toml` and `app.__version__` set to `1.0.0`.

## Addendum (2026-07-04): post-1.0 live acceptance outcomes

The 12-turn acceptance deferred above was rerun as a live acceptance campaign on
2026-06-12/13, closing the loop this report left open. Outcomes from that campaign
and the follow-up live runs, all shipped as defaults since:

- **26B profile required** — the small profile's curator paraphrased memories lossily
  (it dropped the seeded before-dawn commitment at 50-turn scope); the 26B profile
  (`LOCAL_MODEL_PROFILE=26b`) is the supported model for long sessions.
- **`LOCAL_LLM_MAX_RETRIES=1`** — with the old default of 0, a single transient
  client-to-server stall became a session-killing 504; one bounded retry is now the
  default.
- **`LOCAL_STRUCTURED_MAX_TOKENS` raised 350 → 640** — verbose models' critic JSON
  truncated mid-string at the 350-token cap; 640 eliminated the parse failures.
- **`LOCAL_LLM_TIMEOUT_SECONDS=300`** — late-session 26B calls legitimately run long;
  tighter caps cancelled real work.
- **Dual-query recall verified** — retrieval was rebuilt as a dual-query pass (context
  blob plus bare user message) after query-blob dilution buried indirect callbacks; a
  50-turn run then validated all five seeded story-event gates (turns 13/31/38/45/50).
  See [docs/05_rag_memory_design.md](05_rag_memory_design.md).
- **Auto-gating rejected** — `CURATOR_GATING=auto` missed a rule-phrased durable event
  at turn 38 (its heuristic matches commitment verbs only), regressing 50-turn recall;
  `CRITIC_GATING` / `CURATOR_GATING` stay `always` deliberately.
