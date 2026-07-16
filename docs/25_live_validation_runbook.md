# 25 — Live Validation Runbook

> Reviewed: 2026-07-14 @ 3d5e18f

How to clear the four live-validation caveats that are shipped, gate-green, and byte-identical
by default, but still missing the one kind of evidence the deterministic offline gate cannot
produce: a real `llama-server` + Qdrant run. Chains
[docs/BACKLOG.md](BACKLOG.md) **#48**/**#67** (CLI/API composition parity),
**#69** (context-accounting preflight), **#6** (recency boost), and
[docs/22 § P2.2](22_rag_scaling_roadmap.md#p22-long-campaign-preset-enable-the-shipped-but-off-machinery-with-evidence)
(the long-campaign preset, which also exercises § C2's min-age/batch-cap consolidation knobs)
into one sitting, cheapest first.

## Why

All four items were shipped additive, opt-in, and byte-identical-by-default, and passed the
full deterministic gate (`ruff`/`mypy`/`pytest`/`regression_runner`) — per
[CLAUDE.md](../CLAUDE.md)'s measure-first convention, that gate proves the plumbing but cannot
prove live behavior: it uses fake providers, keyword embeddings, and `InMemoryVectorStore` by
design. Each item's own acceptance criterion explicitly calls for a live run the authoring
environment doesn't have (no `llama-server`, no Qdrant). This runbook is for the machine that
does — the owner's — and it chains all four passes into one sitting instead of four separate
ones, ordered so a cheap 8-turn run surfaces an infrastructure problem (wrong model alias, port
clash, missing profile model) before the expensive ~100-turn run at the end.

This runbook only prescribes what to run and what to read; it does not lower the bar to pass.
See [If something fails](#if-something-fails).

## Prerequisites

- **`llama-server` + a profile model.** See the root README's
  [Local Model Setup](../README.md#local-model-setup) section for installing `llama.cpp` and
  fetching a model. `LOCAL_MODEL_PROFILE=26b` (`-c 8192`) is the recommended acceptance model;
  `26b-mtp` (`-c 16384`, local GGUF files only, faster decode) is a good alternative if you
  have the files — both are defined in
  [scripts/lib/local-model-profile.sh](../scripts/lib/local-model-profile.sh).
- **Docker**, for the disposable Qdrant container `scripts/live-smoke.sh` starts and tears
  down itself on every run.
- **The repo venv and `npm ci`** (`python -m pip install -e ".[dev]"`) — the script's
  Playwright UI smoke needs the built SPA and a browser; set `LIVE_SKIP_BROWSER=1` to skip it
  for a faster loop through phases A–C.

**Approximate wall-clock.** These are not all directly measured; where a real measurement
exists it's cited, otherwise the estimate is derived from docs/16's per-turn figures and
flagged as such.

| Phase | Turns | Approx wall-clock |
|---|---|---|
| A — baseline | 8 | A few minutes once the model and Qdrant image are already local. Most of it is fixed startup (model load, Qdrant pull, Playwright browser), not the 8 turns themselves — docs/16's slower (non-MTP) per-turn mean of 59.4s puts 8 turns of generation alone at **≈8 min** (derived, not directly measured at 8 turns). |
| B — context accounting | two 8-turn runs | Roughly twice phase A. |
| C — recency | one or two 8-turn runs | Roughly phase A, ×1 or ×2. |
| D — long-campaign preset | 100 | **~86.5–99 min measured** (docs/16: 86.5 min `26b-mtp`/draft-on, 99.0 min `26b`/draft-off, same 100-turn Rose Gallery scenario). That run was on the author's Apple Silicon hardware and flagged as single-run/not-thermally-controlled (docs/16 Caveats) — treat it as a ballpark, not a guarantee, and budget the better part of two hours including setup. |

Run phases A→D in order; each is strictly cheaper than the next, so a broken model alias or a
Qdrant port clash costs minutes, not the better part of Phase D's two hours.

## Phase A — baseline pass (#48, #67)

```bash
bash scripts/live-smoke.sh
```

No special environment needed — this is the plain acceptance run both items ask for. Set
`LOCAL_MODEL_PROFILE=26b` (or `26b-mtp`) first if you don't want the `small` default profile
live-smoke otherwise falls back to.

**What this actually exercises, precisely.** `#48`/`#67` fixed `cli._build_services` to
delegate outright to `composition.build_services`/`build_orchestrator_config` — the same
functions the API composition root uses — so the two roots can no longer drift on config or
collaborators (`canon_repository`, `structured_failure_sink`, `memory_embedding_provider`).
None of live-smoke.sh's own CLI steps exercise that fixed path directly: `doctor` and
`smoke-run` call separate diagnostic modules (`run_doctor`/`run_smoke`), and `ingest` uses its
own direct builder aliases (`_build_embedding_provider`/`_build_vector_store`) that #67 left
untouched — none of the three ever calls `_build_services`. The N-turn conversation checkpoint
drives the API's HTTP surface (`POST /sessions/{id}/turns`) instead, which calls
`composition.build_services` directly (not through the CLI at all). So a clean live-smoke run
is evidence that the *shared* function both roots now call behaves correctly against a real
model — it does not literally invoke `python -m app.cli turn`/`start-session`. For the CLI's
own commands exercised end to end, run the manual flow from the README's
[Runtime Verification](../README.md#runtime-verification) section afterward against your own
`docker compose up -d qdrant` + local provider (live-smoke tears down what it started on exit,
so that's a separate stack, not a continuation of the live-smoke run) — this is the only way to
directly observe `cli._build_services`' delegation rather than infer it from the shared
function.

**PASS** = the script exits `0` and `${LIVE_ARTIFACT_DIR:-/tmp/rolerag-live-test}/report.md`
lists every step as `- PASS ...` with none `FAIL` — `set -Eeuo pipefail` aborts the script at
the first `FAIL`, so a report that stops partway through is itself a fail signal.

**On success**, edit [docs/BACKLOG.md](BACKLOG.md):

- **#48** — replace *"Still worth a live-smoke pass (it changes the CLI structured-token
  budget)."* with a dated confirmation, e.g. *"Live-smoke validated YYYY-MM-DD (docs/25 Phase
  A): CLI structured-token budget change confirmed under a real model + Qdrant."*
- **#67** — replace *"Live-smoke caveat, same as #48: this changes real CLI turn behavior
  (canon facts now injected, structured failures now logged, semantic dedup now reachable) —
  a live `bash scripts/live-smoke.sh` pass is advisable before relying on it in a real
  session, since the deterministic gate can't exercise real canon-fact retrieval quality or a
  live embedding backend."* with a dated confirmation, e.g. *"Live-smoke validated YYYY-MM-DD
  (docs/25 Phase A): canon injection, structured-failure logging, and semantic-dedup
  reachability confirmed under a real model + Qdrant."*

## Phase B — context accounting (#69)

Two runs, same profile as Phase A; only `MODEL_CONTEXT_WINDOW_TOKENS` changes.
`MODEL_CONTEXT_WINDOW_TOKENS` is an **app-side accounting ceiling**
([app/config.py](../app/config.py)) — it never touches the real `-c` size llama-server is
launched with, so neither run changes what the model can actually hold; both only change when
the app's own preflight/actual-usage warnings fire.

**Sanity run** — deliberately far too small, to prove the plumbing fires at all:

```bash
MODEL_CONTEXT_WINDOW_TOKENS=1024 bash scripts/live-smoke.sh
```

Read `${LIVE_ARTIFACT_DIR:-/tmp/rolerag-live-test}/raw/conversation-checkpoint.json` →
`quality_metrics.context_preflight_warning_count`. **It must be `> 0`.** Docs/22 § P0.1
estimates the actor prompt at several thousand tokens even for this demo scenario — far past
`0.85 × 1024` — so zero here means the warning wiring is broken, not that the prompt was small.

**Real run** — set the ceiling to what the model actually has:

```bash
MODEL_CONTEXT_WINDOW_TOKENS=8192 bash scripts/live-smoke.sh    # small/26b profile (-c 8192)
# MODEL_CONTEXT_WINDOW_TOKENS=16384 bash scripts/live-smoke.sh   # 26b-mtp, or 26b + LLAMA_CPP_CTX_SIZE=16384
```

Match whichever `-c` the server actually launched with: the profile default (`8192` for
`small`/`26b`, `16384` for `26b-mtp` —
[scripts/lib/local-model-profile.sh](../scripts/lib/local-model-profile.sh)) unless you also
passed `LLAMA_CPP_CTX_SIZE`, in which case match that override instead. Leave
`CONTEXT_WARN_RATIO` at its default (`0.85`).

**Acceptance**, per docs/22 § P0.1's own validate step: either the warning fires and reflects
a genuinely large prompt, or it legitimately never fires because prompts stayed under 85% of
the window — both are acceptable outcomes. What must hold either way: the real llama-server
log shows **no context-shift eviction**. If `scripts/live-smoke.sh` started the server itself
(the "managed llama.cpp" path, not the "reuse an existing provider" path), that log is:

```bash
grep -i "context shift" /tmp/rolerag-live-test/raw/llama-server.log
```

Expect no matches. If you instead pointed the run at an already-running external
`llama-server`, `raw/llama-server.log` won't exist — live-smoke never started or captured that
process — so check the external server's own log/terminal output instead.

**On success**, edit:

- [docs/BACKLOG.md](BACKLOG.md) **#69** — the paragraph starting *"**PENDING:** the llama.cpp
  validation half of the acceptance criterion … is NOT yet done. … it requires `bash
  scripts/live-smoke.sh` against a real local model with `MODEL_CONTEXT_WINDOW_TOKENS` set to
  that model's actual context size, which has not been run for this change."* — replace with a
  dated confirmation, both runs' `context_preflight_warning_count`/`context_actual_warning_count`
  values, and the grep result.
- [docs/22 § P0.1](22_rag_scaling_roadmap.md#p01-token-aware-context-accounting-today-zero-token-counting-anywhere) —
  flip the **Validate.** line's `— [ ]` to `— [x]` and add the same dated confirmation.

## Phase C — recency decision (#6)

```bash
RAG_RECENCY_WEIGHT=0.02 bash scripts/live-smoke.sh
```

If time allows, a second point at the other end of docs/22 § P2.2's suggested range, in its
own artifact dir so it doesn't overwrite the first (the script wipes `LIVE_ARTIFACT_DIR` at
the start of every run):

```bash
RAG_RECENCY_WEIGHT=0.04 LIVE_ARTIFACT_DIR=/tmp/rolerag-live-test-recency-04 bash scripts/live-smoke.sh
```

Compare each run's `raw/conversation-checkpoint.json` → `quality_metrics` against Phase A's
baseline (implicitly `RAG_RECENCY_WEIGHT=0.0`): `callback_recall_misses`,
`retrieval_selection_misses`, and `retrieval_miss_ranks`. Recency should not make any of these
worse than baseline — that comparison is the whole decision.

**Record the decision either way** in [docs/BACKLOG.md](BACKLOG.md) **#6** — replace
*"offline sweep 85/85, validate via live-smoke before enabling"* with the live numbers and the
decision (enable at a specific weight, or keep the `0.0` default and say why not). **If
enabling**, set `RAG_RECENCY_WEIGHT` in your own `.env` — not `.env.example` — the shipped
default stays `0.0` unless this becomes a repo-wide recommendation rather than a
validated-for-your-campaign choice.

## Phase D — long-campaign preset (P2.2 and C2)

The full suggested preset. Every value below is an unvalidated suggestion until this run:
`MEMORY_CONSOLIDATION_THRESHOLD`/`MEMORY_CONSOLIDATION_MAX_IMPORTANCE`/
`RAG_WRITE_DEDUP_COSINE_THRESHOLD`/`RAG_RECENCY_WEIGHT` from docs/22 § P2.2's own text;
`MEMORY_CONSOLIDATION_MIN_AGE`/`MEMORY_CONSOLIDATION_BATCH_CAP` (the age-floor/batch-size
knobs § C2 added) don't have suggested numbers in docs/22 itself, so the values here (`10`,
`15`) are this runbook's suggestion for actually exercising both at `THRESHOLD=40` — treat
them as equally unvalidated, not as a documented recommendation:

```bash
MEMORY_CONSOLIDATION_THRESHOLD=40 \
MEMORY_CONSOLIDATION_MAX_IMPORTANCE=2 \
MEMORY_CONSOLIDATION_MIN_AGE=10 \
MEMORY_CONSOLIDATION_BATCH_CAP=15 \
RAG_WRITE_DEDUP_COSINE_THRESHOLD=0.92 \
RAG_RECENCY_WEIGHT=0.02 \
LIVE_TURN_COUNT=100 \
bash scripts/live-smoke.sh
```

**What the checkpoint asserts at 100 turns.** Every `StoryEvent` in
[`app/diagnostics/live_checkpoint.py`](../app/diagnostics/live_checkpoint.py) is in scope,
including the three late-recall probes added for exactly this validation: `amber_ring_token`
(defined turn 55, called back turn 65), `north_stair_rendezvous` (70→80), and
`hollow_bookend_note` — the long-gap probe, defined turn 17 and called back turn 95, the
"still-recallable after 100 turns and a consolidation pass or two" case. Separately, the
SQLite/Qdrant parity check (`persisted.memory_count == qdrant.session_memory_count`) already
excludes `CONSOLIDATED_TAG`-marked rows on both sides (the 2026-07-11 fix), so consolidation
firing does not by itself break that assertion.

**The real risk this run is designed to surface:** the per-event checks are not softened for
a memory that consolidation later folds. `matching_memory_ids` is computed against *every*
SQLite row for the session, tagged or not, so a folded original (or its roll-up summary) still
has to satisfy `semantic_match` against the event's terms, and — whenever any match exists —
`indexed_memory_ids` must still equal `matching_memory_ids` **unconditionally**, not just when
`LIVE_FAIL_ON_STRUCTURED_WARNINGS=1`. If `MEMORY_CONSOLIDATION_MIN_AGE=10` isn't a large enough
rolling window to keep `hollow_bookend_note`'s memory indexed through turn 95, the run fails
outright with a `CheckpointError`, not a quietly elevated miss count. That failure mode *is*
this phase's validation question, not a bug in the checkpoint — see
[If something fails](#if-something-fails).

**Read from `raw/conversation-checkpoint.json`:**

- `persisted.consolidated_memory_count` / `persisted.consolidation_summary_count` — both
  `> 0` proves consolidation actually fired during the run (folded originals / summaries
  written respectively). If both are `0`, `THRESHOLD=40` never tripped in this scenario at
  this turn count — record that too; it means this run validated a 100-turn session without
  consolidation engaging, not the preset's growth-control behavior.
- `quality_metrics.callback_recall_misses`, `retrieval_selection_misses`,
  `retrieval_miss_ranks` — compare against Phase A/C baselines.
- `quality_metrics.latency` (`total_seconds`/`p50_seconds`/`p95_seconds`) — consolidation adds
  an extra LLM call on the turns it fires, worth comparing against an un-consolidated 100-turn
  baseline if you have one.

**Success** = the live-smoke run is green (script exits `0`, report all-PASS) **and** late recall is
intact **with consolidation actually active** (`consolidated_memory_count > 0` — a green run
with it at `0` only proves a long *unconsolidated* session works, which isn't what P2.2 is for).

**On success**, edit:

- [docs/22 § P2.2](22_rag_scaling_roadmap.md#p22-long-campaign-preset-enable-the-shipped-but-off-machinery-with-evidence) —
  replace the **Change.** paragraph's preset example and the **Validate.** line's `— [ ]` with
  the validated values and a dated confirmation; docs/22 § C2 gets the same for its
  min-age/batch-cap knobs specifically.
- [docs/BACKLOG.md](BACKLOG.md) — record the result under the C2/P2.2 items if there's a
  natural entry, or add a new dated note.
- Consider adding the validated preset to `.env.example` as a documented **optional** profile
  comment (mirroring the existing "Recommended for long sessions…" block near the top of the
  file) — defaults stay off regardless; this only makes the validated preset easy to copy.
- `CHANGELOG.md` `Unreleased`, only if you also flip any shipped *default* as a result.

**CI-dispatch alternative — Phase A only.**
[`.github/workflows/live-smoke.yml`](../.github/workflows/live-smoke.yml)'s
`workflow_dispatch` exposes a fixed input list (`turn_count`, model/server settings,
`skip_browser`, `fail_on_structured_warnings`, …) with no passthrough for arbitrary
environment variables, so it can run Phase A's plain baseline on a self-hosted runner
(`turn_count` defaults to `8` and accepts up to `100`) but has no way to set
`MODEL_CONTEXT_WINDOW_TOKENS`, `RAG_RECENCY_WEIGHT`, or any `MEMORY_CONSOLIDATION_*`/
`RAG_WRITE_DEDUP_COSINE_THRESHOLD` knob. Dispatching `turn_count: 100` without those env vars
would run a 100-turn *baseline* (no preset), which is a different, less useful data point than
this phase — Phases B, C, and D are local-only for now.

## Phase E — docs/26 Stage 5: Lanes A+B live validation (#80)

Phases A–D's setup, environment, and reading conventions are assumed from here on — this phase
does not repeat them. It validates the [docs/26](26_memory_retrieval_redesign.md) redesign
stages shipped 2026-07-14 (#75 harness fail-fast, #76 provenance attribution, #77 best-match
tag/importance fold, #78 Lane A canon pinning, #79 Lane B lexical slice quotas, plus this
document's own #80 session-side wiring: the definition-turn retry and its offset-aware
bookkeeping) against a real `llama-server` + Qdrant 100-turn run. Nothing below is new machinery
to write — every mechanism it exercises is already shipped, gate-green, and
byte-identical-by-default; this phase is where it earns a live default flip the same way Phases
A–D earned theirs.

### The command

The full Phase D preset, plus the three docs/26 knobs Stage 5 exists to validate:
`CANON_TAG_PINNING=true` (Lane A, #78), `RAG_SLICE_LEXICAL_QUOTA=2` (Lane B, #79), and
`LIVE_DEFINITION_RETRIES=1` (the harness-local retry, #80 — confirmed harness-scoped, not a
Settings field, by docs/26 §8 Q4).

```bash
LIVE_ARTIFACT_DIR=/tmp/rolerag-live-test-stage5-run1 \
MEMORY_CONSOLIDATION_THRESHOLD=40 \
MEMORY_CONSOLIDATION_MAX_IMPORTANCE=2 \
MEMORY_CONSOLIDATION_MIN_AGE=10 \
MEMORY_CONSOLIDATION_BATCH_CAP=15 \
RAG_WRITE_DEDUP_COSINE_THRESHOLD=0.92 \
RAG_RECENCY_WEIGHT=0.02 \
CANON_TAG_PINNING=true \
RAG_SLICE_LEXICAL_QUOTA=2 \
LIVE_DEFINITION_RETRIES=1 \
LIVE_TURN_COUNT=100 \
bash scripts/live-smoke.sh
```

### At least two clean runs

Run the command above **twice**, end to end, before touching any default — not once. Docs/26
§6 Stage 5 is explicit about why: MTP's non-bit-determinism re-rolls the curator's paraphrase
and tag choices on every scripted rerun, so a single green run is weak evidence for a default
flip (the same reasoning that made the original P2.2 preset require two-plus runs' worth of
scrutiny). Point the second run at its own artifact dir so the first isn't overwritten
(`live-smoke.sh` wipes `LIVE_ARTIFACT_DIR` at the start of every run) — change only that one
variable, keep every other env var identical:

```bash
LIVE_ARTIFACT_DIR=/tmp/rolerag-live-test-stage5-run2 \
MEMORY_CONSOLIDATION_THRESHOLD=40 \
MEMORY_CONSOLIDATION_MAX_IMPORTANCE=2 \
MEMORY_CONSOLIDATION_MIN_AGE=10 \
MEMORY_CONSOLIDATION_BATCH_CAP=15 \
RAG_WRITE_DEDUP_COSINE_THRESHOLD=0.92 \
RAG_RECENCY_WEIGHT=0.02 \
CANON_TAG_PINNING=true \
RAG_SLICE_LEXICAL_QUOTA=2 \
LIVE_DEFINITION_RETRIES=1 \
LIVE_TURN_COUNT=100 \
bash scripts/live-smoke.sh
```

### What to read, per run

Both runs produce the same shape of artifact — substitute each run's own `LIVE_ARTIFACT_DIR`:
`${LIVE_ARTIFACT_DIR}/raw/conversation-checkpoint.json`.

- **Blue-seal-class pinning evidence (Lane A, #78).** Walk `turns[].standing_facts_count` /
  `turns[].standing_facts_chars` turn by turn. Once the blue-seal-class fact (or any
  durable-tagged fact) has been established, its pinned block must never drop back to
  `0`/`null` for the rest of the run — that is the whole guarantee `CANON_TAG_PINNING` is
  buying. Cross-check against the verified-offline headroom (docs/26 §3.3: 3 tag-eligible
  items, 352 chars against the D3 pool) — a live campaign growing past `canon_max_items=8` /
  `canon_max_chars=900` is docs/26 §8's open question 2, not a failure by itself, but it
  should be visible here if it happens.
- **Lexical slice activity (Lane B, #79).** In each callback turn's
  `turns[].retrieval.selected[]`, read `slice_score` / `slice_matched_terms` /
  `slice_guaranteed` on the entries that carry them. A `slice_guaranteed: true` entry is
  Lane B claiming a reserved slot; note which callback turns it fires on and whether the
  matched terms are genuinely rare (the offline D3 replay predicted the blue-seal memory
  ranking #1/15 on `{messag, rule, trust}` — does a live run's matched-term set look
  similarly targeted, or is the slice firing on common scene vocabulary, which would mean
  `min_slice_score` needs to be set higher than "unset/no floor" once you pick a real value).
- **Definition-turn retries — the measured survival delta (#80).** Read
  `quality_metrics.definition_retries_allowed` and `quality_metrics.definition_retries_used`
  (per-event; also mirrored per event under `events[].definition_retries_used`). For every
  event where a retry was consumed, cross-reference `turns[]` for the two (or more) entries
  sharing that `turn_index` — the labeled `is_definition_retry`/`retry_attempt` entries — and
  record whether the retry succeeded. This is the raw material for the delta docs/26 §4 and §7
  insist on **measuring, not assuming**: the naive `0.067² ≈ 0.45%`/`~96.5%` independence math
  was rejected specifically because retry outcomes may correlate with the original failure
  (same model state, same draft class). Two runs give you at most a couple of data points per
  probe — say so plainly when you record this rather than presenting it as a stable rate.
- **Context-preflight warning counts (Lane A pressure).**
  `quality_metrics.context_preflight_warning_count` /
  `quality_metrics.context_actual_warning_count`. Lane A's pinned Standing-facts block adds
  prompt tokens every turn now, unconditionally (not only when a callback needs them) —
  confirm this does not push a real ~100-turn campaign over the context-accounting ceiling in
  a way Phase B's baseline didn't already show.
- **Recall/selection misses vs. Phase A/D baselines.**
  `quality_metrics.callback_recall_misses`, `retrieval_selection_misses`,
  `retrieval_miss_ranks` — compare against your own Phase A and Phase D numbers (or
  [docs/22 § P2.2](22_rag_scaling_roadmap.md#p22-long-campaign-preset-enable-the-shipped-but-off-machinery-with-evidence)'s
  recorded runs if you have not re-run A/D recently). Lanes A+B exist to make these go down,
  or at minimum not go up, relative to the dense-only baseline; a regression here is a finding,
  not noise to average away.

### Owner-side extras this environment could not run

Neither of these needs a fresh live run — both replay against artifacts this phase (or the
preserved D3 artifact) already produces — but both need a real embedding model and/or a
disposable Qdrant this authoring environment does not have:

- `rolerag semantic-benchmark --model sentence-transformers/all-MiniLM-L6-v2` through the
  slice-enabled retriever (`RAG_SLICE_LEXICAL_QUOTA=2` set), holding docs/22 § P0.4's
  calibrated floors (recall@10 ≥ 0.75, nDCG@10 ≥ 0.70, German recall@10 ≥ 0.55) — confirms
  Lane B does not regress the offline-measured semantic-quality floors now that it reorders
  the live selection window, not just the D3 replay.
- `reindex-memories` of the D3 pool
  (`docs/artifacts/live-validation-D3-2026-07-12.db`) into a disposable Qdrant, then
  `inspect_story_event` with both lanes' quotas on, for an end-to-end (not offline-replay)
  confirmation that the blue-seal memory `354b8d98` lands in the selected top-5 for the
  callback query. Docs/26 §6 Stage 4 already ran this once during Stage 4's own validation;
  re-running it here, after Stage 5's two live runs, closes the loop against the exact preset
  this phase validates rather than Stage 4's narrower one.

### PASS criteria

Both runs: `live-smoke.sh` exits `0`, `report.md` is all-`PASS`, and — since this preset keeps
`LIVE_FAIL_ON_STRUCTURED_WARNINGS` at its default `1` — every checkpoint assertion held,
including the offset-aware ones (#80): the persisted-turn-count check accounted for whatever
`definition_retries_used` totalled, and provenance attribution resolved against the
actually-successful attempt on every retried event. Both are checkpoint-internal invariants —
if either were wrong, the run would have failed outright with a `CheckpointError`, not silently
produced a wrong number. Beyond exit code: the pinning/slice/retry evidence above must read as
*working as designed*, not merely as "the run did not crash" — a green run where
`standing_facts_count` never rises above `0`, or where `slice_guaranteed` never fires once in
100 turns, is a finding to record and investigate, not a pass.

### On success (both runs clean)

- **Derive and record `min_slice_score`.** Pull the summed-IDF `slice_score` values observed
  above (both runs) for every `slice_guaranteed: true` hit and every near-miss (matched but
  not guaranteed). Set a concrete `RAG_SLICE_MIN_SCORE` default from that observed
  distribution — docs/26 §3.4 is explicit that no numeral ships pre-measured; this is where
  one gets chosen, WITH the observed numbers that justified it recorded alongside it, not just
  the final value.
- **Flip the two runtime defaults** — `CANON_TAG_PINNING` and `RAG_SLICE_LEXICAL_QUOTA`
  (`app/config.py` / `.env.example`) from `false`/`0` to `true`/`2`, plus the newly-derived
  `RAG_SLICE_MIN_SCORE`. Re-run the deterministic gate after flipping (`ruff check . && mypy .
  && pytest && python -m app.evals.regression_runner`) — the quotas=0/flag-off golden tests pin
  the *old* default's byte-identity, not the new one, so confirm nothing else in the suite
  implicitly assumed the old defaults.
- **[docs/22 § P2.2](22_rag_scaling_roadmap.md#p22-long-campaign-preset-enable-the-shipped-but-off-machinery-with-evidence)** —
  add the dated Stage 5 confirmation and both runs' key numbers, plus — required, not
  optional, per docs/26 §6 Stage 5 — the explicit **#73 acceptance reinterpretation**: the
  acceptance bar ("blue-seal reaches the selected top-5") is satisfied via Lane A's
  retrieval-free pinning path (and/or Lane B's lexical slice) for this instance, **not** via a
  rank improvement in dense retrieval. Record that reinterpretation in words, not only the
  passing numbers — docs/26 §6 Stage 5 calls this out by name as something that must be
  recorded explicitly, not left implicit behind a green checkmark.
- **[docs/BACKLOG.md](BACKLOG.md) #80** — flip to `[x]`, dated, with both runs' artifact paths,
  the measured retry survival delta (stated as an observation from N data points, not a
  rate), and the derived `min_slice_score`.
- **`CHANGELOG.md` `Unreleased`** — one entry for the default flips, exactly the case Phase D's
  own Recording matrix row calls for ("only if you also flip any shipped default").
- **Add a row to the [Recording matrix](#recording-matrix)** below, matching the existing
  rows' shape (Phase, Issue(s), On success update).

### Contract-tier attribution semantics (owner decision 2026-07-16)

Phase E run 1 (94 turns, 7/7 prior inspections green) failed at the turn-95 long-gap
probe on exactly the consolidation question Phase D names above: the preset folded the
turn-17 memory and the 15→1 roll-up dropped the fact. Per the owner's decision (recorded
in [docs/22 § P2.2](22_rag_scaling_roadmap.md)), `_validate_attribution` now applies the
docs/26 §8 Q1 contract tiers instead of blanket index-set equality: a matching memory
absent from Qdrant **without** a `consolidated` tag is still a hard failure (unchanged);
a folded original whose roll-up summary carries the probe match (and is indexed) is
satisfied through the summary; a folded original whose roll-up **lost** the match is a
hard failure only for guarantee-tier (durable-commitment tag family) facts and a
**recorded loss** (`quality_metrics.consolidation_lost_matches`, plus the per-event
`folded_lost_ids`) for best-effort facts — the long-gap probe thereby becomes the
compression-quality meter for the preset rather than a gate on a promise the docs/26 §8
Q1 contract deliberately does not make. Read `consolidation_lost_matches` on every run
and judge it; a non-empty value is evidence to weigh, not noise.

### No loosening, and a retry-consumed run is not a re-roll

The [no-loosening rule](#if-something-fails) applies here exactly as in Phases A–D: a Phase E
failure — the pinned block dropping to `0`, the slice never firing, a retry-exhausted
`CheckpointError`, a semantic-benchmark floor regression — is a finding about Lanes A/B or the
preset, not a bug in the checkpoint to route around. Do not lower
`LIVE_FAIL_ON_STRUCTURED_WARNINGS`, do not weaken `_validate_attribution` or the offset-aware
persisted-turn-count check, and do not treat a run where `LIVE_DEFINITION_RETRIES=1` consumed a
retry as disqualified, or as a free re-roll to discard in favor of a "cleaner" one. Per docs/26
§8 Q4 (owner-confirmed 2026-07-14): a consumed retry is a **labeled scenario variant** — the
harness modeling a real player's natural retry on an errored turn — not a re-roll of the run.
Count it, record its outcome in `definition_retries_used`, and let it stand as one of your two
data points.

## Recording matrix

| Phase | Issue(s) | On success, update |
|---|---|---|
| A | #48, #67 | docs/BACKLOG.md #48 and #67 caveat sentences |
| B | #69 | docs/BACKLOG.md #69 PENDING paragraph; docs/22 § P0.1 Validate line |
| C | #6 | docs/BACKLOG.md #6 caveat sentence; your own `.env` if enabling (not `.env.example`) |
| D | P2.2, C2 | docs/22 § P2.2 Change/Validate text; docs/22 § C2 confirmation; optionally `.env.example` as a documented optional preset; CHANGELOG.md `Unreleased` only if a shipped default changes |
| E | #80 (docs/26 Stage 5) | docs/22 § P2.2 dated Stage 5 confirmation + #73 acceptance reinterpretation; docs/BACKLOG.md #80 → `[x]`; `CANON_TAG_PINNING`/`RAG_SLICE_LEXICAL_QUOTA`/`RAG_SLICE_MIN_SCORE` defaults + `.env.example`; CHANGELOG.md `Unreleased` |

## If something fails

A failure is a finding, not a bug to route around. Record it — in the relevant docs/22
subsection and/or the relevant docs/BACKLOG.md item — with:

- the exact command and environment you ran,
- the checkpoint JSON path (`${LIVE_ARTIFACT_DIR:-/tmp/rolerag-live-test}/raw/conversation-checkpoint.json`).
  Even a run that aborts mid-way leaves usable evidence here: `live_checkpoint.py` writes this
  file incrementally after every turn and every scripted event inspection
  (`write_progress`/`progress_writer`), so the last state before the failure survives with
  `"status": "in_progress"`,
- what specifically failed — the assertion message, and which turn/event it names.

**Do not loosen the checkpoint to force a pass**: not `_validate_attribution`'s assertions,
not `LIVE_FAIL_ON_STRUCTURED_WARNINGS`, not the SQLite/Qdrant parity check. A Phase D failure
where consolidation folds a memory a late callback still needs is exactly the kind of result
this pass exists to surface — the fix is tuning the preset (a larger
`MEMORY_CONSOLIDATION_MIN_AGE`, a higher `THRESHOLD`, or reconsidering the preset for that
scenario shape) and documenting the finding, not editing the assertion.
