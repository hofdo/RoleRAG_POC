# 25 — Live Validation Runbook

> Reviewed: 2026-07-12 @ b854814

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

## Recording matrix

| Phase | Issue(s) | On success, update |
|---|---|---|
| A | #48, #67 | docs/BACKLOG.md #48 and #67 caveat sentences |
| B | #69 | docs/BACKLOG.md #69 PENDING paragraph; docs/22 § P0.1 Validate line |
| C | #6 | docs/BACKLOG.md #6 caveat sentence; your own `.env` if enabling (not `.env.example`) |
| D | P2.2, C2 | docs/22 § P2.2 Change/Validate text; docs/22 § C2 confirmation; optionally `.env.example` as a documented optional preset; CHANGELOG.md `Unreleased` only if a shipped default changes |

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
