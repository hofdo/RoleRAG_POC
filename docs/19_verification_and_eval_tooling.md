# 19 — Verification & Eval Tooling

> Reviewed: 2026-07-12 @ b854814

## Purpose

This document is the reference for the live-verification and offline-evaluation tooling
that lives outside the request path: the live-smoke checkpoint, the managed llama.cpp
model profiles, the model bake-off / secret-containment / RAG-knob harnesses, and the
diagnostics modules that back them. None of it changes application behavior — every script
here drives a disposable local stack or reads already-written artifacts. Use it to answer
"does a real local model still pass?", "which model recalls best and leaks least?", and
"did a RAG-knob change move retrieval?".

The safe, fast diagnostics (`health`, `doctor`, `smoke-run`) and the day-to-day run
instructions stay in the root [README](../README.md#runtime-verification); config values
live in [`.env.example`](../.env.example) + [app/config.py](../app/config.py). This doc
carries the deep operational detail those keep a summary of.

## Scripts at a glance

| Script | What it does | Key env |
|--------|--------------|---------|
| [`scripts/live-smoke.sh`](../scripts/live-smoke.sh) | Full disposable-stack live checkpoint: Qdrant + FastAPI + a real local model, an N-turn Rose Gallery conversation, and a Playwright UI smoke | `LOCAL_LLM_MODEL`, `LIVE_TURN_COUNT`, `LIVE_ARTIFACT_DIR`, `LOCAL_MODEL_PROFILE`, `LLAMA_CPP_*` |
| [`scripts/bakeoff.sh`](../scripts/bakeoff.sh) | Aggregates recall + containment + latency across model run dirs into one comparison table | `BAKEOFF_DIR`, run-dir args |
| [`scripts/secret-probe.sh`](../scripts/secret-probe.sh) | Adversarial secret-containment probe against the "A Bride for Sarnhold" pack | `RUN_DIR`, `LOCAL_LLM_MODEL` |
| [`scripts/ab-sweep.sh`](../scripts/ab-sweep.sh) | One-factor-at-a-time (OFAT) sweep over RAG memory/retrieval knobs, wrapping `live-smoke.sh` | `MODEL_PROFILE`, `TURNS`, `TOP_K`, `OUT` |
| [`scripts/test-local-model-matrix.sh`](../scripts/test-local-model-matrix.sh) | Deterministic checks once, then the full **live** stack sequentially for the `small` and `26b` profiles (hours of real generation), aggregated by `app.diagnostics.model_comparison`; see [14](14_local_model_comparison_2026-06-08.md) and the section below | `PYTHON`, `MODEL_COMPARE_ARTIFACT_DIR`, `MODEL_COMPARE_TURN_COUNT` |
| [`scripts/dev-up.sh`](../scripts/dev-up.sh) / [`scripts/dev-down.sh`](../scripts/dev-down.sh) | Bring the full dev stack up/down (used by `make dev`) | `DEV_API_PORT` |
| [`scripts/semantic-benchmark.sh`](../scripts/semantic-benchmark.sh) | One-command real-embedding recall@k/nDCG/MRR benchmark over the graded semantic corpus (docs/22 P0.4; walkthrough: [docs/24](24_semantic_benchmark_runbook.md)) — no Qdrant/API/model server needed. Wraps the resilient `rolerag semantic-benchmark --json` CLI, prints suggested floors via `scripts/lib/suggest_floors.py`, and optionally runs the opt-in `-m semantic` pytest tier | `MODELS`, `TOP_K`, `INCLUDE_KEYWORD`, `RUN_PYTEST`, `ARTIFACT_PATH` |

The scripts source shared helpers from `scripts/lib/`: process lifecycle
(`process-lifecycle.sh`), the local-model launch profiles
([`local-model-profile.sh`](../scripts/lib/local-model-profile.sh)), and the A/B progress
readers (`ab_progress.py`, `ab_row.py`).

## Live-smoke checkpoint

`scripts/live-smoke.sh` is the flagship live-acceptance run. It stands up an isolated
stack, exercises the real HTTP and browser paths against a real local model, and records
detailed diagnostics — without touching your working data.

**What it does, in order:** checks required commands → ensures a local provider exposing
`LOCAL_LLM_MODEL` is available (reuse or managed startup) → starts a disposable Qdrant
(default `127.0.0.1:6334`) → runs `doctor` and a `smoke-run --real-runtime` → ingests demo
lore → starts FastAPI (default `127.0.0.1:18080`) → runs a live HTTP API flow → runs a
Playwright UI smoke (unless skipped) → runs the N-turn Rose Gallery conversation checkpoint
(`app.diagnostics.live_checkpoint`). It writes a report and per-step raw artifacts, removes
its Qdrant container on exit, and stops only a llama.cpp process it started itself.

**Provider reuse vs. managed startup.** It first probes `/v1/models`; if the configured
model is already served it reuses that provider and never stops it. Otherwise it starts a
managed llama.cpp from the selected profile (see below) and stops it on exit with
`SIGTERM` → wait `LLAMA_CPP_STOP_TIMEOUT` (default `15`) → `SIGKILL`.

**Key environment:**

| Var | Default | Meaning |
|-----|---------|---------|
| `LIVE_ARTIFACT_DIR` | `/tmp/rolerag-live-test` | Root for the report and all raw artifacts; the dir is wiped and recreated each run |
| `LOCAL_LLM_MODEL` | `chatgpt-onnechan` | Model id the provider must expose (matched against `/v1/models`) |
| `LIVE_TURN_COUNT` | `8` | Rose Gallery turns; accepts `5`–`100` (`LIVE_LONG_TURN_COUNT` is a fallback when unset) |
| `LIVE_FAIL_ON_STRUCTURED_WARNINGS` | `1` | Fail on critic / memory-curation / memory-indexing / retrieval warnings; set `0` for report-only |
| `LIVE_SKIP_BROWSER` | `0` | Set `1` to skip the Playwright UI smoke |
| `LIVE_HTTP_TIMEOUT_SECONDS` | `420` | Per-request HTTP timeout for the harness drivers (`live_checkpoint`, `secret_probe`, and the in-script API flow) — dense models warm up slowly |
| `API_PORT` / `QDRANT_PORT` | `18080` / `6334` | Isolated ports so a live run never collides with the dev stack |

`LIVE_ARTIFACT_DIR` is the supported way to redirect the hardcoded `/tmp/rolerag-live-test`
path (it also names where the workflow uploads artifacts from).

## CI: the Live Smoke workflow

[`.github/workflows/live-smoke.yml`](../.github/workflows/live-smoke.yml) runs the live
checkpoint on a self-hosted runner. Triggers and knobs:

- **Manual `workflow_dispatch`** with inputs `turn_count` (validated `5`–`100` at the
  workflow level, matching the script's own `LIVE_TURN_COUNT` range),
  `llama_server_path`, `llama_hf_model`, `llama_model_path`, `llama_ctx_size`, and
  `llama_n_gpu_layers` — the `llama_*` inputs map onto the `LLAMA_CPP_*` env vars below.
- **Weekly `schedule`** that is a silent no-op unless the repository variable
  `ENABLE_SCHEDULED_LIVE_SMOKE` is set to `true`.
- **Artifacts** are uploaded unconditionally from the live artifact dir: `report.md`, the
  conversation-checkpoint JSON, the API flow JSON, llama-server logs, and Playwright traces
  on failure.

## The two-model comparison (`test-local-model-matrix.sh`)

`PYTHON=.venv/bin/python bash scripts/test-local-model-matrix.sh` runs the deterministic
checks once (ruff, mypy, pytest, frontend tests, regression runner), then the complete live
stack sequentially for the `small` and `26b` profiles — real model runs, hours of wall
clock. Outputs land isolated under `/tmp/rolerag-model-comparison/{small,26b}` (override
with `MODEL_COMPARE_ARTIFACT_DIR`) with `comparison.json`, `comparison.md`, and a
turn-aligned transcript. `MODEL_COMPARE_TURN_COUNT` must be `20` (default) or `50`; other
values are rejected. Quality findings are report-only; deterministic, infrastructure,
persistence, indexing, and retrieval-visibility failures exit nonzero.

## Model profiles & the `LLAMA_CPP_*` matrix

Managed llama.cpp startup is driven by a named profile
([`scripts/lib/local-model-profile.sh`](../scripts/lib/local-model-profile.sh) is the
source of truth for the exact per-profile flags):

| `LOCAL_MODEL_PROFILE` | Served model | Notes |
|-----------------------|--------------|-------|
| `small` (default) | `DavidAU/gemma-4-E4B-it-…-Thinking-GGUF:Q8_0` via `-hf` | Uses a patched chat template so grammar-constrained JSON is not starved by thought-channel output |
| `26b` | `HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M` via `-hf` | The recommended acceptance model |
| `26b-mtp` | Same 26B Balanced base from local GGUF files, plus an MTP speculative-draft model | Faster decode (`-md … --spec-type draft-mtp --spec-draft-n-max 3`, 16384-token context); point `MODEL_26B_MTP_DIR` at the directory holding the two GGUFs (default `~/models/gemma4-26b-qat-balanced-mtp`). See [16_mtp_speculative_decoding_2026-06-29.md](16_mtp_speculative_decoding_2026-06-29.md) |

All named profiles pass `--jinja`, disabled reasoning/thinking, full GPU offload (`-ngl
all`), flash attention, a `q8_0` K cache, a `q4_0` V cache, and seed `424242` (8192-token
context for `small`/`26b`, 16384 for `26b-mtp`). Sampling stays app-controlled — the router
sets temperature per request — so the profiles carry no `--temp`/`--top-k`/etc.

The `LLAMA_CPP_*` variables override the launch when you need to point at a specific binary,
weights, or hardware:

| Var | Effect |
|-----|--------|
| `LLAMA_CPP_SERVER_PATH` | `llama-server` binary to launch (default from `PATH`) |
| `LLAMA_CPP_MODEL_PATH` | Switches managed startup from `-hf` to a local `-m` GGUF path |
| `LLAMA_CPP_HF_MODEL` | Hugging Face GGUF repo:quant for `-hf` startup (defaults to the profile's model) |
| `LLAMA_CPP_HOST` / `LLAMA_CPP_PORT` | Bind address for the managed server (default `127.0.0.1:8080`) |
| `LLAMA_CPP_CTX_SIZE` | Optional `-c` context override |
| `LLAMA_CPP_N_GPU_LAYERS` | Optional `--n-gpu-layers` override |
| `LLAMA_CPP_SERVER_ARGS` | Whitespace-separated extra args appended to the launch command |
| `LLAMA_CPP_STOP_TIMEOUT` | Grace period before `SIGKILL` on managed shutdown (default `15`) |

## Model bake-off

The bake-off answers "of several models, which recalls best and leaks least?". Each model
gets its own **run directory** carrying two artifacts:

- `raw/conversation-checkpoint.json` — the N-turn recall run, written by
  `scripts/live-smoke.sh` (`app.diagnostics.live_checkpoint`).
- `secret-probe.json` — the containment run, written by `scripts/secret-probe.sh`.

`scripts/secret-probe.sh` writes its `secret-probe.json` into the *same* `RUN_DIR` as the
recall run without clobbering the recall artifacts, so both live in one directory. It stands
up a disposable Qdrant + FastAPI against the Sarnhold pack, reuses an already-running local
server (launch it with `--alias ${LOCAL_LLM_MODEL}`), ingests the scenario lore, and runs
`app.diagnostics.secret_probe`. It requires `RUN_DIR` and `LOCAL_LLM_MODEL` (which must
equal the server's `--alias`).

`scripts/bakeoff.sh` then aggregates. With no arguments it reads the three standard run
dirs under `${BAKEOFF_DIR:-/tmp/rolerag-bakeoff}` (`A-gemma26b`, `B-qwen35b`, `C-qwen27b`);
otherwise pass `LABEL:DIR` specs directly:

```bash
bash scripts/bakeoff.sh A:/tmp/run-a B:/tmp/run-b
```

It runs `app.diagnostics.model_bakeoff`, which reads each run's recall checkpoint, secret
probe, and structured-failure log and emits one side-by-side table (`bakeoff.json` +
`bakeoff.md` under `BAKEOFF_DIR`). Lower is better for every miss / leak / latency /
structured-failure / finish-length column; no winner is declared automatically — the
recall-vs-leak trade-off and prose quality are a human call.

## Secret-containment, in one line

The probe measures **confabulation, not parroting**. A persona's own `secrets`,
`forbidden_knowledge`, and `private_description` are structurally absent from the actor and
memory-extraction prompts, and GM-only lore is double-filtered out of retrieval — so a model
cannot repeat a secret it was handed. The realistic failure mode is a weaker model
*inventing or confirming* the hidden truth under a leading question. Detection is a
conservative high-precision screen (affirmative phrase groups), every response is recorded
for human adjudication, and `total_leaks` is a comparison metric, not a hard gate. See
[17_content_authoring_reference.md](17_content_authoring_reference.md) for how hidden fields
drive containment.

## RAG-knob A/B sweep

`scripts/ab-sweep.sh` tunes retrieval by running the live Rose Gallery checkpoint once per
config — one knob off baseline (OFAT) — each into its own isolated artifact dir, in
report-only mode (a retrieval miss does not abort the run). It wraps `live-smoke.sh` and
changes no application behavior; it only sets env knobs for each run and then prints a
comparison table built from each run's `conversation-checkpoint.json` `quality_metrics`.

Default sweep axes (edit `CONFIGS` in the script): `RAG_RECENCY_WEIGHT`,
`RAG_INDEX_IMPORTANCE_FLOOR`, `RAG_WRITE_DEDUP_COSINE_THRESHOLD`, and
`CANON_IMPORTANCE_FLOOR`, each compared against the application-default baseline. Configure
the run via env (defaults shown):

```bash
MODEL_PROFILE=26b TURNS=50 TOP_K=10 OUT=/tmp/rolerag-ab \
PYTHON=.venv/bin/python bash scripts/ab-sweep.sh
```

This sweep is the source of the RAG-tuning guidance in
[`.env.example`](../.env.example) — the 2026-06-15 38-turn A/B run on the 26B model found
`top_k=10` (the default) recalled every seeded story event while the other knobs showed no
gain, which is why their defaults are recommended unchanged.

## Backing diagnostics modules

The scripts are thin drivers; the logic lives in `app/diagnostics/` (10 modules; the
package also holds `runtime_checks.py` and `smoke_runner.py` behind `doctor`/`smoke-run`):

| Module | Role |
|--------|------|
| [`live_checkpoint.py`](../app/diagnostics/live_checkpoint.py) | Runs the N-turn Rose Gallery conversation over HTTP, checks recall/persistence/retrieval, and writes `conversation-checkpoint.json`. Owns the `5`–`100` turn-count bounds (`resolve_turn_count`) |
| [`secret_probe.py`](../app/diagnostics/secret_probe.py) | Adversarial confabulation screen for the Sarnhold pack; writes `secret-probe.json` |
| [`model_bakeoff.py`](../app/diagnostics/model_bakeoff.py) | N-way read-only aggregator over run dirs; never starts a server or model |
| [`model_comparison.py`](../app/diagnostics/model_comparison.py) | The hardwired two-model comparison behind `scripts/test-local-model-matrix.sh` (see [14](14_local_model_comparison_2026-06-08.md)) |
| [`retrieval_miss.py`](../app/diagnostics/retrieval_miss.py) | Ranks expected chunk ids across the full candidate set so a tuning run sees *how far off* a wanted memory ranked, not just pass/fail |
| [`embedding_ab.py`](../app/diagnostics/embedding_ab.py) | LLM-free real-embedding rank A/B over the seeded corpus; backs the `embedding-ab` CLI command |
| [`structured_failures.py`](../app/diagnostics/structured_failures.py) | Appends raw structured-output failures to a JSONL log for offline analysis (the bake-off's `struct_fail` column) |
| [`eval_runs.py`](../app/diagnostics/eval_runs.py) | Read-only aggregation of run artifacts behind `GET /diagnostics/eval-runs` (see below) |

## Eval Runs API & the SPA Eval page

The same run artifacts feed a read-only dashboard. `GET /diagnostics/eval-runs` scans the
eval results directory — read from the `EVAL_RESULTS_DIR` environment variable (deliberately
not a `Settings` field), default `<repo>/eval-runs` — for run subdirectories each containing
`raw/conversation-checkpoint.json` (the bake-off layout) or a top-level
`conversation-checkpoint.json`, and surfaces headline `quality_metrics` (recall, latency,
warnings). The SPA Eval page (`/app`) reads exactly these artifacts, and its empty-state
copy names the same `EVAL_RESULTS_DIR` layout. Full endpoint shapes and error codes are in
[12_api_contract.md](12_api_contract.md#eval-runs).

## See also

- root [README — Runtime Verification](../README.md#runtime-verification) — the quickstart
  summary and safe diagnostics.
- [25_live_validation_runbook.md](25_live_validation_runbook.md) — chains the pending
  live-validation passes (#48/#67 baseline, #69 context accounting, #6 recency, the P2.2
  long-campaign preset) into one guided sitting on top of the tooling described here.
- [12_api_contract.md](12_api_contract.md#eval-runs) — the Eval Runs HTTP surface.
- [16_mtp_speculative_decoding_2026-06-29.md](16_mtp_speculative_decoding_2026-06-29.md) —
  the `26b-mtp` speculative-decoding measurement.
- [14_local_model_comparison_2026-06-08.md](14_local_model_comparison_2026-06-08.md) — the
  two-model comparison behind `test-local-model-matrix.sh`.
