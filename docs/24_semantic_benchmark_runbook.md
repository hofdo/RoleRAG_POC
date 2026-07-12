# 24 — Semantic Benchmark Runbook

> Reviewed: 2026-07-12 @ ea31def

How to run the first real-embedding-model pass through the graded semantic corpus. This is
the measurement gate described in
[docs/22 § P0.4](22_rag_scaling_roadmap.md#p04-eval-assets-before-retrieval-upgrades-the-measurement-gate);
the harness, CLI, and one-command runner below are the tooling side of that gate, not the
gate itself — the gate opens once real numbers exist (see the [run log](#run-log)).

## Why

[docs/22 § P0.4](22_rag_scaling_roadmap.md#p04-eval-assets-before-retrieval-upgrades-the-measurement-gate)
is the measurement gate for everything downstream of it: all of **P1** (hybrid sparse+dense
retrieval, the multilingual embedding swap, structure-aware chunking) and **P2.5**
(cross-encoder rerank) are *unmeasurable*, not just unmeasured, until one real-embedding run
exists on the graded corpus — every one of those items validates against P0.4 metrics.

The harness's offline half (graded corpus, recall@k/nDCG/MRR metrics, the
`app/diagnostics/semantic_benchmark.py` harness, the `rolerag semantic-benchmark` CLI, the
opt-in `-m semantic` pytest tier) already shipped. What it never got was a run against a real
embedding model: the authoring environment's proxy blocks FastEmbed downloads outright, so no
real numbers were ever produced there. This runbook is for the machine that doesn't have that
constraint — the owner's.

That first run does two things at once: its numbers **open the P1 gate**, and its output
**calibrates** the provisional, deliberately generous floors in
[`tests/evals/test_semantic_benchmark_opt_in.py`](../tests/evals/test_semantic_benchmark_opt_in.py).

## Prerequisites

Just the repo venv — `make install`, or the
[README quickstart](../README.md#fresh-clone-setup). Explicitly, none of these are needed:

- **No Qdrant.** The harness indexes the graded corpus into an `InMemoryVectorStore`.
- **No `llama-server` / model server.** The benchmark is LLM-free — it scores embeddings and
  ranking only; nothing generates text.
- **No API server.** The CLI (and the opt-in pytest tier) call
  `app/diagnostics/semantic_benchmark.py` in-process.

The one thing that does touch the network: the first embed call for a given `--model` lazily
downloads it (~90 MB for the default `sentence-transformers/all-MiniLM-L6-v2`) into the local
Hugging Face cache. Every run after that is offline. `--keyword` never downloads anything (see
[Reading the numbers](#reading-the-numbers) for why its scores don't count toward calibration).

## The one command

```bash
bash scripts/semantic-benchmark.sh
```

Configured by environment variables, no flags:

| Env var | Default | Meaning |
|---|---|---|
| `MODELS` | `sentence-transformers/all-MiniLM-L6-v2` | Space-separated FastEmbed model names, one `--model` per word. `MODELS=""` (explicitly set to empty) means "no real models" — pair it with `INCLUDE_KEYWORD=1` for a download-free smoke run, otherwise the script exits 1 explaining there's nothing to benchmark. |
| `TOP_K` | `10` | Candidate depth (`--top-k`); must be `>= 10` since recall@10/nDCG@10 need it. |
| `INCLUDE_KEYWORD` | `0` | `1` also benchmarks the deterministic keyword provider (`--keyword`, no downloads). |
| `RUN_PYTEST` | `1` | `1` also runs the opt-in `-m semantic` pytest tier against the *first* word in `MODELS`, after the CLI run finishes. Skipped automatically when `MODELS` is empty. |
| `ARTIFACT_PATH` | `docs/artifacts/semantic-benchmark-<today>.json` | Where the `--json` output is written. |

Raw CLI equivalent (what the script wraps):

```bash
python -m app.cli semantic-benchmark --model sentence-transformers/all-MiniLM-L6-v2 --json
```

Pytest-tier equivalent (what `RUN_PYTEST=1` runs, once a model is configured):

```bash
ROLERAG_SEMANTIC_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    pytest -m semantic tests/evals/test_semantic_benchmark_opt_in.py
```

**Failure semantics.** Each `--model`/`--keyword` provider runs independently: a failed
download or an unknown model name is logged to stderr as `[failed] <label>:
<ExceptionType>: <message>` and skipped, without aborting providers still queued. The artifact
is still written for whichever providers succeeded (an empty `[]` if all of them failed). The
CLI — and therefore the script — still exits non-zero if *any* provider failed, so scripting
can distinguish "fully clean" from "partial" from the exit code alone.

## Reading the numbers

Each provider's report scores two retrieval paths over two query subsets:

- **reranked** — the production path: `ActorContextRetriever`'s dual-query retrieval plus the
  full additive boost rerank. This is the primary number — it's what a real turn returns.
- **raw** — single-query, dense-only, no boosts. Isolates how much of the reranked score comes
  from the embedding model itself versus the deterministic boost layer; useful when comparing
  candidate models later, since the boost layer is identical across all of them.
- **overall** — every graded query.
- **german** — the 9-query German subset only. The default `all-MiniLM-L6-v2` is English-only
  (docs/22 § P1.2), so a weaker score here is *expected*, not a regression — closing that gap
  is P1.2's motivation.

Per path/subset, five metrics: `recall@5`, `recall@10`, `recall@5(strict)`, `ndcg@10`, `mrr`.
`recall@5(strict)` only counts `judgment==2` (directly relevant) chunks, unlike the plain
recall columns which count any `judgment>=1`. The corpus also bakes in five same-proper-noun
distractor clusters (same name, three distinct facts each, one query per fact), so a high
score reflects actually discriminating *which* fact answers a query — not just matching the
proper noun.

`--keyword` (`INCLUDE_KEYWORD=1`) reports the same table shape, but its vectors are
deterministic keyword counts, not a real embedding model — it exercises the harness plumbing,
not embedding quality. `scripts/lib/suggest_floors.py` knows this and never prints floor
suggestions for the keyword provider's numbers.

## Calibrating the floors

1. Run the benchmark against the model you want to calibrate against — normally the shipped
   default, `all-MiniLM-L6-v2`.
2. Read `suggest_floors.py`'s output (the script runs it for you automatically against the
   artifact it just wrote): a compact aggregate table per provider, then, for real providers,
   paste-ready lines shaped like this (numbers illustrative, not real results):

   ```
   suggested floors for tests/evals/test_semantic_benchmark_opt_in.py:
   _MIN_RECALL_AT_10 = 0.45
   _MIN_NDCG_AT_10 = 0.35
   _MIN_GERMAN_RECALL_AT_10 = 0.30
   ```

3. **Hand-edit** `_MIN_RECALL_AT_10`, `_MIN_NDCG_AT_10`, and `_MIN_GERMAN_RECALL_AT_10` in
   [`tests/evals/test_semantic_benchmark_opt_in.py`](../tests/evals/test_semantic_benchmark_opt_in.py)
   to the suggested values, and update that module's "PROVISIONAL" docstring wording so it
   reflects a calibrated run instead of an untested guess. `suggest_floors.py` only prints —
   it never edits the test file itself.
4. Re-run the pytest tier (`RUN_PYTEST=1`, or the raw command above) to confirm the edited
   floors pass against the run you calibrated them from.
5. Commit the test edit, the dated artifact under `docs/artifacts/`, and a new row in the
   [run log](#run-log) together, in one commit.

**If a floor fails on the first run,** that is itself a P0.4 finding worth recording, not a
license to lower the floor and move on — per the test module's own docstring, investigate
first. These floors are generous enough that a MiniLM-class model failing one most likely
means something in the retrieval or boost path is broken, not that the floor was set wrong.

## Run log

One row per real run against this corpus. Artifacts live under `docs/artifacts/`, committed
alongside the row that references them.

| date | commit | model | reranked overall recall@10 | reranked overall nDCG@10 | reranked german recall@10 | artifact | action taken |
|------|--------|-------|-----------------------------|----------------------------|-----------------------------|----------|---------------|
| | | | | | | | |

## Next: P1.2 candidates

The same script benchmarks several models in one invocation, which is how to compare the
[P1.2 multilingual candidates](22_rag_scaling_roadmap.md#p12-embedding-model-upgrade-path-multilingual)
once the default model's floors above are calibrated:

```bash
MODELS="sentence-transformers/all-MiniLM-L6-v2 paraphrase-multilingual-MiniLM-L12-v2 jinaai/jina-embeddings-v2-base-de" \
    bash scripts/semantic-benchmark.sh
```

`intfloat/multilingual-e5-large` stays excluded from any such run until Settings-driven
query/passage prefixes exist — see the prefix caveat in
[docs/22 § P1.2](22_rag_scaling_roadmap.md#p12-embedding-model-upgrade-path-multilingual). Any
default-model swap that follows from these numbers goes through
[docs/23](23_embedding_migration_runbook.md) (`reset-index` → re-ingest → `reindex-memories`),
per the measure-first workflow — swapping the shipped default needs P0.4 evidence first.
