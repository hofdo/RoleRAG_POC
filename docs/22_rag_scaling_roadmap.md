# 22 — RAG Scaling Roadmap: Larger Scenarios on ~27B Local Models

> Reviewed: 2026-07-19 @ 760c9e4
>
> **Update 2026-07-14 (docs/26 synthesis).** The four Phase D live failures recorded under
> [§ P2.2](#p22-long-campaign-preset-enable-the-shipped-but-off-machinery-with-evidence) were
> synthesized into [docs/26](26_memory_retrieval_redesign.md) — two guarantee lanes
> (tag-eligible canon pinning; lexical slice quotas) over the existing pipeline, staged as
> [backlog #75–#80](BACKLOG.md#planned-2026-07-14--docs26-memoryretrieval-redesign-from-the-phase-d-findings).
> P2.2's unblocking path changes: **no longer blocked on P1.1** (see the P2.2 update note);
> P1.1 becomes a conditional escalation rung pulled by post-Stage-5 live evidence.
>
> **Update 2026-07-11 (P0.4, offline half).** Graded-relevance corpus + recall@k/nDCG
> benchmark harness shipped (see the
> [P0.4 subsection](#p04-eval-assets-before-retrieval-upgrades-the-measurement-gate)):
> `app/evals/semantic_corpus.py` (~85 chunks across the three collection shapes, five
> same-proper-noun distractor clusters, a German query subset), `app/evals/retrieval_metrics.py`
> (recall@k incl. a strict judgment==2 variant, nDCG@k, MRR — pure functions, hand-computed
> unit tests), `app/diagnostics/semantic_benchmark.py` (indexes the corpus and scores every
> query through the production `ActorContextRetriever` path plus a raw dense-only baseline),
> and CLI `rolerag semantic-benchmark --model <fastembed-name>|--keyword [--json]`. An opt-in
> `-m semantic` pytest tier (`tests/evals/test_semantic_benchmark_opt_in.py`) asserts
> provisional floors against a real FastEmbed model, gated behind `ROLERAG_SEMANTIC_MODEL` so
> the default gate never downloads one. **Pending:** the real-model benchmark run itself (real
> FastEmbed downloads are blocked in the authoring environment; single command for the owner's
> machine: `rolerag semantic-benchmark --model sentence-transformers/all-MiniLM-L6-v2`), floor
> calibration from that run, and transcript-derived queries (needs real play exports via
> `export-session`) — all noted in the P0.4 section below.
>
> **Update 2026-07-11.** Added late-recall `StoryEvent`s to the live checkpoint (see the
> *Eval methodology* bullet under
> [unverified candidates](#unverified-candidates-from-the-2026-07-07-sweep-verify-before-building))
> so a 100-turn live-smoke run asserts recall past turn 50, including a long-gap probe.
>
> **Update 2026-07-08.** A follow-up code-grounded review re-checked the RAG core and memory
> lifecycle. It **confirmed** two of the [unverified candidates](#unverified-candidates-from-the-2026-07-07-sweep-verify-before-building)
> (consolidation age guard; pinned-canon ↔ retrieved-chunk double-spend) and added three new
> findings — all folded into
> [§ 2026-07-08 review](#2026-07-08-review-confirmations--new-findings) below. **The two
> highest-value findings, C1 and N1, are now shipped** (`64db602`, `0c11c29`; byte-tested +
> live-smoke no-regression) — see their subsections. Engine-quality,
> test, and ops recommendations from the same review live in
> [docs/BACKLOG.md](BACKLOG.md#review-2026-07-08--engine-quality-testing-ops).
>
> Authored 2026-07-07 by Claude Fable 5 from a first-hand code analysis plus a multi-agent
> improvement sweep with per-finding adversarial verification (method described in
> [docs/21](21_fable_handoff_reasoning.md)). This is a **living roadmap** — check items off,
> move corrections in, and re-verify priorities as live evidence lands.
>
> **Verification coverage note.** The adversarial verification pass was cut short by a
> session rate limit after the embeddings/hybrid lens: those findings are fully verified
> (deltas folded in below, incl. the new P1.4), the P0 facts were verified first-hand by
> the author, and the remaining sweep findings are listed under
> [Unverified candidates](#unverified-candidates-from-the-2026-07-07-sweep-verify-before-building)
> — verify each against code before building on it.

## Scope and stance

Target: scenarios with **10–100× the demo lore corpus** and **100+-turn campaigns**, played
on a **~27B-parameter local model** (reference: the `26b` llama.cpp profile,
`-c 8192`, or `26b-mtp` at 16384 — [scripts/lib/local-model-profile.sh](../scripts/lib/local-model-profile.sh)).

This roadmap deliberately extends the "personal-use scope" stance in
[docs/BACKLOG.md](BACKLOG.md) ("Not doing: corpus-scale micro-opts") **at the owner's
request**: larger scenarios are now in scope. Everything here still follows the
[docs/10](10_next_steps_after_mvp.md) decision rule (deterministic orchestration, state in
application code, visibility boundaries, measured benefit; no LangChain/LangGraph, no
autonomous loops) and the house style for risky knobs: **additive, opt-in, byte-identical
defaults** unless live evidence justifies a default flip.

Ordering rationale (the full chain of thought is in
[docs/21 § How I reasoned about the RAG core](21_fable_handoff_reasoning.md#how-i-reasoned-about-the-rag-core)):
**budget → measurement → reach → prompt → growth**. Budget first because the recommended
config already runs near the context ceiling; measurement second because every retrieval
claim after it is unverifiable without better eval assets.

---

## P0 — before authoring bigger scenarios

### P0.1 Token-aware context accounting (today: zero token counting anywhere)

> **Validation procedure ready 2026-07-12** —
> [docs/25](25_live_validation_runbook.md) Phase B chains the sanity run (deliberately tiny
> `MODEL_CONTEXT_WINDOW_TOKENS`, proving the warning fires) and the real-ctx run (matching the
> profile's actual context window) this item's own Validate step calls for, and states exactly
> what to read from the checkpoint JSON and the llama-server log. The runs themselves are
> still pending (docs/BACKLOG.md #69).

**Problem.** Every budget in the engine is characters or item counts
([app/orchestration/context_budget.py](../app/orchestration/context_budget.py),
`RECENT_DIALOGUE_MAX_MESSAGE_CHARS`, `CANON_MAX_CHARS`); there is no tokenizer or token
estimate anywhere in `app/` (verified by grep). With the recommended
`RAG_DEFAULT_TOP_K=10` × 800 chars, 8 recent turns × 2 messages × ≤900 chars, plus
persona/scene/canon (~2K chars), the actor prompt is ≈24–26K chars ≈ **6–7K tokens against
the 26b profile's `-c 8192`** with `LOCAL_LLM_MAX_TOKENS=500` reserved for the reply.
Longer authored personas/scenes or long player messages overflow silently: llama.cpp
context-shifts (evicts oldest KV) and the app never knows — the "Standing facts" block and
persona header are exactly what gets shifted out first.

**Change.**
1. Add a deterministic token *estimator* (`len(chars)/4` is adequate for budget guardrails;
   keep it provider-agnostic, no tokenizer dependency — this preserves deterministic tests).
2. Compute an estimated prompt-token total in the generation stage and attach it to
   `stage_timings`-style turn diagnostics + a turn `warning` when the estimate exceeds a
   configurable fraction (e.g. 85%) of a new `LOCAL_LLM_CONTEXT_TOKENS` setting.
3. Surface the number in the RAG Inspector / Analytics pages (additive API field).
4. **Free exact numbers (verified 2026-07-07):** the provider already returns real
   `prompt_tokens`/`completion_tokens` per call — `LlmResponse.usage` is populated at
   [app/llm/openai_compatible.py:113-116](../app/llm/openai_compatible.py) and read
   **nowhere** (grep `\.usage` finds no consumer). Persist it into turn diagnostics
   alongside the estimate; the estimator then covers pre-flight warnings, the provider
   number covers ground truth and calibrates the estimator.

**Non-goals.** No dynamic re-budgeting yet — first make overflow *visible*, then tune.

**Validate.** Unit tests on the estimator + a live-smoke run with a deliberately oversized
scenario; check the warning fires and llama-server logs show no context shift afterwards.
Effort S–M. — [x] live-validated 2026-07-12 (docs/25 Phase B, `26b-mtp`): sanity run at
`MODEL_CONTEXT_WINDOW_TOKENS=1024` fired 7 preflight + 5 actual-usage warnings (wiring
proven); real run at `16384` (matching `-c`) fired 0/0 (prompts legitimately under the 85%
ratio); `grep -i "context shift" raw/llama-server.log` empty on both runs.

### P0.2 Raise the local context window deliberately (and document the VRAM math)

**Problem.** `-c 8192` for the `26b` profile is the binding constraint; the same profile
family already runs 16384 for `26b-mtp`. A 27B-class model on 16–24 GB with
`--cache-type-k q8_0 --cache-type-v q4_0` (already the profile default) can afford more
context, especially for an A4B-style MoE.

**Change.** Raise `ctx_size` for the `26b` profile to 16384 after a live check (VRAM
headroom + latency), and record the decision + measurements in this doc. Pair with P0.1 so
the app-side budget actually knows the ceiling (`LOCAL_LLM_CONTEXT_TOKENS`).

**Validate.** `scripts/live-smoke.sh` with `LIVE_TURN_COUNT=50`+; watch prefill latency
(stage_timings.generation) — bigger ctx without prompt-shape work (P2.3) raises worst-case
prefill. Effort S. — [ ]

### P0.3 Sentence-boundary chunk trimming in the prompt budget

> **Shipped 2026-07-09.** `_truncate_text` and `_clip_line` now trim at the last sentence
> boundary (`.`/`!`/`?` followed by whitespace-or-end) inside the cap window; if none exists,
> fall back to the last word boundary; if neither exists, hard-cut. The `"..."` omission marker
> is always kept when trimming occurs, and under-cap text is byte-identical to before. Both
> functions changed in lockstep with the same regex/fallback logic. Unit-tested byte-for-byte
> (under-cap identity, sentence-boundary trim, word-boundary fallback, pathological no-boundary
> hard-cut, marker presence, tiny-cap edge case). Full deterministic gate + regression runner
> pass unchanged — ranking evals are unaffected because trim happens after selection.
>
> **Fix note (2026-07-11, cross-review P1).** The sentence boundary won unconditionally
> whenever *any* `.`/`!`/`?` existed in the cap window, even a few characters in (e.g. an
> abbreviation like "Mr."), collapsing the whole chunk to noise while still spending a
> prompt slot. Both `_truncate_text` and `_clip_line` now require a boundary to retain at
> least half the budget before it wins (checked for both the sentence and word tiers),
> falling to a full hard-cut otherwise. Unit-tested: early-terminator repro, exact-half
> boundary, below-half boundary.

**Problem.** `_truncate_text` cuts retrieved chunks mid-sentence at 800 chars with `"..."`
([app/orchestration/context_budget.py:36-41](../app/orchestration/context_budget.py)) —
retrieval can rank the right chunk first and the prompt still loses the fact if it sits
past the cut. Same pattern in the retrieval-query clip (`_clip_line`,
[app/rag/retriever.py:171-174](../app/rag/retriever.py)).

**Change.** Trim at the last sentence boundary (fallback: word boundary) before the cap;
keep the explicit omission marker. Deterministic, testable byte-for-byte.

**Validate.** Unit tests; ranking evals unchanged (trim happens after selection).
Effort S. — [x]

### P0.4 Eval assets before retrieval upgrades (the measurement gate)

> **Shipped 2026-07-11 (this commit) — offline half.** Item 1 (graded corpus), item 2
> (recall@k/nDCG extension), and item 3 (optional `-m semantic` tier) landed; item 4's
> transcript-derived queries remain open (needs real play exports). Corpus:
> `app/evals/semantic_corpus.py` — a single fictional scenario ("Ashen Hold" on the "Ember
> March", tonally close to `bride-for-sarnhold` but original content so nothing collides with
> real scenario data) with 84 chunks: 36 `canon_lore` (incl. a 10-chunk German subset authored
> as Scholar Albrecht's archive), 24 `session_memory` forming an in-session mystery arc, and 24
> `persona_memory` across four NPCs. 36 queries carry 0/1/2 graded judgments (0 implicit —
> absence from the mapping), including a 9-query German subset and **five** same-proper-noun
> distractor clusters (Captain Meravelle, Quartermaster Udo, Orin, the Amber Ring, Thornwell
> Bridge — three facts each, one query isolates each fact) so an embedding that matches only the
> proper noun without discriminating *which* fact answers the query loses recall/nDCG even
> though it would look perfect on the old 9-item pool. Every chunk carries the real payload
> fields production filters key on (`visibility=player` always; exactly one of
> `world_id`/`session_id`/`persona_id` per collection shape). Metrics:
> `app/evals/retrieval_metrics.py` — `recall_at_k` (binary, judgment>=1, plus a strict
> `relevance_floor=2` variant), `ndcg_at_k` (standard log2-discounted DCG/IDCG over the graded
> judgments), `mrr` — pure functions with no embedding dependency, hand-computed unit tests
> (`tests/unit/test_retrieval_metrics.py`). Harness: `app/diagnostics/semantic_benchmark.py`
> (sibling to `embedding_ab.py`) indexes the corpus into an `InMemoryVectorStore` under a given
> embedding provider and scores every query through **two** paths: the production
> `ActorContextRetriever.retrieve_for_actor_with_diagnostics` (dual-query, per-collection
> oversampling, the full additive boost rerank — this doc's own "prefer measuring through
> ActorContextRetriever" guidance, since that is what a real turn actually returns) and a raw
> single-query dense-only merge (cheap to compute from the same index; isolates how much of the
> reranked score is the embedding model itself vs. the deterministic boost layer, useful when
> comparing candidate models for P1.2 since the boost layer is identical across all of them).
> Reports recall@5, recall@10, strict recall@5, nDCG@10, and MRR, aggregated overall and over
> the German subset separately, per path. CLI: `rolerag semantic-benchmark --model
> <fastembed-name> [--model ...] [--keyword] [--top-k N] [--json]` — this is the single command
> for a real embedding-model run. Opt-in tier: `tests/evals/test_semantic_benchmark_opt_in.py`
> (file name deviates from the plan's suggested `test_semantic_benchmark.py` — mypy's module
> resolution collides on identical basenames across `tests/unit/` and `tests/evals/` with no
> `__init__.py` in either, the same reason `test_retrieval_miss_eval.py` isn't named
> `test_retrieval_miss.py`), marker `semantic` (registered in `pyproject.toml`), gated behind
> `ROLERAG_SEMANTIC_MODEL=<fastembed-name>` (unset by default, so the deterministic gate never
> attempts a download) — when run against a real model it asserts **provisional, deliberately
> generous** floors (recall@10 >= 0.3, nDCG@10 >= 0.2) that must be calibrated on the first real
> run, not trusted as a quality bar yet. Deterministic coverage: metric-math unit tests, corpus
> integrity tests (`tests/unit/test_semantic_corpus.py` — unique ids, every judgment resolves to
> a real chunk, German subset non-empty, distractor clusters verified by string-matching the
> shared proper noun across pairwise-distinct facts and confirming no single query conflates two
> facts from the same cluster), and an end-to-end benchmark smoke run with the deterministic
> keyword provider (`tests/unit/test_semantic_benchmark.py` — proves the harness plumbing works;
> keyword-provider scores are explicitly documented as not semantically meaningful). Full gate
> green (737 pytest tests, +35 over the pre-P0.4 baseline; only the 4 opt-in `semantic` tests
> skip, nothing downloads; regression runner unchanged at 85 checks). **Pending (the "online
> half" and beyond):** the real-model benchmark run itself — FastEmbed downloads are blocked in
> the authoring environment (proxy 403), so no real numbers exist yet; floor calibration in the
> opt-in tier once that run lands; item 4's transcript-derived query subset (needs real play
> exports via `export-session`), explicitly out of scope for this commit.
>
> **Runbook shipped 2026-07-12** — [docs/24](24_semantic_benchmark_runbook.md) walks the first
> real-model run end to end: a resilient `rolerag semantic-benchmark` CLI (per-provider failure
> tolerance), the one-command `scripts/semantic-benchmark.sh` runner (writes the `--json`
> artifact, prints suggested floors via `scripts/lib/suggest_floors.py`, optionally runs the
> opt-in pytest tier), and the floor-calibration procedure for
> `tests/evals/test_semantic_benchmark_opt_in.py`.

**Problem.** The deterministic harness uses keyword embeddings + `InMemoryVectorStore` — it
pins engine logic, **not semantic quality**; `embedding-ab` ranks a small seeded event set
(BACKLOG #10 ended "candidates tied" — on fixtures too small to discriminate). No
graded-relevance corpus, no distractor-heavy fixtures, no German queries existed until this
commit's corpus shipped — every improvement below was unmeasurable before it, and remains
unmeasured (not unmeasurable) until the owner runs a real embedding model through it.

**Change.**
1. Build a fixture scenario pack ~10× `bride-for-sarnhold` (LLM-generated lore is fine)
   with **graded relevance judgments** (query → expected chunk ids, graded 0–2) including
   adversarial distractors (same proper nouns, different facts) and a German query subset.
2. Extend `embedding-ab` to run recall@k / nDCG over this pack with *real* FastEmbed
   models (offline, no LLM), not just the seeded events.
3. Optional pytest tier behind a marker (`-m semantic`, excluded from CI default) running
   the same corpus with real embeddings.
4. Derive part of the query set from real play transcripts (`export-session`) to counter
   fixture-author bias; unit-test the new recall@k/nDCG math with the deterministic keyword
   provider so the metric layer itself stays pinned.

**Validate.** Self-validating — this *is* the validator. Effort M–L (hand-grading
relevance judgments is the long pole; today's `embedding-ab` pool is only 9 items —
5 seeded events + 2 smalltalk + 2 lore chunks — which is why BACKLOG #10 "tied"). —
[~] offline half shipped (corpus, metrics, harness, CLI, `-m semantic` marker — see the
shipped note above); runbook + one-command runner + floor calibration shipped (see
[docs/24](24_semantic_benchmark_runbook.md)); **real-model numbers landed 2026-07-12**
(`all-MiniLM-L6-v2`: reranked overall recall@10 0.824 / nDCG@10 0.761, German recall@10
0.630 — docs/24 run log; floors calibrated to 0.75/0.70/0.55) — **the P1 gate is open**;
item 4's transcript-derived queries still open.

---

## P1 — retrieval reach and quality (gated on P0.4)

### P1.1 Hybrid sparse+dense retrieval in Qdrant

**Problem.** Retrieval is dense-only ([app/rag/vector_store.py](../app/rag/vector_store.py));
the lexical-overlap boost in [app/rag/ranking.py](../app/rag/ranking.py) can only **reorder
chunks the dense search already returned** — a dense miss is unrecoverable. Fantasy lore is
proper-noun-heavy, exactly where a 384-dim MiniLM misses most; at 10–100× corpus size the
misses dominate.

**Change.** Add opt-in sparse vectors (FastEmbed BM25/SPLADE family) as a named vector per
collection, server-side fusion (RRF) in Qdrant; `InMemoryVectorStore` gets a deterministic
in-memory BM25 for test parity (the repo invariant). Keep the deterministic rerank on top —
it stays the explainability layer. Config: `RAG_HYBRID_SEARCH=off|rrf`, default `off`.

**Validate.** P0.4 corpus recall@k before/after; retrieval diagnostics must label which
leg (dense/sparse) surfaced each candidate (RAG Inspector addition). Effort L. — [ ]

**Verified implementation notes (2026-07-07 adversarial pass).**
- The sparse/prefetch legs MUST carry the same visibility/scope `query_filter` as the dense
  leg ([vector_store.py:321-366](../app/rag/vector_store.py)) or the visibility boundary breaks.
- Named-vector layouts break the existing size check: `ensure_collection` reads
  `.config.params.vectors.size` ([vector_store.py:191-193](../app/rag/vector_store.py)),
  which becomes a dict under named vectors — the check needs a per-layout branch.
- fastembed 0.8.0 (in the venv) ships `Qdrant/bm25` with a `language` parameter incl.
  `german`; qdrant-client is 1.18.0.
- The in-memory BM25 parity leg can reuse `content_terms`/`_stem`
  ([ranking.py:250-272](../app/rag/ranking.py)) as its deterministic tokenizer.
- The legacy `client.search` branch is pinned by
  `tests/unit/test_vector_store.py:71-133`; RRF may shuffle
  `event_key_retrieval` top-rank pins — add a hybrid-variant deterministic eval rather than
  loosening the existing one.

### P1.2 Embedding model upgrade path (multilingual)

> **Runbook shipped 2026-07-10** — change item (1) lives in
> [docs/23](23_embedding_migration_runbook.md) (reset-index → re-ingest → reindex-memories,
> fingerprint guards, failure modes, rollback). Change items (2) benchmark and (3) default
> swap remain gated on the P0.4 corpus.

**Problem.** `all-MiniLM-L6-v2` is English-only and the weakest quality lever; scenarios
may be authored/played in German. Swaps are configurationally trivial (`EMBEDDING_MODEL`)
but operationally undocumented — dimension changes break collections
(`VectorStoreDimensionMismatch`).

**Change.** (1) Write the migration runbook: `reset-index` (all collections) →
`reindex-memories` per session → re-`ingest`; (2) benchmark multilingual candidates on the
P0.4 corpus via extended `embedding-ab`; (3) swap the default only on evidence, per house
rules.

**Candidate list (corrected 2026-07-07, verified against fastembed 0.8.0 in the venv):**
`paraphrase-multilingual-MiniLM-L12-v2` (384-dim, symmetric),
`jinaai/jina-embeddings-v2-base-de` (768-dim, symmetric),
`intfloat/multilingual-e5-large` (1024-dim, **query/passage prefixes required**).
`multilingual-e5-small/base` and `bge-m3` are NOT supported by fastembed 0.8.0 — struck.

**Prefix caveat (verified).** fastembed 0.8.0's `query_embed`/`passage_embed` add **no**
prefix for the e5 family, and the app's `EmbeddingProvider` protocol is symmetric-only
([app/rag/embeddings.py:13-19](../app/rag/embeddings.py)). If e5-large is benchmarked,
Settings-driven prefixes must land first (`EMBEDDING_QUERY_PREFIX`/`EMBEDDING_DOCUMENT_PREFIX`,
default `''` = byte-identical), applied at the call sites — query:
[retriever.py:41](../app/rag/retriever.py); documents:
[ingestion.py:69](../app/rag/ingestion.py),
[indexer.py:52,132](../app/memory/indexer.py); dedup both sides:
[memory_dedup.py:79-82](../app/orchestration/stages/memory_dedup.py) — **not** via Protocol
default methods (the ~9 structural test fakes inherit nothing from a Protocol). Otherwise
an e5 benchmark run is invalid. The two symmetric candidates need none of this.

**Validate.** P0.4 metrics incl. the German query subset; end-to-end via live-smoke.
Effort M (runbook S, benchmark M; +S if prefixes needed). — [ ]

### P1.4 Embedding-model identity fingerprint (adversarially verified, new)

> **Shipped 2026-07-09.** Both stores now carry the embedding-model identity *inside* the
> vector store, atomic with the collection lifecycle: `QdrantVectorStore` writes a reserved
> sentinel meta point (fixed `uuid5` id, `__rolerag_sentinel__` payload marker) per collection;
> `InMemoryVectorStore` carries a parity `dict[RagCollection, str]`. `ensure_collection` gained
> an opt-in `model_key: str | None = None` parameter and a new `VectorStoreModelMismatch`
> (`ValueError` subclass, sits beside `VectorStoreDimensionMismatch`, message points at this
> section as the migration runbook). Byte-identical when `model_key` is omitted — the check/adopt
> logic never runs. Real callers (`ingest_document`/`ingest_lore_manifest`, `MemoryIndexer`, and
> every `composition.py`/`cli.py` wiring site) now pass `settings.embedding_model`; the CLI
> `ingest`/`ingest-scenario-lore`/`reindex-memories`/`start-session` paths already had
> `ValueError`/`Exception` catch-alls, so a mismatch fails loud with the runbook hint for free.
> Live-turn indexing (`stages/memory.py`'s `except Exception` around `index_memories`) already
> turns any indexing exception into a write-blocking turn warning without losing the SQLite
> write — verified with a dedicated orchestrator test using the real exception type, not just a
> generic stub error. `doctor --check-qdrant` gained a read-only fingerprint scan
> (`QdrantVectorStore.read_model_fingerprint`, never adopts) across all three collections and
> fails loud on any mismatch, with the runbook hint attached. `drop_collection`/`reset-index`
> clear the fingerprint for both stores (Qdrant: the sentinel lives inside the dropped
> collection; in-memory: an explicit dict pop). Backward compatible: an unfingerprinted
> collection (pre-P1.4 or a caller that never passes `model_key`) adopts the current model
> identity on first fingerprinted contact instead of raising. The sentinel is excluded from
> every Qdrant search two ways — it carries no `visibility` payload field (never matches the
> existing `must` clause) and an explicit `must_not` on the sentinel marker — proven by a
> dedicated embedded-`QdrantClient(":memory:")` test. Unit-tested on both stores (mismatch
> raise, drop-then-recreate clears, fingerprint adoption, sentinel-never-in-results,
> read-is-read-only) plus a `doctor --check-qdrant` mismatch/pass test. Full deterministic gate
> + regression runner pass unchanged.
>
> **Fix note (2026-07-11, cross-review P4).** The sentinel exclusion covered every search
> path but not `app/diagnostics/live_checkpoint.py`'s unfiltered `_qdrant_count` helper: the
> sentinel is a real point, so an unscoped `client.count()` (the canon-lore path) counted it
> too, inflating `canon_lore_count` by 1 and letting the checkpoint's `>= 1 lore` gate pass on
> a stamped-but-empty collection — diverging from `InMemoryVectorStore`, whose fingerprint is a
> separate dict entry, never a counted point. `_qdrant_count` now applies the same `must_not`
> exclusion `_build_qdrant_filter` uses on every search path. Unit-tested against an embedded
> `QdrantClient(":memory:")`: stamped-empty counts 0, stamped+N counts N, session-scoped counts
> still exclude the sentinel.

**Problem (confirmed).** The only guard on the index is vector **size**
([vector_store.py:188-196](../app/rag/vector_store.py) Qdrant, `:72-76` in-memory); the
embedding model's identity is stored nowhere. Swapping `EMBEDDING_MODEL` between
same-dimension models — e.g. the default `all-MiniLM-L6-v2` (384) to P1.2's own candidate
`paraphrase-multilingual-MiniLM-L12-v2` (384) — silently mixes incompatible vector spaces:
old chunks stay, new memories upsert beside them
([indexer.py:41-61](../app/memory/indexer.py)), scores blend meaninglessly in the additive
rerank, and fail-open search hides it. The trap sits directly on P1.2's upgrade path.

**Change.** Store a model fingerprint **inside the vector store** so fingerprint and
collection lifecycles are atomic (sentinel meta point per Qdrant collection; dict in
`InMemoryVectorStore` — parity): `ensure_collection` gains a model-identity check and a new
`VectorStoreModelMismatch`. CLI paths (`ingest`, `reindex-memories`) fail loud with the
P1.2 runbook as the remedy; during a live turn the memory stage already wraps indexing in
`except Exception` ([stages/memory.py:262-264](../app/orchestration/stages/memory.py)), so
in-play protection is write-blocking + warning — additionally surface the mismatch in
`doctor` for a loud signal. `drop_collection`/`reset-index` must clear the fingerprint or
the runbook bricks on a stale one. Effort S–M. — [x]

**Validate.** Unit tests both stores (mismatch raise, drop-then-recreate clears);
`doctor --check-qdrant` surfaces it; P1.2 runbook exercised end-to-end once.

### P1.3 Structure-aware chunking + contextual chunk headers

> **Shipped 2026-07-16 — offline half (opt-in, default off).** `ChunkingConfig` gained
> `structure_aware: bool = False` (`app/rag/chunking.py`), paired with
> `Settings.rag_structure_aware_chunking` / `RAG_STRUCTURE_AWARE_CHUNKING=false`
> (`app/config.py`, `.env.example`) and threaded into every Settings-driven
> `ChunkingConfig(...)` construction site (`app/composition.py`'s
> `auto_ingest_scenario_lore` -- the composition root shared by the CLI's `start-session`
> and the API's `POST /sessions` -- both CLI `ingest`/`ingest-scenario-lore` commands, and
> the deterministic smoke runner). `chunk_text` gained a keyword-only
> `doc_title: str | None = None`, ignored entirely on the legacy path -- proven by a
> golden-baseline test that hardcodes the exact output of the PRE-P1.3 chunker on three
> fixtures (nested headings, an oversized paragraph, plain multi-paragraph), captured
> before `chunking.py` was touched, including regardless-of-doc_title byte-identity.
> Flag on: the document splits on markdown ATX headings (levels 1-6) into sections with a
> hierarchy-aware path (`A › B › C`; a new heading pops every stack entry at its level or
> deeper first); paragraphs accumulate within a section only -- chunks never straddle a
> heading boundary, and overlap seeding resets per section. An oversized block now
> cascades sentence-boundary packing -> word-boundary packing -> the original fixed-window
> hard split (last resort only, e.g. a single unbroken token) -- never a mid-word cut while
> any boundary exists at some tier. Every emitted chunk gets a first line
> `<doc title> › <section path>` plus a blank line, skipping whichever part is absent (no
> header line at all if both are absent), and deduplicating the common lore shape whose
> only H1 is the document's own title -- when the section path's root segment already IS
> the doc title, the title is not repeated (`Title › Sub`, never `Title › Title › Sub`);
> the header is free budget-wise (a chunk may
> exceed `chunk_size_chars` by the header's length -- documented on `chunk_text`, simpler
> and more predictable than shrinking the body budget). `ingest_document` derives
> `doc_title` from the document's first `# ` (H1) line, else the filename stem, and passes
> it unconditionally (harmless on the legacy path).
>
> **Tag decision -- deviation from this section's original "store section path in chunk
> metadata/tags" text, with evidence.** Did NOT add a `section:<path>` chunk tag. Before
> adding it, checked every reader of `chunk.tags` at retrieval/ranking time as this item's
> own validate step implies: `app/rag/ranking.py`'s `_lexical_overlap_boost` folds
> `content_terms(" ".join(chunk.tags))` into the lexical rerank score of **every** chunk
> with no filter gate (`ranking.py:294`) -- so a section-path tag is not inert metadata, it
> would silently perturb rerank scores for any structure-aware-ingested lore chunk whose
> section title happens to share vocabulary with a query. That is exactly the kind of
> semantic-quality effect this whole feature is gated on measuring before shipping, not
> something to introduce as an unmeasured side effect of a "diagnostics-only" tag. (Tag
> *filtering* is separately confirmed inert on this path -- actor retrieval never sets
> `RetrievalFilter.tags`, and `app/rag/lexical.py`'s Lane B scores `MemoryEpisode.tags` for
> session-memory lexical slicing, never lore `RagChunk.tags` -- but ranking's unconditional
> read is the one that actually matters here.) Section identity still reaches the embedded
> text via the contextual header itself, which is this item's core ask regardless of the
> tag question. Pinned by
> `test_ingest_document_structure_aware_does_not_add_section_tags`.
>
> **#86 interplay.** Flipping the flag changes chunk text -> chunk ids
> (`sha256(source:index:text)`), so the content-fingerprint skip (backlog #86) sees a
> different id set on the next ingest per source and correctly falls through to a full
> re-ingest rather than skipping -- pinned by
> `test_ingest_document_structure_aware_flag_flip_changes_chunk_ids_and_reingests`.
>
> **Validation status (mirrors § P0.4's own honesty pattern).** Offline half only.
> Deterministic chunking/ingestion/config/wiring tests are green
> (`tests/unit/test_chunking.py`, `test_ingestion.py`, `test_config.py`,
> `tests/integration/test_cli.py`) -- but per this doc's own measure-first workflow, a
> keyword-embedding suite cannot show whether the contextual header actually helps
> recall/nDCG; it only proves the mechanism is wired and byte-identical when off. Semantic
> validation is pending owner-side: `rolerag semantic-benchmark --model
> sentence-transformers/all-MiniLM-L6-v2` with the flag on vs off against the docs/24
> calibrated floors (recall@10 >= 0.75, nDCG@10 >= 0.70, German recall@10 >= 0.55), then
> live-smoke, before any default flip.

**Problem.** Chunking is blind paragraph accumulation
([app/rag/chunking.py](../app/rag/chunking.py)): chunks straddle markdown section
boundaries; oversized blocks split at fixed character offsets (mid-word,
[chunking.py:42-48](../app/rag/chunking.py)); a chunk's embedded text carries no document
or section identity, so "the treaty" in chapter 3 embeds identically to "the treaty" in an
unrelated document.

**Change.** (1) Split on markdown headings first, accumulate paragraphs within a section;
(2) split oversized blocks at sentence boundaries; (3) prepend a one-line contextual header
(`<doc title> › <section path>`) to the **embedded** text (and keep it in the prompt text —
it aids the model too); store section path in chunk metadata/tags for diagnostics.
Chunk ids already hash `source:index:text`, so re-ingest replaces cleanly.

**Validate.** Chunking unit tests; P0.4 recall (headers typically help both legs of P1.1).
Effort M. — [~]

---

## P2 — scale, latency, and long campaigns

### P2.1 Qdrant payload indexes (and optional quantization)

> **Shipped 2026-07-16.** `QdrantVectorStore.ensure_collection` now creates keyword payload
> indexes for all six fields every search filters on (`visibility`, `world_id`, `session_id`,
> `persona_id`, `scene_id`, `tags`) via the existing `_require_qdrant_models()` lazy-import
> pattern -- on **both** the freshly-created-collection path and the already-exists
> early-return path, so a pre-P2.1 collection gets indexed on its next `ensure_collection`
> contact, not just brand-new ones (the roadmap's own callout). Idempotent -- repeat contact
> re-requests the same six indexes, which Qdrant accepts as a no-op -- and fail-open for the
> index step specifically: an unexpected `create_payload_index` error is caught around the
> whole per-collection loop (index creation is a query-speed optimization, not a correctness
> requirement -- filtering still works without an index -- so it can never break collection
> creation/use). Opt-in INT8 scalar quantization shipped alongside: new Settings field
> `qdrant_scalar_quantization` (`QDRANT_SCALAR_QUANTIZATION`, default `false`) threads
> through `app.composition.build_vector_store`, the single function that actually
> constructs `QdrantVectorStore` -- `app.cli._build_vector_store` is a plain alias of that
> same function, not an independent construction path, now pinned by an identity test in
> `tests/unit/test_composition_config_parity.py` against the #48/#67 drift class. When
> enabled, `create_collection` receives `quantization_config=ScalarQuantization(scalar=
> ScalarQuantizationConfig(type=ScalarType.INT8))`; when off (default) the call is
> byte-identical to pre-P2.1 -- no `quantization_config` key at all, not even `None`. Applies
> only at first collection creation, as documented on the setting (`rolerag reset-index` +
> re-ingest to apply it to an existing collection). `InMemoryVectorStore` unchanged, as
> anticipated -- indexes/quantization are pure Qdrant-side filtering/storage details, so
> search semantics and the existing parity tests are unaffected. Unit-tested: both
> `ensure_collection` paths call `create_payload_index` for all six fields (incl. idempotent
> repeat contact on the already-exists path); the quantization kwarg is present with INT8
> when enabled and entirely absent when off; a monkeypatched `create_payload_index` failure
> leaves `ensure_collection` succeeding and the store usable (upsert + search still work
> afterward). Validated: full deterministic gate (`ruff`, `mypy --strict`, pytest, regression
> runner) green. Caveat: qdrant-client's embedded local mode (`QdrantClient(":memory:")`,
> used by every test here, including the filter-parity suites) accepts `create_payload_index`
> as a no-op with a `UserWarning`, so the fail-open fallback is exercised only by the
> dedicated monkeypatched-failure test, not incidentally by the others. **Still open
> (owner-side):** live latency numbers on a real Qdrant server at the P0.4 corpus scale --
> this doc's own validation ask -- since payload indexes only pay off against a real
> server's query planner, not the embedded local-mode client the deterministic suite runs
> against.

**Problem.** Collections are created with vectors only
([vector_store.py `ensure_collection`](../app/rag/vector_store.py)); every search filters
on `visibility` + a scope id without payload indexes — fine at POC scale, a full-scan
filter cost at 10k–1M points.

**Change.** Create keyword payload indexes for `visibility`, `world_id`, `session_id`,
`persona_id`, `scene_id`, `tags` in `ensure_collection` (idempotent — and it must also run
on the **already-exists early-return path**, [vector_store.py:190-196](../app/rag/vector_store.py),
or existing collections never get indexed). Expose optional scalar quantization behind
config, default off. `InMemoryVectorStore` needs no change (indexes are an implementation
detail of filtering, semantics identical — parity holds).

**Validate.** Existing tests (semantics unchanged); latency numbers on the P0.4 corpus.
Effort S. — [x]

### P2.2 Long-campaign preset (enable the shipped-but-off machinery, with evidence)

> **Validation procedure ready 2026-07-12** —
> [docs/25](25_live_validation_runbook.md) Phase D chains the full preset (including § C2's
> min-age/batch-cap knobs) with `LIVE_TURN_COUNT=100`, and states exactly what evidence to read
> from the checkpoint JSON: `persisted.consolidated_memory_count` /
> `consolidation_summary_count` (proves consolidation fired) and the late-recall callback
> misses (proves it didn't cost recall). The run itself is still pending.
>
> **First run 2026-07-12 — FAILED at turn 65 (finding recorded, docs/25 § If something
> fails).** Full preset on `26b-mtp` (`THRESHOLD=40 MAX_IMPORTANCE=2 MIN_AGE=10 BATCH_CAP=15
> DEDUP=0.92 RECENCY=0.02 LIVE_TURN_COUNT=100`, artifact `/tmp/rolerag-live-D`): 13/14 report
> steps PASS, then `CheckpointError: event amber_ring_token has no persisted matching memory`
> at the turn-65 callback inspection; checkpoint JSON `status: in_progress`, 64 turns.
> **Root cause is probe vocabulary, not memory loss and not consolidation**: the turn-55 fact
> was extracted, persisted (importance 2, `player`, un-consolidated), and its summary reads
> *"The player gave Iria Vale an amber ring as a symbol of a pact/promise"* — `semantic_match`
> requires one term from each group and the third group is `("wear","sign","token")`, none of
> which survive the extractor's legitimate paraphrase ("symbol … pact/promise"). The pipeline
> did its job; the harness's synonym list didn't anticipate the paraphrase. Positive evidence
> from the same run before the abort: consolidation **fired** (15 rows tagged `consolidated` —
> exactly one `BATCH_CAP=15` batch — plus 1 roll-up summary), all five pre-turn-50 probes
> passed inspection, and the turn-17 `hollow_bookend_note` memory **survived the consolidation
> pass un-folded and still probe-matchable** at turn 64 (its turn-95 callback never ran).
> Open decision before re-running: widen `amber_ring_token`'s third term group (e.g. add
> "symbol"/"pledge"/"promise") as a harness-vocabulary fix — the docs/25 no-loosening rule
> targets assertions and thresholds; a synonym gap that misses a semantically intact memory is
> a probe false-negative — or keep the vocabulary strict and accept paraphrase-sensitivity as
> part of what the probe measures. Validate stays unchecked either way until a full 100-turn
> pass exists.
>
> **Second run 2026-07-12 (probes widened) — FAILED at turn 31 with a REAL defect (fixed as
> BACKLOG #72).** `CheckpointError: event silver_compass has no persisted matching memory` —
> and this time the memory truly wasn't there: turn 21's diagnostics show the curator
> extracted the compass-gift episode and the **always-on lexical write-dedup dropped it**
> (`memory dedup dropped 1 duplicate candidate(s)`, `memory_written: False`). At the shared
> `COVERAGE_THRESHOLD=0.5`, every natural terse phrasing of the gift scores 0.50–0.56
> coverage against the compass-free dawn-promise memory on frame vocabulary alone
> ({player, iria, vale, keep, return}) — verified by running `is_covered_by_summaries`
> against the run's own SQLite rows. Run-dependent: run 1's longer phrasing survived, which
> is also why 8–50-turn acceptance runs never tripped it. Not preset-related — the semantic
> `0.92` knob was innocent (measured cosine for the pair: 0.50); this dedup is active in
> every configuration. Fix (#72): dedicated `WRITE_DEDUP_COVERAGE_THRESHOLD=0.75` for the
> dedup call site, deterministic-fallback coverage unchanged at 0.5, and both dedup warnings
> now name the dropped summaries. Third run pending with the fix in place.
>
> **Third run 2026-07-12 (#72 fix in) — FAILED at turn 38; P2.2 validation now BLOCKED ON
> P1.1 (finding #73).** `silver_compass` **passed** (#72 verified live), then
> `CheckpointError: event blue_seal_trust_rule was not selected by callback retrieval`. The
> memory existed, matched, and was indexed; it ranked **12 of 17** candidates (top-5
> selected) for the callback "I ask Iria what rule we agreed to use before trusting any
> message" — whose direct answer it is. Offline reproduction (`reindex-memories` into a
> disposable Qdrant + `inspect_story_event` against the run's SQLite): dense score 0.3221
> vs 0.45–0.58 for scene-vocabulary chatter — `all-MiniLM-L6-v2` misses the
> "rule for trusting messages" ↔ "only trust blue-wax-sealed messages" paraphrase; the
> lexical leg boosted it +0.15, not enough. **Recency exonerated**: identical rank at
> `RAG_RECENCY_WEIGHT=0.0` and `0.02`. Root cause is pool size: the #72 fix (correctly)
> persists more memories — 48 indexed by turn 37 vs 36 pre-fix — and dense-only retrieval
> stops surfacing direct answers at that scale; run 1 passed the same probe on a luckier
> phrasing/pool. Candidate-swap experiment on the live case: `paraphrase-multilingual-
> MiniLM-L12-v2` ranks it **worse** (18/48 dense vs MiniLM's 10/48), and
> `jinaai/jina-embeddings-v2-base-de` fails to load under current onnxruntime (broken graph
> fusion) — so P1.2 alone does not fix this class; **P1.1 hybrid sparse+dense and/or P2.5
> rerank are the load-bearing fixes**, with this query/pool as their live acceptance case.
> The preset machinery itself validated fine across the three runs: consolidation fired
> (15 folded + 1 summary, run 1), SQLite/Qdrant parity held, the turn-17 probe survived a
> consolidation pass. Re-run P2.2 after P1.1 lands.
>
> **Fourth run 2026-07-13 — FAILED at turn 65; not a recall defect (harness-clarity gap
> filed as #74).** Run 4 cleared every probe that killed runs 2–3 (compass, blue seal, key,
> three-tap all passed; consolidation fired at turn 54, "rolled up 15 memories into 1
> summary"), then `amber_ring_token has no persisted matching memory` — because turn 55,
> the probe's *definition* turn, ended as a **controlled failure** (critic rejected the
> draft; `outcome: controlled_failure`, "No memory or world state was changed"). Invariant
> #4 working exactly as designed: the fact never entered the world, so the turn-65 callback
> correctly found nothing. This is the known ~6.7% local fail-closed rate landing on one of
> the 8 probe-definition turns — placement luck, expected to hit a definition turn in a
> substantial fraction of 100-turn runs. Two consequences: (1) a fully-green 100-turn P2.2
> run needs P1.1 (#73) *plus* either placement luck or #74's precise handling; re-rolling
> until green is exactly what docs/25 forbids, so P2.2 stays blocked on P1.1 with #74 as
> the harness-accuracy fix. (2) The checkpoint's current report for this case — a recall
> miss at the callback, 10 turns after the actual event — is misleading; #74 makes it fail
> fast at the definition turn with the real cause named.
>
> **Update 2026-07-14 — unblocking path revised; P2.2 is no longer blocked on P1.1.** The
> four-run dossier above was synthesized into [docs/26](26_memory_retrieval_redesign.md)
> (four independent redesigns, twelve adversarial judge verdicts, one recommended target).
> Its key empirical finding, verified against the preserved D3 artifact: #73's blue-seal
> instance is a **dead `canon_importance_floor` predicate on validated-good
> `build_standing_facts` machinery**, not a retrieval ceiling — the memory carries the right
> durable-fact tags but the D3 pool's curator importance distribution (9×1 / 38×2 / 1×3,
> zero at the floor of 4) means the floor never admits it. Lane A tag-eligible canon pinning
> (backlog **#78**) fixes that instance retrieval-free; Lane B lexical slice quotas (**#79**)
> cover the 94%-non-canon remainder of the pool; #74's harness fail-fast lands first
> (Stage 0, **#75**). The P2.2 re-run is docs/26 Stage 5 (**#80**): at least two clean
> 100-turn runs before any default flips, with the #73 acceptance reinterpretation (the fact
> reaches the actor prompt via pinning, not via dense rank) recorded explicitly here. P1.1
> drops to a conditional escalation rung (docs/26 Stage 6), pulled only by a live miss
> neither lane covers — its spec below is unchanged, plus docs/26's RRF score-scale caveat.
>
> **Phase E run 1 attempt (2026-07-14, `26b-mtp`, full Stage 5 preset) — ABORTED at the
> turn-38 blue-seal inspection; root cause split in two, one fixed session-side, one
> recorded as open.** `CheckpointError: event blue_seal_trust_rule was not selected by
> callback retrieval` (artifact `/tmp/rolerag-live-E1`, checkpoint `status: in_progress`,
> 37 turns; run DB recovered from its WAL on a scratch copy for forensics — the artifact
> itself untouched). Both lanes were demonstrably live at turn level before the abort:
> the standing-facts block grew 1→5 pinned facts (557 chars) and never dropped, and 65
> slice-guaranteed selections fired across 37 turns. The two causes:
> (1) **Harness fidelity gap — FIXED.** `inspect_story_event` replayed selection
> dense-only: Lane B's slice quotas apply only inside `TurnRetrievalStage`, so the
> checkpoint asserted the pre-#79 dense-only bar — a selection path the engine no longer
> serves prompts from. Offline replay against this run's own 47-memory pool ranks the
> blue-seal memory **#1 of 16** lexically (score 6.83, matched `{messag, trust}`), i.e.
> the real turn-38 selection would have served it; only the inspection replay could not.
> Fix: the inspection now mirrors the stage's slice application exactly (same scorer
> inputs, settings-driven quotas, same prompt window; quota 0 stays a byte-identical
> no-op; deliberately NOT fail-open — a scorer error in a validator fails loudly).
> Pinned by `test_inspect_story_event_applies_lexical_slice_quotas_like_the_turn_stage`.
> The assertion itself is untouched — a #74-class precision fix, not a loosening.
> (2) **Lane A tag-vocabulary drift — OPEN, recorded.** This run's curator re-roll
> tagged the blue-seal memory `["player_decision", "quest_rule", "information_filter"]`
> (D3: plain `rule`); `CANON_TAGS` eligibility is exact-set-intersection, so
> `quest_rule` ∉ family and the fact was **not pinned** — Lane A's guarantee is
> curator-tag-vocabulary-dependent under MTP non-determinism. The #73 acceptance case
> stays covered by Lane B (rank #1 ≤ quota 2), so this does not block the Phase E
> re-runs; it does mean the acceptance reinterpretation may need to lean on the "and/or
> Lane B" clause docs/25 Phase E already provides. Candidate hardening (owner decision,
> deliberately not built mid-validation): token-split tag matching (`quest_rule` →
> `{quest, rule}` ∩ family) or a deterministic-extractor trigger for rule-declaration
> phrasings ("we will trust only …") emitting canonical tags at importance 4; either
> widens what the flag-on guarantee pins and interacts with §3.3.1's supersession scope.
>
> **Phase E run 1 (2026-07-15, `26b-mtp`, full Stage 5 preset, inspection slice-parity fix
> in) — FAILED at the turn-95 `hollow_bookend_note` inspection with the docs/25-anticipated
> consolidation finding; 94/94 turns success and 7/7 prior inspections PASSED first.**
> `CheckpointError: event hollow_bookend_note has extracted memories missing from Qdrant`
> (artifact `/tmp/rolerag-live-E1`, preserved incl. run DB; two earlier aborts the same
> evening were environment, not engine: an Ollama-pinned 9.8 GB model OOMing Metal at first
> decode, then a battery-power idle sleep freezing turn 73's wall clock into a 504).
> The failing probe is exactly docs/25 Phase D's named validation question — and the answer
> is measured, not assumed: **the preset (`THRESHOLD=40 MAX_IMPORTANCE=2 MIN_AGE=10
> BATCH_CAP=15`) does not preserve long-gap best-effort facts.** Consolidation fired 3×15
> batches (45 of 109 memories folded); the turn-17 bookend memory (`178d4dc3`, importance 2,
> tags `item/location/secret` — not durable-commitment family, so neither `_PRESERVE_TAGS`
> nor Lane A protected it, correctly per the docs/26 §8 Q1 minimal contract) was folded, and
> its batch's roll-up summary (`d4335302`, 15→1 compression) retains none of
> `bookend`/`note`/`shelf` — the fact is gone from every prompt-reachable path (dense index,
> Lane B pool, Lane A block), surviving only as the audited SQLite tombstone + `source_ids`
> tag. Positive evidence from the same run: blue-seal passed its inspection through the Lane
> B slice (the 2026-07-14 fidelity fix verified live), Lane A saturated the 8/900 cap at
> turn 56 with one visible eviction (docs/26 §8 Q2's case, observed), 173 slice-guaranteed
> selections with summed-IDF scores 0.69–19.08 (deciles 2.0/3.04/3.7/4.03/4.91/5.42/6.29/
> 8.58/10.46 — run 1's half of the `min_slice_score` derivation data), zero definition
> retries needed, zero fail-closed turns. **Open decision (owner):** the probe predates the
> Q1 minimal contract and asserts hard index-retention for a fact the contract classes as
> best-effort; no knob set both exercises consolidation inside 100 turns and preserves an
> unprotected turn-17 fact to turn 95 (MIN_AGE>78 or MAX_IMPORTANCE=1 each starve the
> threshold). Candidate paths: (a) make the probe's definition turn a durable declaration so
> the fact enters the protected family (scenario change, assertions untouched); (b) refine
> the assertion to contract-tier semantics (folded original whose roll-up carries the match
> = pass, roll-up lost it = recorded loss, gating only for guarantee-tier facts — the same
> correction the SQLite/Qdrant count parity check received 2026-07-11); (c) tame the preset
> and give up in-run consolidation evidence; (d) decouple: validate Lanes A+B with
> consolidation off, consolidation preset separately. Recorded here pending the owner call —
> no assertion, threshold, or preset value was changed to route around the failure.
> **Owner decision 2026-07-16: (b) contract-tier assertion.** `_validate_attribution` now
> classifies matching-but-unindexed memories via `build_event_attribution`: non-consolidated
> missing = hard fail (unchanged); folded + roll-up carries the match = satisfied through
> the summary; folded + roll-up lost it = hard fail for guarantee-tier (durable-commitment
> family, `effective_canon_tags`) facts, recorded loss
> (`quality_metrics.consolidation_lost_matches`) for best-effort facts. The long-gap probe
> becomes the preset's compression-quality meter. Semantics + the four-case split pinned by
> `test_build_event_attribution_contract_tier_consolidation_split`; docs/25 Phase E carries
> the same note. Phase E runs restart under these semantics.
>
> **Phase E restart run (2026-07-16, contract-tier semantics in) — failed fast at turn 55
> with the first live retry-survival data point.** `amber_ring_token`'s definition turn
> ended `controlled_failure`; the #80 retry was consumed and ALSO ended
> `controlled_failure` — identical failure envelope (same scripted message, same critic
> rejection text, same warnings) on both attempts. This is a fully correlated double
> failure, exactly the correlation docs/26 §4 warned about when it rejected the naive
> `0.067² ≈ 0.45%` independence math — recorded as data point #1 for the measured delta
> (N=1: retry did not rescue). #75's fail-fast named the cause at turn 55 instead of a
> misleading recall miss at 65; #80's offset bookkeeping held (both attempts recorded,
> step-keyed). 55 prior turns: all success, 5/5 inspections passed. Artifact preserved
> (scratchpad E1-attempt5). Not a preset or lane defect — the known ~6.7% fail-closed
> rate landing on a probe definition turn twice; rerun follows, each attempt recorded.
> **Second occurrence same day (post-rebase run): identical correlated double
> controlled_failure on the same turn — 5 of 7 historical attempts on this line have now
> fail-closed (~70% observed vs ~6.7% global).** A definition line with a ~70% fail-closed
> rate measures critic strictness on that draft class ("press a ring into her palm and
> tell her to wear it as a sign of my word"), not recall, and no retry budget rescues it.
> **Owner decision 2026-07-16: probe-repair reword** to the silver_compass-shaped
> declarative gift ("I give Iria an amber ring as a token of my promise to return."),
> same fact/term-groups/callback — per the run-1 vocabulary-widening precedent (probe
> false-negative repair, not an assertion or threshold change). The critic-rejection
> root cause on the original phrasing stays unknown by design (drafts are never
> persisted, invariant #4); a critic-reason diagnostics field was considered and
> deferred — reopen if the reworded line ever fail-closes again. Retry-delta ledger so
> far: N=2 consumed retries, both correlated double failures, zero rescues.
>
> **Phase E run 1 (E1) — first CLEAN 100-turn pass (2026-07-19, `26b-mtp`, reworded probe,
> contract-tier semantics in).** All 100 turns success (0 controlled failures — the ~6.7%
> rate would predict ~7), 8/8 probes attributed, `report.md` all-PASS, `status: pass`. Lane
> A: 8 pinned facts at cap, **never dropped, zero cap evictions** across 100 turns. Lane B:
> 184 guaranteed slice selections, summed-IDF score deciles
> 2.15/3.04/3.47/3.89/4.80/5.60/6.47/7.28/9.29 (run-1 half of the `min_slice_score`
> derivation set). Consolidation fired hard: 60 folded + 4 roll-up summaries. The two
> `callback_recall_misses`/`retrieval_selection_misses` both decompose to non-defects:
> `before_dawn_promise` was SELECTED into the prompt (actor reply paraphrased outside the
> term groups — response variance, not retrieval); `hollow_bookend_note` is the recorded
> contract-tier consolidation loss (`consolidation_lost_matches`), metering as designed;
> `key_hiding_place` had a duplicate extraction, one copy selected and recalled. **E2 did
> not produce a second clean run** — it was killed after the laptop hibernated on a dead
> battery mid-run (52 h frozen), an environment failure, not a code result. Per docs/25's
> two-run bar the default flip stays gated on a genuine second clean run; the harness/probe
> fixes merged to main byte-identical-at-default in the meantime.
>
> **Owner-side extras (2026-07-19, offline, no live run):** both PASS.
> (1) `semantic-benchmark --model sentence-transformers/all-MiniLM-L6-v2` with
> `RAG_SLICE_LEXICAL_QUOTA=2 CANON_TAG_PINNING=true` set — reranked production path
> recall@10 **0.824** / nDCG@10 **0.761** / German recall@10 **0.630**, all clearing the
> P0.4 floors (0.75 / 0.70 / 0.55) and identical to the docs/24 baseline: Lane B/pinning are
> post-retriever, so they do not perturb the retriever-level semantic quality the floors
> gate. (2) D3 pool (`live-validation-D3-2026-07-12.db`, 48 memories) reindexed into a
> disposable Qdrant (all-MiniLM-L6-v2) + `inspect_story_event` end-to-end, both lanes on:
> the blue-seal memory `354b8d98` lands in the selected set (`missed=[]`); **lanes OFF, the
> same query reproduces the original #73 miss end-to-end** (`354b8d98` in `missed`, not
> selected). This is the #73 acceptance case confirmed against real Qdrant + real
> embeddings, not just the offline replay — the fact reaches the actor prompt via Lane A
> pinning / Lane B slice, **not** via a dense-rank improvement (the explicit #73
> reinterpretation docs/26 §6 Stage 5 requires). Artifact opened via a scratch copy;
> `docs/artifacts/` untouched (no `-wal`/`-shm` sidecars).

**Problem.** Consolidation, semantic write-dedup, importance floor, and recency boost are
implemented and OFF (deliberately — offline evals can't prove live benefit; a hard index
cap regressed 50-turn recall, [app/config.py:99-105](../app/config.py)). A 100+-turn
campaign will eventually need growth control, and the sanctioned mechanism is
consolidation, **not caps**.

**Change.** Define and live-validate one documented preset (e.g.
`MEMORY_CONSOLIDATION_THRESHOLD=40`, `MEMORY_CONSOLIDATION_MAX_IMPORTANCE=2`,
`RAG_WRITE_DEDUP_COSINE_THRESHOLD=0.92`, `RAG_RECENCY_WEIGHT=0.02–0.04`) via long
live-smoke runs (`LIVE_TURN_COUNT=100`, recall probes late in the run). Record results
here and in `.env.example` comments. Defaults stay off until the evidence says otherwise.

**Validate.** Live-smoke long-run recall (the only arbiter for these knobs). Effort M
(mostly measurement time). — [ ]

### P2.3 Prefix-cache-friendly prompt shape (27B prefill latency)

**Problem.** The actor prompt is one system message ending with per-turn retrieved context
([app/orchestration/context_builder.py](../app/orchestration/context_builder.py)), so every
turn invalidates the llama.cpp prefix cache at the first changed byte → full ~6K-token
re-prefill per turn on a 27B. The sliding 8-turn dialogue window invalidates from the
oldest message onward once the window is full.

**Change (investigate, then decide).** Move volatile content (retrieved context, standing
facts) out of the system message into the *latest* user message; keep persona/scene as the
stable system prefix. Measure llama-server `--cache-reuse` interaction. This changes
generation behavior (position of context) — treat as a live A/B (bake-off harness) before
any default change.

**Validate.** stage_timings.generation deltas over a 20-turn live run; role-consistency
evals + bake-off quality comparison. Effort M, decision-heavy. — [ ]

### P2.4 World-scoped durable memory (engages the deferred "Milestone 4" decision)

**Problem.** Memories are session-scoped (persona memories cross sessions since v1.2);
multi-session campaigns in one world have no world-level continuity, by explicit decision
([docs/BACKLOG.md](BACKLOG.md) "Milestone 4 deferred") — to be built only when live
evidence shows recall degrading because facts live in session episodes.

**Change.** Do nothing yet; instrument first. Add a live-smoke probe that starts session B
in the same world after session A establishes facts, and measures what B can recall. If it
degrades, the design conversation in BACKLOG reopens with data. Effort S (probe only). — [ ]

> **Update 2026-07-14 — the design conversation happened ahead of the data.**
> [docs/27](27_world_chronicle_design.md) records the decided target (automatic *boundary*
> chronicle, world-scoped persona memory, tag-based carry-over; backlog **#81–#83**). The
> probe above is unchanged and still comes first — it is chronicle Stage C0 (**#81**) and
> its result is the baseline #83 is measured against. Build stays gated on the probe plus
> docs/26 Stages 0–5.

### P2.5 Optional cross-encoder rerank pass

**Problem.** At larger candidate pools the additive boosts saturate; a cross-encoder is the
standard next quality step but costs latency and explainability.

**Change.** Opt-in FastEmbed reranker over the top ~30 fused candidates, off by default,
scores exposed in diagnostics as another labeled component (preserving the "original score
survives" rule — the CE score must NOT replace `chunk.score`, or `original_score`
diagnostics silently change meaning and the additive boost constants lose their cosine
calibration). A CE model-load/inference failure must follow the retrieval fail-open
contract: degrade to the un-reranked order with a turn warning. Only worth doing after
P1.1/P1.2 land and P0.4 shows remaining headroom. fastembed 0.8.0 verified candidates:
`jinaai/jina-reranker-v2-base-multilingual`, `Xenova/ms-marco-MiniLM-L-6-v2`.

**Validate.** P0.4 nDCG + live latency budget. Effort M. — [ ]

---

## Verified small fixes (do anytime)

- **Tags-filter parity divergence (verified first-hand).** `InMemoryVectorStore` required a
  chunk to carry **all** filter tags (`issubset`) while Qdrant matched **any** (`MatchAny`).
  Actor retrieval never sets `tags`, so gameplay was unaffected — but any future tag-scoped
  feature would pass deterministic tests and behave differently live. **Done 2026-07-08**
  (BACKLOG #50): Qdrant now ANDs tags (one `MatchValue` per tag), and
  `tests/unit/test_vector_store_parity.py` pins parity across *every* filter dimension through
  both stores (Qdrant via embedded `:memory:` local mode). — [x]

## Unverified candidates from the 2026-07-07 sweep (verify before building)

The adversarial-verification pass was rate-limited before reaching these analyst findings.
Anchors were reported by the analysts but **not independently confirmed** — treat each as a
hypothesis: verify the anchors first, then promote it into the numbered roadmap or strike
it with a note.

- *Chunking/ingestion:* stale-chunk orphans — source identity is the raw path string;
  nothing sweeps chunks of removed/renamed manifest documents (re-ingest only replaces
  matching paths) — **confirmed 2026-07-16 and shipped as BACKLOG #87** (`ingest-scenario-lore
  --prune`, opt-in/default-off, path-prefix scoped so other scenarios' lore and non-lore
  collections stay untouchable). / CLI `start-session` re-embeds the whole manifest corpus
  every start (no content fingerprint to skip unchanged docs) — cost grows linearly with
  corpus size — **confirmed 2026-07-16 and shipped as BACKLOG #86** (content-fingerprint skip
  via chunk-id set equality, default on, byte-identical end state, `--force` escape hatch,
  fail-open on a store-read failure).
- *Context budget:* the recent-dialogue window is the largest prompt consumer (~up to 3.6K
  tokens) with uniform 900-char clipping — importance-uneven turns get equal budget.
- *Memory lifecycle:* consolidation summaries may leak into durable cross-session
  `persona_memory`; the curator prompt has no importance rubric although importance=4
  gates persona memory, canon, and eviction order; consolidation has no age guard (can
  swallow memories written moments ago — **confirmed 2026-07-08, see § below**);
  pinned-canon + retrieved-chunk duplication can double-spend context in an 8K window
  (**confirmed 2026-07-08 + new slot-displacement mechanism, see § below**).
- *Qdrant/vector store:* deleting a session with Qdrant unreachable can orphan
  still-retrievable `persona_memory` vectors (fail-open delete, nothing re-sweeps);
  `replace_source` is delete-then-upsert — a brief retrieval outage window per re-ingest.
- *Eval methodology:* **fixed 2026-07-11** — the live checkpoint's recall probes all landed
  before turn ~50, so a 100-turn run asserted nothing about late recall. Three late
  `StoryEvent`s are now wired into `app/diagnostics/live_checkpoint.py`
  (`amber_ring_token` 55→65, `north_stair_rendezvous` 70→80, and a long-gap probe
  `hollow_bookend_note` 17→95 — an old fact stated in the opening act, recalled 78 turns
  later) covered by deterministic unit tests; live-run validation (does the local model
  actually recall it) is still pending a `LIVE_TURN_COUNT=100` run on real hardware.
  Retrieval-miss floors still measure absolute score rather than margin-over-best-distractor.
- *Query construction:* `build_retrieval_query` puts the user message **last** after up to
  ~1.3K chars of framing ([retriever.py:145-168](../app/rag/retriever.py)) — near MiniLM's
  ~256-token input truncation the message can fall off the embedded text entirely (the
  dual-query bare-message pass currently masks this; reordering message-first would make
  the framed query robust on its own).

## 2026-07-08 review: confirmations + new findings

A follow-up review read the RAG core, memory lifecycle, and prompt assembly against this
roadmap. Anchors below were verified first-hand in code. All respect the invariants
(deterministic transparent ranking, fail-open retrieval, visibility gate) and the house style
(additive/opt-in, byte-identical defaults). None are catchable by the keyword-embedding
deterministic suite except the pure prompt-assembly ones (unit-testable byte-for-byte) — the
rest are gated on the **P0.4 graded corpus** and/or live-smoke, per the measure-first workflow.

### C1 — Standing-facts ↔ retrieved-chunk double-spend now *displaces* distinct facts (confirmed, promote)

> **Shipped 2026-07-08 (`64db602`).** `select_retrieved_chunks_for_prompt` now excludes
> chunks whose normalized text matches a standing fact, and `TurnRetrievalStage` over-fetches
> `top_k + len(standing_facts)` so the freed slot is recovered. Byte-identical when there are
> no standing facts; slot-recovery is unit-tested byte-for-byte. Live-smoke (8-turn) showed
> zero `retrieval_selection_misses` / `retrieval_miss_ranks` regression. Full distinct-fact-count
> gain awaits the **P0.4** graded corpus (not yet built).

Confirms the unverified "pinned-canon + retrieved-chunk duplication" candidate, and adds the
mechanism that makes it a **recall** regression, not just token waste. A durable,
high-importance PLAYER memory tagged `promise`/`rule`/`agreement` is pinned verbatim into the
"Standing facts" block by `build_standing_facts` (`canon_builder.py:26-50`) *and* near-certain
to win rerank (it collects importance + session + lexical boosts), so it also lands in the
retrieved set. `select_retrieved_chunks_for_prompt` (`context_budget.py:22-33`) dedups **only by
`chunk.id`** against other retrieved chunks — never against the standing-facts text. Because
retrieval returns exactly `top_k` and the prompt budget is the *same* number (default 5,
`retrieval.py:74` ← `context_budget.retrieved_chunks`), the duplicate consumes one of only five
slots, evicting a distinct fact. The most load-bearing facts are exactly the ones that
double-spend. **Fix:** exclude standing-facts text from the retrieved set (normalized-text match;
derived facts also carry the memory id) **before** truncation, and retrieve `top_k +
len(standing_facts)` so the freed slot is actually recovered (otherwise 5→4). Post-selection
prompt step; does not touch ranking determinism or the visibility gate. Effort S–M; measure
distinct-fact count in the actor prompt on live-smoke.

### C2 — Consolidation folds the whole eligible backlog with no age floor (age-guard confirmed; one-shot compression new)

> **Shipped 2026-07-09 (`0feb94b`).** Added `min_age` and `batch_cap` keyword params to
> `select_consolidatable` (rank-based age, derived from the existing oldest-first
> `created_at`-then-`id` ordering — no wall-clock dependency). `min_age` holds the N newest
> eligible memories out of the foldable pool (the rolling recent window); `batch_cap` folds
> only the oldest N of what's left per pass. Wired through `MemoryConsolidator` →
> `TurnMemoryStage` → `TurnOrchestratorConfig` → `composition.py` exactly like the existing
> `consolidation_threshold`/`consolidation_importance_ceiling` knobs, with paired
> `MEMORY_CONSOLIDATION_MIN_AGE` / `MEMORY_CONSOLIDATION_BATCH_CAP` Settings + `.env.example`
> keys. Both default to 0 (no-op): the threshold gate itself now runs against the
> age-floored pool (so a fresh low-importance memory can't be swallowed the instant the raw
> backlog count crosses threshold), and `batch_cap` only bounds how many of that pool are
> folded in a given pass, never whether consolidation triggers — at the 0/0 defaults this
> reproduces the pre-C2 selection byte-for-byte, proven by unit tests at both the
> `select_consolidatable` primitive and the `TurnMemoryStage` integration level.
> Consolidation itself still ships OFF by default (`MEMORY_CONSOLIDATION_THRESHOLD=0`); live
> validation of non-zero `min_age`/`batch_cap` rides with the [P2.2](#p22-long-campaign-preset-enable-the-shipped-but-off-machinery-with-evidence)
> long live-smoke run per the measure-first workflow, not this change.
>
> **Fix note (2026-07-11, cross-review P2 + P3).** Two bugs surfaced once consolidation
> actually rolls up a batch, both fixed without touching the min-age/batch-cap behavior
> above. **P2:** `mark_memories_consolidated` tags originals but never deletes the SQLite
> row, so `live_checkpoint.py`'s unfiltered `persisted_memory_count` diverged from the
> Qdrant-indexed count on the first roll-up and would fail the checkpoint's SQLite/Qdrant
> parity assertion on every `LIVE_TURN_COUNT=100` **P2.2** validation run thereafter —
> `inspect_live_state` now excludes `CONSOLIDATED_TAG`-marked rows from that count. **P3:**
> `_persist_consolidation` indexed the summary *before* marking the originals consolidated,
> so an index failure (e.g. a P1.4 fingerprint mismatch) left the summary committed but the
> same oldest-N backlog still eligible — re-tripping the threshold and minting a second
> summary of the same memories on the very next turn, one orphan summary row per turn.
> Reordered to SQLite-first (persist + mark + cache-invalidate) with indexing/unindexing now
> best-effort afterward, degrading to a warning on failure (invariant #4: Qdrant is a
> rebuildable derived index). Unit-tested: SQLite parity count excludes consolidated rows;
> two consecutive consolidation passes with a forced index failure on the first prove no
> second summary is minted.
>
> Confirms the "no age guard" candidate and adds a second coupled issue. (a) **No age floor:**
`select_consolidatable` (`consolidation.py:49-66`) sorts oldest-first but has no minimum-age
filter, so a low-importance untagged memory written *this turn* is swallowed the moment the
backlog crosses threshold — it never gets a chance to be retrieved on its own. (b) **Whole-backlog
one-shot:** once `len(candidates) >= threshold`, `memory_consolidation.py:60-95` passes **all**
eligible candidates into a single 2–4 sentence summary (or one large `"Earlier in this session:
…"` blob in the deterministic fallback) — a heavy one-shot information loss. A partial roll-up
(fold only the oldest N, keep a rolling recent window, require a minimum age) is gentler and
matches the "sleep cycle" intent. This matters because consolidation is the sanctioned growth
control for [P2.2](#p22-long-campaign-preset-enable-the-shipped-but-off-machinery-with-evidence);
if enabling it costs a recall cliff for recent-but-trivial facts, the owner avoids the one tool
meant for scale. **Fix:** min-age param + batch-size cap on `select_consolidatable`. Off by
default already; validate on the P2.2 long live-smoke run; byte-identical when `threshold==0`.

### N1 — Write-dedup false-drops distinct durable facts, and the extractor's own framing prefix inflates the ratio (new)

> **Shipped 2026-07-08 (`0c11c29`).** Took the cheapest of the three listed fixes: strip the
> constant `The player stated:` framing before computing coverage terms (candidate + each
> existing summary). The stored summary is unchanged; a no-op for model/author summaries. Unit
> tests prove a framing-only false-drop now writes and an identical framed fact still dedups (no
> new false-writes). Live-smoke (8-turn) showed zero `memory_extraction_misses` regression. If
> the **P0.4** adversarial-distractor subset (once built) still shows residual false-drops, escalate
> to rare-term weighting or a threshold bump.

`is_covered_by_summaries` (`deterministic_extractor.py:112-132`, `COVERAGE_THRESHOLD = 0.5`,
consumed by write-dedup at `memory.py:168` and `memory_dedup.py:50`) drops a new candidate when
≥50% of its content terms appear in some existing summary (reversal markers the only escape). For
short durable facts this false-drops *distinct* events: new `"I promise to guard the bridge"` →
`{player, state, promis, guard, bridg}` vs existing `"…guard the northern gate"` shares
`{player, promis, guard}` = 3/5 = 0.6 → **dropped**, even though bridge ≠ gate is the whole
point. Worse, the deterministic extractor **injects the shared vocabulary itself** by prefixing
every candidate `summary=f'The player stated: "{sentence}"'` (`:101`) — `{player, state}` on every
fact — systematically pushing the coverage ratio toward the drop threshold. A silently dropped
promise never reaches SQLite, so it can be neither retrieved nor pinned as canon: a hard,
warning-only recall loss on the highest-value tag classes. **Fix (cheapest first):** strip the
constant `The player stated:` framing before computing coverage terms; or weight rarer terms over
shared framing terms; or raise the threshold. Precision-tuned layer pinned by memory-regression
tests — gate on the P0.4 distinct-fact / adversarial-distractor subset (same proper noun,
different fact) to prove fewer false-drops without new false-writes.

### N2 — The deterministic lexical + dedup layer is English-only (new; complements P1.2)

[P1.2](#p12-embedding-model-upgrade-path-multilingual) covers multilingual *embeddings* only. But
the deterministic lexical layer — the recall-safety net that catches proper nouns a 384-dim
MiniLM misses (P1.1's own rationale) — is language-blind: `_LEXICAL_STOPWORDS` (`ranking.py:23-34`)
and `content_terms`/`_stem` (`:250-272`) are English. For German play, German function words
(`der/die/das/und/nicht/ist/mit`) aren't stopped, so they inflate lexical overlap with unrelated
chunks (the exact noise the English list suppresses), and German morphology
(`Vertrag`/`Vertrags`/`Vertrages`) isn't stemmed, so real matches are missed — and the same terms
feed `is_covered_by_summaries`, shifting write-dedup math for German memories. (Tokenization
survives: `token.isalpha()` keeps umlauts.) **Distinct from BACKLOG #26** — that SKIP was about
English *framing-word* tuning; this is a *language* gap. **Fix:** a German stopword set behind a
language setting (S); optional German stemming (M). Additive/opt-in; gate on P0.4's German subset.

### N3 — Read-time prompt dedup is id-only; near-duplicate memory *text* co-fills slots when write-dedup is off (new; complements P2.2 / #30)

> **Shipped 2026-07-09, review-round fix same day.** `select_retrieved_chunks_for_prompt`
> tracks the normalized text (the same `_normalize_for_match` helper C1 uses) of every
> chunk it has already selected and skips later chunks whose normalized text repeats,
> regardless of id. First occurrence (highest rank) wins, so ranking order and determinism
> are unchanged. **Correction:** the initial landing claimed no separate over-fetch was
> needed; a review confirmed that was false whenever the fixed-size ranked window (sized
> only for the C1 exclusion count) contained more than one duplicate-text pair — the freed
> slot then had no replacement candidate available and the block silently shrank below
> `budget.retrieved_chunks` (repro: 5 candidates / 2 duplicate pairs / budget=5 → 3
> selected). Fixed by widening `TurnRetrievalStage.run`'s over-fetch with a bounded
> worst-case margin, `retrieved_chunks - 1`, on top of the existing C1 standing-facts
> compensation — deterministic, no new setting, byte-identical final prompt output (the
> extra fetched candidates are only ever used as backfill; the block still caps at
> `budget.retrieved_chunks`). Unit-tested: the `top_k` formula at the stage level, an
> end-to-end boundary-condition repro (proven non-tautological against the pre-fix
> formula), duplicate-text/distinct-id chunks filling exactly one slot each, legitimately
> distinct chunks left untouched, and the C1 exclusion interaction still holding. Full
> deterministic gate + regression runner pass.
>
> **Fix note (2026-07-11, cross-review P5 + P6).** **P5:** the text-dedup key was checked
> only on raw, pre-truncation chunk text, so two chunks diverging only past
> `budget.max_retrieved_chunk_chars` (e.g. sharing an 850-char prefix under an 800-char cap)
> passed as distinct and then rendered as byte-identical truncated prompt blocks.
> `select_retrieved_chunks_for_prompt` now also dedups on the normalized *post-truncation*
> text. **P6 (honesty only, no behavior change):** re-examined whether the `retrieved_chunks
> - 1` over-fetch margin above is a genuine guarantee against under-fill. It isn't — it
> covers the specific worst case it was sized for (every slot but one in the first
> `retrieved_chunks` window is a duplicate of an earlier one), but heavy near-duplicate
> pileup of a *single* fact (more copies than the margin) can still push genuinely distinct
> chunks out of `rerank_chunks`'s truncated `top_k` window entirely — confirmed with a
> reproduction (one fact under 9 ids outranking 4 distinct facts, `retrieved_chunks=5`,
> margin=4 → only 1 selected). Tightened the `retrieval.py`/`context_budget.py` wording to
> state that scope precisely instead of implying an unconditional guarantee; that residual
> gap is a write-side/consolidation concern, not something a larger fixed margin closes, and
> is deliberately not addressed here (no fetch-retry machinery added).

Credit where due: the persona-memory dual-write and consolidation paths are already id-safe (the
indexer keeps `id=memory.id`; originals are `CONSOLIDATED_TAG`-unindexed), so those are *not* a
problem. The real gap: with semantic write-dedup shipping OFF by default
(`RAG_WRITE_DEDUP_COSINE_THRESHOLD=1.0`) and always-on curation writing ~1.7 memories/turn,
near-identical *distinct-id* memories accumulate and can co-occupy the 5 retrieved slots
(`context_budget.py` and `ranking._deduplicate_ranked_chunks` both dedup strictly by `chunk.id`).
Same slot-scarcity logic as C1. **Fix:** a cheap read-time exact/normalized-text dedup in
`select_retrieved_chunks_for_prompt` — a safety net independent of the write-side toggle. Effort
S; deterministic; measure on P0.4 for any accidental drop of legitimately-distinct chunks. Lower
priority than C1/N1 — do alongside P2.2.

### Triage

~~Do first (small, high value, both recall losses on the highest-value facts): **C1**, **N1**.~~
**C1 + N1 shipped 2026-07-08** (`64db602`, `0c11c29`) — byte-tested + live-smoke no-regression.
**N3 shipped 2026-07-09, over-fetch sizing corrected same day in review round** — byte-tested;
same normalizer as C1, no live-smoke needed (pure prompt-assembly, unit-testable byte-for-byte).
**C2 shipped 2026-07-09** (`0feb94b`) — byte-tested defaults; live validation of non-zero
`min_age`/`batch_cap` still rides with the long-campaign P2.2 live-smoke run, not this change.
Do when German play lands: **N2**.

## Explicitly not proposed (decision record honored)

- **Hard session-memory index caps** — regressed 50-turn recall; consolidation is the
  mechanism ([app/config.py:99-105](../app/config.py), BACKLOG #29).
- **Cross-provider fallback for any task** — violates the session-bound provider invariant.
- **Moving ranking policy into Qdrant** — explainability/testability rule in
  [docs/05](05_rag_memory_design.md) ("keep ranking policy in application code").
- **LangChain/LangGraph or framework churn** — [docs/10](10_next_steps_after_mvp.md) Work To Avoid.
- **Token streaming before critic validation** — the SSE boundary is a security feature.

## Measure-first workflow (applies to every item above)

1. Name the metric and harness **before** coding (P0.4 corpus / embedding-ab / live-smoke).
2. Land the measurement if it doesn't exist.
3. Implement additive + opt-in; defaults byte-identical.
4. Gate: `ruff check . && mypy . && pytest && python -m app.evals.regression_runner`.
5. Live-validate (live-smoke; long-run for memory/ranking changes).
6. Record the evidence here and flip defaults only on it.
