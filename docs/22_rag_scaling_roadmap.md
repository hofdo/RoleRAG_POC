# 22 — RAG Scaling Roadmap: Larger Scenarios on ~27B Local Models

> Reviewed: 2026-07-11 @ 4f6822b
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
Effort S–M. — [ ]

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
shipped note above); still open: the real-model run itself, floor calibration from it, and
item 4's transcript-derived queries.

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
Effort M. — [ ]

---

## P2 — scale, latency, and long campaigns

### P2.1 Qdrant payload indexes (and optional quantization)

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
Effort S. — [ ]

### P2.2 Long-campaign preset (enable the shipped-but-off machinery, with evidence)

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
  matching paths). / CLI `start-session` re-embeds the whole manifest corpus every start
  (no content fingerprint to skip unchanged docs) — cost grows linearly with corpus size.
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
