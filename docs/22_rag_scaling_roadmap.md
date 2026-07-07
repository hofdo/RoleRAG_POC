# 22 — RAG Scaling Roadmap: Larger Scenarios on ~27B Local Models

> Reviewed: 2026-07-07 @ 7888ee7
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

**Problem.** `_truncate_text` cuts retrieved chunks mid-sentence at 800 chars with `"..."`
([app/orchestration/context_budget.py:36-41](../app/orchestration/context_budget.py)) —
retrieval can rank the right chunk first and the prompt still loses the fact if it sits
past the cut. Same pattern in the retrieval-query clip (`_clip_line`,
[app/rag/retriever.py:171-174](../app/rag/retriever.py)).

**Change.** Trim at the last sentence boundary (fallback: word boundary) before the cap;
keep the explicit omission marker. Deterministic, testable byte-for-byte.

**Validate.** Unit tests; ranking evals unchanged (trim happens after selection).
Effort S. — [ ]

### P0.4 Eval assets before retrieval upgrades (the measurement gate)

**Problem.** The deterministic harness uses keyword embeddings + `InMemoryVectorStore` — it
pins engine logic, **not semantic quality**; `embedding-ab` ranks a small seeded event set
(BACKLOG #10 ended "candidates tied" — on fixtures too small to discriminate). No
graded-relevance corpus, no distractor-heavy fixtures, no German queries. Every improvement
below is unmeasurable today.

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
5 seeded events + 2 smalltalk + 2 lore chunks — which is why BACKLOG #10 "tied"). — [ ]

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
the runbook bricks on a stale one. Effort S–M. — [ ]

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

- **Tags-filter parity divergence (verified first-hand).** `InMemoryVectorStore` requires a
  chunk to carry **all** filter tags (`issubset`,
  [vector_store.py:386-387](../app/rag/vector_store.py)) while Qdrant matches **any**
  (`MatchAny`, [vector_store.py:359-365](../app/rag/vector_store.py)). Actor retrieval
  never sets `tags`, so gameplay is unaffected — but any future tag-scoped feature would
  pass deterministic tests and behave differently live. Pick one semantic (AND is the safer
  read of the in-memory intent), implement it in both stores, and pin it with a
  paired-store test. Effort S. — [ ]

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
  swallow memories written moments ago); pinned-canon + retrieved-chunk duplication can
  double-spend context in an 8K window.
- *Qdrant/vector store:* deleting a session with Qdrant unreachable can orphan
  still-retrievable `persona_memory` vectors (fail-open delete, nothing re-sweeps);
  `replace_source` is delete-then-upsert — a brief retrieval outage window per re-ingest.
- *Eval methodology:* the live checkpoint's recall probes all land before turn ~50 — a
  100-turn run asserts nothing about late recall (add late second-callback StoryEvents);
  retrieval-miss floors measure absolute score rather than margin-over-best-distractor.
- *Query construction:* `build_retrieval_query` puts the user message **last** after up to
  ~1.3K chars of framing ([retriever.py:145-168](../app/rag/retriever.py)) — near MiniLM's
  ~256-token input truncation the message can fall off the embedded text entirely (the
  dual-query bare-message pass currently masks this; reordering message-first would make
  the framed query robust on its own).

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
