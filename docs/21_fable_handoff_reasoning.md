# 21 — Predecessor-Agent Handoff: Reasoning & Chains of Thought (Claude Fable 5, 2026-07-07)

> Reviewed: 2026-07-07 @ 7888ee7
>
> **Living doc with a historical core.** The reasoning chains in here are a point-in-time
> record of how the 2026-07-07 analysis was derived — do not rewrite them. The pointers and
> working agreements should be kept current like any living doc.

## Purpose and how to use this document

This document was written by Claude Fable 5 on its last available day for this project, as
an explicit knowledge transfer to the successor models (Claude Opus 4.8, Claude Sonnet 5)
that will continue the work. [docs/08](08_agent_handoff.md) tells you *what* to read and
*where* to change things. This document tells you **how the predecessor reasoned** — the
chains of thought behind the analysis, the reconstructed WHYs of the architecture, and the
thinking patterns that keep changes safe here.

Use it like this:

- Before your first change: read [Mental model](#mental-model-as-i-verified-it), then
  [Reconstructed design reasoning](#reconstructed-design-reasoning-chains) for the subsystem
  you're touching.
- Before a RAG change: read [How I reasoned about the RAG core](#how-i-reasoned-about-the-rag-core)
  here, then take the concrete work items from [docs/22](22_rag_scaling_roadmap.md).
- When you're unsure whether an idea is new: check
  [Ideas already tried, rejected, or deliberately deferred](#ideas-already-tried-rejected-or-deliberately-deferred)
  — this repo has an unusually complete decision record and re-proposing rejected ideas is
  the most likely way for a fresh agent to waste a day.

## How this analysis was produced (method, reproducible)

Chain of thought, so you can re-run or extend it:

1. **Inline first-hand reading before any delegation.** I read the load-bearing files myself
   rather than trusting summaries: `README.md`, `docs/08`, all of `app/rag/` (retriever,
   ranking, chunking, embeddings, vector_store, ingestion), `app/config.py`,
   `app/orchestration/{context_builder,context_budget,turn_orchestrator}.py`,
   `scripts/lib/local-model-profile.sh`, `.env.example`, `docs/README.md`, `docs/BACKLOG.md`.
   Rationale: the handoff writer must have first-hand ground truth; delegated summaries are
   for breadth, not for the spine.
2. **Fan-out for breadth.** Eight parallel reader agents mapped the subsystems
   (orchestration stages, memory lifecycle, LLM layer, persistence/domain, evals/diagnostics,
   docs decision-history, API/frontend, ops/verification), each returning a structured map
   (purpose, data flow, invariants, gotchas, scaling limits, test anchors) with file:line
   anchors.
3. **Improvement analysis as six independent lenses**, each constrained to the question
   "what limits larger scenarios on a ~27B local model, in THIS codebase": embeddings/hybrid
   retrieval, chunking/ingestion, context budget for 8–16K ctx, memory lifecycle over long
   campaigns, Qdrant configuration at scale, and evaluation methodology.
4. **Adversarial verification of every finding.** Each proposed improvement was independently
   re-checked against the code and the decision record by a skeptical verifier: evidence
   accurate? already implemented behind a flag? already rejected with evidence? breaks an
   invariant? Only survivors made it into docs/22, with corrections applied.
5. **Synthesis by the same agent that did step 1**, so the final documents are anchored in
   first-hand reading, not agent hearsay.

If you have workflow/multi-agent orchestration available, this fan-out → verify → synthesize
shape is the right one for this repo: it is small enough to map completely, and its decision
record is good enough that verification against docs catches most false recommendations.

## Mental model (as I verified it)

The one-paragraph version: **a bounded, deterministic engine wrapped around three
single-task LLM calls.** The orchestrator runs a fixed stage sequence per turn — session →
retrieval → routing → generation → validation → critique (→ repair) → containment →
persistence → memory. LLM calls happen only inside generation/critique/repair/memory-extraction;
everything else is deterministic Python. SQLite owns all authoritative state; Qdrant holds a
rebuildable index of curated memories and ingested lore; the provider (local llama.cpp or
cloud) is fixed per session at creation.

The load-bearing asymmetry to internalize: **retrieval is fail-open, the critic is
fail-closed.** A turn without retrieved context is acceptable (degraded but safe); a turn
with unvalidated text is not (it might leak hidden authored content). Every "should this
error propagate?" decision in the codebase follows from that asymmetry. When you add a new
stage or data path, decide which side of it you're on first.

Second asymmetry: **prompt-side prevention vs output-side containment.** Hidden content is
kept out of player-facing prompts by construction (visibility filtering at retrieval +
prompt assembly), and *additionally* scanned for on the way out (`secret_guard.scan_reply`,
verbatim redaction + paraphrase flagging). Neither layer alone is trusted.

## Reconstructed design reasoning chains

These are the WHYs I reconstructed from code comments, git history, BACKLOG entries, and doc
cross-references. They are the constraints your changes must satisfy or consciously revisit.

**Why session-bound provider (no per-turn escalation)?** Earlier versions had per-turn
cloud escalation paths. They were removed (plan `2026-07-02-session-bound-provider`) because
mixed-provider sessions made privacy reasoning impossible (which provider saw what?) and
routing nondeterministic. The chain: privacy invariant needs a static provider → bind at
creation → `CLOUD_MODE` only gates creation → router becomes trivially deterministic.
Anything you add that "falls back to cloud" re-opens the hole the removal closed.

**Why is retrieval fail-open?** The product is a solo roleplay session; a dead Qdrant
container should degrade the experience, not end it. The cost of fail-open is silent
retrieval loss — which is why every fail-open path emits a turn `warning` and diagnostics
persist per turn. If you change failure semantics, the warnings surface is the contract to
maintain.

**Why deterministic additive reranking instead of a learned/cross-encoder reranker?**
(a) Explainability — the RAG Inspector shows exactly which boost moved which chunk;
(b) testability — evals pin ranking behavior byte-for-byte with keyword embeddings;
(c) latency on a 26B-busy machine. The constants in `app/rag/ranking.py` are canonical;
`Settings` mirrors them so tuning is possible without editing source. A cross-encoder pass
is a legitimate future option (see docs/22) but must stay optional and diagnosable.

**Why dual-query retrieval?** A single context-framed query (scene + persona + recent
dialogue + message) buries indirect callbacks ("what rule did we agree on?") in long
sessions — the framing dominates the embedding. The fix searches twice (framed + bare
player message) and dedups by chunk id in reranking (`app/rag/retriever.py:113-118`).
Remember this when you touch query construction: the two queries have different jobs.

**Why is `session_memory_max_episodes` 0 (unbounded)?** A hard cap was implemented and
**measurably regressed 50-turn recall** — evicted memories stop being retrievable, and
importance-ranking plus dual-query keep the unbounded index usable at POC scale
(`app/config.py:99-105`, BACKLOG #29). The general lesson encoded there: **for episodic
memory, silent eviction is worse than growth** until proven otherwise. Consolidation
(summarize-then-replace, `MEMORY_CONSOLIDATION_THRESHOLD`) is the sanctioned growth-control
mechanism, not caps.

**Why are consolidation, semantic dedup, and recency boost all opt-in (off by default)?**
Each was implemented, validated offline as behavior-preserving in its default, and left off
because offline evals could not prove live benefit (keyword embeddings don't measure
semantic quality). The repo's philosophy: **defaults change only on live evidence**
(live-smoke / bake-off runs), knobs ship early. Respect that bar when you want to flip one.

**Why one system message containing everything (persona/scene/canon/chunks)?** Simplicity
and provider compatibility (any OpenAI-compatible server). Known cost: any retrieved-context
change invalidates the llama.cpp prefix cache for the whole prompt → full re-prefill on a
26B model each turn. This is quantified and addressed as a roadmap item in docs/22 rather
than fixed opportunistically, because prompt reshaping shifts generation behavior and needs
live A/B validation.

**Why chars, not tokens, for every budget?** No tokenizer dependency in the engine keeps
tests deterministic and providers swappable. It held up at POC scale; it is the first thing
that breaks at 8K ctx with a bigger `RAG_DEFAULT_TOP_K` — see docs/22 P0 items.

## How I reasoned about the RAG core

My chain of thought when asked "will this handle larger scenarios on a 27B local model?":

1. **Budget arithmetic first.** `.env.example` recommends `RAG_DEFAULT_TOP_K=10` × 800 chars
   + 8 recent turns × 2 messages × ≤900 chars + persona/scene/canon (~2K chars) ≈ 24–26K
   chars ≈ 6–7K tokens — against `-c 8192` in the 26b profile with 500 tokens reserved for
   the response. Conclusion: the recommended config **already sails within ~1K tokens of the
   context ceiling**; any scenario with longer authored personas/scenes or longer player
   messages can silently overflow. There is no token counting anywhere — overflow appears
   as llama.cpp context-shift truncation the app never sees. This made "token-aware budget +
   raise `-c`" the top priority in docs/22.
2. **Then retrieval reach.** Dense-only search with `all-MiniLM-L6-v2` (384-dim, English,
   ~256-token effective window) is the quality ceiling. The lexical boost only reorders what
   dense search already returned — a dense miss is unrecoverable downstream (verified in
   `app/rag/retriever.py` + `ranking.py`: boosts apply post-retrieval). At demo-corpus scale
   the ceiling is invisible; at 10–100× lore it becomes the dominant failure mode
   (proper-noun-heavy fantasy queries are exactly where MiniLM is weakest). Hence: hybrid
   sparse+dense in Qdrant and/or a stronger multilingual dense model are the highest-leverage
   retrieval changes — with embedding-ab + a bigger fixture corpus as the gate.
3. **Then what reaches the prompt.** Chunks are truncated mid-sentence at 800 chars with
   `"..."` — retrieval can win and the prompt still loses the fact if it sat past the cut.
   Chunking is blind to markdown structure; a chunk can straddle a section boundary and
   embed as mush. Both are cheap fixes with existing-harness validation, so they rank high
   on effort-adjusted impact.
4. **Then memory growth.** Unbounded episodic index + everything-indexed floor works at POC
   scale by evidence (see above). For 100+-turn campaigns the sanctioned levers already
   exist (consolidation, dedup, importance floor, recency) but are unproven live — so
   docs/22 frames them as "enable + measure" work items, not new code.
5. **Always: how would we know it worked?** The deterministic eval harness uses keyword
   embeddings — it pins *engine logic*, not semantic quality. Any embedding/ranking change
   is invisible to it by design. The only real arbiters are `embedding-ab` (offline, seeded
   events) and live-smoke recall over long sessions. That's why docs/22 puts eval assets
   (larger fixture corpus, graded relevance judgments, German queries) *before* the flashy
   retrieval upgrades: without them every improvement claim is vibes.

The meta-pattern for successors: in this repo, reason **budget → reach → prompt → growth →
measurement**, in that order. It matches how the data flows and it front-loads the changes
that are cheap to validate.

## Ideas already tried, rejected, or deliberately deferred

Do not re-propose these without new evidence; the record lives in
[docs/BACKLOG.md](BACKLOG.md), [docs/10](10_next_steps_after_mvp.md), and `.env.example`
comments:

- **Hard cap on session memories** — regressed 50-turn recall; consolidation is the
  sanctioned mechanism (BACKLOG #29, `app/config.py:99-105`).
- **Memory-extraction cloud retry / cross-provider fallback** — conflicts with the
  session-bound provider invariant (BACKLOG #9, dropped).
- **Roleplay-aware stopword changes** — reverses a documented lexical-boost design choice;
  benefit unprovable offline (BACKLOG #26, skipped).
- **Embedding model swap to bge-small** — evaluated via `embedding-ab`; candidates tied on
  the seeded fixtures, so the default stayed (BACKLOG #10). Note for docs/22: "tied on small
  seeded fixtures" is weak evidence either way — the fixture corpus is the thing to fix first.
- **Narrowing broad `except Exception` handlers** — they are intentional fail-open seams
  (BACKLOG #18, skipped).
- **Structured `TurnResult.errors` beyond the additive API `errors` field** — YAGNI for a
  single-user POC (BACKLOG #19, shipped additively instead).
- **Token streaming / pre-validation SSE** — deliberately excluded; SSE only carries
  already-validated text (README, docs/12). The critic boundary is the reason.

## Danger zones, restated with reasoning

- `TurnOrchestrator` + `app/orchestration/stages/` — every cross-cutting concern (warnings,
  timings, controlled failure, persona-switch commit ordering) threads through here. The
  persona-switch commit is *deliberately* deferred until after successful persistence
  (`turn_orchestrator.py:444-453`); the deferred-memory job *deliberately* reloads the
  scene/persona the turn ran under, not the session's live ones (`run_deferred_memory`
  docstring). Both fix real bugs; both look like refactor bait. Leave the ordering alone.
- `app/rag/ranking.py` constants are mirrored in `Settings` — change both or the
  "defaults reproduce canonical constants byte-for-byte" contract breaks
  (`test_config.py` / `test_retrieval_ranking.py` pin this).
- The `include_hidden` gate (`stages/critique.py`) is the single point where hidden text is
  allowed into any prompt, and only when the route is local. Treat every diff touching it as
  security-sensitive; `tests/evals/test_provider_binding_regressions.py` is the tripwire.
- Chunk identity is `sha256(source:index:text)` (`app/rag/ingestion.py:119-121`) and
  ingestion replaces **by source path**. Renaming a lore file orphans nothing in SQLite but
  leaves stale chunks in Qdrant under the old source until re-ingest; `reset-index` +
  re-ingest is the clean path. Keep that in mind for any "edit lore mid-campaign" feature.

## Working agreements for successor models

1. **Gate every change**: `ruff check . && mypy . && pytest && python -m app.evals.regression_runner`
   — then the CLI surface checks in docs/08 if you touched commands/settings, then live-smoke
   if you touched anything the deterministic harness can't see (embeddings, prompts, providers).
2. **Docs move with code** (sweep rule, freshness headers). New numbered docs get indexed in
   [docs/README.md](README.md). `.env.example` and `app/config.py` change together.
3. **Measure-first for RAG quality claims.** If your change claims better retrieval/recall,
   name the harness run that would show it (embedding-ab / live-smoke / new eval) *in the PR
   or commit body*, and prefer landing the measurement before the change.
4. **Additive, opt-in, byte-identical defaults** is the house style for risky knobs — copy
   the `RAG_RECENCY_WEIGHT` pattern (default preserves old behavior exactly; comment explains
   when and how to enable; live validation named).
5. **When in doubt, the decision record wins.** BACKLOG/docs/10/.env comments over instinct.

## Pointers

- Verified improvement roadmap (what to build next in the RAG core): [docs/22](22_rag_scaling_roadmap.md)
- Onboarding + where-to-change-things: [docs/08](08_agent_handoff.md)
- Subsystem map: [docs/09](09_current_architecture_map.md)
- RAG/memory design as-built: [docs/05](05_rag_memory_design.md)
- Verification tooling (live-smoke, profiles, bake-off): [docs/19](19_verification_and_eval_tooling.md)
