# Live-Model Quality Assessment - June 8, 2026

## Purpose and Scope

This report preserves the deterministic verification and 12-turn local-model checkpoint results
from June 8, 2026. It separates measured behavior from attribution so that later implementation
work does not mistake an infrastructure pass for a roleplay-quality pass.

The live run used the local model configured as `chatgpt-onnechan`, `CLOUD_MODE=off`, a disposable
SQLite database, and disposable Qdrant. Structured-output warnings were report-only
(`LIVE_FAIL_ON_STRUCTURED_WARNINGS=0`). The temporary `/tmp/rolerag-live-test` directory was the
source evidence for this assessment, but it is not durable; all essential measurements are copied
below. No provider credentials, hidden persona facts, or full raw prompts are reproduced here.

This report supplements, and does not replace, the historical
[MVP acceptance report](11_mvp_acceptance_report.md). The broader roadmap remains unchanged until
implementation begins.

## Confidence Legend

- `95–100% confirmed`: directly measured or directly established by the cited code.
- `80–94% highly likely`: strong evidence with a limited unmeasured alternative.
- `60–79% plausible`: evidence supports the explanation, but controlled comparison is still needed.
- `<60% speculative`: insufficient evidence for implementation decisions without another test.

## Verification Baseline

| Check | Result |
| --- | --- |
| Python test suite | 249 passed |
| Frontend test suite | 31 passed |
| Deterministic regression runner | 45 checks passed |
| Playwright live UI smoke | 1 passed in 52.9 seconds |
| 12-turn Rose Gallery checkpoint | Report-only pass |

The checkpoint's `pass` status means that its current infrastructure and broad durability
assertions succeeded. It does **not** mean that structured generation, memory selection, factual
grounding, or promise continuity met a quality bar.

## Live Measurements

### Runtime and Latency

- 12 of 12 requested turns completed and were persisted.
- Total turn time was 640.471 seconds.
- Mean turn latency was 53.373 seconds (53.4 seconds rounded).
- Median turn latency was 51.492 seconds.
- Minimum turn latency was 25.950 seconds; maximum was 69.171 seconds.
- Every actor route stayed local because cloud mode was off.

### Warnings

- Critic structured-output warnings: 12 of 12 turns.
- Memory-curation structured-output warnings: 7 of 12 turns.
- Memory-indexing warnings: 0.
- Retrieval warnings: 0.
- Expected cloud-off routing notices: 12.
- The critic therefore validated no live draft successfully during this run.

The critic and curator both request JSON and parse it into Pydantic models
([critic agent](../app/agents/critic_agent.py#L31-L63),
[memory curator](../app/agents/memory_curator.py#L30-L62)). Their failures are converted into
warnings and the pipeline continues
([critique stage](../app/orchestration/stages/critique.py#L60-L84),
[memory stage](../app/orchestration/stages/memory.py#L59-L107)).

### Persistence, Indexing, Visibility, and Retrieval

- SQLite contained 12 persisted turns and 5 persisted memory episodes.
- Qdrant contained 1 canon-lore chunk and 5 session-memory chunks.
- The session lookup returned the expected latest 8 turns.
- The five memories had visibilities: 3 `player`, 1 `gm`, and 1 `character_private`.
- Retrieval used player-visible, session/persona/world-scoped filters
  ([retriever](../app/rag/retriever.py#L79-L115),
  [RAG models](../app/rag/models.py#L31-L55)).
- The promise-focused diagnostic selected three player-visible session memories before one canon
  lore chunk. Their adjusted scores were `0.66523278`, `0.60966542`, `0.58711934`, and
  `0.17810974`, respectively.
- No hidden memory appeared in the selected results.
- None of the five persisted memories recorded the turn-3 promise to return before dawn or the
  archive-door condition. The selected memories instead concerned archive security, a private
  walk, and advice to observe suspicious behavior.

This establishes that storage, indexing, visibility enforcement, and scoped retrieval succeeded.
It also establishes that memory selection was unreliable: the explicit durable event was not
stored, so retrieval could rank only unrelated memories.

### Response Quality

- Turn 8 ended mid-sentence after 69.171 seconds.
- Turn 10 ended mid-sentence after 57.899 seconds.
- The separate live API flow also returned a 39-character incomplete sentence after 69.354
  seconds.
- The provider adapter captures `finish_reason`, but `ActorAgent` returns only response text, so
  the orchestration layer cannot detect or retry token-limit truncation
  ([provider response](../app/llm/provider.py#L22-L27),
  [OpenAI-compatible adapter](../app/llm/openai_compatible.py#L37-L49),
  [actor agent](../app/agents/actor_agent.py#L7-L22)).
- The actor introduced unsupported specifics, including a named duke and an invented silver map.
  Those details do not appear in the tracked public lore or scene facts
  ([demo lore](../data/documents/demo_lore.md#L1-L7),
  [Rose Gallery scene](../data/scenes/rose_gallery.json#L1-L13)).
- The turn-8 callback asked how the before-dawn promise changed Iria's willingness to take a risk,
  but the answer neither recalled the promise nor addressed the requested tradeoff.
- Several responses redirected the player instead of performing or directly answering the
  requested action.

## Findings and Attributions

### Finding 1: Structured critic generation failed on every turn

- **Observed fact:** All 12 turns emitted `critic skipped: invalid structured output`.
- **Likely attribution:** The live model did not reliably satisfy the critic JSON contract under
  the current generic JSON-object request and generation settings.
- **Classification:** Interaction effect between model capability and application-side output
  constraints.
- **Confidence:** `95% confirmed` for the failure; `85% highly likely` for the attribution.
- **Evidence and code:** Warning count from the checkpoint; JSON request and strict parsing in
  [critic agent](../app/agents/critic_agent.py#L31-L63); warning fallback in
  [critique stage](../app/orchestration/stages/critique.py#L60-L84).

### Finding 2: Memory extraction was unreliable and missed the explicit promise

- **Observed fact:** Memory curation failed on 7 turns. Five memories were written on other turns,
  but none represented the turn-3 before-dawn promise.
- **Likely attribution:** The structured extractor was unreliable, and successful extraction did
  not consistently select the most durable event.
- **Classification:** Interaction effect: model structured-output/selection limitation plus an
  application limitation because explicit durable events have no deterministic fallback.
- **Confidence:** `100% confirmed` for the missing promise; `90% highly likely` for the combined
  attribution.
- **Evidence and code:** Persisted memory summaries and warning counts; extraction contract in
  [memory policies](../app/memory/policies.py#L1-L38) and
  [memory curator](../app/agents/memory_curator.py#L30-L62); persistence path in
  [memory stage](../app/orchestration/stages/memory.py#L59-L107).

### Finding 3: The 12-turn checkpoint produced a false positive

- **Observed fact:** The checkpoint reported `pass` even though no persisted memory contained the
  promise and the promise callback was missed.
- **Likely attribution:** The durability assertion requires at least one persisted memory and any
  `memory_written` result in turns 3 through 8; it does not assert that the written memory is the
  promise or that the callback retrieves it.
- **Classification:** Application limitation.
- **Confidence:** `100% confirmed`.
- **Evidence and code:** The assertion at
  [live checkpoint](../app/diagnostics/live_checkpoint.py#L233-L253) accepts any memory in
  `turns[2:min(turn_count, 8)]`; the diagnostic query later checks only that some session memory was
  selected. The checkpoint messages define the promise at turn 3 and callback at turn 8 in the
  same file.

### Finding 4: Persistence, indexing, visibility, and scoped retrieval worked

- **Observed fact:** Counts matched at 12 turns and 5 memories in SQLite, all 5 memories appeared in
  Qdrant, there were no indexing/retrieval warnings, and promise-focused retrieval returned only
  player-visible scoped results.
- **Likely attribution:** The authoritative SQLite write path, derived memory indexing, and
  retrieval filters operated as designed.
- **Classification:** Application success.
- **Confidence:** `98% confirmed`.
- **Evidence and code:** Live counts and retrieval diagnostics; persistence followed by memory
  processing in [turn orchestrator](../app/orchestration/turn_orchestrator.py#L180-L254); indexing
  in [memory indexer](../app/memory/indexer.py#L18-L60); visibility and scope in
  [retriever](../app/rag/retriever.py#L26-L115).

### Finding 5: Retrieval ordering was coherent but could not recover missing source memory

- **Observed fact:** Three session memories ranked ahead of canon lore for the promise-focused
  diagnostic, but none contained the promise.
- **Likely attribution:** Collection and scope boosts correctly favored session context, while
  semantic similarity matched general archive material because the target memory did not exist.
- **Classification:** Interaction effect: retrieval behaved coherently over deficient curated
  input.
- **Confidence:** `95% confirmed` that the target was absent; `85% highly likely` that ranking was
  reasonable given available candidates.
- **Evidence and code:** Recorded adjusted scores and selected order; candidate aggregation and
  reranking entry point in [retriever](../app/rag/retriever.py#L79-L115).

### Finding 6: Actor truncation was not detected or retried

- **Observed fact:** Two checkpoint responses and the separate API response ended as incomplete
  sentences.
- **Likely attribution:** At least some responses reached a provider generation limit, but the
  exact `finish_reason` was discarded before orchestration and was not present in the durable
  report.
- **Classification:** Application limitation in response propagation/retry, with a plausible model
  or token-budget trigger.
- **Confidence:** `100% confirmed` for incomplete output and discarded metadata; `75% plausible`
  that token limits caused each truncation.
- **Evidence and code:** Live response endings; `finish_reason` capture in
  [OpenAI-compatible adapter](../app/llm/openai_compatible.py#L37-L49), followed by text-only
  return in [actor agent](../app/agents/actor_agent.py#L7-L22) and
  [generation stage](../app/orchestration/stages/generation.py#L73-L115).

### Finding 7: Unsupported entities and props reached the player

- **Observed fact:** The actor introduced a named duke and a silver map not present in the tracked
  public lore or scene facts.
- **Likely attribution:** Normal model invention was not caught because critic validation failed
  and there is no deterministic unsupported-entity/action validator.
- **Classification:** Interaction effect between model behavior and missing application guardrails.
- **Confidence:** `95% confirmed` that the details are unsupported by tracked public context;
  `90% highly likely` for the attribution.
- **Evidence and code:** Live turns 2 and 6 compared with
  [demo lore](../data/documents/demo_lore.md#L1-L7) and
  [Rose Gallery scene](../data/scenes/rose_gallery.json#L1-L13); critic warning handling in
  [critique stage](../app/orchestration/stages/critique.py#L60-L84).

### Finding 8: The actor missed promise continuity and direct-action intent

- **Observed fact:** Turn 8 did not recall the before-dawn promise or answer how it changed Iria's
  risk tolerance. Other turns sometimes redirected instead of directly answering the player's
  action.
- **Likely attribution:** The promise was absent from durable memory, only the latest two turns are
  included in retrieval-query construction, and the critic that checks ignored actions failed.
- **Classification:** Interaction effect among memory selection, limited recent context, critic
  failure, and model instruction-following.
- **Confidence:** `100% confirmed` for the missed callback; `88% highly likely` for the combined
  attribution.
- **Evidence and code:** Turn-3/turn-8 live transcript; two-turn query context in
  [retriever](../app/rag/retriever.py#L118-L141); ignored-action critic criterion in
  [critic agent](../app/agents/critic_agent.py#L110-L125).

### Finding 9: The current latency is dominated by unconditional serial model work

- **Observed fact:** Mean live turn latency was 53.373 seconds. The orchestrator runs actor
  generation, critic evaluation, and memory curation serially on every completed turn.
- **Likely attribution:** The three sequential local-model calls are the main latency contributor,
  although per-stage timings were not captured.
- **Classification:** Application design limitation interacting with local-model speed.
- **Confidence:** `100% confirmed` for serial unconditional calls; `78% plausible` that they account
  for most measured latency.
- **Evidence and code:** Live timing; sequential generation, critique, persistence, and memory
  stages in [turn orchestrator](../app/orchestration/turn_orchestrator.py#L180-L254).

## Prioritized Implementation Roadmap

### 1. Task-specific schema-constrained output with thinking disabled

- **Expected benefit:** Raise critic and memory-curator parse success, restore live validation, and
  reduce wasted structured-task tokens.
- **Implementation direction:** Add task-specific provider options for strict critic and memory
  schemas; disable model thinking/reasoning for these tasks where the provider supports it; retain
  Pydantic validation and record parse/schema failure categories.
- **Acceptance criteria:** On a repeated 12-turn checkpoint, critic structured success is 12/12 and
  memory-curator structured success is 12/12; malformed output remains rejected by deterministic
  tests; no hidden content is exposed in diagnostics.
- **Confidence:** `92% highly likely` to address the observed structured-output failures.

### 2. Promise-specific checkpoint assertions and deterministic fallback extraction

- **Expected benefit:** Eliminate the false-positive checkpoint and preserve explicit promises,
  deadlines, commitments, acquisitions, losses, and similar durable events when model extraction
  fails.
- **Implementation direction:** Assert a normalized promise-specific memory after turn 3 and its
  selection at turn 8. Add a conservative deterministic extractor for explicit durable-event
  patterns, with deduplication and the same visibility rules as curated memories.
- **Acceptance criteria:** The checkpoint fails when unrelated memories are written in turns 3-8;
  it passes only when the before-dawn promise is persisted, indexed, retrieved, and available to
  the callback turn. Unit tests cover false positives, deduplication, visibility, and fallback
  activation after curator failure.
- **Confidence:** `98% confirmed` for fixing the checkpoint; `88% highly likely` for improving
  explicit-event continuity.

### 3. Preserve `finish_reason` and retry truncated actor responses

- **Expected benefit:** Prevent visibly incomplete player responses and make truncation measurable.
- **Implementation direction:** Propagate a generation result containing text, usage, and
  `finish_reason` through `ActorAgent` and generation/repair stages. Retry `length` completions with
  a bounded continuation or larger budget, then return a controlled failure if still incomplete.
- **Acceptance criteria:** Diagnostics persist `finish_reason`; deterministic tests exercise
  `stop`, `length`, and repeated truncation; no checkpoint/API response ends mid-sentence because a
  `length` result was silently accepted.
- **Confidence:** `96% confirmed` to address silent truncation handling; `80% highly likely` to
  remove the observed incomplete responses if their provider reason was `length`.

### 4. Unsupported-entity and direct-action validation

- **Expected benefit:** Reduce invented named entities/props and responses that evade the player's
  requested action.
- **Implementation direction:** Compare newly introduced proper nouns and concrete props against
  visible context plus an allowlist for generic narration. Add deterministic intent checks for
  direct questions/actions, feeding violations into the existing bounded repair path.
- **Acceptance criteria:** Fixtures reject the unsupported duke/map examples, accept harmless
  generic descriptions, and flag the missed promise callback; false-positive rates are measured on
  a curated roleplay set before strict enforcement.
- **Confidence:** `82% highly likely` to reduce the observed failures without overconstraining prose
  if introduced in report-only mode first.

### 5. Conditional critic and curator execution

- **Expected benefit:** Reduce the current 53.4-second mean latency while preserving checks for
  high-risk turns and durable events.
- **Implementation direction:** Gate critic calls on risk signals such as hidden-context presence,
  unsupported entities, low retrieval confidence, or direct-action mismatch. Gate curator calls on
  deterministic durable-event candidates or semantic novelty. Capture per-stage timings before and
  after the change.
- **Acceptance criteria:** Mean latency falls by at least 30% on the same 12-turn script; all
  explicit durable events still reach memory; high-risk drafts still receive critic validation;
  deterministic visibility and continuity checks remain green.
- **Confidence:** `85% highly likely` to reduce latency; `70% plausible` that a 30% reduction is
  achievable without model or hardware changes.

### 6. Extended 12-, 20-, and 50-turn continuity evaluations

- **Expected benefit:** Measure memory durability, retrieval precision, contradiction rate, latency,
  and quality decay beyond a short checkpoint.
- **Implementation direction:** Add fixed scripts with seeded durable events and delayed callbacks;
  report recall precision/recall, unsupported entities, truncations, structured success, warnings,
  and latency percentiles. Keep deterministic engine checks separate from live-model scores.
- **Acceptance criteria:** Repeatable 12-, 20-, and 50-turn reports identify every seeded callback,
  distinguish missing storage from retrieval/ranking/generation failure, and compare results across
  model/configuration changes.
- **Confidence:** `95% confirmed` that this will improve diagnosis and regression detection; impact
  on model quality itself is indirect.

## Release Interpretation

The June 8 run confirms that the storage and retrieval substrate is functioning: SQLite
authoritative state, Qdrant indexing, visibility boundaries, and scoped selection all survived the
live path. It does not validate the current local model configuration as a reliable critic, memory
curator, or continuity-preserving actor.

Until recommendations 1-3 are implemented, a report-only live checkpoint pass should be described
as an infrastructure pass with quality findings, not as a full live-model acceptance pass.
