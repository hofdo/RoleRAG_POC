# Glossary of Project Terms

> Reviewed: 2026-07-04 @ 571acc8

The vocabulary the docs, the code, and the commit history lean on. Each entry links
the doc that owns the concept in depth; this page only defines the term. Terms that
were removed by a design change are collected under [Removed concepts](#removed-concepts)
so a stale reference is self-identifying.

## Terms

**Fail-closed / controlled failure.** The turn pipeline's safe-failure mode: when the
critic cannot validate a draft — repair was exhausted, or the critic itself raised any
exception — the orchestrator withholds the unvalidated draft and returns a fixed
controlled-failure message instead of serving unchecked text
([app/orchestration/stages/critique.py](../app/orchestration/stages/critique.py) lines
146–163). The persisted turn is recorded with `outcome="controlled_failure"` and
`critic_status="rejected"` (`TurnOutcome.CONTROLLED_FAILURE` in
[app/domain/models.py](../app/domain/models.py)). "Fail-closed" is the invariant;
"controlled failure" is the observable outcome. See
[04_agent_workflows](04_agent_workflows.md) and
[12_api_contract](12_api_contract.md).

**Critic.** The reviewing agent that checks the actor's draft for consistency and
hidden-fact leakage before the reply is emitted, and can reject it (triggering repair).
On a local session the critic is shown hidden authored fields for leak detection; on a
cloud session those fields are stripped from its prompt, so it checks prose only
([app/orchestration/stages/critique.py](../app/orchestration/stages/critique.py) line
130). See [04_agent_workflows](04_agent_workflows.md).

**Gating (`always` vs. `auto`).** Whether an agent runs on every turn or only on
turns judged risky. `CRITIC_GATING`/`CURATOR_GATING` default to `always` (run every
turn); set to `auto` to skip low-risk turns
([app/config.py](../app/config.py); `GATING_MODES` in
[app/orchestration/stages/critique.py](../app/orchestration/stages/critique.py)). In
`auto` mode the **risk predicate** (`_is_risky_turn`) treats a turn as risky when the
draft validator flagged it, retrieval confidence is missing or below
`LOW_RETRIEVAL_CONFIDENCE`, scene complexity is at least `HIGH_SCENE_COMPLEXITY`, or the
session provider is cloud — otherwise the turn is served unvalidated by design. Auto
gating is off by default because it regressed 50-turn recall (see
[05_rag_memory_design](05_rag_memory_design.md)).

**Containment (input-side vs. output-side).** Keeping author-hidden facts (persona
`secrets`, `forbidden_knowledge`, `private_description`, scene `gm_private_summary`) out
of player-facing output. *Input-side* containment never puts hidden fields into the
actor prompt: retrieval requests only `player`-visible chunks. *Output-side* containment
is the deterministic `secret_guard` backstop that runs after generation — it redacts any
verbatim or per-sentence echo of a hidden fact and flags likely paraphrase
([app/agents/secret_guard.py](../app/agents/secret_guard.py): `redact_hidden_facts` on
critic output, `scan_reply` on the actor reply). See
[17_content_authoring_reference](17_content_authoring_reference.md) and
[18_security_privacy_and_backups](18_security_privacy_and_backups.md).

**Canon / standing facts.** The "Standing facts" block pinned verbatim into the actor
prompt so load-bearing commitments survive out of the recent-dialogue window. Author-pinned
canon facts come first; derived facts must be `player`-visible, have importance at least
`CANON_IMPORTANCE_FLOOR` (default `4`), and carry at least one durable `CANON_TAGS` tag
(promise, entrusted, deadline, oath, …). The combined list is ordered by importance then
recency, deduplicated, and bounded by `CANON_MAX_ITEMS`/`CANON_MAX_CHARS`, independent of
vector retrieval ([app/orchestration/canon_builder.py](../app/orchestration/canon_builder.py)).
See [05_rag_memory_design](05_rag_memory_design.md).

**Dual-query retrieval.** The retrieval pass runs two queries and unions the results: a
framed query (the scene/lore context blob) anchors scene relevance, and a second query
with the bare player message keeps indirect callbacks ("what rule did we agree?")
retrievable when the framed blob would otherwise bury them. The union is deduplicated by
chunk id before reranking ([app/rag/retriever.py](../app/rag/retriever.py)). See
[05_rag_memory_design](05_rag_memory_design.md).

**Deferred memory curation.** On both API turn endpoints, memory curation and indexing
run in a post-response background job rather than inline: `memory_written` is `false` in
every live turn response, a `memory curation deferred: runs after this response` warning
is expected on every successful turn, and the live response's `stage_timings` has no
`memory` key. The persisted turn's memory diagnostics are updated once the job completes
and are visible via the turn-detail endpoints
([app/orchestration/turn_orchestrator.py](../app/orchestration/turn_orchestrator.py):
`run_deferred_memory`). See [12_api_contract](12_api_contract.md).

**Live checkpoint.** The end-to-end diagnostic that drives a real session through the
running server for a configured number of turns and reports recall, latency, and
containment against seeded events — the engine behind the live-smoke workflow
([app/diagnostics/live_checkpoint.py](../app/diagnostics/live_checkpoint.py)). It is a
report-only metric, not part of the request path. See
[19_verification_and_eval_tooling](19_verification_and_eval_tooling.md).

**Session-bound provider.** The provider (`local` or `cloud`) is chosen once at
`POST /sessions` and is immutable for the session's lifetime; every task — actor, repair,
critic, memory extraction — runs on that bound provider, with no escalation, fallback, or
per-turn override (`choose_route` in [app/llm/router.py](../app/llm/router.py)).
`CLOUD_MODE` gates creation only: `off` rejects a cloud session with `400
cloud_unavailable`, `ask` prompts for one interactive confirmation at creation, `auto`
binds silently. See [06_local_cloud_model_strategy](06_local_cloud_model_strategy.md).

**Visibility levels.** The three-valued label on a chunk or memory that controls who may
see it: `player` (safe to surface to the player), `gm` (GM-only), and `character_private`
(private to a character) — the full `Visibility` enum
([app/domain/visibility.py](../app/domain/visibility.py)). Actor prompt construction
accepts only `player`-visible content; the critic may inspect the hidden levels for leak
detection. See [05_rag_memory_design](05_rag_memory_design.md).

## Removed concepts

These terms appear in older commits, plans, and reports but no longer exist in the code.
A live reference to any of them is a documentation bug.

**Per-turn cloud confirmation** *(removed 2026-07-02)*. There was once a two-phase
confirm-per-turn flow (`confirmation_required` status, a `cloud_confirmed` request field).
Confirmation now happens at most once, at session creation, under `CLOUD_MODE=ask`; there
is no per-turn confirmation. See [06_local_cloud_model_strategy](06_local_cloud_model_strategy.md).

**Cloud escalation / fallback** *(removed 2026-07-02)*. The router once had automatic
cloud paths — local-failure fallback, a cloud repair ladder, scene-complexity escalation,
and retrieval-confidence escalation. All four were removed when provider became
session-bound: cloud is a peer choice made at creation, never a rescue mechanism
([app/llm/router.py](../app/llm/router.py)). See
[06_local_cloud_model_strategy](06_local_cloud_model_strategy.md).

**`request_cloud` / `force_local`** *(removed 2026-07-02)*. Per-turn request fields that
asked for or forced a specific provider mid-session. No such field exists; the provider is
fixed at creation. See [12_api_contract](12_api_contract.md).
