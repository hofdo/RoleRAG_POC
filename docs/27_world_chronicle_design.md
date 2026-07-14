# 27 — World Chronicle: Cross-Session World Continuity (Design)

> Reviewed: 2026-07-14 @ afa3266

Design record of the 2026-07-14 owner/agent design conversation that resolves
[docs/26 §8](26_memory_retrieval_redesign.md#8-open-questions-for-the-owner) question 6
("should memories carry across sessions?") and engages the deferred **Milestone 4**
decision ([docs/BACKLOG.md](BACKLOG.md#decisions-2026-07-01-audit), "shared world state:
deferred with rationale") with a concrete target design. **Status: decided design, not a
scheduled build.** Construction stays evidence-gated per the house rule — pulled by a real
second-campaign need, and strictly after the
[docs/26 §6 stages](26_memory_retrieval_redesign.md#6-staged-migration-plan) land, because
tags and provenance are this design's substrate. Backlog items: **#81–#84**.

## 1. The decided game model

One sentence: **a persistent world whose consequences and relationships survive the story
that created them — discovered in play, never recapped.**

Campaign 1 ends. Later, a new session starts in the same world with a **new protagonist**.
The village the last hero burned is still ash. Iria does not know the new face — but she
still keeps the blue-seal rule, and if this hero earns her trust she may speak of the one
who entrusted the silver compass to her. Nothing is narrated as already known; the world
simply remembers.

Decision record (owner answers, two structured rounds, 2026-07-14):

| # | Question | Decision |
|---|----------|----------|
| 1 | Primary new-session scenario | **New hero, same world** — a different protagonist in a world shaped by earlier campaigns |
| 2 | World model | **Automatic chronicle** — the engine promotes durable facts to world scope itself |
| 3 | Persona-memory scope | **World-scoped** — an NPC's memories stay in the world where they formed (today's persona-only filter leaks across worlds) |
| 4 | What carries | **Hard commitments + relationship-defining moments**, selected by tags — not the dead importance floor |
| 5 | When promotion runs | **At the session boundary** — a batch pass at campaign end / first new session in that world; later chronicles supersede earlier ones in order |
| 6 | Authorial control | **Automatic + editable after** — no mandatory review; chronicle entries are listable, editable, deletable |
| 7 | New-hero session start | **Nothing — discover in play** (a "years later" recap was explicitly declined) |
| 8 | Default visibility of promoted facts | **NPC-held** — only explicitly consequence-class facts become public world lore |
| 9 | Rumor truth-link *(addendum)* | **Derived from real facts, distortion allowed** — a generated rumor always links to a chronicled fact or session memory; free fabrications are authored-only |
| 10 | Rumor authorship *(addendum)* | **Session-bound provider, background, flag-gated** — hearsay prose is LLM-written off the turn path; a dedicated small auxiliary model was assessed and parked (§3.5.3) |
| 11 | When rumors arise *(addendum)* | **Mid-session, about current events** — a cadence-driven background pass; the boundary pass reconciles but does not generate (boundary generation noted as an available extension, not selected) |
| 12 | Rumor lifecycle *(addendum)* | **Boundary reconciliation** — confirmed / debunked-but-lingering / faded-toward-legend at chronicle passes; spread simulation explicitly rejected |

Rows 9–12 were added the same day in a follow-up round: the owner asked for a **living
world** — rumors and similar soft memories — on top of the chronicle. They also resolve a
tension rows 7–8 left open: with no recap and NPC-held defaults, rumors are the mechanism
by which a world that *remembers* also *speaks*, without spoiling hidden truth.

The load-bearing consequence of decision 1: a new hero implies a **session boundary**, so
"automatic" never has to mean live/continuous promotion. The boundary pass sees the
*settled* end-state of a closed story — which is what makes the automatic model buildable
here at all, given that live supersession is the mechanism class docs/26 §7 rejected.

## 2. Current state (verified 2026-07-14 against source)

Retrieval searches three collections every turn
([app/rag/retriever.py](../app/rag/retriever.py)): `SESSION_MEMORY` filtered by
`session_id` (emergent play memory — never crosses sessions), `PERSONA_MEMORY` filtered by
`persona_id` **only** (cross-session NPC memory), and `CANON_LORE` filtered by `world_id`
(authored lore — already world-persistent). Author-pinned `CanonFact` rows are
session-scoped. Sessions are resumable indefinitely, so one campaign = one session needs
none of this; the design targets the deliberate new-session moments (a new protagonist,
or the immutable session-bound provider forcing a new session).

Two verified defects this design fixes:

1. **Floor starvation.** `PERSONA_MEMORY_IMPORTANCE_FLOOR = 4`
   ([app/memory/indexer.py](../app/memory/indexer.py)) vs. a live curator importance
   distribution of 9×1 / 38×2 / 1×3, zero at 4+ (docs/26 §3.3, D3 artifact) — so only
   deterministic-extractor commitments (importance 4) ever cross sessions today. The
   designed channel exists and works; the gate starves it to the hard-commitments subset.
   docs/26 §5 recorded this coupling and deferred it behind P2.4 instrument-first: this
   doc is the design half of that resolution, the probe (#81) is still the instrument half.
2. **World leak.** The persona filter carries no `world_id`, so reusing a persona in a
   different world imports its memories across universes. Decision 3 makes world scoping
   the contract (#82).

## 3. Target design

### 3.1 Three continuity channels, mapped onto existing seams

| Layer | Example | Mechanism |
|---|---|---|
| Public consequence | "the village stays burned" | New `world_facts` SQLite table (authoritative — the deferred Milestone 4 layer), indexed into `CANON_LORE` with `world_id` and `source_type="world_chronicle"`; retrieval already searches that collection every turn with the right filter |
| NPC-held history | "Iria keeps the blue-seal rule" | The existing `PERSONA_MEMORY` channel, world-scoped (#82), with the selection predicate moved from the importance floor to tag eligibility (#83) |
| Hidden history | true but undiscovered | `gm`/private visibility on promoted rows — dormant until the author surfaces it; existing visibility enforcement keeps it out of player-facing prompts |

Default is **NPC-held** (decision 8): a promoted fact becomes public world lore only when
it carries an explicit consequence-class tag; everything else lives with the NPC who
witnessed it and surfaces through dialogue retrieval. This is the conservative failure
direction — a wrongly-classified fact stays *less* visible, never more — and it preserves
the fiction for the new protagonist: nothing from the old campaign is narrated as if the
new hero knew it.

### 3.2 The boundary chronicle pass

Runs once per session at its boundary: when a campaign is declared ended, or lazily when
the first new session is created in the same world. Inputs: the closed session's
tag-eligible memories (the same `CANON_TAGS` family that docs/26 Stages 2–3 make reliable;
the exact tag set is parameterized by the still-open docs/26 §8 Q1 contract discussion)
plus their `source_turn_id` provenance (docs/26 Stage 1). Output: `world_facts` rows and
world-scoped persona-memory entries, each carrying source session/turn provenance.

Promotion is **deterministic selection over already-tagged rows — no new LLM call site
for facts.** The LLM's influence ends at the tags it already wrote during play, and making
those tags trustworthy is exactly what the docs/26 fold/pinning work does. LLM-written
chronicle prose required its own design conversation — which happened the same day for the
rumor layer specifically (§3.5): hearsay prose may be generated; **fact promotion itself
stays selection-only.**

**Supersession happens only here, in boundary order.** Chronicle passes are ordered by
session; a later session's promoted fact can supersede an earlier one's within the same
tag family (marked superseded, not deleted — auditable). No live/in-play supersession
exists anywhere in this design.

### 3.3 Audit surface (decision 6)

`world_facts` is an authoritative SQLite table with list/edit/delete via CLI and API; the
vector side is derived and rebuildable, exactly like memories today. Automatic by default,
curable on demand — the author remains the owner of truth (invariant 1) without mandatory
epilogue homework. Every promotion and supersession is logged with provenance.

### 3.4 The new hero's experience (decisions 7–8)

Session B starts cold: no recap, no chronicle dump. World history reaches the new
protagonist only through play — public facts via lore retrieval when relevant, NPC-held
facts via persona retrieval when that NPC is present and the conversation warrants it,
hidden facts when the author chooses to surface them. The declined alternative (a
player-visible "years later" recap distilled from public facts) is recorded as a
**rejected feature**, not a deferred one; revisit only if cold starts prove disorienting
in real play.

### 3.5 Living-world layer: rumors (same-day addendum, decisions 9–12)

**A rumor is a player-visible hearsay row whose underlying truth stays hidden.** It is
non-authoritative *by type*: a `world_rumors` row (or `world_facts` row with
`kind="rumor"` — build-time choice) links `derived_from` → the memory episode
(mid-session) or world fact it distorts, and carries `truth_status`
(accurate/distorted/contradicting) plus lifecycle `status`
(circulating/confirmed/debunked/faded). The actor can always voice it safely because it
enters the prompt labeled as what-people-say ("Rumors circulating: …"), never as narrator
truth — which is exactly what lets a world with NPC-held defaults and no recap still
*speak* about its history. World-scoped from birth, so rumors born in campaign 1 persist
into campaign 2 under boundary reconciliation.

**3.5.1 Generation (decisions 9–11).** A cadence-driven background pass
(`RUMOR_INTERVAL` turns, flag-gated, default off) rides the existing deferred-job path —
zero turn-latency cost, the same pattern as deferred memory curation. Source selection is
deterministic (recent player-visible durable-class memories); one structured call on the
**session's bound provider** phrases the hearsay, with distortion allowed but anchored: a
deterministic validation check requires the rumor's content terms to be drawn from the
source fact plus known world entities — a hallucinated foreign name fails the check, the
rumor is simply not written, and a warning is logged (fail-open, consistent with the
memory pipeline's posture). Structured-output failure likewise skips rather than
degrading to a flat template. Rumor rows are auditable/editable like all chronicle
content.

**3.5.2 Surfacing and reconciliation (decision 12).** Rumor rows index into `CANON_LORE`
(world-filtered, `source_type="world_rumor"`) and flow through retrieval as ordinary
labeled chunks — no ranking special-case, no new vector-store feature. Within a running
session, play that debunks a rumor needs no engine action: the debunking scene is session
memory and out-competes contextually. Status changes happen only at chronicle boundaries:
a rumor whose fact became public turns *confirmed* (or retires), a contradicted one turns
*debunked-but-lingering* (or retires), and old ones *fade toward legend* — the same
ordered-boundary trick as supersession, no spread simulation. The boundary pass
**reconciles but does not generate** (decision 11 selected mid-session arising only);
boundary-generated "echo" rumors from freshly promoted facts would reuse identical
machinery and are recorded as an available extension, not scope.

**3.5.3 The dedicated-small-model question (decision 10), assessed and parked.** The
owner asked whether a small LLM — analogous to the small embedding model — should write
rumors. Assessment: the embedding model itself literally cannot (it is an encoder, not a
generator). A separate small *generative* model (1–4B GGUF) is feasible — llama.cpp
already runs an MTP draft model alongside the 26B (docs/16) — but it does not pay rent
here: rumor calls are rare and already off the turn path via the deferred-job pattern, so
a small model would save latency nobody experiences, while costing a second server
process and profile, a new config surface, weaker instruction-following/German quality at
that size, a higher hallucination risk against the anchor check, and — most importantly —
a new "auxiliary task provider" concept that reopens the session-bound-provider decision
record for all background tasks (curator included), a far bigger door than rumors
justify. **Parked with a named trigger:** revisit only if live evidence shows background
rumor generation contending with turn latency or memory on the owner's hardware.
Pre-scouted candidates (2026-07-14 web scan, NOT verified against the task): **Qwen3.5-2B**
(Apache 2.0, strongest small-tier multilingual/German, right at the 2B cap),
**Gemma 3 1B** (smallest/fastest, same model family as the 26B roleplay model — shared
template/style lineage), **EuroLLM-1.7B** (EU-language-first, strong for its size,
Apache 2.0). All have GGUF/llama.cpp paths (~1 GB at Q4_K_M). Candidate choice is not the
open question; the trigger firing is.

## 4. Explicitly not in this design

- **Live supersession / continuous promotion** — same verdict as docs/26 §7's
  world-state-ledger rejection; the session boundary is the whole trick.
- **A mandatory review gate** ("propose-then-confirm") — declined in favor of
  automatic + editable (decision 6).
- **A session-start recap** — declined (decision 7).
- **A new LLM call site for facts** — promotion is pure selection (§3.2). The rumor layer
  adds one flag-gated *background* call site for hearsay prose (§3.5), deliberately
  decided in the same-day addendum; it writes fiction, never truth.
- **Rumor-spread simulation** — per-region/per-NPC knowledge of who has heard what,
  evolving over time — rejected (decision 12); lifecycle lives at boundaries only. The
  related deeper frontier (per-NPC knowledge gating for *session* memories generally —
  today every NPC in a session effectively shares the player-visible memory pool) is
  likewise out of scope: it is a living-world *simulation engine*, not a memory feature.
- **A dedicated small rumor-writing model** — assessed and parked with a named trigger
  (§3.5.3).
- **Legacy-hero-as-NPC** (promoting the previous protagonist to a persona/lore figure) —
  attractive v2 idea, out of scope for v1.
- **`CanonFact` changes** — author-pinned canon stays session-scoped; world-level
  authored truth already has homes (lore documents, and now `world_facts`).

## 5. Staging and gates (backlog #81–#83)

Strictly after docs/26 Stages 0–5 (#75–#80): tags and provenance are the substrate, and
the docs/26 measurement stack is what makes promotion auditable. Pulled by a real
second-campaign need — do not build speculatively.

- **#81 — C0, the P2.4 probe (S).** Unchanged from
  [docs/22 § P2.4](22_rag_scaling_roadmap.md#p24-world-scoped-durable-memory-engages-the-deferred-milestone-4-decision):
  session A establishes facts, session B starts in the same world, measure what B recalls.
  Run before building; today's expected result (B recalls only importance-4 commitments
  through a shared persona, nothing else) becomes the baseline the chronicle is measured
  against.
- **#82 — C1, world-scoped persona memory (S).** `world_id` payload on persona-memory
  chunks + the retrieval filter, rebuildable via `reindex-memories`, with the
  InMemoryVectorStore parity test the house convention requires. Independent of the
  chronicle — fixes the cross-world leak regardless of when #83 happens.
- **#83 — C2, chronicle v1 (M/L).** `world_facts` table + boundary promotion pass +
  tag-based persona selection replacing the floor + audit CLI/API + diagnostics.
  Deterministic tests offline (fake-provider pipeline end-to-end); live acceptance is the
  #81 probe re-run showing session B recalling the promoted set — and nothing it
  shouldn't.
- **#84 — C3, living-world rumor layer (M).** §3.5 in full: rumor rows + cadence-driven
  background generation on the bound provider (flag-gated, default off, deferred-job
  path) + deterministic anchor validation + hearsay prompt framing + boundary
  reconciliation. Depends on #83's substrate (storage, audit surface, boundary pass).
  Deterministic tests via fake-provider rumor output and anchor-check units; live
  acceptance: a session with generation on produces only anchored, labeled rumors, and
  the next boundary pass reconciles them correctly.

## 6. Invariants check

1. **LLM never owns authoritative state** — `world_facts` is SQLite-authoritative, the
   index derived and rebuildable, promotion deterministic. Holds.
2. **Visibility boundary** — NPC-held/hidden defaults are the conservative direction;
   public requires an explicit tag; existing enforcement layers apply unchanged. Holds.
3. **Session-bound provider** — untouched; the chronicle incidentally gives the clean
   answer to "continue a campaign on the other provider": new session, same world. Holds.
4. **Fail-open retrieval / fail-closed critic** — chronicle facts ride the existing
   fail-open retrieval channels; a failed boundary pass leaves the world unchanged and is
   retryable. Holds.
5. **Deterministic, transparent ranking** — chronicle facts arrive as ordinary
   chunks/canon lines; no score manipulation. Holds.
6. **Orchestration out of routes/agents** — the boundary pass is its own module invoked
   from thin CLI/API surfaces, mirroring consolidation's placement. Holds.

## 7. Open details, deferred to build time

The promotion tag set (parameterized by the docs/26 §8 Q1 continuity-contract discussion,
still open); the deterministic consequence-class ("public") tag rule; time-skip semantics
(does scene `current_time` advance between campaigns?); whether a chronicle entry links
the personas present (probably yes — cheap provenance); export/import interaction (once
worlds carry state beyond lore, `export-session` alone no longer captures a whole world).
For the rumor layer specifically: the `RUMOR_INTERVAL` cadence and per-session rumor cap
(measure, don't guess); the exact anchor-validation rule (term subset vs. entity list);
boundary "echo" generation of rumors from freshly promoted facts (machinery-identical,
not selected — see §3.5.2); reputation aggregation ("the butcher of the vale") as a
derived stance summary — deferred until rumors exist to aggregate.
