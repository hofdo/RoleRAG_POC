# 26 — Memory/Retrieval Redesign: Synthesis and Target Architecture

> Reviewed: 2026-07-14 @ a7aa00d

Synthesis of four independently-designed, adversarially-judged memory/retrieval
redesigns (world-state-ledger, retrieval-guarantees, event-sourced, minimal-invasive)
into one recommended target architecture and staged migration plan. Source material:
the four 100-turn live validation runs of 2026-07-12/13 (docs/22 P2.2 subsection,
[docs/25_live_validation_runbook.md](25_live_validation_runbook.md)), the failure
dossier reproduced below, and the twelve judge verdicts (three lenses × four designs).
Reads alongside — does not restate —
[docs/21_fable_handoff_reasoning.md](21_fable_handoff_reasoning.md) (why the
architecture is shaped this way) and
[docs/22_rag_scaling_roadmap.md](22_rag_scaling_roadmap.md) (the verified roadmap this
proposal slots into, extends, or in one place overrides with live evidence).

This document does not re-derive invariants; see the CLAUDE.md list and docs/21. It is
decisive: one recommended architecture, not a menu. Where a judge disproved a claim in
one of the four input designs, that claim is not resurrected here even in weakened form.

This doc is indexed in [docs/README.md](README.md) (numbered-docs table and the
living-docs `17`–`26` range) as of this revision.

---

## 1. Why now

The 2026-07-04 verdict ("RAG assessed done — leave it be, escalation ladder if recall
regresses") was conditional, and the condition has been met: four live 100-turn runs on
2026-07-12/13 (26B local + Qdrant + MiniLM, preserved artifacts under
`docs/artifacts/`) produced two live-only defects invisible to the deterministic gate —
#72 (write-dedup false-drop, fixed, class remains) and #73 (dense-only selection
ceiling, open, structural) — plus one measurement defect (#74, harness mis-attributing
a fail-closed definition turn as a recall miss) and one durable-fragility observation
(curator paraphrase is the substrate every downstream mechanism keys on, and MTP
decode's non-determinism means every scripted rerun re-rolls that substrate from
scratch). None of this is visible offline: the deterministic gate runs on keyword
embeddings by design and cannot see semantic-quality regressions (CLAUDE.md
verification note; docs/22 measure-first workflow). Per the escalation-ladder rule
itself, recall regressing live is exactly the sanctioned trigger to resume work — but
complexity must still pay rent for a one-person POC, which is the standard this
synthesis holds every proposed mechanism to.

## 2. Diagnosis recap

1. **Paraphrase fragility** — the curator stores LLM paraphrases of facts; write-dedup
   coverage math, recall-probe term matching, and dense embeddings all key on that
   surface text, and MTP's non-bit-determinism re-rolls it every run.
2. **Dedup identity (#72, fixed; class remains)** — write-dedup identity is fuzzy
   surface-term overlap, not fact identity; the 0.75/0.5 threshold split and reversal-
   marker escape hatch are the shipped mitigation, not a structural fix.
3. **Selection ceiling (#73, open)** — a fixed `top_k=5` window loses a fight against
   a growing dense-scored pool; the blue-seal rule memory ranked 12/17 for the exact
   callback it answers because cosine similarity does not track "is this the direct
   answer," and no additive boost constant can fix that without breaking the boost
   layer's cosine calibration.
4. **Fail-closed placement (#74, open)** — ~6.7%/turn controlled-failure rate is
   invariant #4 working correctly, but when it lands on a probe's *definition* turn the
   fact never enters the world and the harness reports it ten turns later as a
   misleading "recall miss" instead of a correctly-named write-side non-event.

## 3. Target architecture

### 3.0 Naming and shape

**Two independent guarantee lanes over the existing pipeline, one shared provenance
substrate, and a corrected measurement layer.** No new table, no new store, no new
LLM call site, no new vector-store feature. Every mechanism below is either (a) a
predicate flip on machinery that already exists and is validated-good, (b) one
nullable SQLite column via the established `_ensure_column` pattern, or (c) a pure
function over data the pipeline already loads every turn.

```
                          write path                          read path
turn succeeds                                          TurnSessionLoader.load
  → containment scan                                     ├─ canon_facts (author-pinned)
  → persist turn (turns.id, .turn_index)                  ├─ ALL session memories
  → deterministic extractor (user_message)  ─┐            │    (existing, unconditional)
       + tag/importance fold (3.2)           ├─ candidates└─ build_standing_facts()
  → curator (LLM, unchanged prompt)         ─┘                 Lane A: tag-eligible
       source_turn_id stamped on all (3.1)                     canon pinning (3.3)
  → write-dedup 0.75/0.5 (UNCHANGED)                    TurnRetrievalStage.run
  → persist (memory_episodes + source_turn_id)             ├─ dense dual-query (UNCHANGED)
  → index (Qdrant, UNCHANGED)                              ├─ additive boost rerank (UNCHANGED)
  → consolidation (UNCHANGED, carries                      └─ Lane B: lexical slice
       source_turn_id forward per 3.1.1)                        quota reorder (3.4)
                                                         select_retrieved_chunks_for_prompt
                                                           (UNCHANGED — dedupes against
                                                            Lane A's standing-facts block)
```

### 3.1 Provenance substrate — `memory_episodes.source_turn_id`

Every design independently converged on this; adopt it as the shared foundation.

```sql
ALTER TABLE memory_episodes ADD COLUMN source_turn_id INTEGER;  -- nullable, additive
```

Fifth instance of the existing idempotent `_ensure_column` migration (four precedents
in `app/persistence/sqlite.py`). `MemoryEpisode` / `MemoryCandidate`
(`app/domain/models.py`) gain `source_turn_id: int | None = None` — old rows
deserialize as `None`, same additive contract as `TurnDiagnostics.token_usage`.

Threading is a **three-signature, two-call-site** change (both call sites already have
a turn id in scope — do not understate this as "one extra parameter"):
`TurnMemoryStage.run` → `_run_extraction` → `_persist_and_index` each gain an optional
`turn_id: int | None` kwarg; the inline call site
(`turn_orchestrator.py` ~line 486, `persistence.turn.id`) and the deferred call site
(`run_deferred_memory`, `DeferredMemoryJob.turn_id`) both pass it. Every candidate
producer (curator, deterministic extractor, curator-failure fallback) stamps it on the
`MemoryCandidate` before persistence.

**3.1.1 Consolidation must carry provenance forward, not drop it.** A judge-verified
gap in the event-sourced design: `_persist_consolidation` builds a fresh candidate with
no turn linkage, so a folded fact's `source_turn_id` is lost the moment consolidation
fires — silently reintroducing exactly the measurement false-negative provenance was
built to remove, for any probe fact that gets folded mid-run. Fix: the consolidation
summary candidate gets `source_turn_id = min(source_turn_id of folded originals)`
(the earliest-established turn, since that's what a "when was this established" query
means) plus a `tags += ["source_ids:<comma-joined-original-episode-ids>"]` audit trail
reusing the existing tags_json column — no new schema.

**None-handling is specified, not left to Python's `min()`.** Legacy/pre-Stage-1
`memory_episodes` rows (and any row written during the transition period before this
stage lands) have `source_turn_id IS NULL`, and a consolidation pass can fold one of
those together with a newly-provenanced row from the same session — `min()` over a set
containing `None` is undefined as specified, the same class of None-handling hazard
three judges already caught elsewhere in this codebase (e.g. the `created_at`-None sort
hazard). The rule: take `min()` over the non-null subset of folded originals'
`source_turn_id` values; if that subset is empty (every folded original is legacy/
unprovenanced), the summary's `source_turn_id` stays `None` — never silently coerce a
missing value to a sentinel turn id. Cheap, and closes the hole three judges flagged
independently.

**What this does NOT become:** a fact-identity key, a dedup key, or a quote/fact_key
hash. The event-sourced design's `fact_key = sha256(turn_id + normalized_quote)`
approach was constructed-and-broken by two separate judges: NPC-side quotes are drawn
from MTP-nondeterministic assistant text (so "fact identity" still re-rolls per run for
half the dialogue), and the exact-tier dedup false-drops distinct facts extracted from
one multi-fact sentence sharing an identity. Provenance here is attribution-only:
"which turn produced this row," nothing more.

### 3.2 Deterministic tag/importance fold (best-match, not first-match)

At the curator-coverage-drop site (`app/orchestration/stages/memory.py:169-177`),
today a deterministic fallback candidate covered by a curated summary (0.5 threshold)
is silently discarded — including its guaranteed `promise`/`entrusted`/`agreement`/
`deadline` tag and its importance=4. Replace the drop with a fold onto the
**best-matching** (highest-coverage) curated summary, not the first one encountered in
list order:

```python
for candidate in fallback_candidates:
    covering = best_covering_summary(candidate.summary, curated)  # argmax coverage, not first-match
    if covering is None:
        extras.append(candidate)
    else:
        idx = covering.index
        curated[idx] = curated[idx].model_copy(update={
            "tags": ordered_union(curated[idx].tags, candidate.tags),
            "importance": max(curated[idx].importance, candidate.importance),
        })
        warnings.append(f"deterministic candidate folded (best-match, coverage={covering.score:.2f}): {candidate.summary[:80]}")
```

`best_covering_summary` requiring the *maximum*-coverage match rather than accepting
the first summary that clears 0.5 directly closes a judge-constructed failure: with
first-match, a two-fact turn ("I'll take the compass and I promise to return before
dawn") can fold the compass candidate's tags onto the dawn-promise summary if that
summary happens to be checked first and also clears 0.5 — donating tags to the *wrong*
fact while losing the compass event's own tag/importance entirely. Best-match reduces
but does not eliminate this (a pathological case with two near-equal coverage scores
remains possible and is left as an audited warning, not silently resolved).

Effect: any player-stated durable event is guaranteed canon-taggable and clears
`PERSONA_MEMORY_IMPORTANCE_FLOOR=4` even when the curator forgets the tag — removing
the LLM's tag vocabulary from the critical path for the first-person-stated class only.
The 0.75/0.5 threshold split, reversal-marker escape, and framing-strip are untouched.

### 3.3 Lane A — tag-eligible canon pinning (retrieval-free guarantee)

**The core empirical finding, verified against the preserved D3 artifact
(`docs/artifacts/live-validation-D3-2026-07-12.db`, session `1a3f65da…`), that makes
this lane cheap:** the guaranteed-inclusion path already exists
(`build_standing_facts`, wired every turn, retrieval-free, already deduped against
retrieved chunks with slot backfill already handled by the existing over-fetch
arithmetic) and already carries the exactly-correct durable-fact tags on the memory
that failed #73 — but is dead because `canon_importance_floor=4` while the curator's
importance distribution across the 48-memory pool is 9×1 / 38×2 / 1×3 / **zero at 4+**.
The blue-seal memory is `importance=2, tags=["rule", ...]`. #73's concrete instance is
not a retrieval ceiling — it is a dead predicate on validated-good machinery.

```python
# app/orchestration/canon_builder.py — build_standing_facts eligibility
eligible = [
    m for m in memories
    if m.visibility == Visibility.PLAYER
    and CANON_TAGS.intersection(m.tags)
    and (m.importance >= importance_floor or settings.canon_tag_pinning)
]
```

New Settings field `canon_tag_pinning: bool = False` (`CANON_TAG_PINNING` in
`.env.example`, house opt-in pattern identical to `RAG_RECENCY_WEIGHT`). Byte-identical
at default. `CANON_TAGS` gains German aliases (`regel`, `versprechen`, `abmachung`,
`frist`, `schwur`, `eid`, `anvertraut`), mirrored into consolidation's `_PRESERVE_TAGS`
so German-tagged durable facts are never folded either — one frozenset edit, itself
behind the same flag to preserve the byte-identical-at-default contract.

**3.3.1 Stale-fact safeguard — gated on `canon_tag_pinning`, mandatory whenever that
flag is on.** This safeguard is **not** a change to `build_standing_facts`'s existing,
already-shipped eligibility/dedup behavior at default. Verified against the current
code: `DETERMINISTIC_EVENT_IMPORTANCE = 4` (`app/memory/deterministic_extractor.py:17`)
already equals `canon_importance_floor`'s default of 4 (`app/config.py:127`), and the
deterministic extractor's tags (`promise`/`entrusted`/`agreement`/`deadline`) already
intersect `CANON_TAGS` — so deterministic-extracted durable facts are canon-eligible
**today**, with `CANON_TAG_PINNING=false`, purely via the existing importance-floor
predicate. `canon_tag_pinning=true` only widens eligibility for curator-tagged,
sub-floor-importance facts (the blue-seal case); it does not change how
already-floor-eligible facts are deduped. `build_standing_facts` today does only
exact-text dedup (`app/orchestration/canon_builder.py:55-66`) — no supersession logic
exists, and this proposal does not add any at default.

The new tag-family-supersession drop rule below **only fires when
`settings.canon_tag_pinning` is true.** It exists because unconditional canon pinning
(the flag's whole purpose — surfacing sub-floor curator-tagged facts) plus the
write-dedup coverage math (unchanged, see §3.5) can make a *superseded* fact immortal
instead of merely rank-decayed once pinning widens the eligible set. Concretely: player
says "the rule is now blue seal plus a knock"; term overlap with the stored "blue seal
only" memory clears 0.75 with no `_REVERSAL_MARKERS` hit (no "not/never/no longer" —
it's an amendment, not a negation), so the amendment gets write-deduped away while the
*original* rule keeps being pinned every turn once tag-pinning surfaces it. This is the
existing #72-class bug, not a new one — pinning just changes its consequence from
"decays by rank" to "repeats forever until a human notices," and only once the flag is
on. Mitigation, shipped in the same stage as pinning and active only when
`canon_tag_pinning` is true: `build_standing_facts` dedups pinned lines not just by
exact text (existing, flag-independent C1 behavior) but drops an OLDER canon-eligible
entry whose (`kind`-equivalent tag set, loosely: same CANON_TAG) sibling was superseded
by a strict term-superset candidate within the same tag family and a newer
`created_at` — a narrow, auditable rule, gated behind the flag, not a general
supersession engine. This is intentionally conservative: it fires only on
near-identical tag-family term supersets, logs every application, and leaves ambiguous
cases pinned (over-inclusion, not silent loss, remains the fallback failure direction).

**Caps and overflow:** `canon_max_items=8` / `canon_max_chars=900` (existing
constants) are the entire growth policy at v1. Verified headroom against the D3
artifact: 3 tag-eligible items, 352 chars. Overflow (past ~8 concurrent durable facts
in one campaign) silently drops the oldest/lowest-importance line today; this proposal
adds a `standing_facts_count`/`standing_facts_chars` diagnostic (additive
`TurnDiagnostics` fields) so overflow is *visible*, and defers any supersession/
consolidation-of-canon mechanism until a live campaign actually saturates the cap (see
§8, open question 2) — building it now is exactly the kind of complexity that must
earn its rent with evidence first.

### 3.4 Lane B — lexical slice quotas (retrieval-time guarantee, general case)

Lane A only rescues facts that (a) carry a `CANON_TAG` and (b) fit under the cap. The
D3 pool is 94% non-canon (45/48 memories) — lore, lower-stakes color, anything the
curator didn't tag durably. That class stays fully exposed to the untouched dense-only
ceiling unless something else covers it. This is the retrieval-guarantees design's
core mechanism, adopted with the bug fixes every judge round found, because it scored
highest of the four across all three lenses (6/8/8.5) and is the only mechanism in the
whole pool that is (a) general-purpose — not gated on tags — and (b) falsifiable
offline, before any live run, against the preserved D3 artifact.

**Scorer** (`app/rag/lexical.py`, new, pure functions, reuses `content_terms()` —
no new tokenizer):

```python
def score_memories_lexical(*, query_text: str, memories: Sequence[MemoryEpisode]) -> list[LexicalHit]:
    pool = [m for m in memories
            if m.visibility == Visibility.PLAYER
            and CONSOLIDATED_TAG not in m.tags]          # FIX 1 — exclude retired originals
    docs = {m.id: content_terms(m.summary) | content_terms(" ".join(m.tags)) for m in pool}
    idf = idf_over(docs)                                  # session-pool-relative IDF
    q = content_terms(query_text)
    hits = [LexicalHit(m.id, sum(idf[t] for t in (q & docs[m.id])), matched=tuple(sorted(q & docs[m.id])))
            for m in pool if q & docs[m.id]]
    return sorted(hits, key=lambda h: (-h.score, -h.importance, -h.created_ordinal, h.memory_id))
```

IDF is computed against the session's own memory pool, so frame vocabulary (the exact
`{player, iria, vale, keep, return}` #72 poison set) is automatically cheap and rare
terms (`blue, wax, seal, compass`) automatically expensive — self-calibrating, no
language-specific stopword list required (a genuine, if partial, German mitigation:
document-frequency weighting discounts frequent function words even without a German
stopword table, though the shared `content_terms`/`_stem` tokenizer stays
English-biased for suffix stemming — N2 remains explicitly deferred, see §8).
`CONSOLIDATED_TAG` exclusion (fix 1) closes a judge-verified bug: without it, a memory
consolidation deliberately unindexed from Qdrant gets resurrected into the actor prompt
by the lexical injection path, silently bypassing the consolidation policy.

**Selection** (`app/rag/ranking.py`, ranking policy stays in `app/rag` per invariant 5):

```python
@dataclass(frozen=True)
class SliceQuotas:
    lexical: int = 0            # opt-in; 0 = byte-identical no-op
    min_slice_score: float | None = None    # FIX 2 — floor, not just >=1 matched term; see below
```

`quotas.min_slice_score` (fix 2) closes the second judge-verified bug: with only
`min_lexical_matches=1` and no score floor, the 2 lexical slots fill on nearly every
turn in a large pool (almost any message shares one content term with *something*),
making the "guaranteed slice" a permanent ~40% cut of dense slots rather than a
targeted rescue. A minimum summed-IDF floor means only genuinely rare-term matches
claim a slot; weak/common-term hits spill to the fused fill exactly as if the slice
were empty. **The value shown above (`None`) is deliberate — this document does not
ship a numeral for `min_slice_score`.** An earlier draft of this proposal listed
`1.5` as a concrete default; unlike the 0.75/0.5 write-dedup thresholds, which are
empirically grounded and cited against live data, no basis for any specific IDF-floor
value has been measured yet. Treat `min_slice_score` as a placeholder to be set from
Stage 5 live-validation data (§6), not a decided default — Stage 4 ships the mechanism
with the floor gated behind an explicit opt-in (`None`/disabled behaves as "no floor,"
matching pre-fix behavior, so the fix's presence doesn't silently change semantics
before a real value is chosen), and Stage 5 is where a measured value gets recorded
and only then promoted into the shipped default.

Selected members are injected as `RankedChunk(original_score=0.0, applied_boosts={},
slice_score=hit.score)` — **fix 3**: `slice_score` is a dedicated field, not folded
into `applied_boosts`, because `adjusted_score == original_score + sum(applied_boosts)`
is an identity relied on elsewhere in diagnostics; a labeled side-channel preserves that
identity for every chunk instead of silently breaking it for injected ones.

Quota reordering happens on the **final** `prompt_window` (=`budget.retrieved_chunks`,
i.e. 5), not the inflated ~13-item rerank window — this placement detail is the single
most load-bearing insight across all three judge rounds of this design: without it, a
guaranteed member at fused position 11 never survives `select_retrieved_chunks_for_prompt`'s
walk, and the whole mechanism is decorative. `select_retrieved_chunks_for_prompt`
(context_budget.py) stays byte-identical; validated-good code is not touched.

**Fail-open ordering (fix 4):** the stage computes `lexical_hits` from
`context.session_memories` (already loaded every turn for standing facts — zero extra
query) **before** calling the retriever, not inside it. A Qdrant/dense-search failure
then degrades to lexical+recency-only (strictly better than today's empty-chunks
fallback) instead of losing the lexical leg along with the dense one — closing the
third judge-verified bug (as originally specified, lexical hits flowed through the
retriever call and died with any exception it raised).

**Confidence-gating interaction (flagged, must be tested, not left as a risk bullet):**
`retrieval_confidence = max(chunk.score)` over player-visible chunks feeds
`low_retrieval_confidence` behavior. An injected slice member carries
`original_score=0.0`, so a turn rescued entirely by the lexical slice can still read as
low-confidence and trigger hedging behavior on exactly the turn the mechanism exists to
rescue. Stage 4 (§6) ships a named test asserting the intended confidence semantics
(recommendation: confidence computation includes `slice_score`-bearing chunks at a
capped equivalent, not zero) rather than deferring the decision.

**Defaults:** `RAG_SLICE_LEXICAL_QUOTA=0` (off; live preset uses 2),
`min_slice_score` unset/disabled at v1 ship and tuned during Stage 5 live validation,
not guessed (see above). Settings/.env.example pairing per the test-enforced
convention.

### 3.5 What Lane A + Lane B deliberately do NOT touch

Write-dedup identity (`is_covered_by_summaries`, 0.75/0.5 split, reversal markers,
framing-strip) is **unchanged** by both lanes. Two designs proposed touching it —
IDF-weighting the coverage math (retrieval-guarantees' rider) and quote-based coverage
terms (event-sourced) — and both were judge-broken: IDF-weighted coverage makes
genuinely duplicate memories score *low* overlap once their shared motif terms go
IDF-cheap in a long campaign, silently re-inflating the pool the whole redesign exists
to relieve; quote-based coverage inflates false-drop rates for distinct German
sentences via the English-only stopword/stemmer pair, converting the documented
run-dependent #72 jitter into a *deterministic* per-language bug. Neither rider ships.
The 0.75/0.5 split stays exactly as #72 left it.

### 3.6 Turn-pipeline integration summary

Write path: unchanged through persistence except for (a) `source_turn_id` stamped on
every candidate producer and carried forward by consolidation, (b) the best-match tag/
importance fold at the coverage-drop site. Read path: unchanged through dense
retrieval and the additive boost rerank; Lane B reorders the final selection window
using data already loaded for Lane A; Lane A renders as today via
`TurnSessionLoader.load` → `LoadedTurnContext.standing_facts` → `build_actor_messages`,
with one predicate change. Both lanes are pure, deterministic, fail-open (Lane A rides
the session-loader path whose SQLite failure already fails the turn load, matching
today's canon-facts posture; Lane B degrades to fused-only on any scorer exception).
Neither lane adds an LLM call site, a new table, or a new vector-store feature.

## 4. How each failure mechanism dies

Honest classification: **impossible** (structurally cannot recur), **unlikely**
(residual rate measured or bounded, with an observable/recoverable failure direction),
or **unchanged** (mechanism untouched by design, with the reason stated).

**1. Paraphrase fragility.**
*Measurement half* — **unlikely→rare**, not impossible: provenance-based probe
attribution (source_turn_id-linked, §6) removes the substring-matching false negative
that produced the run-1 "symbol of a pact/promise" miss, for any definition-turn fact
that survives to a persisted row and isn't later folded by consolidation without
provenance carry-forward (§3.1.1 closes that specific hole). Attribution must be
**provenance OR phrasing-match**, never provenance-only — a judge showed pure
provenance-only attribution over-attributes on multi-memory turns (a definition turn
producing several memories, only one of which is the probe fact, would register as
"extracted" even if the curator missed the actual fact). *Retrieval half* — **unlikely**
for tagged facts (Lane A bypasses embedding entirely) and **unchanged-but-mitigated**
for everything else (Lane B's rare-term slice reduces but does not eliminate
dense-embedding sensitivity to curator wording). *Write half* — **unchanged**: the
curator still stores paraphrases; MTP still re-rolls them; no design in the pool
offered a write-side fix for this that survived judging (quote-anchoring broke on
NPC-side non-determinism and multi-fact false-drops). Residual: an untagged,
non-lexically-rare, curator-only fact still depends on embedding proximity surviving a
paraphrase re-roll — accepted, because it degrades to flavor-recall miss, not a broken
promise.

**2. Dedup identity (#72 class).** **Unchanged as a mechanism, by decision** — the
0.75/0.5 threshold split and reversal-marker escape are validated post-fix (zero
false-drops observed across four 100-turn runs at 0.75) and every rider proposed to
"improve" the coverage math (IDF-weighting, quote-based terms) was judge-broken.
Hardened only at the edges: the tag/importance fold (§3.2) means a candidate *dropped*
by curator-coverage still donates its durable tag to the best-matching survivor, so a
coverage-drop can no longer silence a fact's canon eligibility even though the
duplicate-summary row itself is still gone. Residual failure direction: rare, auditable
(drop warnings name the summary), and recoverable (the player can restate a fact, or it
can be author-pinned via the existing `canon_facts` API).

**3. Selection ceiling (#73).** Two lanes, two honesty levels. **Lane A: impossible
once tagged and under the cap** — for facts that (i) carry a `CANON_TAG` and (ii) fit
under the cap, pinned facts bypass embedding, vector search, and reranking entirely;
the D3 blue-seal case passes by construction, verified against the live artifact (not
asserted). But this classification is conditional on the tag arriving in the first
place: MTP re-rolls the curator's free-form tag choice per run exactly as it re-rolls
paraphrase (a judge's sharpest finding against the source design — "3/3 tag precision
is one dice roll"), mitigated but not eliminated by the deterministic fold guaranteeing
tags for the player-first-person-stated subclass. So the honest label is
**conditionally impossible: impossible once a `CANON_TAG` has survived onto the fact
and it fits under the cap; the tag's arrival itself is not deterministic** for the
curator-tagged subclass (it is deterministic for the §3.2 fold's first-person-stated
subclass). This document's own §7 penalizes a competing design for leading with a bare
"impossible" that glossed over exactly this kind of precondition; the same standard
applies here. **Lane B: unlikely**, general-purpose, independent of tags: a rare-term
direct answer occupies a reserved slot regardless of cosine rank or pool size, verified
falsifiable offline against the D3 artifact (query terms `{trust, message}` appear in
only 4 of 48 pool memories — a clean IDF win). Residual for both lanes: motif-heavy
long campaigns where many memories share the query's rare terms (bounded, auditable via
diagnostics — matched terms and slice ranks are shown, never a silent cosine loss);
German morphology (Lane B weaker there, stemmer is English-only; Lane A helped by tag
aliases only). Canon-cap saturation (Lane A, past ~8 concurrent tagged facts) is an
open, deferred, *visible* edge (§3.3.1). **P1.1 hybrid sparse+dense stays on the
roadmap as the escalation rung** for whatever residual survives both lanes after live
validation — pulled by evidence of a live miss on a fact neither lane covers, not
pushed now; see §6 Stage 6.

**4. Fail-closed placement (#74).** **Unchanged as engine behavior, correctly** — the
critic's fail-closed contract is invariant #4 and this proposal does not touch it. What
changes is attribution: the harness fails fast at the definition turn, naming the real
cause ("fact never entered the world: turn N failed closed") instead of reporting a
recall miss ten-to-forty turns later. A single scripted retry of the definition turn's
message (mirroring real player behavior on an errored turn) is included, labeled
honestly: the naive `0.067² ≈ 0.45%`-per-probe / `~96.5%`-run-survival math assumes
retry-independence that a judge correctly challenged (critic rejections plausibly
correlate within a retry pair, since the same draft class comes from the same model
state) — this proposal does **not** adopt that number as a target. The retry ships as
a documented, labeled, harness-local scenario-semantics change (`LIVE_DEFINITION_RETRIES=1`,
runs using it flagged in `quality_metrics`; see §3.4's closing note and §6 Stage 5 for
where this is defined and scoped), and its actual observed survival rate is what Stage
5 measures and records, not what is assumed going in.

## 5. What stays untouched, and why

Everything on the docs/22 "validated-good" list, unconditionally: consolidation
machinery (selection policy, SQLite-first persist ordering, `CONSOLIDATED_TAG`
lifecycle, parity accounting) — no design in the pool, including this one, touches it,
other than the provenance carry-forward fix in §3.1.1 which is additive, not a policy
change. The deterministic additive-boost rerank in `app/rag/ranking.py` — every
proposal to modify it (bigger lexical caps, RRF-fused scores flowing into the same
additive math) was judge-broken on cosine-scale calibration grounds; the boost
constants, their Settings/`.env.example` mirroring, and the `original_score`-preserving
diagnostic contract are untouched. Write-dedup (§3.5, above). `MemoryCurator`'s prompt,
`curator_gating=always`, and the fail-open-to-deterministic ladder — no prompt change
anywhere in this proposal (the fold in §3.2 operates purely on outputs already
returned). All four visibility-enforcement layers. The deferred-memory job flow and
drain-before-delete reroll semantics. SQLite-authoritative / Qdrant-derived split and
`reindex-memories` rebuildability. `PERSONA_MEMORY_IMPORTANCE_FLOOR=4` and its
cross-session-NPC-memory coupling — flagged as a real collateral finding (curated
memories currently never reach persona memory either, for the identical importance-
floor reason as #73's canon case) but deliberately **not** fixed here: it has no live
probe yet, and P2.4 (world-scoped multi-session probe) is the pre-authorized
instrument-first path docs/22 already specifies — build the probe before touching the
floor, not the other way around. The entire measurement stack (P0.4 corpus and its
calibrated floors, `event_key_retrieval` dense pins, `retrieval_miss` floors,
`memory_continuity`, provider-binding/containment suites, `InMemoryVectorStore` parity
convention) — extended additively (one new regression category, one new offline
replay script, additive diagnostic fields) and never loosened, per the docs/25
no-loosening rule. Every rejected-idea decision on record: no session cap, no
auto-gating, no cross-provider fallback, recency stays opt-in at 0.0, `top_k` stays 5
by default (crowding, not capacity, was the diagnosis — both lanes attack crowding).

## 6. Staged migration plan

Each stage: additive, default-off where it changes runtime behavior, full deterministic
gate (`ruff check . && mypy . && pytest && python -m app.evals.regression_runner`) plus
docs/08 CLI surface checks required to close, before any live run.

**Stage 0 — Instruments first** (maps to docs/22's "land the measurement before
coding" rule; the missing offline write-path benchmark identified independently by
every design's gap analysis).
- *Scope*: new `regression_runner` category `memory_write_lifecycle` (template:
  `memory_continuity.py`) — scripted transcript through the real deterministic
  extractor + real dedup + a task-aware fake curator, asserting durable-fact survival,
  the pinned compass/dawn adversarial pair still dedups correctly, and (once Stage 3
  lands) tag-presence post-fold and standing-facts inclusion. #74 harness fail-fast:
  `run_checkpoint` checks each probe's definition-turn `outcome`; on `controlled_failure`
  it fails immediately naming the cause instead of deferring to a later recall check.
  A ~30-line offline replay script (`app.diagnostics.replay_selection` or similar) that
  loads the preserved D3 artifact read-only (see rollback note below) and checks Lane
  A/B eligibility without any model or live Qdrant.
- *Deterministic tests*: the new eval category itself, plus unit tests for the fail-fast
  logic using fake `EventInspector`s (existing Protocol-typed seam).
- *Live measurement*: none required — this stage is instrumentation, not behavior
  change.
- *Rollback*: trivial; new test/tooling files only, zero runtime code touched.
  **Mandatory precondition**: copy the D3 artifact (`docs/artifacts/live-validation-D3-2026-07-12.db`)
  before any replay tooling opens it, or open read-only (`mode=ro`) — `git status`
  already shows untracked `.db-shm`/`.db-wal` files against this exact artifact from a
  prior read-write open; the canonical repro must not drift between stages.
- *Effort*: ~1.5-2 days.
- *Backlog mapping*: closes the harness half of #74; lands the offline write-path
  benchmark docs/22's gap analysis calls for.

**Stage 1 — Provenance substrate**
- *Scope*: §3.1 in full — `_ensure_column`, model fields, three-signature thread
  through both call sites, consolidation carry-forward (§3.1.1, including the
  None-handling rule: min of the non-null subset, or `None` if no folded original is
  provenanced). Harness attribution updated to provenance-OR-phrasing (not
  provenance-only, per §4's correction).
- *Deterministic tests*: repository round-trip unit test; deferred-job integration
  test asserting persisted `source_turn_id == job.turn_id`; consolidation test
  asserting a folded summary's `source_turn_id` equals the earliest original's; a
  **second** consolidation test covering the mixed-NULL case — one legacy/unprovenanced
  original (`source_turn_id IS NULL`) folded together with one provenanced original in
  the same pass — asserting the result is `min()` of the non-null subset, and a third
  asserting the all-NULL-subset case resolves to `None` rather than raising or
  coercing to a sentinel.
- *Live measurement*: replay Stage-0's offline script against a **fresh** smoke-run DB
  (not D3 — D3 predates this column and will read all-NULL, which is expected and not
  a failure) to confirm the column populates correctly end-to-end.
- *Rollback*: additive column, no behavior change if reverted — set flag/thread-through
  can be reverted independently of the column existing.
- *Effort*: ~1.5-2 days.

**Stage 2 — Deterministic tag/importance fold**
- *Scope*: §3.2, best-match version, at the curator-coverage-drop site.
- *Deterministic tests*: unit tests with the pinned compass/dawn fixtures plus a
  synthetic two-fact-per-turn fake-curator case proving best-match beats first-match
  (the judge-constructed wrong-donation scenario).
- *Live measurement*: none required pre-Stage-5; folds are inspectable in any
  smoke-run's diagnostics warnings.
- *Rollback*: pure function change at one call site; revert is a diff revert.
- *Effort*: ~0.5-1 day.

**Stage 3 — Lane A: tag-eligible canon pinning**
- *Scope*: §3.3 in full, including the mandatory (when the flag is on) stale-fact
  safeguard (§3.3.1) and German `CANON_TAGS`/`_PRESERVE_TAGS` aliases, both behind
  `CANON_TAG_PINNING=false`. `standing_facts_count`/`standing_facts_chars`
  diagnostics.
- *Deterministic tests*: `build_standing_facts` is a pure function —
  **flag-off byte-identity golden test that includes two same-tag-family
  deterministic-extractor facts in the fixture pool** (e.g. two distinct
  `promise`-tagged facts, both already floor-eligible at `importance=4` under today's
  default predicate) — this is the regression case fix 2 requires: it asserts that
  with `CANON_TAG_PINNING=false`, `build_standing_facts` output for that fixture is
  byte-identical to today's pre-change output (no supersession/drop logic fires,
  because §3.3.1's rule is gated on the flag and does not apply at default); new
  eligibility-with-flag-on tests; `memory_write_lifecycle` (Stage 0) gains the
  tag-presence-post-fold and standing-facts-inclusion assertions. Separate unit tests
  for the stale-fact safeguard's narrow supersession rule with the flag **on** (must
  fire on the constructed amendment case, must NOT fire on ambiguous cases, and must
  NOT fire at all — regardless of fixture — when the flag is off).
- *Live measurement*: **offline first** — replay script from Stage 0 asserts the
  blue-seal summary appears in `build_standing_facts(...)` output with the flag on,
  against the D3 pool, no model required. This is checkable today.
- *Rollback*: flag flip to `false`; zero data migration involved.
- *Effort*: ~1-1.5 days.
- *Backlog mapping*: this stage is the concrete fix for the #73 acceptance case's
  canon-tagged instance (docs/22 P2.2 run 3).

**Stage 4 — Lane B: lexical slice quotas**
- *Scope*: §3.4 in full, with all four judge-identified fixes baked in from the start
  (CONSOLIDATED_TAG exclusion, min-score floor left unset/disabled at ship rather than
  seeded with an unmeasured numeral, dedicated `slice_score` field, pre-retriever
  fail-open ordering) plus the confidence-gating test named explicitly rather than
  deferred. `LIVE_DEFINITION_RETRIES` is a harness-local scenario-semantics parameter
  (see Stage 5), not an `app.config.Settings` field — it is not subject to the
  `.env.example` pairing test because it does not affect runtime engine behavior, only
  how the live-validation harness scripts a probe's message sequence.
- *Deterministic tests*: IDF math, tie-breaks, quota permutations (empty slices,
  overlap with dense results, injection of non-dense-fetched hits), quotas=0 byte-
  identical golden test against `rerank_chunks`, visibility filtering in the scorer,
  a confidence-computation test with an injected `slice_score`-only chunk.
- *Live measurement*: offline first — replay script asserts memory `354b8d98` (the
  blue-seal repro id) ranks within the lexical quota for the callback query against
  the D3 pool (verified feasible: query terms `{trust, message}` occur in 4/48 pool
  memories). Then `reindex-memories` the D3 pool into disposable Qdrant + run
  `inspect_story_event` with quotas on to confirm selected-top-5 membership
  end-to-end. Then the P0.4 semantic-benchmark suite through the slice-enabled
  retriever, holding the calibrated floors (recall@10≥0.75, nDCG@10≥0.70, German
  recall@10≥0.55) — plus a pool-size-scaled (50/100/200) corpus variant to reproduce
  the #73 regime offline with a real embedding model.
- *Rollback*: `RAG_SLICE_LEXICAL_QUOTA=0`; byte-identical no-op, proven by the golden
  test.
- *Effort*: ~2-3 days (widest blast radius: new module, `LoadedTurnContext` field,
  stage plumbing, ranking-layer change, config pairing, diagnostics).
- *Backlog mapping*: the general-purpose fix for #73 beyond the canon-tagged subclass;
  independently satisfies the spirit of docs/22's request for a non-embedding-model-
  swap answer to the dense-only ceiling (P1.2 model swaps were separately rejected by
  live evidence — multilingual variants ranked the blue-seal case *worse*).

**Stage 5 — Live validation and default flip**
- *Scope*: no engine code change. Defines `LIVE_DEFINITION_RETRIES` explicitly as a
  harness-local parameter of the docs/25 Phase D runner (not a `Settings`/`.env.example`
  field, per Stage 4's note) and wires it through the preset. Full `docs/25` Phase D
  100-turn preset with `CANON_TAG_PINNING=on`, `RAG_SLICE_LEXICAL_QUOTA=2`,
  `LIVE_DEFINITION_RETRIES=1`, and the Stage-1 provenance-based (OR-phrasing)
  attribution active throughout. This run is also where a measured `min_slice_score`
  value is derived from observed IDF-score distributions and only then recorded as the
  shipped default (§3.4) — it is not decided in this document.
- *Deterministic tests*: n/a — this is the live arbiter stage per CLAUDE.md's
  verification ladder.
- *Live measurement*: **at least two** Phase D runs (not one — a judge correctly
  flagged that MTP's non-bit-determinism makes a single green run weak evidence for a
  default flip); record per-run: blue-seal-class fact pinned-every-turn (Lane A),
  lexical-slice hit rate and matched-term diagnostics (Lane B), quote/tag validation
  rates, definition-turn retry usage and its *actual* observed survival delta (not the
  assumed 96.5%), context-preflight warning counts (prompt-budget pressure from Lane
  A's pinned block, verified headroom ~352/900 chars but must hold under a real
  campaign), and P0.4 floors. Flip `CANON_TAG_PINNING` and
  `RAG_SLICE_LEXICAL_QUOTA` defaults, and set a measured `min_slice_score` default,
  only if both runs are clean; record evidence in docs/22 per the house rule,
  including the reinterpretation that the #73 acceptance bar ("blue-seal in selected
  top-5") is satisfied via Lane A's pinning path for the canon-tagged instance, not via
  rank improvement in dense retrieval — a different door, same purpose (fact reaches
  the actor prompt every turn), and this reinterpretation must be recorded explicitly,
  not silently.
- *Rollback*: both flags flip back to `false`; no data migration to undo (columns
  stay, just unused).
- *Effort*: 1-2 elapsed days including two multi-hour live runs.
- *Backlog mapping*: closes P2.2 (docs/22), which was blocked on #73; this proposal
  unblocks it via Lane A+B rather than requiring P1.1 first, per §6 Stage 6 below.

**Stage 6 — P1.1 hybrid sparse+dense (conditional, pulled not pushed)**
- *Scope*: unchanged from the existing docs/22 P1.1 specification and its danger
  zones — Qdrant named sparse vectors (FastEmbed BM25), server-side RRF,
  `RAG_HYBRID_SEARCH=off|rrf` default off, deterministic in-memory BM25 for
  `InMemoryVectorStore` parity reusing `content_terms`/`_stem` as tokenizer, identical
  visibility/scope filters on every leg, deterministic rerank kept **on top** as the
  explainability layer (not fused-score-first — the event-sourced design's attempt to
  put RRF-fused scores directly under the existing additive-boost math was
  judge-broken: RRF scores (~0.02-0.03) are an order of magnitude below the cosine
  range the boost constants are calibrated against, silently gutting the "preserves
  original vector score" half of invariant 5 — this stage must re-score fused
  candidates against dense cosine, or use scale-aware weights, before the boost layer
  ever sees them).
- *Trigger, explicit and measurable*: a live probe miss, after Stage 5 is green, on a
  fact that is (a) not tag-eligible for Lane A and (b) not covered by Lane B's lexical
  slice (i.e., genuinely a cross-embedding-distance miss with no lexical anchor). Do
  not build this speculatively.
- *Deterministic tests / live measurement / rollback*: as specified in docs/22 P1.1 —
  unchanged, this proposal adds only the trigger condition and the score-scale fix
  above.
- *Effort*: ~1 week (docs/22's own estimate; this proposal's contribution is only
  gating it correctly, not doing the work).

## 7. Explicitly rejected alternatives

**World-state-ledger's new `ledger_entries` table with (`kind`, `subject_key`,
`object_key`) supersession identity.** Rejected. The retrieval-free-bypass *insight*
is correct and is absorbed into this proposal via Lane A (§3.3) — reusing the existing
canon-pinning path rather than building a second guaranteed-inclusion mechanism next to
it. The table itself is rejected because its identity scheme is degenerate as specified:
the deterministic mirror hard-codes `subject='player', object=<persona name>`, so *all*
promises/agreements/entrusted-items to the same NPC share one identity — a judge showed
this makes every second distinct promise silently evict the first from the prompt
("every second promise, gift, or rule involving the same NPC silently evicts the
first... worse than the #72 class it replaces because it fires on DISSIMILAR facts").
A second judge independently found the design's own #73-acceptance claim
("by construction") false against the actual scripted transcript: the blue-seal
definition turn matches no deterministic-mirror regex, and the mirror false-positives
on the callback *question itself*, superseding the real rule on the exact turn being
measured. German reversal markers and stopwords are entirely absent from the reused
primitives, making stale facts *immortal* rather than merely stale in German campaigns.

**Event-sourced's quote-anchored fact identity and RRF-fused-under-boost integration.**
Rejected. The provenance-column half of this design is adopted (§3.1) — every judge
independently reached the same conclusion, so it is a convergent, not contested,
recommendation. The quote/`fact_key` dedup mechanism is rejected: NPC-side quotes are
drawn from MTP-nondeterministic assistant text, so the design's own "fact identity
becomes IMPOSSIBLE to re-roll" claim is false for half the dialogue (a judge
constructed this directly); the exact-match tier, advertised as false-drop-impossible,
false-drops distinct facts extracted from one multi-fact sentence sharing an
establishing quote (fixed here differently, via best-match folding in §3.2, not
quote-hash identity); and `content_terms`/`_stem` being English-only means quote-based
coverage math *inflates* false-drops for distinct German sentences, turning a
run-dependent bug deterministic — worse, not better, for exactly the stated
German+English operating constraint. The RRF-under-boost integration is rejected for
the score-scale reason detailed in Stage 6 above; if/when P1.1 ships, it must re-score
fused candidates to the boost layer's calibration, not feed RRF scores directly into
additive constants tuned for cosine.

**Minimal-invasive's tag-driven-pinning-only strategy, unsupplemented.** Partially
adopted (it is the empirical basis for Lane A, §3.3, and its diagnosis method —
querying the D3 artifact rather than theorizing — is the standard this whole synthesis
follows) but explicitly not adopted **as the sole mechanism**, because a judge found
its headline claim ("selection failure becomes structurally impossible for the canon
class") over-broad: tag eligibility depends on an uncontrolled, MTP-re-rolled curator
tag vocabulary, and the design's own "3/3 tag precision" evidence is one non-deterministic
sample, not a proof. Left standing alone, it also leaves the 94%-non-canon remainder of
the D3 pool fully exposed to the untouched dense ceiling — a judge called this "one
mechanism conditionally suppressed... two surviving essentially intact." This proposal
pairs it with Lane B specifically to cover that gap rather than accepting the
tag-vocabulary bet as the only line of defense.

**Retrieval-guarantees' sliced-selection mechanism as originally specified (unfixed).**
Not rejected — adopted, but only with the four bug fixes detailed in §3.4, each
independently verified against the source design by a judge: the unfiltered
`list_memories_for_session` call resurrecting `CONSOLIDATED_TAG` rows into the prompt;
`min_lexical_matches=1` with no score floor making the guaranteed slice a near-permanent
dense-slot tax rather than a targeted rescue; the `applied_boosts` additive-identity
break for injected chunks; and the fail-open ladder being asserted rather than actually
wired (a Qdrant exception, as specified, kills the lexical hits too, because they flow
through the retriever call rather than being computed before it). None of these were
fatal to the design's core mechanism — all four are cheap, targeted amendments — which
is why it is adopted-with-fixes rather than rejected outright, unlike the other three.

**The definition-turn retry's advertised survival-rate math (`0.067² ≈ 0.45%` per
probe, `~96.5%` run survival).** Rejected as a *claim*; the retry mechanism itself is
kept (§4, §6 Stage 5). A judge correctly identified that this assumes
retry-independence for controlled failures on an identical scripted message, which is
plausibly false (correlated failure modes from the same model state) — this proposal
requires Stage 5 to *measure* the actual delta, not assume the multiplication holds.

## 8. Open questions for the owner

> **Owner answers 2026-07-14** (the questions below are kept verbatim; answers recorded here
> rather than rewriting them):
>
> 1. **Open — separate discussion requested.** The continuity-contract framing is not yet
>    confirmed; Stage 3 (backlog #78) stays gated on it.
> 2. **Wait.** No preemptive canon-consolidation mechanism; raise the caps per-campaign if
>    the new `standing_facts_count`/`_chars` diagnostics ever show saturation in live play.
> 3. **Keep both.** The conservative over-inclusion trade-off is confirmed: ambiguous
>    supersessions stay pinned; only the narrow, audited tag-family rule may drop an older
>    fact.
> 4. **Yes to both.** The single labeled definition-turn retry reads as modeling a real
>    player's retry on an errored turn, and `LIVE_DEFINITION_RETRIES` stays harness-local
>    (not a `Settings` field).
> 5. **English is the main language for now.** N2 stays deferred; the partial mitigations
>    (Lane A tag aliases, Lane B pool-IDF) are the accepted interim position.
> 6. **Open — conceptual deep dive requested** on whether memories should carry across
>    sessions at all (whether a world outlives one session is the underlying question).
>    Stays behind the P2.4 instrument-first gate meanwhile.

1. **Continuity-contract framing.** This proposal's philosophical bet — durable,
   tagged facts (promises/rules/agreements/deadlines) are the player-facing contract
   that must never silently drop, while everything else (lore, color, flavor) is
   best-effort — is exactly minimal-invasive's framing, and it is consistent with the
   evidence (all four live failures were canon-class or harness artifacts, none were
   pure lore-recall misses). Confirm this is the right bar before Stage 3 ships, since
   it determines how hard Lane A vs. Lane B each need to work.
2. **Canon-cap saturation.** At `canon_max_items=8`/`canon_max_chars=900`, a campaign
   accumulating more than ~8 concurrent durable facts starts silently dropping the
   oldest/lowest-importance pinned line (now *visible* via the new diagnostic, but not
   prevented). Acceptable to raise the cap per-campaign (bounded by the existing #69
   preflight warning) until this is actually observed live, or is a canon-consolidation
   mechanism (folding low-priority pinned facts, distinct from episodic consolidation)
   worth building preemptively for long single-campaign use?
3. **Stale-canon-fact safeguard scope.** §3.3.1's narrow tag-family-supersession rule
   (active only when `canon_tag_pinning` is true) is deliberately conservative
   (over-inclusion over silent loss). Confirm this trade-off, or would a stricter rule
   (accepting some silent-loss risk to keep the canon block smaller) be preferred for
   this specific single-player use case?
4. **Definition-turn retry as a documented scenario-semantics change.** Confirm this
   reads as "the harness now models a real player's natural retry on an errored turn"
   rather than as re-rolling a run, per docs/25's no-loosening spirit — the mechanism
   changes what a scripted message maps to (possibly two turns instead of one), not any
   assertion or threshold. Also confirm the harness-local (non-Settings) scoping of
   `LIVE_DEFINITION_RETRIES` from Stage 4/5 is the intended home, versus promoting it
   to a real Settings field if a future need arises to vary it outside the live-
   validation harness.
5. **German lexical layer priority.** Both lanes have partial-but-incomplete German
   mitigations (tag aliases for Lane A; IDF's stopword-independence for Lane B), and
   N2 (a full German stopword/stemming layer) stays explicitly deferred per existing
   roadmap decision. Given German+English is a stated operating constraint, is this
   an acceptable interim position, or should N2 be pulled forward ahead of Stage 6?
6. **Persona-memory importance-floor coupling.** Confirmed but deliberately deferred
   (§5): curated (non-deterministic-extractor) memories currently never reach
   cross-session NPC memory either, for the identical importance-floor reason as
   #73's canon case. Leave fully behind the P2.4 instrument-first gate as
   recommended, or does this warrant the owner's attention sooner given it affects
   every NPC's long-game memory, not just one probe class?
