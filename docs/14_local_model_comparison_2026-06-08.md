# Local Model Comparison - June 8, 2026

## Scope

This study compares the repository's small and 26B local-model profiles with the same application
snapshot, prompts, seed, inference settings, hardware, Qdrant image, and live-stack workflow. It
does not declare an overall prose winner. Findings are separated into application defects, model
limitations, model-application interactions, and infrastructure limitations.

Automated results come from
[`app.diagnostics.live_checkpoint`](../app/diagnostics/live_checkpoint.py) and
[`app.diagnostics.model_comparison`](../app/diagnostics/model_comparison.py). Manual judgments are
identified explicitly and were made from aligned transcripts under `/tmp`.

## Reproducibility

The immutable configuration is preserved in the
[run manifest](artifacts/local-model-comparison-2026-06-08/run-manifest.md). Key values:

| Item | Value |
| --- | --- |
| Git HEAD | `22f23ffdb30e9da0bc13921a9042b157a45c9493` |
| Tracked diff SHA-256 | `be8ab452972d5772419d240b4af97de834319432fd4183cd896f1482d1dd82b8` |
| Host | Apple M5, 32 GB, macOS 26.3 |
| llama.cpp | build `8680` (`15f786e65`) |
| Seed | `424242` |
| Context | `8192` tokens |
| Actor generation | `500` tokens, temperature `0.75` |
| Structured generation | `350` tokens, temperature `0.0` |
| Cache | K `q8_0`, V `q4_0`, flash attention on |
| Retrieval | top 5, 800 characters per chunk |
| Cloud | off |

Models:

| Profile | Model | Quantization |
| --- | --- | --- |
| small | `DavidAU/gemma-4-E4B-it-The-DECKARD-Expresso-Universe-HERETIC-UNCENSORED-Thinking-GGUF` | `Q8_0` |
| 26B | `HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced` | `Q4_K_M` |

Before live execution, `ruff`, `mypy`, 260 Python tests, frontend tests, and 45 deterministic
regression checks passed. Docker, llama.cpp, both cached models, disk space, and ports `8080`,
`18080`, and `6334` were verified.

## Execution Record

| Run | UTC interval | Exit | Runtime artifacts |
| --- | --- | ---: | --- |
| 20-turn small primary | 11:44-12:01 | 1 | `/tmp/rolerag-model-comparison-20/small` |
| 20-turn 26B | 12:01-12:28 | 0 | `/tmp/rolerag-model-comparison-20/26b` |
| 20-turn small rerun | 12:31-13:02 | 0 | `/tmp/rolerag-model-comparison-20/small-rerun-1` |
| 20-turn small tie-breaker | 13:03-13:41 | 0 | `/tmp/rolerag-model-comparison-20/small-rerun-2` |
| 50-turn small primary | 13:41-17:11 | 1 | `/tmp/rolerag-model-comparison-50/small` |
| 50-turn 26B | 18:59-20:34 | 0 | `/tmp/rolerag-model-comparison-50/26b` |
| 50-turn small rerun | 20:37-23:08 | 1 | `/tmp/rolerag-model-comparison-50/small-rerun-1` |

The primary commands are recorded in the manifest. The original small 20-turn run returned an
empty actor response on turn 10. Two identical reruns completed, so the exact empty response was
not reproducible. The small 50-turn run then timed out while requesting turn 38; its identical
rerun timed out while requesting turn 30. The timeout category therefore reproduced even though
the exact turn varied.

Sanitized machine-readable summaries:

- [20-turn comparison](artifacts/local-model-comparison-2026-06-08/comparison-20.json)
- [50-turn comparison](artifacts/local-model-comparison-2026-06-08/comparison-50.json)

## 20-Turn Qualification

| Dimension | Small primary | Small rerun 1 | Small tie-breaker | 26B |
| --- | ---: | ---: | ---: | ---: |
| Exit | 1 | 0 | 0 | 0 |
| Completed checkpoint turns | 10 | 20 | 20 | 20 |
| Persisted memories | not finalized | 11 | 12 | 31 |
| Critic warnings | not finalized | 20 | 20 | 20 |
| Curator warnings | not finalized | 10 | 10 | 0 |
| `length` finishes | not finalized | 7 | 13 | 0 |
| Mean latency | not finalized | 82.847 s | 91.453 s | 76.283 s |
| p50 latency | not finalized | 83.379 s | 95.758 s | 82.576 s |
| p95 latency | not finalized | 105.091 s | 117.858 s | 92.118 s |
| Promise extracted | unavailable | no | no | yes |
| Promise selected | unavailable | no | no | no |
| Automated callback recall | unavailable | no | no | no |

The 20-turn gate was opened because deterministic checks passed, both providers loaded repeatedly,
Qdrant and persistence invariants held in every completed run, and the small hard failure did not
repeat in two identical reruns. Quality warnings, truncations, and callback misses remained
report-only as specified.

## 50-Turn Extension

| Dimension | Small | 26B |
| --- | ---: | ---: |
| Primary exit | 1 | 0 |
| Completed turns | 37, then timeout on 38 | 50 |
| Identical rerun | 29, then timeout on 30 | not required |
| Persisted memories | incomplete run | 71 |
| Qdrant session memories | incomplete run | 71 |
| Critic warnings | incomplete run | 50 |
| Curator warnings | incomplete run | 0 |
| Indexing warnings | 0 before failure | 0 |
| Retrieval warnings | 0 before failure | 0 |
| `length` finishes | summary unavailable after failure | 0 |
| Mean latency | summary unavailable after failure | 83.047 s |
| p50 latency | summary unavailable after failure | 85.451 s |
| p95 latency | summary unavailable after failure | 102.384 s |
| Total measured turn latency | summary unavailable after failure | 4152.327 s |

The small result is an operational limitation, not a quality score for 50 turns. Both failures were
read timeouts around a long actor request. In the primary run the provider timeout became an
unhandled API exception and HTTP 500; in the rerun the checkpoint client's 240-second timeout
expired. The provider client is constructed without an application-specific timeout or retry policy
([provider adapter](../app/llm/openai_compatible.py#L11-L35)).

## Attribution Funnels

### 26B, 50 Turns

| Seeded event | SQLite extraction | Exact Qdrant ID | Selected | Automated recall | Manual judgment |
| --- | --- | --- | --- | --- | --- |
| Before-dawn promise | yes | yes | no | no | Partial: remembered a promise and returning, but omitted the before-dawn condition |
| Silver compass | no | no | no | no | Partial action: returned an unnamed metal item without identifying the compass |
| Blue-seal trust rule | yes | yes | no | no | Incorrect: invented a three-source verification rule |
| Third-rose-pedestal key | yes, two matches | yes, both | no | no | Incorrect: invented a hollow western urn |
| Three-tap west-door signal | yes | yes | no | no | Incorrect: invented a two-part cuff gesture |

Four events reached SQLite and Qdrant but none was selected. Their earliest failed stage is
retrieval/ranking, an application defect. The silver compass failed first at extraction, a
model-curation limitation under the current schema and prompt. No selected result had hidden
visibility.

The automated matcher did not create a false negative for any fully correct callback. It did miss
some partial semantic behavior, such as returning an unnamed metal object, but manual review still
found the seeded fact incomplete or wrong.

### Small

| Run and event | Extraction | Indexing | Selection | Recall | Interpretation |
| --- | --- | --- | --- | --- | --- |
| 20-turn rerun 1, promise | no | not applicable | no | no | Earliest failure: model extraction/schema interaction |
| 20-turn tie-breaker, promise | no | not applicable | no | no | Same reproducible extraction miss |
| 50-turn primary, promise | not durably reported | not durably reported | not durably reported | manual miss | Run failed before final report |
| 50-turn primary, compass | not durably reported | not durably reported | not durably reported | manual miss | Callback reached, but report was lost on later timeout |
| Later 50-turn callbacks | unobserved | unobserved | unobserved | unobserved | Run terminated before completing the study |

The failed small 50-turn runs do not support claims about all five event funnels. Missing values are
reported as unavailable rather than inferred from partial state.

## Structured Output

- Both profiles produced invalid critic output on every completed turn: 20/20 for each qualified
  run and 50/50 for 26B. Because the failure is shared, attribution is a prompt/schema/parser
  interaction, not two independent model failures.
- The small profile also failed memory-curation structure on 10/20 turns in both successful
  qualification reruns.
- The 26B profile produced valid curator structure on every completed turn, but still omitted the
  silver compass from durable memory. Valid JSON therefore did not guarantee correct event
  selection.
- Indexing and retrieval emitted no warnings in completed runs.

## Completion and Latency

The small model reached `finish_reason=length` on 7/20 and 13/20 turns in its two successful
qualification runs. The 26B model returned `stop` on all 20- and 50-turn completions. Under the
shared 500-token actor budget, this is a reproducible small-profile behavior rather than a broad
application token-budget failure.

Small response lengths were unstable. Rerun 1 ranged from 8 to 2388 characters; rerun 2 ranged
from 334 to 2447. Several `length` responses visibly ended without completing the requested action.
The 26B 50-turn responses ranged from 948 to 1707 characters and all reported `stop`.

The 26B profile was not faster in absolute terms at 50 turns: median latency was 85.451 seconds and
measured turn time totaled 69.2 minutes. The small profile was unable to complete the extension
because individual requests eventually exceeded the effective timeout path.

## Manual Transcript Review

These observations are manual, not automated metrics:

- **Directness:** Both profiles often answered with atmospheric generalities instead of the
  requested concrete fact. The 26B callbacks at turns 38, 45, and 50 were direct but confidently
  wrong.
- **Persona consistency:** 26B generally retained Iria's precise, dry voice. Small frequently
  drifted into ornate court narration, changed the regent's gender, and introduced unrelated
  political subplots.
- **Invented facts:** After filtering the noisy proper-noun detector manually, unsupported small
  examples included the Duke of Valois, Countess of Astrea, Atheria, Veridia, Lorraine, Burgundy,
  Castille, Guise, Mornay, and Vaudreuil. Unsupported 26B examples included Countess Elara, guard
  Kaelen, and the Count of Oakhaven. None appears in the visible lore, scene, or persona files.
- **Callback correctness:** 26B demonstrated fluent continuity language without reliable factual
  continuity. It often sounded certain while substituting a new rule, location, or signal.

## Issue Ledger

| Symptom | Evidence | Affected profile | Attribution | Confidence | Recommended application action |
| --- | --- | --- | --- | --- | --- |
| Exact seeded memories never selected | 0/4 extracted 26B events selected at 50 turns | shared application path | Application retrieval/ranking defect | High | Persist per-candidate scores and filters; add exact durable-event boost or event-key retrieval test |
| Critic invalid on every turn | 20/20 small, 20/20 and 50/50 26B | both | Model-schema-parser interaction | High | Use schema-constrained generation, record parser failure type, add deterministic critic fallback |
| Small curator invalid on half of turns | 10/20 in both successful reruns; 0/20 for 26B | small | Model limitation under current schema | High | Tighten curator schema/prompt and add conservative deterministic durable-event extraction |
| Small promise never extracted | Two identical successful 20-turn runs | small | Model extraction limitation | High | Add explicit promise/deadline fallback extraction and exact event assertions |
| 26B compass never extracted | Valid curator outputs but no matching memory | 26B | Model content-selection limitation | Medium-high | Add acquisition/entrustment examples and deterministic item-transfer extraction |
| Small truncates frequently | `length` on 7/20 and 13/20 turns | small | Model behavior under shared token budget | High | Bound response style, propagate length handling, continue/retry or return controlled failure |
| Small long session times out | Primary at turn 38, rerun at turn 30 | small | Model-application/infrastructure interaction | High | Configure explicit provider timeouts, bounded retries, cancellation, and controlled 504/503 response |
| Empty small actor response | Primary turn 10; absent in two reruns | small | Inconclusive model-application interaction | Medium | Reject empty provider content with structured diagnostics and one bounded retry |
| Unsupported named entities | Manually verified examples in both transcripts | both, more extensive in small | Model invention plus ineffective critic | High | Add report-only unsupported-entity checks against visible context |
| Fluent but false callbacks | 26B invented rule, hiding place, and signal | 26B | Model context-use limitation after retrieval miss | High | Treat retrieval miss as uncertainty; prevent confident callback claims without selected evidence |
| Partial-run diagnostics lost | Failed checkpoints produced no final JSON summary | small 50-turn runs | Application diagnostic defect | High | Write an atomic checkpoint after every turn and preserve event inspections before each request |

## Conclusions by Dimension

- **Operational endurance:** 26B completed 50 turns. Small repeatedly timed out before completion.
- **Structured memory curation:** 26B satisfied the JSON contract reliably; small did not.
- **Durable-event extraction:** 26B extracted four of five events; small missed the qualification
  promise in both successful reruns.
- **Indexing and visibility:** No defect was observed. Exact extracted IDs were present in Qdrant,
  SQLite/Qdrant counts matched, and hidden memories were never selected.
- **Retrieval:** The application failed to select every exact seeded memory in the 26B extension.
- **Callback use:** Neither profile demonstrated reliable factual callback correctness.
- **Completion control:** Small showed frequent token-limit completion and reproducible long-run
  timeouts; 26B completed with `stop`.
- **Prose:** 26B was more consistently in persona, while both profiles invented unsupported facts.
  This is a dimensional observation, not an overall winner declaration.

## Limitations and Open Questions

- Only one story, seed, hardware host, llama.cpp build, and quantization per profile were tested.
- The primary small 20-turn run failed, so small qualification metrics come from the required
  identical reruns.
- Failed 50-turn runs did not emit final checkpoint JSON; partial state cannot reconstruct
  time-correct retrieval selection after cleanup.
- The proper-noun detector has many sentence-initial false positives. Only named examples verified
  against the visible source files are called unsupported.
- Retrieval diagnostics record selected IDs but not all rejected candidates and score components,
  limiting root-cause precision.
- Per-stage actor, critic, curator, embedding, and indexing timings were not captured.
- The timeout boundary is imposed by layered OpenAI/httpx/checkpoint clients and was not isolated
  to one configured duration in this study.

Full transcripts, SQLite databases, Qdrant state, Playwright results, and llama.cpp logs remain
under `/tmp` and are intentionally not committed.
