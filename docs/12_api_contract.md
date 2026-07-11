# 12 - API Contract

> Reviewed: 2026-07-11 @ a20bfc5

## Scope

The FastAPI surface is a local-use API covering five concern areas: session lifecycle (creation,
lookup, recent list, mid-session scene switch), turn execution (non-streaming, buffered
streaming, last-turn deletion), author surfaces (session memories, pinned canon facts),
diagnostics (runtime status, per-turn and bulk turn details, eval runs), and the public content
catalog. It delegates engine behavior to shared composition and orchestration services.

Available endpoints:

- `GET /runtime/status`
- `GET /content/catalog`
- `GET /sessions`
- `POST /sessions`
- `GET /sessions/{session_id}`
- `POST /sessions/{session_id}/scene`
- `GET /sessions/{session_id}/turns/{turn_index}`
- `GET /sessions/{session_id}/turn-details`
- `GET /sessions/{session_id}/memories`
- `GET /sessions/{session_id}/canon`
- `POST /sessions/{session_id}/canon`
- `DELETE /sessions/{session_id}/canon/{fact_id}`
- `POST /sessions/{session_id}/turns`
- `POST /sessions/{session_id}/turns/stream`
- `DELETE /sessions/{session_id}/turns/last`
- `GET /diagnostics/eval-runs`
- `GET /diagnostics/eval-runs/{run_id}`

The Angular SPA is served as static files at `/app` (the root `/` redirects there); it is not an
API endpoint and is excluded from OpenAPI.

FastAPI also serves a generated, always-current API reference at `/docs` (Swagger UI) and
`/openapi.json`. The generated schema is the ground truth for the endpoint inventory and exact
payload shapes; this hand-written contract covers what OpenAPI cannot express — SSE framing,
error-code semantics, visibility and exposure rules, and deferred-memory semantics.

API routes and the local browser UI do not own retrieval, persistence, routing, prompt
construction, or visibility logic.
SQLite remains authoritative state. Qdrant remains a derived retrieval index.

## Runtime Status

`GET /runtime/status` returns safe, shallow, non-diagnostic runtime metadata for the local web-UI
status panel. It does not call LLMs, instantiate providers, open SQLite, probe Qdrant reachability,
query retrieval collections, or perform deep runtime checks.

The response shape is:

```json
{
  "app_name": "rolerag-poc",
  "app_version": "<app.__version__>",
  "environment": "local",
  "cloud_mode": "ask",
  "retrieval_configured": true,
  "content_catalog_available": true,
  "local_provider_configured": true,
  "cloud_provider_configured": false
}
```

The boolean fields mean required settings are present, not that external systems are reachable.
`content_catalog_available` may validate the configured catalog enough to return a boolean, but
catalog errors are collapsed to `false` without diagnostic details.

Use `python -m app.cli doctor` and `python -m app.cli smoke-run --real-runtime` for deeper
operational checks. Those tools remain the place for configuration diagnostics, SQLite checks,
Qdrant checks, provider reachability, and real-runtime smoke validation.

Runtime status responses explicitly exclude API keys, provider URLs, model secrets,
`content_root`, file paths, SQLite paths, Qdrant URLs, prompts, retrieved chunks,
GM/private fields, hidden context, and internals.

## Content Catalog

`GET /content/catalog` returns public session-start metadata from the backend process-level
`CONTENT_ROOT` only. It is a read-only catalog for the local web-UI selectors and does not accept
query parameters, request bodies, per-request content roots, or frontend scenario-pack selection.

The response groups public metadata into:

- `worlds`: public world identifiers, names, default scene identifiers, and referenced scene and
  persona identifiers
- `scenes`: public scene identifiers, titles, locations, player-visible summaries, and active
  persona identifiers
- `personas`: public persona identifiers, names, roles, public descriptions, and speaking styles

Catalog responses explicitly exclude private or internal fields: `gm_private_summary`,
`private_description`, `secrets`, `forbidden_knowledge`, `content_root`, raw file paths, hidden
lore, raw prompts, retrieved chunks, SQLite internals, Qdrant internals, and provider internals.

If configured content files cannot form a public catalog, the API returns `400
invalid_content_catalog` using the standard error envelope.

## Session Creation

`POST /sessions` accepts:

```json
{
  "world_id": "demo_world",
  "scene_id": "rose-gallery",
  "player_name": "Avery",
  "active_persona_id": "archivist",
  "provider": "local",
  "skip_lore_ingest": false
}
```

`provider` is `"local"` (default) or `"cloud"`. It is chosen once, at creation, and is
immutable for the session's entire lifetime — every task type (actor, repair, critic,
memory extraction) for that session runs on this provider; there is no per-turn override
and no mid-session switch.

Creating a `cloud` session is gated by `CLOUD_MODE`: `off` rejects the request with `400
cloud_unavailable`; `auto` allows it silently; `ask` is enforced by the CLI/SPA callers
(interactive confirmation before the request is sent), not by this endpoint itself. A
`cloud` request also requires a configured cloud API key regardless of `CLOUD_MODE`.

`skip_lore_ingest` (default `false`, additive) mirrors the CLI's `start-session
--skip-lore-ingest`: unless set to `true`, session creation best-effort auto-ingests the
scenario's `documents/manifest.json` lore (idempotent, fail-open) via the shared
`app.composition.auto_ingest_scenario_lore` helper. A scenario with no manifest is a silent
no-op. A failed ingest (unreachable vector store, missing embedding backend, or an
embedding-model fingerprint mismatch — docs/22 P1.4) never fails session creation; it
surfaces as a string in the response's `warnings`.

The response contains safe session identifiers, the bound provider, and any auto-ingest
warnings:

```json
{
  "session_id": "<id>",
  "world_id": "demo_world",
  "active_scene_id": "rose-gallery",
  "active_persona_id": "archivist",
  "provider": "local",
  "warnings": []
}
```

The API uses the process-level `CONTENT_ROOT` settings value when creating sessions. API requests
do not accept a per-request content root or per-request scenario-pack selection. CLI scenario-pack
support through `--content-root` remains available.

## Scene Switch

`POST /sessions/{session_id}/scene` switches the session's active scene mid-session. It accepts:

```json
{
  "scene_id": "rose-gallery"
}
```

`scene_id` is required (1–200 characters; unknown request fields are rejected). It must be one of
the session world's `scene_ids` and must load as a valid scene; otherwise the request fails with
`400 invalid_scene`. On success the response has the same shape as the `POST /sessions` response,
carrying the updated `active_scene_id`. Unknown session ids return `404 session_not_found`.

## Recent Sessions

`GET /sessions` is a local-use recent-session list (used by the CLI `list-sessions` flow and the
SPA's session pickers). It returns
only safe metadata from SQLite authoritative state, ordered by `updated_at` descending with
`created_at` as the deterministic tie-breaker, and capped at 10 sessions. It accepts no query
parameters and no request body.

The fixed response shape is:

```json
{
  "sessions": [
    {
      "session_id": "<id>",
      "world_id": "demo_world",
      "active_scene_id": "rose-gallery",
      "active_persona_id": "archivist",
      "player_name": "Avery",
      "created_at": "2026-06-03T10:00:00Z",
      "updated_at": "2026-06-03T10:05:00Z"
    }
  ]
}
```

This endpoint does not instantiate turn services, call LLMs, query Qdrant, load turns, load
memory, run retrieval, or include diagnostics. It explicitly excludes messages, prompts,
`content_root`, file paths, retrieved chunks, memory, SQLite details, Qdrant details, provider
data, hidden/private fields, and storage diagnostics.

## Session Lookup

`GET /sessions/{session_id}` returns safe session identifiers and a bounded recent-turn list.
It does not return player names, content roots, hidden scene state, private persona state, SQLite
internals, or Qdrant diagnostics.

The CLI `resume` command uses this endpoint, and the SPA's session store consumes it for resume;
the returned `recent_turns` are rendered as transcript entries and remain backend-owned.

## Turn Detail

`GET /sessions/{session_id}/turns/{turn_index}` returns the persisted record for a single turn,
combining the stored turn fields (`turn_index`, `outcome`, `scene_id`, `persona_id`,
`user_message`, `assistant_message`, `route`, `created_at`) with the diagnostics captured at turn
time (`finish_reason`, `memory_written`, `critic_status`, `warnings`, `errors`, `stage_timings`,
and an optional `retrieval` ranking). See [Turn Execution](#turn-execution) for the `outcome` and
`errors` semantics. The `retrieval` block reuses the same metadata-only candidate shape as the
turn-execution response (`query`, `selected`, `rejected`), so no chunk text is exposed. Turns
persisted before diagnostics existed report empty diagnostic defaults.

It returns `404 session_not_found` for an unknown session and `404 turn_not_found` for an unknown
`turn_index`.

## Bulk Turn Details

`GET /sessions/{session_id}/turn-details` returns every stored turn's full diagnostics in one
call — `{"session_id": "<id>", "turns": [<turn detail>, …]}` with each entry shaped exactly like
the single-turn `GET /sessions/{session_id}/turns/{turn_index}` response, ordered by
`turn_index`. It exists so the SPA's Analytics and RAG Inspector pages don't fan out N requests
per session. The same metadata-only exposure rules apply: no chunk text, no prompts, no hidden
fields. Unknown session ids return `404 session_not_found`.

## Eval Runs

`GET /diagnostics/eval-runs` is a read-only scan of the process-level eval results directory for
per-run `conversation-checkpoint.json` artifacts. The directory is read from the
`EVAL_RESULTS_DIR` environment variable (deliberately not a `Settings` field), defaulting to
`<repo>/eval-runs`; each run is a subdirectory containing either
`raw/conversation-checkpoint.json` (the bake-off layout) or a top-level
`conversation-checkpoint.json`. It returns `results_dir` plus a list of run
summaries: `id`, `status`, `turn_count`, `recall_misses`, `extraction_misses`,
`retrieval_misses`, `total_seconds`, `p50_seconds`, `p95_seconds`, and `warning_total`. It does
not call LLMs or touch SQLite/Qdrant.

`GET /diagnostics/eval-runs/{run_id}` returns the full `conversation-checkpoint.json` payload for
one run (the SPA Eval page's drill-down). Unknown run ids return `404 eval_run_not_found`. Run
artifacts are local diagnostic files produced by the live checkpoint harness; the endpoint serves
them verbatim and is local-use only, like the rest of the surface.

The tooling that produces these run artifacts — the live-smoke checkpoint, the model bake-off,
and the RAG-knob sweep — is documented in
[19_verification_and_eval_tooling.md](19_verification_and_eval_tooling.md).

## Session Memories

`GET /sessions/{session_id}/memories` returns the persisted durable-memory episodes for a
session as metadata: `id`, `scene_id`, `actor_id`, `summary`, `importance`, `visibility`, and
`tags`. The web UI renders this in a read-only "Memory" panel and the CLI exposes the
same data via `inspect-memories`. Unknown session ids return the standard `404` envelope.

The endpoint deliberately returns episodes of every visibility level (`player`, `gm`,
`character_private`): it is a single-user authoring surface, so the author can inspect their own
GM-only notes. Actor-facing leakage is prevented by retrieval and prompt visibility filtering,
not by filtering this endpoint (see [Exposure Boundaries](#exposure-boundaries)).

## Session Canon

The canon endpoints are an author surface for pinning durable "Standing facts" into the actor
prompt by hand, alongside the auto-derived canon. Pinned facts lead the Standing-facts block.

`GET /sessions/{session_id}/canon` returns the pinned facts as `{ "session_id": "<id>", "facts":
[{ "id": "<id>", "text": "<fact>" }] }`.

`POST /sessions/{session_id}/canon` accepts `{ "text": "<1-500 chars>" }` and returns `201` with
the created `{ "id": "<id>", "text": "<fact>" }`. It returns `503 canon_unavailable` when the
session was opened without a canon repository.

`DELETE /sessions/{session_id}/canon/{fact_id}` returns `204` on success and `404
canon_fact_not_found` for an unknown fact id.

All three return `404 session_not_found` for an unknown session.

## Turn Execution

`POST /sessions/{session_id}/turns` accepts:

```json
{
  "message": "What have you heard about the regent?",
  "active_persona_id": "archivist"
}
```

`active_persona_id` is optional. When it is provided and differs from the stored session
persona, the turn performs a mid-session persona switch: the id must belong to the session
world's persona list (otherwise `400 invalid_turn_request`), the turn runs under the new
persona, and the switch becomes durable for subsequent turns only after this turn persists — a
failed turn never commits it.
The request body carries no provider or routing flags: the turn always runs on the
session's provider, bound once at `POST /sessions` time and immutable thereafter (see
[Session Creation](#session-creation)). There is no way to request cloud, force local, or
override the route on a per-turn basis.

The success response includes generated text, the turn outcome, route metadata, the provider
finish reason, memory-write status, critic validation status, and warnings in both free-form
and structured form:

```json
{
  "status": "completed",
  "outcome": "success",
  "text": "<actor response>",
  "route": {
    "provider": "local",
    "model": "local-model",
    "reason": "session provider: local"
  },
  "finish_reason": "stop",
  "memory_written": false,
  "critic_status": "accepted",
  "warnings": ["memory curation deferred: runs after this response"],
  "errors": [
    {
      "category": "warning",
      "stage": "general",
      "message": "memory curation deferred: runs after this response",
      "suggestion": null
    }
  ],
  "stage_timings": {
    "session": 0.002,
    "retrieval": 0.041,
    "routing": 0.0,
    "generation": 12.318,
    "validation": 0.001,
    "critique": 8.077,
    "persistence": 0.004
  },
  "retrieval": {
    "query": "<retrieval query>",
    "selected": [
      {
        "id": "public-lore",
        "source": "demo_lore.md",
        "source_type": "lore",
        "collection": "canon_lore",
        "visibility": "player",
        "tags": [],
        "original_score": 0.61,
        "adjusted_score": 0.61,
        "applied_boosts": {},
        "selected_rank": 1
      }
    ],
    "rejected": []
  }
}
```

`outcome` distinguishes a normal reply (`"success"`) from a persisted controlled failure
(`"controlled_failure"`). It appears on the non-streaming turn response, on turn-detail
responses, and on the streaming `failure` payload; the streaming `final` payload carries no
`outcome` — a `final` frame implies success.

`warnings` reports fail-open runtime behavior, such as skipped retrieval or skipped memory
indexing. A turn can return HTTP `200` with warnings when actor generation still completed.
`errors` is the structured, branch-friendly view of the same warnings: each entry is a
`TurnError` with `category`, `stage`, `message` (the original warning text), and an optional
`suggestion` remediation hint, derived from the warning strings by `classify_warnings` in
`app/orchestration/turn_errors.py`. It appears alongside `warnings` on the non-streaming turn
response, turn-detail responses, and the streaming `final` and `failure` payloads.

Both API turn endpoints always defer memory curation to a post-response background job:
`memory_written` is `false` in every live turn response, the `memory curation deferred: runs
after this response` warning is expected on every successful turn, and the live response's
`stage_timings` contains no `memory` key. The persisted turn's diagnostics (`memory_written`,
memory warnings) are updated after the background job completes and are visible via the
turn-detail endpoints.

`critic_status` reports how critic validation concluded for the returned text:

- `accepted`: the critic validated the initial draft.
- `repaired`: the critic rejected the draft and the returned text is a validated repair.
- `rejected`: the critic rejected the draft and repair was exhausted, or the critic itself
  errored — any critic exception fails the turn closed, withholding the unvalidated draft; the
  text is a controlled failure message.
- `skipped`: the critic did not run — either `CRITIC_GATING=auto` judged the turn low-risk (the
  draft is served unvalidated by design) or the turn failed before critique ran.

There is no per-turn routing control and no two-phase confirmation flow. The route for
every turn (`route.provider`, `route.model`) is always the session's bound provider from
creation (see [Session Creation](#session-creation)); critic and memory-extraction tasks
follow the same bound provider. `status` is always `"completed"` on success — there is no
`confirmation_required` status, and no request field (`request_cloud`, `cloud_confirmed`,
`force_local`) exists to ask for or approve a different provider mid-turn.

`stage_timings` is a report-only diagnostic mapping each executed pipeline stage to its
wall-clock duration in seconds. `repair` appears only when a repair attempt ran. The same
object appears in the streaming `final` and `failure` payloads.

`retrieval` is a report-only ranking diagnostic. It lists the selected chunks and the rejected
candidates with their score components, contains metadata only, and never includes chunk text.
It is `null` when retrieval is not configured or the retriever does not expose diagnostics. The
same object appears in the streaming `final` and `failure` payloads.

## Buffered Turn Streaming

`POST /sessions/{session_id}/turns/stream` accepts the same request body as
`POST /sessions/{session_id}/turns` and returns `Content-Type: text/event-stream`.

Player-visible text is intentionally buffered. The route awaits the existing validated
`TurnOrchestrator.run_turn()` pipeline before emitting player-visible text, which arrives
only after retrieval, routing, generation, critic validation, repair, and persistence
complete (memory curation is deferred past the response; see
[Turn Execution](#turn-execution)). This endpoint does not add provider token streaming,
pre-validation token emission, or a second orchestration path.

While the pipeline runs, the stream emits live `stage` progress frames — metadata-only signals
naming the pipeline stage that just started, carrying no player-visible text:

```text
event: stage
data: {"stage":"retrieval"}

```

Stage names follow the pipeline order: `session`, `retrieval`, `routing`, `generation`,
`validation`, `critique`, `repair` (only when a repair pass runs), and `persistence`. No
`memory` stage frame is emitted on the API path because memory curation is deferred past the
response.

After the pipeline completes, a successful stream emits the validated text (by default as
exactly one `text` event) followed by one `final` event:

```text
event: text
data: {"text":"<validated final text>"}

event: final
data: {"route":{"provider":"local","model":"local-model","reason":"..."}, "finish_reason":"stop", "memory_written":false, "critic_status":"accepted", "warnings":["memory curation deferred: runs after this response"], "errors":[{"category":"warning","stage":"general","message":"memory curation deferred: runs after this response","suggestion":null}], "stage_timings":{"generation":12.318,"critique":8.077}}

```

Validated fragmentation is implemented behind `SSE_TEXT_CHUNK_CHARS` (default `0` = a single
`text` frame): when set above `0`, the validated text is emitted as multiple ordered `text`
fragments, each a slice of the already-validated text, so the critic-before-emission boundary
is preserved. Reconstructed equivalence means concatenated `text` event content plus `final`
metadata matches the existing non-streaming `CreateTurnResponse` (minus its `status` and
`outcome` fields — a `final` frame implies success).

When the orchestrator returns a safe controlled failure after critic and repair attempts, the
stream emits no `text` frames and never emits rejected draft text — the stage frames are
followed by one terminal `failure` event:

```text
event: failure
data: {"text":"<safe controlled failure>", "outcome":"controlled_failure", "route":{"provider":"local","model":"local-model","reason":"..."}, "finish_reason":"length", "memory_written":false, "critic_status":"rejected", "warnings":[], "errors":[], "stage_timings":{"generation":12.318,"critique":8.077,"repair":11.402}}

```

Runtime failures discovered while streaming (unknown session, invalid turn request, provider
unavailable, provider timeout) arrive as a terminal `error` event instead of a JSON error
response — the HTTP status is already committed to `200` once streaming starts:

```text
event: error
data: {"code":"session_not_found","message":"Unknown session id: <id>","status":404}

```

`status` carries the HTTP status code the same failure would produce on the non-streaming
endpoint. Only `422` request-body validation fails before the stream starts and returns the
standard JSON envelope (see [Errors](#errors)).

## Turn Deletion (Reroll)

`DELETE /sessions/{session_id}/turns/last` deletes the most recent stored turn — the undo
primitive behind the SPA's reroll button. Before deleting, it drains any pending deferred
memory-curation jobs so a still-running job cannot resurrect the deleted turn's memories; it
then deletes the turn row, deletes the memories recorded after that turn was persisted
(memory provenance is by timestamp), and unindexes them from the retrieval index (SQLite stays
authoritative; index cleanup is best-effort).

The response reports what was removed:

```json
{
  "session_id": "<id>",
  "deleted_turn_index": 4,
  "user_message": "<the deleted turn's player message>",
  "deleted_memory_count": 2
}
```

`user_message` is returned so clients can restore the player's message into the composer for a
reroll. A session with no turns returns `404 no_turns`; unknown session ids return `404
session_not_found`.

## Errors

Handled `400`, `404`, `422`, `503`, and `504` responses use one envelope:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "details": []
  }
}
```

Stable error codes:

- `invalid_content_catalog` (`400`)
- `cloud_unavailable` (`400`)
- `invalid_session_request` (`400`)
- `invalid_scene` (`400`)
- `invalid_turn_request` (`400`)
- `session_not_found` (`404`)
- `turn_not_found` (`404`)
- `no_turns` (`404`)
- `canon_fact_not_found` (`404`)
- `eval_run_not_found` (`404`)
- `validation_error` (`422`)
- `provider_unavailable` (`503`)
- `canon_unavailable` (`503`)
- `provider_timeout` (`504`)

A turn whose provider request exceeds the configured provider timeout returns
`504 provider_timeout`; a turn whose model server is unreachable (down, refused, wrong URL)
returns `503 provider_unavailable`. Both use the same envelope and name the provider and model
but never include prompt text.

On the streaming endpoint, only `422` request-body validation fails before the stream starts
and returns the JSON envelope. Every other failure (`400`, `404`, `503`, `504`) is discovered
inside the already-started stream and arrives as a terminal `event: error` frame carrying
`{"code", "message", "status"}` over HTTP `200` — see
[Buffered Turn Streaming](#buffered-turn-streaming). Validation details
contain only `loc`, `type`, and the fixed sanitized message `Request field validation failed`;
reflected input and validator context are excluded.

## Exposure Boundaries

Responses do not expose raw prompt contents, hidden retrieved chunk text, or the GM-only and
character-private fields of the authored content definitions: `gm_private_summary`,
`private_description`, `secrets`, `forbidden_knowledge`, `content_root`, raw file paths,
hidden lore, raw prompts, retrieved chunks, provider secrets, SQLite internals,
Qdrant internals, provider internals, and Qdrant diagnostics stay out of every response.

Persisted memory episodes are governed separately: `GET /sessions/{session_id}/memories`
deliberately returns episodes of all visibility levels as a single-user authoring surface (see
[Session Memories](#session-memories)); actor-facing leakage is prevented by retrieval and
prompt visibility filtering.

## Known Limitations

The local web UI does not provide authentication, multi-user isolation, browser-local
authoritative state, or frontend scenario-pack selection. The API does not provide provider token
streaming, pre-validation token emission, per-request content-root selection, per-request
scenario-pack selection, hidden-context diagnostics, or retrieval payloads containing chunk text;
the `retrieval` diagnostic field is metadata-only.
