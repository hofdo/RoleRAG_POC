# 12 - API Contract

## Scope

The FastAPI surface is intentionally small. It exposes session creation, session lookup, and turn
execution while delegating engine behavior to shared composition and orchestration services.

Available endpoints:

- `GET /runtime/status`
- `GET /content/catalog`
- `GET /sessions`
- `POST /sessions`
- `GET /sessions/{session_id}`
- `GET /sessions/{session_id}/turns/{turn_index}`
- `GET /sessions/{session_id}/turn-details`
- `GET /sessions/{session_id}/memories`
- `GET /sessions/{session_id}/canon`
- `POST /sessions/{session_id}/canon`
- `DELETE /sessions/{session_id}/canon/{fact_id}`
- `POST /sessions/{session_id}/turns`
- `POST /sessions/{session_id}/turns/stream`
- `GET /diagnostics/eval-runs`
- `GET /diagnostics/eval-runs/{run_id}`

The Angular SPA is served as static files at `/app` (the root `/` redirects there); it is not an
API endpoint and is excluded from OpenAPI.

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
  "provider": "local"
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

The response contains safe session identifiers and the bound provider:

```json
{
  "session_id": "<id>",
  "world_id": "demo_world",
  "active_scene_id": "rose-gallery",
  "active_persona_id": "archivist",
  "provider": "local"
}
```

The API uses the process-level `CONTENT_ROOT` settings value when creating sessions. API requests
do not accept a per-request content root or per-request scenario-pack selection. CLI scenario-pack
support through `--content-root` remains available.

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
combining the stored turn fields (`turn_index`, `scene_id`, `persona_id`, `user_message`,
`assistant_message`, `route`, `created_at`) with the diagnostics captured at turn time
(`finish_reason`, `memory_written`, `critic_status`, `warnings`, `stage_timings`, and an optional
`retrieval` ranking). The `retrieval` block reuses the same metadata-only candidate shape as the
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
per-run `conversation-checkpoint.json` artifacts. It returns `results_dir` plus a list of run
summaries: `id`, `status`, `turn_count`, `recall_misses`, `extraction_misses`,
`retrieval_misses`, `total_seconds`, `p50_seconds`, `p95_seconds`, and `warning_total`. It does
not call LLMs or touch SQLite/Qdrant.

`GET /diagnostics/eval-runs/{run_id}` returns the full `conversation-checkpoint.json` payload for
one run (the SPA Eval page's drill-down). Unknown run ids return `404 eval_run_not_found`. Run
artifacts are local diagnostic files produced by the live checkpoint harness; the endpoint serves
them verbatim and is local-use only, like the rest of the surface.

## Session Memories

`GET /sessions/{session_id}/memories` returns the persisted durable-memory episodes for a
session as metadata: `id`, `scene_id`, `actor_id`, `summary`, `importance`, `visibility`, and
`tags`. The web UI renders this in a read-only "Memory" panel and the CLI exposes the
same data via `inspect-memories`. Unknown session ids return the standard `404` envelope.

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

`active_persona_id` is optional. When provided, it must match the stored session persona.
The request body carries no provider or routing flags: the turn always runs on the
session's provider, bound once at `POST /sessions` time and immutable thereafter (see
[Session Creation](#session-creation)). There is no way to request cloud, force local, or
override the route on a per-turn basis.

The success response includes generated text, route metadata, the provider finish reason,
memory-write status, critic validation status, and warnings:

```json
{
  "status": "completed",
  "text": "<actor response>",
  "route": {
    "provider": "local",
    "model": "local-model",
    "reason": "session provider: local"
  },
  "finish_reason": "stop",
  "memory_written": false,
  "critic_status": "accepted",
  "warnings": [],
  "stage_timings": {
    "session": 0.002,
    "retrieval": 0.041,
    "routing": 0.0,
    "generation": 12.318,
    "validation": 0.001,
    "critique": 8.077,
    "persistence": 0.004,
    "memory": 9.012
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

`warnings` reports fail-open runtime behavior, such as skipped retrieval, skipped memory
indexing, or a deferred memory-curation job. A turn can return HTTP `200` with warnings
when actor generation still completed.

`critic_status` reports how critic validation concluded for the returned text:

- `accepted`: the critic validated the initial draft.
- `repaired`: the critic rejected the draft and the returned text is a validated repair.
- `rejected`: the critic rejected the draft and repair was exhausted; the text is a controlled
  failure message.
- `skipped`: the returned text was never successfully validated, either because the critic
  errored (see `warnings`) or because the turn failed before critique ran.

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

The implementation is intentionally buffered. The route awaits the existing validated
`TurnOrchestrator.run_turn()` pipeline before emitting player-visible text. First output arrives
only after retrieval, routing, generation, critic validation, repair, persistence, and memory
handling complete. This endpoint does not add provider token streaming, pre-validation token
emission, or a second orchestration path.

A successful stream currently emits exactly one `text` event followed by one `final` event:

```text
event: text
data: {"text":"<validated final text>"}

event: final
data: {"route":{"provider":"local","model":"local-model","reason":"..."}, "finish_reason":"stop", "memory_written":false, "critic_status":"accepted", "warnings":[], "stage_timings":{"generation":12.318,"critique":8.077}}

```

The contract permits repeated `text` events for future validated fragmentation. Reconstructed
equivalence means concatenated `text` event content plus `final` metadata matches the existing
non-streaming `CreateTurnResponse`.

When the orchestrator returns a safe controlled failure after critic and repair attempts, the
stream emits only one `failure` event and never emits rejected draft text:

```text
event: failure
data: {"text":"<safe controlled failure>", "route":{"provider":"local","model":"local-model","reason":"..."}, "finish_reason":"length", "memory_written":false, "critic_status":"rejected", "warnings":[], "stage_timings":{"generation":12.318,"critique":8.077,"repair":11.402}}

```

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
- `invalid_turn_request` (`400`)
- `session_not_found` (`404`)
- `canon_fact_not_found` (`404`)
- `validation_error` (`422`)
- `provider_unavailable` (`503`)
- `canon_unavailable` (`503`)
- `provider_timeout` (`504`)

A turn whose provider request exceeds the configured provider timeout returns
`504 provider_timeout`; a turn whose model server is unreachable (down, refused, wrong URL)
returns `503 provider_unavailable`. Both use the same envelope and name the provider and model
but never include prompt text.

Failures known before streaming starts retain the same JSON envelopes: `400 invalid_turn_request`,
`404 session_not_found`, `422 validation_error`, `503 provider_unavailable`, and
`504 provider_timeout`. Validation details
contain only `loc`, `type`, and the fixed sanitized message `Request field validation failed`;
reflected input and validator context are excluded.

## Exposure Boundaries

Responses do not expose raw prompt contents, hidden retrieved chunk text, GM-only fields,
character-private fields, `gm_private_summary`, `private_description`, `secrets`,
`forbidden_knowledge`, `content_root`, raw file paths, hidden lore, raw prompts, retrieved chunks,
provider secrets, SQLite internals, Qdrant internals, provider internals, or Qdrant diagnostics.

## Known Limitations

The local web UI does not provide authentication, multi-user isolation, browser-local
authoritative state, or frontend scenario-pack selection. The API does not provide provider token
streaming, pre-validation token emission, per-request content-root selection, per-request
scenario-pack selection, hidden-context diagnostics, or retrieval payloads containing chunk text;
the `retrieval` diagnostic field is metadata-only.
