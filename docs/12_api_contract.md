# 12 - API Contract

## Scope

The FastAPI surface is intentionally small. It exposes session creation, session lookup, and turn
execution while delegating engine behavior to shared composition and orchestration services.

Available endpoints:

- `GET /play` for the local HTML play surface, excluded from OpenAPI
- `GET /runtime/status`
- `GET /content/catalog`
- `GET /sessions`
- `POST /sessions`
- `GET /sessions/{session_id}`
- `POST /sessions/{session_id}/turns`
- `POST /sessions/{session_id}/turns/stream`

API routes and the local browser UI do not own retrieval, persistence, routing, prompt
construction, or visibility logic.
SQLite remains authoritative state. Qdrant remains a derived retrieval index.

## Runtime Status

`GET /runtime/status` returns safe, shallow, non-diagnostic runtime metadata for the local `/play`
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
`CONTENT_ROOT` only. It is a read-only catalog for the local `/play` selectors and does not accept
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
  "active_persona_id": "archivist"
}
```

The response contains safe session identifiers:

```json
{
  "session_id": "<id>",
  "world_id": "demo_world",
  "active_scene_id": "rose-gallery",
  "active_persona_id": "archivist"
}
```

The API uses the process-level `CONTENT_ROOT` settings value when creating sessions. API requests
do not accept a per-request content root or per-request scenario-pack selection. CLI scenario-pack
support through `--content-root` remains available.

## Recent Sessions

`GET /sessions` is a local-use recent-session list for the `/play` resume selector. It returns
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

The local `/play` UI uses this endpoint when a user resumes from the recent-session selector or
pastes a `session_id` into `Resume session`; the returned `recent_turns` are rendered as transcript
entries and remain backend-owned.

## Turn Execution

`POST /sessions/{session_id}/turns` accepts:

```json
{
  "message": "What have you heard about the regent?",
  "active_persona_id": "archivist",
  "request_cloud": false
}
```

`active_persona_id` is optional. When provided, it must match the stored session persona.

The success response includes generated text, route metadata, the provider finish reason,
memory-write status, and warnings:

```json
{
  "text": "<actor response>",
  "route": {
    "provider": "local",
    "model": "local-model",
    "reason": "default local route"
  },
  "finish_reason": "stop",
  "memory_written": false,
  "warnings": [],
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

`warnings` reports fail-open runtime behavior, such as skipped retrieval or skipped cloud routing.
A turn can return HTTP `200` with warnings when actor generation still completed.

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
data: {"route":{"provider":"local","model":"local-model","reason":"..."}, "finish_reason":"stop", "memory_written":false, "warnings":[]}

```

The contract permits repeated `text` events for future validated fragmentation. Reconstructed
equivalence means concatenated `text` event content plus `final` metadata matches the existing
non-streaming `CreateTurnResponse`.

When the orchestrator returns a safe controlled failure after critic and repair attempts, the
stream emits only one `failure` event and never emits rejected draft text:

```text
event: failure
data: {"text":"<safe controlled failure>", "route":{"provider":"local","model":"local-model","reason":"..."}, "finish_reason":"length", "memory_written":false, "warnings":[]}

```

## Errors

Handled `400`, `404`, and request-validation `422` responses use one envelope:

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

- `invalid_content_catalog`
- `invalid_session_request`
- `invalid_turn_request`
- `provider_timeout`
- `session_not_found`
- `validation_error`

A turn whose provider request exceeds the configured provider timeout returns
`504 provider_timeout` with the same envelope. The message names the provider and model but never
includes prompt text.

Failures known before streaming starts retain the same JSON envelopes: `400 invalid_turn_request`,
`404 session_not_found`, `422 validation_error`, and `504 provider_timeout`. Validation details
contain only `loc`, `type`, and the fixed sanitized message `Request field validation failed`;
reflected input and validator context are excluded.

## Exposure Boundaries

Responses do not expose raw prompt contents, hidden retrieved chunk text, GM-only fields,
character-private fields, `gm_private_summary`, `private_description`, `secrets`,
`forbidden_knowledge`, `content_root`, raw file paths, hidden lore, raw prompts, retrieved chunks,
provider secrets, SQLite internals, Qdrant internals, provider internals, or Qdrant diagnostics.

## Known Limitations

The local play UI does not provide authentication, multi-user isolation, browser-local
authoritative state, or frontend scenario-pack selection. The API does not provide provider token
streaming, pre-validation token emission, per-request content-root selection, per-request
scenario-pack selection, hidden-context diagnostics, or retrieval payloads containing chunk text;
the `retrieval` diagnostic field is metadata-only.
