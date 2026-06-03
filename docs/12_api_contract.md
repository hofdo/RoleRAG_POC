# 12 - API Contract

## Scope

The FastAPI surface is intentionally small. It exposes session creation, session lookup, and turn
execution while delegating engine behavior to shared composition and orchestration services.

Available endpoints:

- `GET /play` for the local HTML play surface, excluded from OpenAPI
- `POST /sessions`
- `GET /sessions/{session_id}`
- `POST /sessions/{session_id}/turns`
- `POST /sessions/{session_id}/turns/stream`

API routes and the local browser UI do not own retrieval, persistence, routing, prompt
construction, or visibility logic.
SQLite remains authoritative state. Qdrant remains a derived retrieval index.

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

## Session Lookup

`GET /sessions/{session_id}` returns safe session identifiers and a bounded recent-turn list.
It does not return player names, content roots, hidden scene state, private persona state, SQLite
internals, or Qdrant diagnostics.

The local `/play` UI uses this endpoint only when a user pastes a `session_id` into `Resume
session`; the returned `recent_turns` are rendered as transcript entries and remain backend-owned.

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

The success response includes generated text, route metadata, memory-write status, and warnings:

```json
{
  "text": "<actor response>",
  "route": {
    "provider": "local",
    "model": "local-model",
    "reason": "default local route"
  },
  "memory_written": false,
  "warnings": []
}
```

`warnings` reports fail-open runtime behavior, such as skipped retrieval or skipped cloud routing.
A turn can return HTTP `200` with warnings when actor generation still completed.

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
data: {"route":{"provider":"local","model":"local-model","reason":"..."}, "memory_written":false, "warnings":[]}

```

The contract permits repeated `text` events for future validated fragmentation. Reconstructed
equivalence means concatenated `text` event content plus `final` metadata matches the existing
non-streaming `CreateTurnResponse`.

When the orchestrator returns a safe controlled failure after critic and repair attempts, the
stream emits only one `failure` event and never emits rejected draft text:

```text
event: failure
data: {"text":"<safe controlled failure>", "route":{"provider":"local","model":"local-model","reason":"..."}, "memory_written":false, "warnings":[]}

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

- `invalid_session_request`
- `invalid_turn_request`
- `session_not_found`
- `validation_error`

Failures known before streaming starts retain the same JSON envelopes: `400 invalid_turn_request`,
`404 session_not_found`, and `422 validation_error`. Validation details contain only `loc`, `type`,
and the fixed sanitized message `Request field validation failed`; reflected input and validator
context are excluded.

## Exposure Boundaries

Responses do not expose raw prompt contents, hidden retrieved chunk text, GM-only fields,
character-private fields, `content_root`, provider secrets, SQLite internals, or Qdrant
diagnostics.

## Known Limitations

The local play UI does not provide authentication, multi-user isolation, browser-local
authoritative state, or frontend scenario-pack selection. The API does not provide provider token
streaming, pre-validation token emission, per-request scenario selection, hidden-context
diagnostics, or raw retrieval/debug payloads.
