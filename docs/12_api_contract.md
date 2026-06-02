# 12 - API Contract

## Scope

The FastAPI surface is intentionally small. It exposes session creation, session lookup, and turn
execution while delegating engine behavior to shared composition and orchestration services.

Available endpoints:

- `POST /sessions`
- `GET /sessions/{session_id}`
- `POST /sessions/{session_id}/turns`

API routes do not own retrieval, persistence, routing, prompt construction, or visibility logic.
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

## Exposure Boundaries

Responses do not expose raw prompt contents, hidden retrieved chunk text, GM-only fields,
character-private fields, `content_root`, provider secrets, SQLite internals, or Qdrant
diagnostics.

## Known Limitations

The API does not provide a frontend, streaming, authentication, multi-user isolation, per-request
scenario selection, hidden-context diagnostics, or raw retrieval/debug payloads.
