# RoleRAG Web UI (Angular 19 SPA)

The browser surface for RoleRAG, served by FastAPI at `/app` (the API root `/` redirects
there). It is a **thin client** over the same-origin API: the browser owns no orchestration,
retrieval, validation, routing, persistence, memory, or hidden context — see the exposure
boundaries in [docs/12_api_contract.md](../docs/12_api_contract.md).

## Pages

| Route | What it does |
|-------|--------------|
| `/app/play` | Session setup from the public content catalog with a Local/Cloud provider choice (`CLOUD_MODE=ask` confirms once at creation), resume picker for existing sessions, turn loop over buffered SSE with live stage progress, last-turn reroll, mid-session scene switch + per-turn persona override, memory + canon side panels |
| `/app/inspector` | Per-session turn timeline with retrieval drill-down (query, selected/rejected candidates, scores, boosts) via the bulk `GET /sessions/{id}/turn-details` endpoint |
| `/app/analytics` | Turn latency and per-stage timing statistics for a session |
| `/app/eval` | Eval-run trends from `GET /diagnostics/eval-runs`, with per-run drill-down |

All pages are lazy-loaded standalone components ([src/app/app.routes.ts](src/app/app.routes.ts)).

## Architecture

- [src/app/session-store.ts](src/app/session-store.ts) — root-provided signal store; the logic
  hub for session/turn/memory/canon state. Components inject it and read signals; there is no
  NgRx. Side-panel failures surface via `memoryError`/`canonError` instead of failing silently.
- [src/app/api.service.ts](src/app/api.service.ts) — fetch-based client, including the buffered
  SSE turn parser (POST + `ReadableStream`; `HttpClient` doesn't fit SSE-over-POST). Malformed
  frames surface as typed `ApiError`s.
- [src/app/play-model.ts](src/app/play-model.ts), [src/app/analytics-model.ts](src/app/analytics-model.ts)
  — pure view-model helpers, unit-tested without the DOM.
- Design system: "Grimoire Console" — warm ink-on-vellum with a single live-wire accent, defined
  as CSS variables in [src/styles.scss](src/styles.scss).

## Develop

```bash
npm ci
npm start          # ng serve on :4200, proxying API calls to :8000 (proxy.conf.json)
npm test           # Karma unit tests (CI runs: npm test -- --watch=false --browsers=ChromeHeadless)
```

`npm start` expects the backend stack up (`make dev` from the repo root). The production build
is a plain `npx ng build` (the `/app/` baseHref is pinned in [angular.json](angular.json));
`make dev` and the Dockerfile run it — output lands in `dist/frontend/browser`, which
`app/main.py` mounts at `/app` with a deep-link fallback so client-side routes survive a hard
refresh. CI runs the Karma tests but does not build the bundle.

The end-to-end test lives at the repo root ([tests/e2e/spa-play.spec.mjs](../tests/e2e/spa-play.spec.mjs)):
`PLAYWRIGHT_BASE_URL=http://127.0.0.1:8000 npm run test:e2e-spa` (needs the full stack + model).
