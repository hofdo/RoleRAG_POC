# RoleRAG_POC

> Reviewed: 2026-07-10 @ 9097877

Personal-use RoleRAG engine built around a CLI-first roleplay loop, an Angular SPA
(play, RAG inspector, analytics, eval dashboards), a small FastAPI API, SQLite persistence,
Qdrant-backed retrieval, and deterministic local/cloud routing.

## Contents

- [Current State](#current-state)
- [Quickstart](#quickstart) — [Prerequisites](#prerequisites) · [Fresh Clone Setup](#fresh-clone-setup) · [Daily Use](#daily-use-one-command-host-python) · [Run in Docker](#run-in-docker-no-local-python-needed)
- [Local Model Setup](#local-model-setup)
- [Environment and Routing](#environment-and-routing)
- [Runtime Components](#runtime-components)
- [Safety Boundaries](#safety-boundaries)
- [CLI Usage](#cli-usage)
- [Runtime Verification](#runtime-verification)
- [Scenario Authoring](#scenario-authoring)
- [API Usage](#api-usage) · [Web UI](#web-ui-angular-spa)
- [Qdrant and Retrieval](#qdrant-and-retrieval)
- [Tests, Lint, and Evals](#tests-lint-and-evals)
- [Current Limitations](#current-limitations)
- [Safe Next Steps](#safe-next-steps)
- [Additional Documentation](#additional-documentation)

## Current State

A working personal-use engine. Highlights (see [docs/09](docs/09_current_architecture_map.md)
and [CHANGELOG.md](CHANGELOG.md) for the full inventory):

- **Session-bound provider** — `local` or `cloud` is chosen once at session creation and
  every task (actor, repair, critic, memory) for that session runs on it for its whole
  lifetime; `CLOUD_MODE=off|ask|auto` gates cloud **creation** only (see
  [Environment and Routing](#environment-and-routing)). No per-turn escalation, fallback,
  or mid-turn switching.
- **Surfaces** — a 24-command Typer CLI, a small FastAPI API, and an Angular 21 SPA served
  at `/app` (play, RAG inspector, analytics, eval trends; the root `/` redirects to it).
- **Turn pipeline** — retrieval-aware actor prompt with visibility filtering, deterministic
  reranking across `session_memory`/`persona_memory`/`canon_lore`, a bounded critic + repair
  flow with a first-class `critic_status`, deterministic draft validation, and per-stage
  `stage_timings` on every turn.
- **Play features** — session resume, turn reroll (delete-last-turn), mid-session scene
  switching, per-turn persona override, and author-pinned plus auto-derived canon facts.
- **Persistence & retrieval** — SQLite for sessions/turns/memory/canon (with WAL and a
  `backup` command), curated durable memories indexed into Qdrant, fail-open retrieval.
- **Privacy** — hidden authored content (persona `secrets`/`forbidden_knowledge`, scene
  `gm_private_summary`) never leaves the machine on any provider; memory extraction runs on
  the session's bound provider like every other task.
- **Verification** — `doctor`, `smoke-run`, `validate-content`, a deterministic eval harness,
  and a live-stack checkpoint (see [Runtime Verification](#runtime-verification)).

Not implemented: authentication, multi-user support, provider token streaming, and
production deployment hardening.

## Quickstart

Run `make` (or `make help`) to list the common tasks — `make up` (Docker),
`make dev` (host stack), `make install`, `make check`, `make smoke`. Each is just a
thin wrapper over the commands documented below. New to the repo? Read
[Prerequisites](#prerequisites), then [Fresh Clone Setup](#fresh-clone-setup), then
[Local Model Setup](#local-model-setup) before your first turn.

### Prerequisites

- Python `3.12+`
- Node `20+` (builds the Angular SPA; the API runs without it, but serves no UI)
- Docker
- a local OpenAI-compatible model server such as `llama.cpp` or Ollama's compatibility layer
  (see [Local Model Setup](#local-model-setup))

### Fresh Clone Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
(cd frontend && npm ci && npx ng build)  # SPA (baseHref /app/ is pinned in angular.json); `make dev` does this for you
docker compose up -d qdrant
python -m app.cli config
python -m app.cli health
python -m app.cli doctor
python -m app.cli smoke-run
```

`python -m app.cli health` is the dependency-free local configuration check for a fresh clone. It
does not probe SQLite, Qdrant, or model providers.

`python -m app.cli doctor` performs operational checks for settings, temporary SQLite
initialization, demo data loading, and optional Qdrant or local-provider reachability.

`python -m app.cli smoke-run` executes a deterministic end-to-end MVP verification using a
temporary SQLite database, in-memory retrieval, and fake provider responses.

`python -m app.cli validate-content` validates authored worlds, scenes, personas, and optional
lore metadata without calling an LLM or touching runtime state.

`python -m app.cli create-scenario-template --output <path>` generates a minimal standalone
scenario pack root with valid starter files and an optional lore manifest.

### Daily Use (one command, host Python)

With Docker running and a `.venv` installed:

```bash
bash scripts/dev-up.sh
# opens Qdrant + llama-server + API, builds the SPA, then play at http://127.0.0.1:8000/app/
bash scripts/dev-down.sh   # stop everything dev-up started
```

`dev-up.sh` needs `llama-server` on `PATH` (or a matching provider already running); on the
first run it downloads the profile model (see [Local Model Setup](#local-model-setup)).

### Run in Docker (no local Python needed)

The app (FastAPI + `/app` SPA) runs in Docker alongside Qdrant. The model server
stays on the **host**: Docker has no GPU passthrough on macOS, so a local LLM inside
a container would run CPU-only and far too slowly for the recommended 26B model.

1. Start a model server on the host, port 8080 (see [Local Model Setup](#local-model-setup)).
2. Bring up the app + Qdrant:

   ```bash
   docker compose up --build
   ```

   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) (redirects into the SPA at `/app/`,
   which the image builds in its frontend stage). The container reaches
   the host model via `host.docker.internal`, talks to Qdrant over the compose network,
   and bind-mounts `./data` so the SQLite db persists and scenario content is editable on
   the host. A turn sent before the model server is up returns a clear "provider
   unreachable" message instead of failing hard.

Override settings with env vars (a different host port, model, or cloud mode):

```bash
DEV_API_PORT=9000 LOCAL_LLM_MODEL=my-model CLOUD_MODE=off docker compose up --build
```

Run CLI commands inside the container:

```bash
docker compose exec app rolerag start-session --world-id demo_world \
  --scene-id rose-gallery --active-persona-id archivist --player-name Avery
```

### Start a Session

```bash
python -m app.cli start-session \
  --world-id demo_world \
  --scene-id rose-gallery \
  --active-persona-id archivist \
  --player-name Avery
```

That command returns a JSON object containing the `session_id`.

### Run a Turn

```bash
python -m app.cli turn \
  --session-id <session-id> \
  --message "What have you heard about the regent?"
```

### Resume a Session

```bash
python -m app.cli resume --session-id <session-id>
```

### Ingest Demo Lore

```bash
python -m app.cli ingest data/documents/demo_lore.md \
  --visibility player \
  --source-type lore \
  --world-id demo_world \
  --tag palace
```

This manual step is optional when you start sessions through the CLI: `start-session`
auto-ingests manifest-declared lore (idempotent, fail-open; opt out with
`--skip-lore-ingest`). Sessions created through the API or SPA do not auto-ingest, so
run it once if you only play through the web UI.

### Run the API

```bash
.venv/bin/uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000/app/](http://127.0.0.1:8000/app/) for the web UI (the root `/`
redirects there; if the SPA is not built, it returns a hint with the build command instead).
The UI starts new sessions from backend-owned catalog selectors and runs turns over buffered
SSE. Scenario packs are selected only by starting FastAPI with the desired process-level
`CONTENT_ROOT`; the browser has no scenario-pack selector or content-root input.

## Local Model Setup

The default local configuration expects an OpenAI-compatible endpoint:

```env
LOCAL_LLM_BASE_URL=http://127.0.0.1:8080/v1
LOCAL_LLM_API_KEY=local
LOCAL_LLM_MODEL=chatgpt-onnechan
```

**Install a server.** The reference runtime is `llama.cpp` (`brew install llama.cpp` on
macOS, or a prebuilt binary from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)).
`llama-server` must be on your `PATH` for `dev-up.sh` and the live checkpoint to launch it.

**Get a model.** `llama-server -hf <repo>:<quant>` downloads the GGUF from Hugging Face on
first run and caches it, so you do not need a local file. A small starter model to smoke the
stack is `DavidAU/gemma-4-E4B-it-…-Thinking-GGUF:Q8_0` (~4 GB); the recommended acceptance
model is the 26B-A4B `LOCAL_MODEL_PROFILE=26b`
(`HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M`, ~15 GB on disk and roughly
16 GB of RAM/VRAM to run comfortably). The exact per-profile flags — and the faster
`26b-mtp` variant — live in
[scripts/lib/local-model-profile.sh](scripts/lib/local-model-profile.sh) and are described in
[docs/19_verification_and_eval_tooling.md](docs/19_verification_and_eval_tooling.md).

Example `llama.cpp` server flow (serve the starter model, aliased so the app finds it):

```bash
llama-server -hf DavidAU/gemma-4-E4B-it-The-DECKARD-Expresso-Universe-HERETIC-UNCENSORED-Thinking-GGUF:Q8_0 \
  --host 127.0.0.1 \
  --port 8080 \
  --alias chatgpt-onnechan \
  --api-key local
```

Example Ollama-compatible setup:

```env
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_API_KEY=ollama
LOCAL_LLM_MODEL=qwen3:8b
```

The app does not contain provider-specific gameplay logic. It relies on the configured endpoint speaking the OpenAI-compatible chat-completions shape used by [app/llm/openai_compatible.py](app/llm/openai_compatible.py).

## Environment and Routing

The settings model lives in [app/config.py](app/config.py), and [`.env.example`](.env.example)
mirrors every field with its default and per-field commentary. Those two are the single source
of truth for configuration — copy `.env.example` to `.env` and edit there. The recommended
local-session values (26B-friendly structured-token budget, `RAG_DEFAULT_TOP_K=10`, one provider
retry, a 300s timeout for slow dense models) are already set in `.env.example`.

Provider is a **session-bound** choice: `local` or `cloud` is picked once, at session
creation (`POST /sessions` or `rolerag start-session --provider ...`), and every task type
for that session — actor generation, repair, critic evaluation, and memory extraction —
runs on that bound provider for the session's entire lifetime. There is no per-turn
escalation, no automatic cloud fallback, and no mid-turn provider switching; the router
(`app/llm/router.py`) only ever routes to the session's own provider.

`CLOUD_MODE` gates **cloud session creation**, not per-turn behavior:

- `CLOUD_MODE=off`: creating a `cloud` session is rejected (`400 cloud_unavailable`).
  Existing sessions are unaffected — a session's provider never changes after creation.
- `CLOUD_MODE=ask`: creating a `cloud` session requires interactive confirmation at
  creation time — a `typer.confirm` prompt in the CLI, a `window.confirm` prompt in the
  SPA. Once confirmed, the session runs entirely on cloud with no further prompts.
  The confirmation is enforced by the CLI and SPA clients; a raw `POST /sessions` call
  under `CLOUD_MODE=ask` behaves like `auto`.
- `CLOUD_MODE=auto`: cloud sessions are created without an interactive prompt.

Privacy invariant: hidden authored content — persona `secrets`/`forbidden_knowledge`
fields and scene `gm_private_summary` — never leaves the machine on **any** provider,
local or cloud. This is enforced by an `include_hidden` gate (`app/orchestration/stages/
critique.py`: `include_hidden=route.provider == ModelProviderName.LOCAL`) that only ever
allows hidden fields into a prompt when the route is local, plus a provider-binding eval
test. Critic evaluation and memory extraction follow the session's bound provider like
every other task. Critic prompts include hidden fields only when that provider is local
(the `include_hidden` gate); memory-extraction prompts never contain hidden fields on any
provider — they carry only ids, titles, names, and the completed turn's dialogue.

## Runtime Components

- `CLI`: [app/cli.py](app/cli.py) exposes 24 commands — `config`, `health`, `doctor`, `smoke-run`, `validate-content`, `create-scenario-template`, `start-session`, `resume`, `route`, `ingest`, `ingest-scenario-lore`, `reindex-memories`, `retrieve-debug`, `embedding-ab`, `turn`, `turn-history`, `list-sessions`, `backup`, `delete-session`, `export-session`, `import-session`, `inspect-memories`, `reset-db`, and `reset-index`. Run `rolerag --help` for the authoritative list with descriptions.
- `FastAPI API`: [app/main.py](app/main.py) and [app/api/routes.py](app/api/routes.py) expose the same orchestrator through HTTP.
- `TurnOrchestrator`: [app/orchestration/turn_orchestrator.py](app/orchestration/turn_orchestrator.py) owns the bounded turn pipeline.
- `ActorAgent`: [app/agents/actor_agent.py](app/agents/actor_agent.py) performs text generation only.
- `CriticAgent`: [app/agents/critic_agent.py](app/agents/critic_agent.py) evaluates drafts and builds repair prompts.
- `MemoryCurator`: [app/agents/memory_curator.py](app/agents/memory_curator.py) extracts structured durable-memory candidates.
- `RAG`: [app/rag/](app/rag) handles chunking, embeddings, ingestion, retrieval, and vector-store access.
- `SQLite persistence`: [app/persistence/](app/persistence) stores sessions, turns, and memory episodes.
- `Composition`: [app/composition.py](app/composition.py) wires settings, providers, repositories, and retrieval.
- `Diagnostics`: [app/diagnostics/](app/diagnostics) provides runtime environment checks and the deterministic smoke runner.

More detail:

- [docs/08_agent_handoff.md](docs/08_agent_handoff.md)
- [docs/09_current_architecture_map.md](docs/09_current_architecture_map.md)

## Safety Boundaries

- The LLM does not own authoritative state.
- `ActorAgent` does not retrieve, persist, or mutate state directly.
- FastAPI route handlers stay thin and delegate to shared composition and orchestration code.
- Player-facing actor prompts only include `player`-visible retrieved chunks.
- `CriticAgent` may inspect hidden context to detect leakage, but that output is not player-facing.
- Hidden authored content — persona `secrets`/`forbidden_knowledge` and scene
  `gm_private_summary` — never leaves the machine on any provider; memory extraction runs
  on the session's bound provider like every other task.
- Tests and evals use fake/mock providers and in-memory vector stores rather than real external services.

## CLI Usage

Show commands:

```bash
python -m app.cli --help
rolerag --help
```

Inspect resolved settings:

```bash
python -m app.cli config
```

Run the dependency-free health check:

```bash
python -m app.cli health
```

Run runtime diagnostics:

```bash
python -m app.cli doctor
python -m app.cli doctor --check-qdrant
python -m app.cli doctor --check-local-provider
```

Run the deterministic end-to-end smoke path:

```bash
python -m app.cli smoke-run
```

Run optional live dependency checks without real generation:

```bash
python -m app.cli smoke-run --real-runtime
```

Validate authored content:

```bash
python -m app.cli validate-content
python -m app.cli validate-content --world-id demo_world
python -m app.cli validate-content --content-root data/scenarios/iron-archduke
```

Generate a minimal standalone scenario pack:

```bash
python -m app.cli create-scenario-template --output data/scenarios/iron-archduke
python -m app.cli create-scenario-template --name "Iron Archduke" --output data/scenarios/iron-archduke
```

Launch a validated standalone pack directly:

```bash
python -m app.cli start-session \
  --content-root data/scenarios/iron-archduke \
  --world-id iron-archduke \
  --scene-id iron-archduke-opening \
  --active-persona-id iron-archduke-narrator
```

Inspect routing decisions:

```bash
python -m app.cli route --task actor_response
python -m app.cli route --task actor_response --provider cloud
python -m app.cli route --task repair --provider local
```

Session lifecycle:

```bash
python -m app.cli start-session --world-id demo_world --scene-id rose-gallery --active-persona-id archivist --player-name Avery
python -m app.cli resume --session-id <session-id>
python -m app.cli turn --session-id <session-id> --message "Tell me what changed here."
```

RAG ingestion:

```bash
python -m app.cli ingest --help
python -m app.cli ingest data/documents/demo_lore.md --visibility player --source-type lore --world-id demo_world
python -m app.cli ingest-scenario-lore --content-root data/scenarios/iron-archduke
```

Backfill or repair the vector index for existing SQLite memories:

```bash
python -m app.cli reindex-memories --session-id <session-id>
```

Write a timestamped online-consistent copy of the SQLite database (vectors are excluded
on purpose — Qdrant collections rebuild from SQLite via `reindex-memories`/`ingest`):

```bash
python -m app.cli backup
python -m app.cli backup --output-dir /path/to/backups
```

Drop derived vector collections (all, or one of `canon_lore`/`session_memory`/`persona_memory`),
then rebuild with `reindex-memories` or re-ingest lore:

```bash
python -m app.cli reset-index --collection session_memory
```

Rank the seeded durable-memory events with alternative FastEmbed embedding models (LLM-free
offline A/B; models download on first use):

```bash
python -m app.cli embedding-ab --model BAAI/bge-small-en-v1.5
```

Inspect ranked retrieval for a query without calling an LLM:

```bash
python -m app.cli retrieve-debug \
  --session-id <session-id> \
  --query "What did I promise the archivist?"
```

## Runtime Verification

Safe local diagnostics:

```bash
python -m app.cli health
python -m app.cli doctor
python -m app.cli smoke-run
```

Optional live dependency checks:

```bash
docker compose up -d qdrant
python -m app.cli doctor --check-qdrant --check-local-provider
python -m app.cli smoke-run --real-runtime
```

Run the isolated live stack checkpoint against a real local model:

```bash
npm install
npx playwright install chromium
LOCAL_LLM_MODEL=chatgpt-onnechan \
LIVE_TURN_COUNT=8 \
PYTHON=.venv/bin/python \
bash scripts/live-smoke.sh
```

`scripts/live-smoke.sh` stands up a disposable Qdrant + FastAPI stack against a real local
model, runs live doctor/smoke/API checks, an N-turn Rose Gallery conversation checkpoint, and
a Playwright UI smoke, then writes diagnostics under `/tmp/rolerag-live-test` and tears down
only what it started. It reuses an already-running provider or launches a managed
`llama-server` from the selected `LOCAL_MODEL_PROFILE`. The full checkpoint internals — the
managed-server command and shutdown behavior, the `LOCAL_MODEL_PROFILE` matrix
(`small`/`26b`/`26b-mtp`), the `LLAMA_CPP_*` overrides, `LIVE_TURN_COUNT` (`5`–`100`),
`LIVE_FAIL_ON_STRUCTURED_WARNINGS`, and the model bake-off / secret-probe / RAG A/B harnesses —
live in [docs/19_verification_and_eval_tooling.md](docs/19_verification_and_eval_tooling.md).

Operational rules:

- `health` is config-only and never probes SQLite, Qdrant, or providers.
- `doctor` never mutates the configured runtime database. It verifies SQLite using a temporary file.
- `smoke-run` uses a temporary SQLite database, deterministic embeddings, in-memory retrieval, and fake provider responses by default.
- `--real-runtime` adds shallow Qdrant and local-provider reachability checks only. It does not call cloud APIs, real completions, or write to Qdrant.
- `scripts/live-smoke.sh` is the live local-stack checkpoint. It requires Docker, npm, Playwright, and `llama-server` on `PATH` when no matching provider is already running. It reuses an existing provider or starts and safely stops a managed model from the selected profile (see [docs/19](docs/19_verification_and_eval_tooling.md)). Structured-output and retrieval warnings fail by default and can be made report-only with `LIVE_FAIL_ON_STRUCTURED_WARNINGS=0`.
- cloud placeholder keys such as `replace_me` are treated as unusable and reported clearly.

Failure interpretation:

- `status=pass`: the checked path completed successfully.
- `status=warn`: the local MVP can still operate in safe mode, but an optional dependency or cloud intent is incomplete.
- `status=fail`: an expected local runtime requirement is missing or misconfigured.
- `status=skipped`: the check was intentionally not run.

Typical remediations:

```bash
docker compose up -d qdrant
python -m app.cli config
python -m app.cli doctor --check-local-provider
```

If you want to exercise a real local model after the safe smoke passes, use the existing manual flow:

```bash
python -m app.cli ingest data/documents/demo_lore.md --visibility player --source-type lore --world-id demo_world
python -m app.cli start-session --world-id demo_world --scene-id rose-gallery --active-persona-id archivist --player-name Avery
python -m app.cli turn --session-id <session-id> --message "What have you heard about the regent?"
python -m app.cli retrieve-debug --session-id <session-id> --query "What did I promise the archivist?"
```

## Scenario Authoring

Validate the repository demo content or a standalone authored pack:

```bash
python -m app.cli validate-content
python -m app.cli validate-content --world-id demo_world
python -m app.cli validate-content --content-root data/scenarios/iron-archduke
```

Validation output is structured JSON with:

- `status`: `pass`, `warn`, or `fail`
- `errors`: blocking structural or reference problems
- `warnings`: conservative secrecy or metadata concerns
- `checked_files`: files that were inspected

Validation behavior:

- reuses the existing Pydantic world, scene, and persona models
- checks world-to-scene and world-to-persona references
- checks scene `active_personas` references
- warns when persona secrets or forbidden knowledge appear verbatim in `public_description`
- warns when a scene GM summary is duplicated into the player-visible summary
- validates optional lore metadata in `documents/manifest.json`
- warns when lore documents exist without a manifest or are omitted from the manifest
- supports explicit manifest-driven lore ingestion through `ingest-scenario-lore`

Exit semantics:

- `status=pass` or `status=warn`: exit `0`
- `status=fail`: exit `1`

Create a minimal standalone authoring pack:

```bash
python -m app.cli create-scenario-template \
  --name "Iron Archduke" \
  --output data/scenarios/iron-archduke
```

Generated layout:

```text
data/scenarios/iron-archduke/
  README.md
  worlds/iron-archduke.json
  scenes/iron-archduke_opening.json
  personas/iron-archduke-narrator.json
  documents/lore.md
  documents/manifest.json
```

Template rules:

- generation is deterministic and local-only
- existing output directories are not overwritten unless `--overwrite` is supplied
- generated packs are standalone content roots that can be launched with `start-session --content-root`

## API Usage

Start the server:

```bash
.venv/bin/uvicorn app.main:app --reload
```

Implemented endpoints (see [docs/12_api_contract.md](docs/12_api_contract.md) for the full contract):

- `GET /runtime/status`
- `GET /content/catalog`
- `GET /sessions`
- `POST /sessions`
- `POST /sessions/{session_id}/scene`
- `POST /sessions/{session_id}/turns`
- `POST /sessions/{session_id}/turns/stream`
- `DELETE /sessions/{session_id}/turns/last`
- `GET /sessions/{session_id}`
- `GET /sessions/{session_id}/turns/{turn_index}`
- `GET /sessions/{session_id}/turn-details`
- `GET /sessions/{session_id}/memories`
- `GET /sessions/{session_id}/canon`
- `POST /sessions/{session_id}/canon`
- `DELETE /sessions/{session_id}/canon/{fact_id}`
- `GET /diagnostics/eval-runs`
- `GET /diagnostics/eval-runs/{run_id}`

The `canon` endpoints are an author surface for pinning durable "Standing facts"
into the actor prompt by hand, alongside the auto-derived canon. Pinned facts
lead the Standing-facts block.

`POST /sessions/{session_id}/scene` switches the active scene mid-session (the
`scene_id` is validated against the world's scene list). `DELETE
/sessions/{session_id}/turns/last` deletes the most recent turn together with its
derived memories — the reroll/undo primitive.

Create a session (`provider` defaults to `local`; pass `"provider":"cloud"` to bind the
session to cloud for its whole lifetime, subject to the `CLOUD_MODE` creation gate above):

```bash
curl -X POST http://127.0.0.1:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"world_id":"demo_world","scene_id":"rose-gallery","player_name":"Player","active_persona_id":"archivist"}'
```

Run a turn (the turn body carries no provider/routing flags — the session's bound
provider from creation applies automatically):

```bash
curl -X POST http://127.0.0.1:8000/sessions/<session-id>/turns \
  -H "Content-Type: application/json" \
  -d '{"message":"I ask what the locked door hides.","active_persona_id":"archivist"}'
```

Run a buffered SSE turn:

```bash
curl -N -X POST http://127.0.0.1:8000/sessions/<session-id>/turns/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"I ask what the locked door hides.","active_persona_id":"archivist"}'
```

The SSE route emits validated text only after the existing orchestration pipeline completes. It
does not stream provider tokens or expose drafts before critic validation. Setting
`SSE_TEXT_CHUNK_CHARS` above `0` splits that already-validated text into ordered `text` fragments
the client concatenates, for progressive rendering; it never exposes pre-validation tokens.

### Web UI (Angular SPA)

Open [http://127.0.0.1:8000/app/](http://127.0.0.1:8000/app/) after starting FastAPI with a
built frontend (`make dev` builds it; see [frontend/README.md](frontend/README.md) for the SPA
architecture). The browser surface is a thin client over the same-origin API, with four pages:

- **Play** — catalog selectors load public worlds, scenes, and personas from
  `GET /content/catalog`; `Start session` uses the selected catalog world, scene, persona, and
  player name; turns run over buffered SSE with a live stage-progress indicator; side panels
  show read-only memories and editable canon facts (per-turn route, retrieval, critic-status,
  and warning diagnostics live in the RAG Inspector and Analytics pages, not on this page)
- **RAG Inspector** — per-session turn timeline from `GET /sessions/{id}/turn-details` with
  retrieval drill-down (query, selected/rejected chunks, scores, boosts) per turn
- **Analytics** — turn latency and stage-timing statistics for a session
- **Eval** — eval-run trends from `GET /diagnostics/eval-runs` with per-run drill-down
- the setup screen offers a resume picker over recent sessions (`GET /sessions`); resuming
  reloads the full transcript through `GET /sessions/{id}/turn-details`
- during a session, the Play page can reroll the last turn (`DELETE /sessions/{id}/turns/last`,
  which removes the turn and its derived memories), switch the active scene mid-session
  (`POST /sessions/{id}/scene`), and set a persona override that is sent as
  `active_persona_id` with each subsequent turn
- `CLOUD_MODE=ask` shows a one-time `window.confirm` prompt when starting a cloud session
  (a session's provider choice happens once, at creation, in the setup picker); there is
  no per-turn confirmation once a session is running
- scenario packs remain a process-level backend choice through `CONTENT_ROOT`; start FastAPI with
  the desired `CONTENT_ROOT` to use a different scenario pack

The browser does not own orchestration, retrieval, validation, routing, persistence, memory,
scenario-pack selection, hidden context, or browser-local authoritative state. It provides no
frontend scenario-pack selection, no per-request content-root selection, and no backend ownership moved into browser code.

Read session state:

```bash
curl http://127.0.0.1:8000/sessions/<session-id>
```

List recent local sessions:

```bash
curl http://127.0.0.1:8000/sessions
```

## Qdrant and Retrieval

Qdrant is the runtime vector store for real retrieval:

```bash
docker compose up -d qdrant
```

Collections used by the runtime:

- `canon_lore`
- `session_memory`
- `persona_memory`

Important current behavior:

- document ingestion currently supports `.md` and `.txt`
- world, scene, and persona demo data are loaded from JSON files
- Qdrant is a derived runtime index. CLI `start-session` best-effort auto-ingests
  manifest-declared scenario lore (idempotent and fail-open — an unreachable vector store
  only warns; opt out with `--skip-lore-ingest`). API and SPA session creation does not
  auto-ingest; use `ingest` / `ingest-scenario-lore` manually there or for re-indexing
- retrieval is fail-open; if Qdrant or embeddings are unavailable during a turn, generation continues without retrieved context
- actor retrieval gathers candidates from `session_memory`, `persona_memory`, and `canon_lore`, then applies a deterministic reranking pass
- actor retrieval filters to player-visible chunks and scopes lore, session memory, and persona memory by stored session metadata
- reranking preserves vector score and adds transparent boosts for collection, matching session/scene/persona metadata, memory importance, and lexical overlap with the player message
- curated SQLite memory episodes are indexed into `session_memory` after persistence
- memory indexing is fail-open during turns; use `reindex-memories` to backfill after an outage
- `retrieve-debug` prints metadata-only diagnostics for selected chunks and does not expose hidden text
- actor prompts include only the configured bounded recent-dialogue window; older continuity comes
  back through retrieved durable memory when relevant
- prior recent-dialogue messages are clipped to `RECENT_DIALOGUE_MAX_MESSAGE_CHARS`
  characters (default `900`) with an explicit omitted-characters marker before insertion
  into the actor prompt, and any clipping is surfaced as a turn warning; the current
  incoming player message is not clipped

## Tests, Lint, and Evals

Run all checks:

```bash
ruff check .
mypy .
pytest
(cd frontend && npm test -- --watch=false --browsers=ChromeHeadless)
```

The SPA end-to-end test needs the full stack running (`make dev`):

```bash
PLAYWRIGHT_BASE_URL=http://127.0.0.1:8000 npm run test:e2e-spa
```

Run only eval tests:

```bash
pytest tests/evals -q
```

Run the standalone deterministic regression summary:

```bash
python -m app.evals.regression_runner
```

CI runs those deterministic checks on push and pull request. A separate self-hosted `Live Smoke`
workflow runs the live checkpoint on manual dispatch and a gated weekly schedule; its inputs,
the paired `scripts/test-local-model-matrix.sh` comparison, and the bake-off / secret-probe / RAG
A/B harnesses are documented in
[docs/19_verification_and_eval_tooling.md](docs/19_verification_and_eval_tooling.md).

The eval harness covers retrieval quality, visibility boundaries, role consistency, memory curation
behavior, long-session durable-memory continuity, and provider-binding policy. The 16-turn
`memory_continuity` regression verifies that actor history remains window-bounded, hidden memories
stay out of actor prompts, and SQLite-backed memory survives a fresh derived-index rebuild with
scope isolation. It uses fake providers, deterministic keyword embeddings, SQLite, and
`InMemoryVectorStore`. It is engine regression coverage, not proof of live LLM behavior, semantic
embedding quality, Qdrant quality, or generated prose quality.

## Current Limitations

- personal-use MVP only
- no authentication or multi-user isolation
- buffered SSE only; no provider token streaming or pre-validation token emission
- no production deployment hardening
- no browser-local authoritative state
- no frontend scenario-pack selection
- local/cloud behavior depends on the configured providers actually being available
- Qdrant is required for real retrieval behavior
- memory vector indexing is derived from SQLite and may require `reindex-memories` after an outage

## Safe Next Steps

Post-1.0 candidates are tracked in [docs/10_next_steps_after_mvp.md](docs/10_next_steps_after_mvp.md). The short version:

- tune gating thresholds and retrieval heuristics from live eval evidence
- expand retrieval observability if another surface needs it
- consider validated fragmentation only if it preserves the current exposure boundary
- add auth only if the project becomes multi-user

## Additional Documentation

Start at the hub: [docs/README.md](docs/README.md) is the documentation index — architecture
diagrams (components, turn pipeline, routing) and a living-vs-historical listing of every doc.

Most-used direct links:

- [docs/09_current_architecture_map.md](docs/09_current_architecture_map.md) — where each piece lives
- [docs/12_api_contract.md](docs/12_api_contract.md) — the HTTP API contract
- [docs/20_playing_rolerag.md](docs/20_playing_rolerag.md) — player guide and troubleshooting FAQ
- [docs/GLOSSARY.md](docs/GLOSSARY.md) — project terms
