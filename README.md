# RoleRAG_POC

Personal-use RoleRAG MVP built around a CLI-first roleplay loop, a minimal local play UI, a
small FastAPI API, SQLite persistence, Qdrant-backed retrieval, and deterministic local/cloud
routing.

## Current State

Implemented in this repository:

- typed settings with `pydantic-settings`
- local/cloud OpenAI-compatible provider abstraction
- deterministic routing with `CLOUD_MODE=off|ask|auto`
- structured JSON world, scene, and persona loading
- Typer CLI for session lifecycle, routing inspection, lore ingestion, and turns
- FastAPI endpoints for public content catalog lookup, session creation, turn execution, buffered SSE turn streaming, and session lookup
- framework-free local play UI served by FastAPI at `GET /play`
- SQLite persistence for sessions, turns, and durable memory episodes
- automatic indexing of curated durable memories into session-scoped retrieval
- Qdrant-backed vector storage for ingested lore and retrieval
- retrieval-aware actor prompt construction with visibility filtering
- deterministic retrieval reranking across `session_memory`, `persona_memory`, and `canon_lore`
- retrieval diagnostics for selected chunks through a CLI debug command
- runtime diagnostics through `doctor`
- deterministic end-to-end runtime smoke verification through `smoke-run`
- deterministic content validation through `validate-content`
- standalone scenario pack scaffolding through `create-scenario-template`
- bounded critic and repair flow with a first-class `critic_status` in turn responses
- truncation-aware generation that retries `finish_reason=length` once with a larger budget
- conservative deterministic fallback extraction for explicit player promises and handovers
- local-only memory extraction
- deterministic eval harness using fake providers and in-memory retrieval fixtures

Not implemented:

- authentication
- multi-user support
- provider token streaming
- production deployment hardening

## Quickstart

### Prerequisites

- Python `3.12+`
- Docker
- a local OpenAI-compatible model server such as `llama.cpp` or Ollama's compatibility layer

### Fresh Clone Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
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

### Run the API

```bash
.venv/bin/uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000/play](http://127.0.0.1:8000/play) for the local play UI. The UI starts
new sessions from backend-owned catalog selectors, resumes existing sessions by pasted
`session_id`, sends turns over JSON by default, and exposes buffered SSE as an opt-in developer
toggle. Scenario packs are selected only by starting FastAPI with the desired process-level
`CONTENT_ROOT`; the browser has no scenario-pack selector or content-root input.

## Local Model Setup

The default local configuration expects an OpenAI-compatible endpoint:

```env
LOCAL_LLM_BASE_URL=http://127.0.0.1:8080/v1
LOCAL_LLM_API_KEY=local
LOCAL_LLM_MODEL=chatgpt-onnechan
```

Example `llama.cpp` server flow:

```bash
llama-server -m /path/to/model.gguf \
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

The settings model lives in [app/config.py](app/config.py). `.env.example` mirrors the actual fields used by `Settings`.

Key values:

```env
APP_ENV=local
DATABASE_PATH=data/rolerag.db

LOCAL_LLM_BASE_URL=http://127.0.0.1:8080/v1
LOCAL_LLM_API_KEY=local
LOCAL_LLM_MODEL=chatgpt-onnechan
LOCAL_LLM_MAX_TOKENS=500
LOCAL_STRUCTURED_MAX_TOKENS=350
LOCAL_LLM_TEMPERATURE=0.75

CLOUD_MODE=ask
CLOUD_LLM_BASE_URL=https://api.openai.com/v1
CLOUD_LLM_API_KEY=replace_me
CLOUD_LLM_MODEL=gpt-4.1-mini
CLOUD_LLM_MAX_TOKENS=1000
CLOUD_LLM_TEMPERATURE=0.65

QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_DEFAULT_TOP_K=5
RAG_CHUNK_SIZE_CHARS=1000
RAG_CHUNK_OVERLAP_CHARS=120
RAG_MAX_RETRIEVED_CHUNK_CHARS=800
RECENT_DIALOGUE_TURNS=8
RECENT_DIALOGUE_MAX_MESSAGE_CHARS=900
```

Cloud behavior:

- `CLOUD_MODE=off`: cloud is never used. If cloud would otherwise be chosen, the runtime stays local and records a warning.
- `CLOUD_MODE=ask`: cloud is never called silently. The runtime does not open an interactive confirmation prompt; it keeps actor turns local when possible and returns a controlled failure if cloud repair would require confirmation after local attempts are exhausted.
- `CLOUD_MODE=auto`: cloud may be used for explicit `--request-cloud`, low retrieval confidence, high scene complexity, failed local repair, or local-provider failure.

Critic evaluation and memory extraction stay local in all modes.

## Runtime Components

- `CLI`: [app/cli.py](app/cli.py) exposes `config`, `health`, `doctor`, `smoke-run`, `validate-content`, `create-scenario-template`, `start-session`, `resume`, `route`, `ingest`, `reindex-memories`, `retrieve-debug`, and `turn`.
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
- Memory extraction stays local even when actor or repair routing uses cloud.
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
python -m app.cli route --task actor_response --request-cloud
python -m app.cli route --task repair --failed-local-attempts 2
```

Session lifecycle:

```bash
python -m app.cli start-session --world-id demo_world --scene-id rose-gallery --active-persona-id archivist --player-name Avery
python -m app.cli resume --session-id <session-id>
python -m app.cli turn --session-id <session-id> --message "Tell me what changed here."
python -m app.cli turn --session-id <session-id> --message "Give me the highest quality answer." --request-cloud
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

Run the isolated live stack checkpoint. It first reuses a local OpenAI-compatible model server if
`/v1/models` already exposes `chatgpt-onnechan`; otherwise it starts this managed server:

```bash
llama-server \
  -hf DavidAU/gemma-4-E4B-it-The-DECKARD-Expresso-Universe-HERETIC-UNCENSORED-Thinking-GGUF:Q8_0 \
  --host 127.0.0.1 \
  --port 8080 \
  --alias chatgpt-onnechan \
  --jinja \
  --reasoning off \
  -ngl all \
  -c 8192 \
  -fa on \
  --cache-type-k q8_0 \
  --cache-type-v q4_0 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --seed 424242
```

The primary checkpoint command is:

```bash
npm install
npx playwright install chromium
LOCAL_LLM_MODEL=chatgpt-onnechan \
LIVE_TURN_COUNT=8 \
PYTHON=.venv/bin/python \
bash scripts/live-smoke.sh
```

That script starts disposable Qdrant on `127.0.0.1:6334`, uses a temporary SQLite database under
`/tmp/rolerag-live-test`, starts FastAPI on `127.0.0.1:18080`, runs live doctor/smoke/API checks,
runs an eight-turn Rose Gallery conversation checkpoint and a Playwright UI smoke, writes detailed
turn, persistence, Qdrant, and retrieval diagnostics to `/tmp/rolerag-live-test/report.md`, and
removes its Qdrant container on exit. If it starts managed llama.cpp, it writes
`/tmp/rolerag-live-test/raw/llama-server.log` and kills only that managed process on exit. It leaves
existing `data/qdrant` and user runtime data untouched.

Managed shutdown sends `SIGTERM`, waits up to `LLAMA_CPP_STOP_TIMEOUT` seconds (default `15`), then
sends `SIGKILL` only if the process it started did not exit. An already-running matching provider
is reused and never stopped by the checkpoint.

The equivalent manual FastAPI context for the live stack is:

```bash
DATABASE_PATH=/tmp/rolerag-live-test/work/rolerag-live.db \
QDRANT_URL=http://127.0.0.1:6334 \
LOCAL_LLM_BASE_URL=http://127.0.0.1:8080/v1 \
LOCAL_LLM_API_KEY=local \
LOCAL_LLM_MODEL=chatgpt-onnechan \
CLOUD_MODE=off \
.venv/bin/uvicorn app.main:app --reload
```

Managed llama.cpp startup uses `llama-server` from `PATH` and `LOCAL_MODEL_PROFILE=small` by
default. `LOCAL_MODEL_PROFILE=26b` selects
`HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M`.
Both named profiles use `--jinja`, disabled reasoning/thinking, full GPU offload, an 8192-token
context, flash attention, q8_0 K cache, q4_0 V cache, and seed `424242`.
`LLAMA_CPP_SERVER_PATH` can select another binary. `LLAMA_CPP_MODEL_PATH` switches
startup from `-hf` to a local `-m` GGUF path. It also accepts `LLAMA_CPP_HOST` and
`LLAMA_CPP_PORT`, optional `LLAMA_CPP_CTX_SIZE` for `-c`, optional
`LLAMA_CPP_N_GPU_LAYERS`, and whitespace-separated `LLAMA_CPP_SERVER_ARGS`.
`LIVE_TURN_COUNT` accepts `5` through `50` and defaults to `8`; the former
`LIVE_LONG_TURN_COUNT` remains a fallback when `LIVE_TURN_COUNT` is unset.
`LIVE_FAIL_ON_STRUCTURED_WARNINGS=1` is the default and fails the checkpoint on critic,
memory-curation, memory-indexing, or retrieval warnings. Set it explicitly to `0` for report-only
warning handling. Playwright remains enabled unless `LIVE_SKIP_BROWSER=1`.

Operational rules:

- `health` is config-only and never probes SQLite, Qdrant, or providers.
- `doctor` never mutates the configured runtime database. It verifies SQLite using a temporary file.
- `smoke-run` uses a temporary SQLite database, deterministic embeddings, in-memory retrieval, and fake provider responses by default.
- `--real-runtime` adds shallow Qdrant and local-provider reachability checks only. It does not call cloud APIs, real completions, or write to Qdrant.
- `scripts/live-smoke.sh` is the live local-stack checkpoint. It requires Docker, npm, Playwright, and `llama-server` on `PATH` when no matching provider is already running. It reuses an existing provider or starts and safely stops the managed Hugging Face model above. Structured-output and retrieval warnings fail by default and can be made report-only with `LIVE_FAIL_ON_STRUCTURED_WARNINGS=0`.
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

Implemented endpoints:

- `GET /play`
- `GET /content/catalog`
- `GET /sessions`
- `POST /sessions`
- `POST /sessions/{session_id}/turns`
- `POST /sessions/{session_id}/turns/stream`
- `GET /sessions/{session_id}`

Create a session:

```bash
curl -X POST http://127.0.0.1:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"world_id":"demo_world","scene_id":"rose-gallery","player_name":"Player","active_persona_id":"archivist"}'
```

Run a turn:

```bash
curl -X POST http://127.0.0.1:8000/sessions/<session-id>/turns \
  -H "Content-Type: application/json" \
  -d '{"message":"I ask what the locked door hides.","active_persona_id":"archivist","request_cloud":false}'
```

Run a buffered SSE turn:

```bash
curl -N -X POST http://127.0.0.1:8000/sessions/<session-id>/turns/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"I ask what the locked door hides.","active_persona_id":"archivist","request_cloud":false}'
```

The SSE route emits validated text only after the existing orchestration pipeline completes. It
does not stream provider tokens or expose drafts before critic validation.

### Local Play UI

Open [http://127.0.0.1:8000/play](http://127.0.0.1:8000/play) after starting FastAPI. The
framework-free browser surface is a thin client over the same-origin API:

- catalog selectors load public worlds, scenes, and personas from `GET /content/catalog`
- `Create session` uses the selected catalog world, scene, persona, and player name
- `Developer ID fallback` remains available for manual world, scene, and persona IDs; if catalog
  loading fails, the fallback opens and session creation uses the manual IDs
- recent sessions load from `GET /sessions`; `Resume selected` restores through backend session lookup
- `Resume session` accepts an existing `session_id` fallback and renders backend `recent_turns`
- turn execution uses JSON by default
- the developer panel can opt into buffered SSE and displays safe route, memory, critic-status, and warning data
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
- Qdrant is a derived runtime index; scenario startup does not automatically ingest lore
- retrieval is fail-open; if Qdrant or embeddings are unavailable during a turn, generation continues without retrieved context
- actor retrieval gathers candidates from `session_memory`, `persona_memory`, and `canon_lore`, then applies a deterministic reranking pass
- actor retrieval filters to player-visible chunks and scopes lore, session memory, and persona memory by stored session metadata
- reranking preserves vector score and adds transparent boosts for collection, matching session/scene/persona metadata, memory importance, and lexical overlap with the player message
- curated SQLite memory episodes are indexed into `session_memory` after persistence
- memory indexing is fail-open during turns; use `reindex-memories` to backfill after an outage
- `retrieve-debug` prints metadata-only diagnostics for selected chunks and does not expose hidden text
- actor prompts include only the configured bounded recent-dialogue window; older continuity comes
  back through retrieved durable memory when relevant
- individual recent-dialogue messages are not character-truncated when inserted into actor prompts

## Tests, Lint, and Evals

Run all checks:

```bash
ruff check .
mypy .
pytest
npm run test:frontend
```

Run only eval tests:

```bash
pytest tests/evals -q
```

Run the standalone deterministic regression summary:

```bash
python -m app.evals.regression_runner
```

CI runs those deterministic checks on push and pull request. The separate `Live Smoke` workflow is
manual and targets self-hosted runners because GitHub-hosted runners do not provide the required
local GGUF model or llama.cpp binary. The workflow defaults to the managed Hugging Face model and
supports `llama_server_path`, `llama_hf_model`, `llama_model_path`, `llama_ctx_size`, and
`llama_n_gpu_layers` overrides. The live
workflow validates `turn_count` from `5` through `50`, defaults to eight turns and strict warning
handling, and uploads `/tmp/rolerag-live-test` unconditionally. Artifacts include `report.md`, the
conversation checkpoint JSON, raw command outputs, API flow JSON, llama-server logs when managed
startup is used, and Playwright traces on failure.

Run the paired local-model comparison manually:

```bash
PYTHON=.venv/bin/python bash scripts/test-local-model-matrix.sh
```

It runs deterministic checks once, then the complete live stack sequentially for the small and 26B
profiles with the same seed and 20-turn story. Outputs are isolated under
`/tmp/rolerag-model-comparison/{small,26b}` with `comparison.json`, `comparison.md`, and a
turn-aligned transcript. The 50-turn extension is explicit:

```bash
MODEL_COMPARE_TURN_COUNT=50 \
PYTHON=.venv/bin/python \
bash scripts/test-local-model-matrix.sh
```

Quality findings are report-only. Deterministic, infrastructure, persistence, indexing, retrieval
visibility, and other application-invariant failures produce a nonzero exit. The paired run is
manual and is not part of normal CI.

The eval harness covers retrieval quality, visibility boundaries, role consistency, memory curation
behavior, long-session durable-memory continuity, and cloud-routing policy. The 16-turn
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

The next implementation candidates are tracked in [docs/10_next_steps_after_mvp.md](docs/10_next_steps_after_mvp.md). The short version:

- improve integration coverage
- tune retrieval heuristics from eval evidence
- expand retrieval observability if another surface needs it
- consider validated fragmentation only if it preserves the current exposure boundary
- add auth only if the project becomes multi-user

## Additional Documentation

- [docs/01_product_goal.md](docs/01_product_goal.md)
- [docs/02_architecture.md](docs/02_architecture.md)
- [docs/03_implementation_guide.md](docs/03_implementation_guide.md)
- [docs/04_agent_workflows.md](docs/04_agent_workflows.md)
- [docs/05_rag_memory_design.md](docs/05_rag_memory_design.md)
- [docs/06_local_cloud_model_strategy.md](docs/06_local_cloud_model_strategy.md)
- [docs/07_mvp_phases.md](docs/07_mvp_phases.md)
- [docs/12_api_contract.md](docs/12_api_contract.md)
- [docs/13_live_model_quality_assessment.md](docs/13_live_model_quality_assessment.md)
- [docs/14_local_model_comparison_2026-06-08.md](docs/14_local_model_comparison_2026-06-08.md)
