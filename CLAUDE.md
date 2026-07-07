# CLAUDE.md — Working Guide for Coding Agents

> Reviewed: 2026-07-07 @ 7888ee7

RoleRAG_POC is a **personal-use roleplay RAG engine**: Typer CLI + FastAPI + Angular SPA
over a bounded actor/critic/repair turn pipeline, SQLite as authoritative state, Qdrant as
a derived vector index, and a session-bound local/cloud LLM provider (OpenAI-compatible;
reference local runtime is llama.cpp with a Gemma-family ~26B model).

This file is the entry point for agents. It stays short on purpose — the repo's docs are
excellent and current; **link, don't restate** is a repo convention.

## Read this first (in order)

1. [docs/08_agent_handoff.md](docs/08_agent_handoff.md) — onboarding path + safe working rules
2. [docs/09_current_architecture_map.md](docs/09_current_architecture_map.md) — module map
3. [docs/21_fable_handoff_reasoning.md](docs/21_fable_handoff_reasoning.md) — predecessor-agent
   reasoning chains: *why* the architecture is shaped this way, analysis method, danger zones
4. [docs/22_rag_scaling_roadmap.md](docs/22_rag_scaling_roadmap.md) — verified improvement
   roadmap for the RAG core (larger scenarios, ~27B local models)

## Commands

```bash
# setup (Python 3.12+, Node 20+, Docker)
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
docker compose up -d qdrant

# the full deterministic gate — run before claiming any change works
ruff check . && mypy . && pytest && python -m app.evals.regression_runner

# runtime verification ladder (safe → live)
python -m app.cli health        # config-only, no probes
python -m app.cli doctor        # temp SQLite + demo data + optional --check-qdrant --check-local-provider
python -m app.cli smoke-run     # deterministic E2E with fake providers, in-memory retrieval
bash scripts/live-smoke.sh      # real local model + Qdrant + Playwright (needs llama-server)

# frontend
(cd frontend && npm ci && npx ng build)            # build SPA (served at /app)
(cd frontend && npm test -- --watch=false --browsers=ChromeHeadless)
```

`make help` lists wrappers (`make dev`, `make up`, `make check`, `make smoke`).

## Conventions

- Python 3.12, `ruff` (line length 100, rules E/F/I/B), `mypy --strict` on `app/`.
- Tests: pytest under `tests/{unit,integration,evals}`; deterministic only — fake providers,
  keyword embeddings, `InMemoryVectorStore`. Never make a test depend on live Qdrant or a model.
- Every Qdrant/vector-store feature needs an `InMemoryVectorStore` equivalent (test parity).
- Config: every `Settings` field in [app/config.py](app/config.py) mirrors one documented key in
  [.env.example](.env.example) — a test enforces this pairing. Change both together.
- Living docs carry `> Reviewed: YYYY-MM-DD @ <short-sha>` under the H1; refresh headers you
  touch. New numbered docs must be indexed in [docs/README.md](docs/README.md).
- Commit subjects follow conventional style (`feat:`, `fix:`, `docs:`, `chore:`); backlog items
  are tagged `(#N)` matching [docs/BACKLOG.md](docs/BACKLOG.md).

## Invariants — do not break these

Full rationale in docs/08 and docs/21; the short list:

1. **The LLM never owns authoritative state.** SQLite is the source of truth; Qdrant is a
   rebuildable derived index (`reindex-memories`, `ingest`).
2. **Visibility boundary.** Player-facing actor prompts contain only `player`-visible content.
   Hidden authored fields (persona `secrets`/`forbidden_knowledge`, scene `gm_private_summary`)
   never leave the machine on any provider — enforced by the `include_hidden` gate in
   [app/orchestration/stages/critique.py](app/orchestration/stages/critique.py), the secret-guard
   containment scan, and provider-binding eval tests.
3. **Session-bound provider.** `local`/`cloud` is chosen once at session creation and is
   immutable; no per-turn escalation or cross-provider fallback. `CLOUD_MODE` gates *creation* only.
4. **Retrieval and memory indexing are fail-open**; critic failure is fail-closed
   (controlled failure, never unvalidated text). Don't invert either.
5. **Ranking is deterministic and transparent.** Boost policy lives in `app/rag/ranking.py`,
   preserves the original vector score, and is fully explained in diagnostics.
6. **Orchestration logic stays out of API routes and agents.** Routes are thin;
   `ActorAgent` generates text only; `TurnOrchestrator` owns the lifecycle.

## Where things live

| Area | Files |
|------|-------|
| Turn pipeline (highest-leverage code) | [app/orchestration/turn_orchestrator.py](app/orchestration/turn_orchestrator.py) + [app/orchestration/stages/](app/orchestration/stages) |
| Actor prompt assembly / context budget | [app/orchestration/context_builder.py](app/orchestration/context_builder.py), [context_budget.py](app/orchestration/context_budget.py) |
| RAG core (chunk/embed/store/retrieve/rank) | [app/rag/](app/rag) |
| Durable memory lifecycle | [app/memory/](app/memory), [app/agents/memory_curator.py](app/agents/memory_curator.py) |
| LLM routing + providers + structured output | [app/llm/](app/llm) |
| Persistence (SQLite) + domain models | [app/persistence/](app/persistence), [app/domain/](app/domain) |
| DI wiring | [app/composition.py](app/composition.py) |
| Evals + diagnostics harnesses | [app/evals/](app/evals), [app/diagnostics/](app/diagnostics) |
| Local model profiles (llama.cpp flags, ctx sizes) | [scripts/lib/local-model-profile.sh](scripts/lib/local-model-profile.sh) |

## Verification before claiming success

Run the deterministic gate (above). If you touched CLI/API/docs surfaces, also run the
commands listed in [docs/08_agent_handoff.md](docs/08_agent_handoff.md#verification-before-claiming-success).
For retrieval/ranking/embedding changes, additionally consult the measure-first workflow in
[docs/22_rag_scaling_roadmap.md](docs/22_rag_scaling_roadmap.md) — offline evals use keyword
embeddings and will NOT catch semantic-quality regressions; live-smoke is the arbiter.
