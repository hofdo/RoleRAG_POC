# Session-Bound Provider Choice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cloud becomes a peer choice of primary model, bound to a session at creation — and stops being an automatic rescue/escalation mechanism.

**Architecture:** A session gains a `provider` field (`local` | `cloud`) chosen at creation and immutable thereafter. The router collapses: every task (actor, repair, critic, memory) runs on the session's provider, full stop. All four automatic cloud paths die (local-failure fallback, cloud repair ladder, scene-complexity escalation, retrieval-confidence escalation), along with the per-turn `request_cloud`/two-phase-confirmation machinery. Critic and memory follow the session provider so a cloud session is fully standalone (no llama-server needed); the three hidden-content lines (`secrets`, `forbidden_knowledge`, `gm_private_summary`) are stripped from the critic prompt on cloud routes so secret strings never leave the machine on any provider. `CLOUD_MODE` keeps exactly one job: gating cloud-session creation (`off` = cannot create, `ask` = confirm at creation, `auto` = no prompt).

**Tech Stack:** Python 3.12 / FastAPI / SQLite / Angular 19 signals SPA / pytest / karma.

## Global Constraints

- User decisions (final): critic+memory follow the session provider with secrets stripped on cloud; the per-turn "Request cloud" checkbox is dropped; a session is bound to its provider at creation (no mid-session switch).
- Hard privacy invariant this plan must establish and test: **persona `secrets`, `forbidden_knowledge`, and scene `gm_private_summary` never appear in any request sent to the cloud provider.** The deterministic output-side `secret_guard.scan_reply` (local, in `run_turn`) is untouched and remains the containment layer on both providers.
- No silent provider crossover, ever: a local session must never call the cloud provider object; a cloud session must never call the local one. Failure on either provider → controlled failure (persisted with `outcome`, per 41db80d) — symmetric.
- `critic_gating`/`curator_gating` stay `"always"`; `session_memory_max_episodes` stays `0`.
- Existing sessions migrate as `provider='local'` (additive `_ensure_column` default).
- Backend gate per task: `make check`; tasks touching routing/orchestration also `.venv/bin/python -m app.evals.regression_runner && make smoke`. Frontend tasks: `cd frontend && npx ng test --watch=false && npx ng build`.
- Branch `feat/session-bound-provider` from main. Conventional Commits; end commit messages with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Session provider field (domain + schema + repository + loader)

**Files:**
- Modify: `app/domain/models.py:41-50` (SessionState)
- Modify: `app/persistence/sqlite.py` (initialize_database `_ensure_column` block)
- Modify: `app/persistence/repositories.py` (SQLiteSessionRepository create/get/list SQL + row mapping)
- Modify: `app/orchestration/stages/session.py:62-90` (`TurnSessionLoader.create_session`)
- Test: `tests/unit/test_repositories.py`, `tests/unit/test_turn_orchestrator.py`

**Interfaces:**
- Consumes: `ModelProviderName` (`app/llm/router.py:17`, values `local`/`cloud`; already imported by `app/domain/models.py` for ModelRoute).
- Produces: `SessionState.provider: ModelProviderName = ModelProviderName.LOCAL`; `create_session(..., provider: ModelProviderName = ModelProviderName.LOCAL)` on `TurnSessionLoader` and `TurnOrchestrator.create_session` passthrough; sessions column `provider TEXT NOT NULL DEFAULT 'local'`.

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_repositories.py`:

```python
def test_session_provider_round_trips_and_defaults_to_local(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    repository = SQLiteSessionRepository(connection)
    repository.create_session(
        SessionState(
            id="cloud-session",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
            provider=ModelProviderName.CLOUD,
        )
    )
    repository.create_session(
        SessionState(
            id="local-session",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    cloud = repository.get_session("cloud-session")
    local = repository.get_session("local-session")
    assert cloud is not None and cloud.provider == ModelProviderName.CLOUD
    assert local is not None and local.provider == ModelProviderName.LOCAL
    assert {s.id: s.provider for s in repository.list_recent_sessions(10)} == {
        "cloud-session": ModelProviderName.CLOUD,
        "local-session": ModelProviderName.LOCAL,
    }
```

(`ModelProviderName` import already exists in the test file via `app.llm.router`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_repositories.py::test_session_provider_round_trips_and_defaults_to_local -v`
Expected: FAIL — pydantic rejects unknown `provider` field.

- [ ] **Step 3: Implement**

`app/domain/models.py` SessionState gains (after `content_root`):

```python
    # Bound at creation, immutable for the session's lifetime: the ONE provider
    # that runs actor, repair, critic, and memory for every turn.
    provider: ModelProviderName = ModelProviderName.LOCAL
```

`app/persistence/sqlite.py` — add alongside the other `_ensure_column` calls:

```python
    _ensure_column(
        connection,
        table="sessions",
        column="provider",
        ddl="ALTER TABLE sessions ADD COLUMN provider TEXT NOT NULL DEFAULT 'local'",
    )
```

`app/persistence/repositories.py` `SQLiteSessionRepository`: add `provider` to the INSERT column list/values (`session.provider.value`), to the SELECT column lists in `get_session`/`list_recent_sessions`, and to the row→SessionState mapping: `provider=ModelProviderName(row["provider"]) if "provider" in row.keys() else ModelProviderName.LOCAL` (mirror the `outcome` fallback pattern). Import `ModelProviderName` from `app.llm.router` (module already imports `ModelRoute` from there).

`app/orchestration/stages/session.py` `create_session` gains keyword `provider: ModelProviderName = ModelProviderName.LOCAL`, passed into the `SessionState(...)` construction. `TurnOrchestrator.create_session` (turn_orchestrator.py:~270) gains and forwards the same keyword.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_repositories.py tests/unit/test_turn_orchestrator.py -q` then `make check`
Expected: PASS (field is additive; nothing consumes it yet).

- [ ] **Step 5: Commit**

```bash
git add app/domain/models.py app/persistence app/orchestration/stages/session.py app/orchestration/turn_orchestrator.py tests/unit/test_repositories.py
git commit -m "feat(session): provider field — a session is bound to local or cloud at creation"
```

---

### Task 2: Router collapse + delete every automatic cloud path and the per-turn confirm flow

This is the behavioral core. One atomic task because router, routing stage, generation fallback, repair ladder, orchestrator confirmation path, and TurnInput flags form one contract — the suite only returns to green when all move together.

**Files:**
- Modify: `app/llm/router.py` (rewrite `choose_route`; delete `CloudMode` usage from routing — `CloudMode` enum stays, used by config/creation gate)
- Modify: `app/orchestration/stages/routing.py` (methods take `provider`; delete `provider_failure`, `warning_for_skipped_cloud`, confirmation logic, `cloud_mode`)
- Modify: `app/orchestration/stages/generation.py:143-163` (delete local-failure→cloud fallback)
- Modify: `app/orchestration/stages/repair.py` (single same-provider repair pass; delete cloud ladder)
- Modify: `app/orchestration/turn_orchestrator.py` (delete CONFIRMATION_REQUIRED early return + `cloud_mode` property; thread `context.session.provider`)
- Modify: `app/domain/models.py` (`TurnInput` drops `user_requested_cloud`/`cloud_confirmed`/`force_local`; `TurnOutcome` drops `CONFIRMATION_REQUIRED`)
- Modify: `app/api/schemas.py` (`CreateTurnRequest` drops the three flags; delete `StreamConfirmationPayload`), `app/api/sse.py` (delete confirmation frame branch), `app/api/routes.py` (`_run_turn`/`_to_turn_response` drop confirmation mapping and flags)
- Modify: `app/cli.py` (drop `request_cloud`/route-simulation cloud flags from the turn/route commands; keep `route` command but simplify to the new contract)
- Test: `tests/unit/test_router.py` (rewrite), `tests/unit/test_turn_orchestrator.py`, `tests/unit/test_repair_loop.py`, `tests/unit/test_cloud_fallback.py` (delete), `tests/integration/test_cloud_fallback_turn_flow.py` (delete), `tests/integration/test_api_turns.py`, `tests/unit/test_sse.py`

**Interfaces:**
- Consumes: `SessionState.provider` (Task 1).
- Produces: the new router contract every later task relies on:

```python
def choose_route(
    *,
    task: ModelTask,
    session_provider: ModelProviderName,
    local_model: str,
    cloud_model: str,
    local_max_tokens: int,
    cloud_max_tokens: int,
    local_temperature: float,
    cloud_temperature: float,
    local_structured_max_tokens: int | None = None,
) -> ModelRoute
```

and `TurnRoutingStage.actor(provider, scene) -> RoutingStageResult` / `.critic(provider)` / `.repair(provider)` / `.memory(provider)`, each returning a route on exactly `provider`. `ModelRoute.requires_user_confirmation` field is deleted.

- [ ] **Step 1: Rewrite the router test to the new contract**

Replace the escalation cases in `tests/unit/test_router.py` with:

```python
@pytest.mark.parametrize("provider", [ModelProviderName.LOCAL, ModelProviderName.CLOUD])
@pytest.mark.parametrize(
    "task", [ModelTask.ACTOR_RESPONSE, ModelTask.REPAIR, ModelTask.CRITIC, ModelTask.MEMORY_EXTRACTION]
)
def test_every_task_routes_to_the_session_provider(provider, task):
    route = choose_route(
        task=task,
        session_provider=provider,
        local_model="local-m",
        cloud_model="cloud-m",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.7,
        cloud_temperature=0.65,
        local_structured_max_tokens=640,
    )
    assert route.provider == provider
    assert route.model == ("local-m" if provider == ModelProviderName.LOCAL else "cloud-m")
    assert route.reason == f"session provider: {provider.value}"


def test_structured_tasks_pin_zero_temperature_on_both_providers():
    for provider in (ModelProviderName.LOCAL, ModelProviderName.CLOUD):
        for task in (ModelTask.CRITIC, ModelTask.MEMORY_EXTRACTION):
            route = choose_route(
                task=task,
                session_provider=provider,
                local_model="l", cloud_model="c",
                local_max_tokens=700, cloud_max_tokens=1000,
                local_temperature=0.7, cloud_temperature=0.65,
                local_structured_max_tokens=640,
            )
            assert route.temperature == 0.0
            # Grammar-constrained budget applies to the local server only.
            assert route.max_tokens == (640 if provider == ModelProviderName.LOCAL else 1000)
```

Run: `.venv/bin/python -m pytest tests/unit/test_router.py -v` → FAIL (old signature).

- [ ] **Step 2: Rewrite `choose_route`**

Replace the body of `app/llm/router.py` `choose_route` (keep `CloudMode`, `ModelProviderName`, `ModelTask`, `ModelRoute` — delete `requires_user_confirmation` from `ModelRoute`, delete `LOW_RETRIEVAL_CONFIDENCE`/`HIGH_SCENE_COMPLEXITY` constants, and delete the now-unused `INTENT_CLASSIFICATION`/`SUMMARIZATION` ModelTask members):

```python
def choose_route(
    *,
    task: ModelTask,
    session_provider: ModelProviderName,
    local_model: str,
    cloud_model: str,
    local_max_tokens: int,
    cloud_max_tokens: int,
    local_temperature: float,
    cloud_temperature: float,
    local_structured_max_tokens: int | None = None,
) -> ModelRoute:
    """Every task runs on the session's provider. There is deliberately no
    escalation, fallback, or per-turn override: cloud is a peer choice made at
    session creation, never a rescue mechanism (decision 2026-07-02)."""
    structured = task in {ModelTask.CRITIC, ModelTask.MEMORY_EXTRACTION}
    if session_provider == ModelProviderName.CLOUD:
        return ModelRoute(
            provider=ModelProviderName.CLOUD,
            model=cloud_model,
            max_tokens=cloud_max_tokens,
            # Structured tasks pin greedy decoding on both providers.
            temperature=0.0 if structured else cloud_temperature,
            reason="session provider: cloud",
        )
    return ModelRoute(
        provider=ModelProviderName.LOCAL,
        model=local_model,
        max_tokens=(
            (local_structured_max_tokens or local_max_tokens) if structured else local_max_tokens
        ),
        temperature=0.0 if structured else local_temperature,
        reason="session provider: local",
    )
```

- [ ] **Step 3: Rewrite `TurnRoutingStage`**

`app/orchestration/stages/routing.py`: drop ctor args `cloud_mode`, `low_retrieval_confidence`, `high_scene_complexity`; delete `provider_failure`, `warning_for_skipped_cloud`, `build_local_route`, and the confirmation handling in `actor`. New methods:

```python
    def actor(self, *, provider: ModelProviderName, scene: SceneState) -> RoutingStageResult:
        return RoutingStageResult(
            route=self._choose(task=ModelTask.ACTOR_RESPONSE, provider=provider),
            scene_complexity=self.compute_scene_complexity(scene),
            warnings=(),
        )

    def critic(self, *, provider: ModelProviderName) -> ModelRoute:
        return self._choose(task=ModelTask.CRITIC, provider=provider)

    def repair(self, *, provider: ModelProviderName) -> ModelRoute:
        return self._choose(task=ModelTask.REPAIR, provider=provider)

    def memory(self, *, provider: ModelProviderName) -> ModelRoute:
        return self._choose(task=ModelTask.MEMORY_EXTRACTION, provider=provider)

    def _choose(self, *, task: ModelTask, provider: ModelProviderName) -> ModelRoute:
        return choose_route(
            task=task,
            session_provider=provider,
            local_model=self.local_model,
            cloud_model=self.cloud_model,
            local_max_tokens=self.local_max_tokens,
            cloud_max_tokens=self.cloud_max_tokens,
            local_temperature=self.local_temperature,
            cloud_temperature=self.cloud_temperature,
            local_structured_max_tokens=self.local_structured_max_tokens,
        )
```

Keep `compute_scene_complexity` unchanged — it still feeds critic/curator auto-gating heuristics and diagnostics.

- [ ] **Step 4: Delete the generation fallback**

`app/orchestration/stages/generation.py`: in `_dispatch` (the `except Exception` block at ~143-163), delete the entire fallback — the `except` block collapses to re-raise semantics, so remove the `try/except` wrapper around `_generate_complete` entirely. Also delete the `requires_user_confirmation` guard at the top (field no longer exists). The dual-provider dispatch at ~210-221 (`route.provider == LOCAL → self.provider else self.cloud_provider`) stays — it is now the only crossover-free dispatch point.

- [ ] **Step 5: Single same-provider repair pass**

`app/orchestration/stages/repair.py` `resolve`: keep the accepted-draft early return, the validator-forced-only escape hatch (serve original draft when only the heuristic validator forced repair and repair failed), the `_emit_stage(on_stage, "repair")` emission, and the local repair pass + re-critique. Delete everything from the cloud-route block (`cloud_route = self.routing_stage.repair(...)` at ~146 through the cloud re-critique at ~190): after the first repair's re-critique rejects, go straight to controlled failure. The repair route comes from `self.routing_stage.repair(provider=context.session.provider)` — same provider as the actor, so on a cloud session the "local repair" pass IS a cloud repair. Rename internal locals accordingly (`local_*` → plain names).

- [ ] **Step 6: Orchestrator + domain + API cleanup**

- `app/domain/models.py`: `TurnInput` keeps only `session_id`, `message`, `active_persona_id`. `TurnOutcome` drops `CONFIRMATION_REQUIRED` (update the `StoredTurn.outcome` comment: "stored rows are SUCCESS or CONTROLLED_FAILURE").
- `app/orchestration/turn_orchestrator.py`: delete the `if routing.route.requires_user_confirmation:` early return (~line 306); `routing = self.routing_stage.actor(provider=context.session.provider, scene=context.scene)` (retrieval_confidence no longer feeds routing); delete the `cloud_mode` property/setter (~257-262); `TurnOrchestratorConfig` drops `cloud_mode`, `low_retrieval_confidence`/`high_scene_complexity` stay (gating still uses them via critique stage config).
- `app/composition.py` `build_orchestrator_config`: drop `cloud_mode=settings.cloud_mode` line.
- `app/api/schemas.py`: `CreateTurnRequest` keeps `message` + `active_persona_id` only; delete `StreamConfirmationPayload`; `CreateTurnResponse.status` stays (always `"completed"` now).
- `app/api/sse.py`: delete the `CONFIRMATION_REQUIRED` branch in `build_turn_stream_frames`.
- `app/api/routes.py`: `_run_turn` builds `TurnInput(session_id=..., message=..., active_persona_id=...)`; `_to_turn_response` status collapses to `"completed"`.
- `frontend` is NOT touched in this task (Task 5) — the SPA still sends the extra JSON keys; `CreateTurnRequest` uses `extra="forbid"`, so RELAX it for one task by keeping the three fields as deprecated no-ops? No — keep it atomic: this task also updates `frontend/src/app/api.service.ts` `createBufferedTurn` body to send only `{message, active_persona_id}` and `frontend/src/app/models.ts` `CreateTurnRequest`, deferring the visual cleanup (checkbox/banner removal) to Task 5. Run `npx ng build` to confirm compile.
- `app/cli.py`: update the `route` command (line ~406) to the new signature (drop `--request-cloud`, print the session-provider route) and the turn command's cloud flags.

- [ ] **Step 7: Update the turn-flow tests**

- Delete `tests/unit/test_cloud_fallback.py` and `tests/integration/test_cloud_fallback_turn_flow.py` (they pin the rescue behavior being removed).
- `tests/unit/test_turn_orchestrator.py`: delete/replace confirmation-required tests (`test_turn_orchestrator_cloud_request_requires_confirmation`-style, the `count_turns == 0` confirmation assert at ~1044) and any `TurnInput(..., user_requested_cloud=True)` usages. Add the crossover guard:

```python
@pytest.mark.asyncio
async def test_local_session_never_touches_the_cloud_provider(tmp_path: Path) -> None:
    class ExplodingProvider(LlmProvider):
        async def generate(self, request: LlmRequest) -> LlmResponse:
            raise AssertionError("cloud provider must never be called for a local session")

    orchestrator = _build_orchestrator(tmp_path, FakeProvider(), cloud_provider=ExplodingProvider())
    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Hello")
    )
    assert result.outcome == TurnOutcome.SUCCESS
```

(extend the `_build_orchestrator` helper with a `cloud_provider=None` keyword). Add the symmetric cloud-session test: seed the session with `provider=ModelProviderName.CLOUD`, give the CLOUD provider the scripted responses, make the LOCAL provider the exploding one, assert success and `result.route.provider == ModelProviderName.CLOUD`.
- `tests/unit/test_repair_loop.py`: rewrite the two cloud-ladder tests to the single-pass contract (rejected → repaired-once → still rejected → controlled failure, no second provider involved).
- `tests/unit/test_sse.py` / `tests/integration/test_api_turns.py`: remove confirmation-frame cases; remove `request_cloud` keys from POST bodies.

- [ ] **Step 8: Gates**

Run: `make check && .venv/bin/python -m app.evals.regression_runner && make smoke && cd frontend && npx ng build`
Expected: regression_runner FAILS on `cloud_routing` evals — that suite is rewritten in Task 6; if it blocks the runner, temporarily skip is NOT allowed — pull Task 6's eval rewrite forward into this commit instead (the eval harness change is small; see Task 6 Step 1 for the replacement checks).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(routing)!: every task runs on the session provider; delete cloud escalation, fallback, and per-turn confirm flow"
```

---

### Task 3: Critic + memory follow the session provider; secrets stripped from cloud critic prompts

**Files:**
- Modify: `app/llm/provider.py` (shared `resolve_provider` helper)
- Modify: `app/orchestration/stages/critique.py` (ctor gains `cloud_provider`; dispatch by route; `include_hidden`)
- Modify: `app/orchestration/stages/memory.py` (ctor gains `cloud_provider`; dispatch by route; provider from `session.provider`)
- Modify: `app/agents/critic_agent.py:146-196` (`_build_context`/`evaluate` gain `include_hidden: bool = True`)
- Modify: `app/agents/memory_curator.py` (no prompt change — dialogue only; just dispatch)
- Modify: `app/orchestration/turn_orchestrator.py` (pass `cloud_provider` to both stages)
- Test: `tests/unit/test_critic_agent.py`, `tests/unit/test_actor_agent.py` (untouched, listed for awareness), `tests/unit/test_turn_orchestrator.py`

**Interfaces:**
- Consumes: Task 2's `routing_stage.critic(provider=...)`/`memory(provider=...)`.
- Produces: `resolve_provider(route: ModelRoute, *, local: LlmProvider, cloud: LlmProvider | None) -> LlmProvider` in `app/llm/provider.py` (raises `RuntimeError(f"Missing provider for route: {route.provider.value}")` when cloud is None); `CriticAgent.evaluate(..., include_hidden: bool = True)`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_critic_agent.py` (mirror its existing recording-provider fixture that captures `LlmRequest.messages`):

```python
@pytest.mark.asyncio
async def test_critic_prompt_omits_hidden_content_when_include_hidden_false(...):
    # persona fixture has secrets=["cipher key in the gallery clock"],
    # forbidden_knowledge=["the regent ordered the poisoning"];
    # scene fixture has gm_private_summary="A spy waits behind the mirrored column."
    await agent.evaluate(provider=recording_provider, route=route, persona=persona,
                         scene=scene, user_message="hi", draft="a draft",
                         retrieved_chunks=[], include_hidden=False)
    prompt = " ".join(m.content for m in recording_provider.requests[0].messages)
    assert "cipher key" not in prompt
    assert "poisoning" not in prompt
    assert "mirrored column" not in prompt
    assert "Hidden secrets" not in prompt
    assert "Forbidden knowledge" not in prompt
    assert "Hidden scene facts" not in prompt


@pytest.mark.asyncio
async def test_critic_prompt_includes_hidden_content_by_default(...):
    await agent.evaluate(..., include_hidden=True)
    prompt = " ".join(m.content for m in recording_provider.requests[0].messages)
    assert "cipher key" in prompt
```

In `tests/unit/test_turn_orchestrator.py` — the load-bearing privacy + standalone tests:

```python
@pytest.mark.asyncio
async def test_cloud_session_runs_critic_and_memory_on_cloud_without_secrets(tmp_path: Path) -> None:
    # Cloud session: recording cloud provider answers actor+critic+memory;
    # LOCAL provider explodes if touched (cloud sessions are standalone).
    ...seed session with provider=ModelProviderName.CLOUD...
    result = await orchestrator.run_turn(turn_input=TurnInput(session_id="cloud-session", message="Hello"))
    assert result.outcome == TurnOutcome.SUCCESS
    all_cloud_text = " ".join(
        m.content for req in cloud_provider.requests for m in req.messages
    )
    # The demo persona/scene fixtures carry hidden content -- assert the literal
    # strings never reached the cloud provider on ANY request (actor, critic, memory).
    assert "cipher key" not in all_cloud_text        # persona.secrets
    assert "poisoning" not in all_cloud_text          # persona.forbidden_knowledge
    assert "mirrored column" not in all_cloud_text    # scene.gm_private_summary
```

(adjust the literals to the actual `FakeLoader`/fixture hidden strings in that test file — `test_api_turns.py`'s FakeLoader uses exactly these three.)

Run both files → FAIL (`include_hidden` unknown; critique stage still holds only the local provider).

- [ ] **Step 2: Implement**

`app/llm/provider.py`:

```python
def resolve_provider(
    route: ModelRoute, *, local: LlmProvider, cloud: LlmProvider | None
) -> LlmProvider:
    if route.provider == ModelProviderName.LOCAL:
        return local
    if cloud is None:
        raise RuntimeError(f"Missing provider for route: {route.provider.value}")
    return cloud
```

(import `ModelProviderName`, `ModelRoute` from `app.llm.router`; refactor `TurnGenerationStage._generate_complete`'s inline selection to call it — one source of dispatch truth.)

`app/agents/critic_agent.py`: `evaluate(...)` and `_build_context(...)` gain `include_hidden: bool = True`; wrap the three hidden lines:

```python
        if include_hidden:
            if persona.secrets:
                persona_lines.append(f"Hidden secrets: {'; '.join(persona.secrets)}")
            if persona.forbidden_knowledge:
                persona_lines.append(
                    f"Forbidden knowledge: {'; '.join(persona.forbidden_knowledge)}"
                )
        ...
        if include_hidden and scene.gm_private_summary:
            scene_lines.append(f"Hidden scene facts: {scene.gm_private_summary}")
```

`app/orchestration/stages/critique.py`: ctor gains `cloud_provider: LlmProvider | None = None`; in `run`, `route = self.routing_stage.critic(provider=route_provider)` and:

```python
            critique = await self.critic_agent.evaluate(
                provider=resolve_provider(route, local=self.provider, cloud=self.cloud_provider),
                route=route,
                ...,
                # Hidden authored content never leaves the machine: cloud critics
                # check prose/consistency only; the deterministic local
                # secret_guard scan remains the containment layer.
                include_hidden=route.provider == ModelProviderName.LOCAL,
            )
```

(the `redact_hidden_facts` post-processing and `record_structured_failure` hidden-fact redaction stay — they're local and still correct.)

`app/orchestration/stages/memory.py`: ctor gains `cloud_provider: LlmProvider | None = None`; `run` derives `route = self.routing_stage.memory(provider=session.provider)` and dispatches via `resolve_provider`. Same for the consolidation call if it routes separately (it reuses the memory route — verify `stages/memory_consolidation.py:67` and thread the same resolved provider).

`app/orchestration/turn_orchestrator.py`: pass `cloud_provider=cloud_provider` into `TurnCritiqueStage(...)` and `TurnMemoryStage(...)`.

- [ ] **Step 3: Gates + commit**

Run: `make check && .venv/bin/python -m app.evals.regression_runner && make smoke`

```bash
git add app tests
git commit -m "feat(privacy): critic and memory follow the session provider; hidden content stripped from cloud critic prompts"
```

---

### Task 4: Cloud-session creation gate (API + CLI)

**Files:**
- Modify: `app/api/schemas.py` (`CreateSessionRequest.provider`), `app/api/routes.py` (`create_session` validation)
- Modify: `app/cli.py` (`start-session --provider` + `ask` confirmation)
- Test: `tests/integration/test_api_sessions.py`, `tests/integration/test_cli_sessions.py`

**Interfaces:**
- Consumes: Task 1's `create_session(provider=...)`; `is_usable_cloud_api_key` (`app/config.py:143`); `settings.cloud_mode`.
- Produces: `POST /sessions` accepts `"provider": "local" | "cloud"` (default `"local"`); error code `cloud_unavailable` (400) when cloud is requested but `CLOUD_MODE=off` or no usable key; `CreateSessionResponse.provider: str`.

- [ ] **Step 1: Failing API test**

```python
def test_create_cloud_session_rejected_when_cloud_mode_off(...):
    # settings override fixture with cloud_mode="off"
    response = client.post("/sessions", json={..., "provider": "cloud"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "cloud_unavailable"


def test_create_cloud_session_succeeds_when_configured(...):
    # settings override with cloud_mode="auto" and a usable key
    response = client.post("/sessions", json={..., "provider": "cloud"})
    assert response.status_code == 201
    assert response.json()["provider"] == "cloud"
```

- [ ] **Step 2: Implement**

Schema: `provider: str = Field(default="local", pattern="^(local|cloud)$")` on `CreateSessionRequest`; `CreateSessionResponse` gains `provider: str`. Route `create_session`:

```python
    if request.provider == "cloud" and (
        settings.cloud_mode == CloudMode.OFF
        or not is_usable_cloud_api_key(settings.cloud_llm_api_key)
    ):
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="cloud_unavailable",
            message="Cloud sessions need CLOUD_MODE=ask|auto and a configured cloud API key.",
        )
```

then forward `provider=ModelProviderName(request.provider)`. (`create_session` route needs `settings: Annotated[Settings, Depends(get_settings)]` — add it.) `ask` vs `auto` is a client-side concern (Task 5 confirms in the SPA); the API treats both as allowed.

CLI `start-session`: `--provider local|cloud` option; when `cloud` and `settings.cloud_mode == CloudMode.ASK`, `typer.confirm("Send this session's turns to the cloud provider?", abort=True)`; when off/unusable key, exit 1 with the same message as the API.

- [ ] **Step 3: Gates + commit**

Run: `make check`

```bash
git add app tests
git commit -m "feat(api): session provider choice at creation, gated by CLOUD_MODE and key presence"
```

---

### Task 5: SPA — model dropdown at creation; delete the per-turn cloud UI

**Files:**
- Modify: `frontend/src/app/models.ts` (`CreateSessionRequest.provider`, `RuntimeStatus` already has `cloud_mode`/`cloud_provider_configured`), `frontend/src/app/play-model.ts` (`buildSessionRequest`/`buildCatalogSessionRequest` carry provider; `buildTurnRequest` drops requestCloud/cloudConfirmed/forceLocal params), `frontend/src/app/session-store.ts` (provider signal; delete `pendingConfirm`/`confirmCloud`/`forceLocal`), `frontend/src/app/components/setup-picker.component.ts` (Model dropdown + ask-confirm), `frontend/src/app/components/message-input.component.ts` (delete checkbox + confirm banner)
- Test: the four spec files alongside

**Interfaces:**
- Consumes: Task 4's API contract; `RuntimeStatus.cloud_mode: string`, `cloud_provider_configured: boolean` (already served).
- Produces: `SessionStore.sessionProvider: signal<'local' | 'cloud'>` used at creation only.

- [ ] **Step 1: Failing specs**

```ts
it('shows the cloud model option only when cloud is configured and not off', ...)
// runtimeStatus fixture {cloud_mode: 'off', cloud_provider_configured: true} → dropdown hidden/local-only
// {cloud_mode: 'ask', cloud_provider_configured: true} → both options

it('creates the session with the chosen provider', async () => {
  store.sessionProvider.set('cloud');
  await store.createSessionFromSelection('Avery');
  expect(apiMock.createSession.calls.mostRecent().args[0].provider).toBe('cloud');
});

it('sendMessage no longer sends cloud flags', async () => {
  await store.sendMessage('hi', ...);
  const body = apiMock.createBufferedTurn.calls.mostRecent().args[1];
  expect('request_cloud' in body).toBeFalse();
});
```

- [ ] **Step 2: Implement**

- Setup picker, inside the `@else` form after Persona:

```html
        <label>
          Model
          <select #providerSel (change)="store.sessionProvider.set(providerSel.value === 'cloud' ? 'cloud' : 'local')">
            <option value="local" [selected]="store.sessionProvider() === 'local'">Local</option>
            @if (store.cloudAvailable()) {
              <option value="cloud" [selected]="store.sessionProvider() === 'cloud'">Cloud</option>
            }
          </select>
        </label>
```

- Store: `readonly sessionProvider = signal<'local' | 'cloud'>('local');` and `readonly cloudAvailable = computed(() => { const s = this.runtimeStatus(); return !!s && s.cloud_provider_configured && s.cloud_mode !== 'off'; });` (reuse however runtime status is already loaded — `describeRuntimeStatus` consumers show the store holds it; verify the signal name at execution). `createSessionFromSelection` includes `provider: this.sessionProvider()`; when `'cloud'` and `runtimeStatus.cloud_mode === 'ask'`, gate with `if (!window.confirm('Turns in this session will be sent to the cloud provider. Continue?')) return;` — consent once, at creation.
- Delete: `pendingConfirm` signal + `confirmCloud()`/`forceLocal()` from the store, the confirm banner + "Request cloud" checkbox from `message-input.component.ts` (send() keeps the Task-5-v1.2 draft-on-success contract), `requestCloud` signal, and the corresponding spec blocks. `sendMessage(message)` loses its second parameter — update `rerollLast` call site.

- [ ] **Step 3: Gates + commit**

Run: `cd frontend && npx ng test --watch=false && npx ng build` then `npm run test:e2e-spa` only if a dev stack is up (else covered by Task 7's live proof).

```bash
git add frontend
git commit -m "feat(spa): model choice at session creation; remove per-turn cloud controls"
```

---

### Task 6: Provider-binding regression evals (replace cloud_routing)

**Files:**
- Modify: `tests/evals/test_cloud_routing_regressions.py` → rename to `tests/evals/test_provider_binding_regressions.py`
- Modify: `app/evals/fixtures.py` (harness `cloud_mode` param → `session_provider`), `app/evals/regression_runner.py` (category rename)
- Test: itself

**Interfaces:**
- Consumes: everything above.
- Produces: regression category `provider_binding` with checks: `local_session_never_calls_cloud`, `cloud_session_runs_all_tasks_on_cloud`, `cloud_critic_prompt_carries_no_hidden_content`, `structured_tasks_stay_greedy_on_both_providers`.

- [ ] **Step 1: Rewrite the eval**

Follow the existing fixture harness pattern (`app/evals/fixtures.py:282` builds an orchestrator with fake providers): replace the `cloud_mode` scenarios with the four checks above. The hidden-content check reuses Task 3's approach — recording cloud provider, assert the fixture's secret strings absent from every cloud request. Keep the eval deterministic (fakes only). Update `regression_runner.py`'s category list (`cloud_routing` → `provider_binding`) and any check-name pins in `tests/evals/test_regression_runner.py`.

- [ ] **Step 2: Gates + commit**

Run: `make check && .venv/bin/python -m app.evals.regression_runner` — expect `PASS provider_binding: ...` in the output.

```bash
git add app/evals tests/evals
git commit -m "test(evals): provider-binding regressions replace cloud-routing escalation checks"
```

---

### Task 7: Docs + live proof

**Files:**
- Modify: `README.md` (cloud section: session-bound provider, CLOUD_MODE's new single job, privacy invariant), `.env.example` (comment on CLOUD_MODE semantics), `docs/BACKLOG.md` (decision entry, dated 2026-07-02: "cloud is a peer session-bound choice; all escalation/rescue paths removed; secrets never sent to cloud")
- Test: `tests/integration/test_documentation.py` (if it pins README claims — check and update)

- [ ] **Step 1: Update docs** (the three files above; keep README's "Not implemented" list accurate — remove "provider token streaming" only if untrue, don't touch otherwise).

- [ ] **Step 2: Full gates + live proof**

Run: `make check && .venv/bin/python -m app.evals.regression_runner && make smoke && cd frontend && npx ng test --watch=false && npx ng build`
Then: `LIVE_TURN_COUNT=8 LOCAL_MODEL_PROFILE=26b-mtp bash scripts/live-smoke.sh` — local-session path end-to-end with the real model (the checkpoint POSTs no `request_cloud` key after Task 2; verify `scripts/live-smoke.sh`/`app/diagnostics/live_checkpoint.py:425` body was updated in Task 2 — if missed, fix here).
Cloud-session live proof is manual/optional (needs a real key): `rolerag start-session --provider cloud ...` + one turn.

- [ ] **Step 3: Commit**

```bash
git add README.md .env.example docs/BACKLOG.md tests
git commit -m "docs: session-bound provider model and CLOUD_MODE consent-at-creation semantics"
```

---

## Self-Review Checklist

- Spec coverage: session-bound provider (T1/T4/T5), router collapse + no rescue (T2), critic/memory follow provider + secrets stripped (T3), checkbox dropped + two-phase flow deleted (T2/T5), creation gate off/ask/auto (T4/T5), regression protection (T6), docs (T7). Deliberately out of scope: mid-session provider switch (user declined), token streaming, removing `CloudMode` enum (still the creation gate).
- Known executor verifications flagged inline: exact hidden-string literals in test fixtures (T3), runtime-status signal name in the store (T5), `memory_consolidation.py` route reuse (T3), `test_documentation.py` README pins (T7), live_checkpoint POST body key (T2/T7).
- Type consistency: `choose_route` signature (T2) matches `TurnRoutingStage._choose` call; `resolve_provider` signature (T3) matches both stage call sites; `CreateSessionRequest.provider: str` API-side vs `ModelProviderName` domain-side conversion happens once in the route (T4).
