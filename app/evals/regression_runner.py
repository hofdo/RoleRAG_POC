from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from app.agents.critic_agent import CriticAgent
from app.agents.memory_curator import MemoryCurator, MemoryCuratorOutputError
from app.agents.secret_guard import scan_reply
from app.domain import TurnInput, TurnOutcome
from app.evals.event_key_retrieval import evaluate_event_key_retrieval
from app.evals.fixtures import EvalFixture, build_eval_fixture
from app.evals.memory_continuity import evaluate_memory_continuity
from app.evals.memory_recall import evaluate_memory_recall
from app.evals.retrieval_miss import evaluate_retrieval_miss
from app.evals.retrieval_quality import evaluate_retrieval_quality
from app.evals.role_consistency import evaluate_role_consistency
from app.llm.provider import (
    LlmProvider,
    LlmRequest,
    LlmResponse,
    generate_with_truncation_retry,
)
from app.llm.router import ModelProviderName, ModelRoute, ModelTask, choose_route
from app.orchestration.draft_validator import validate_draft


class CategoryResult(BaseModel):
    name: str
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)


class RegressionReport(BaseModel):
    passed: bool
    total_checks: int
    results: list[CategoryResult] = Field(default_factory=list)


def run_regressions() -> RegressionReport:
    fixture = build_eval_fixture()
    results = [
        _retrieval_result(fixture),
        _event_key_retrieval_result(fixture),
        _memory_recall_result(fixture),
        _visibility_result(fixture),
        _role_consistency_result(fixture),
        asyncio.run(_memory_result(fixture)),
        _memory_continuity_result(),
        _draft_validation_result(),
        asyncio.run(_provider_binding_result(fixture)),
        _containment_result(),
        asyncio.run(_structured_resilience_result()),
        _retrieval_miss_result(fixture),
    ]
    total_checks = sum(len(result.checks) for result in results)
    return RegressionReport(
        passed=all(result.passed for result in results),
        total_checks=total_checks,
        results=results,
    )


def _retrieval_result(fixture: EvalFixture) -> CategoryResult:
    result = evaluate_retrieval_quality(fixture)
    return CategoryResult(name="retrieval", passed=result.passed, checks=result.checks)


def _event_key_retrieval_result(fixture: EvalFixture) -> CategoryResult:
    result = evaluate_event_key_retrieval(fixture)
    return CategoryResult(name="event_key_retrieval", passed=result.passed, checks=result.checks)


def _retrieval_miss_result(fixture: EvalFixture) -> CategoryResult:
    result = evaluate_retrieval_miss(fixture)
    return CategoryResult(name="retrieval_miss", passed=result.passed, checks=result.checks)


def _visibility_result(fixture: EvalFixture) -> CategoryResult:
    prompt = fixture.build_actor_prompt()
    query = fixture.build_retrieval_query()
    checks = {
        "prompt_includes_public_lore": fixture.public_lore_text in prompt,
        "prompt_excludes_scene_gm_summary": fixture.scene_gm_only_text not in prompt,
        "prompt_excludes_gm_only_lore": fixture.gm_only_lore_text not in prompt,
        "prompt_excludes_character_private": fixture.character_private_text not in prompt,
        "query_excludes_private_persona_fields": (
            fixture.primary_persona_secret not in query
            and fixture.primary_persona_private_description not in query
        ),
    }
    return CategoryResult(name="visibility", passed=all(checks.values()), checks=checks)


def _memory_recall_result(fixture: EvalFixture) -> CategoryResult:
    result = evaluate_memory_recall(fixture)
    return CategoryResult(name="memory_recall", passed=result.passed, checks=result.checks)


def _role_consistency_result(fixture: EvalFixture) -> CategoryResult:
    result = evaluate_role_consistency(fixture)
    return CategoryResult(name="role_consistency", passed=result.passed, checks=result.checks)


def _memory_continuity_result() -> CategoryResult:
    result = evaluate_memory_continuity()
    return CategoryResult(name="memory_continuity", passed=result.passed, checks=result.checks)


async def _memory_result(fixture: EvalFixture) -> CategoryResult:
    curator = MemoryCurator()
    important = await curator.curate(
        provider=fixture.build_memory_provider(kind="important"),
        route=fixture.memory_route,
        session=fixture.session,
        scene=fixture.scene,
        persona=fixture.primary_persona,
        user_message=fixture.important_turn_user_message,
        assistant_message=fixture.important_turn_assistant_message,
    )
    trivial = await curator.curate(
        provider=fixture.build_memory_provider(kind="trivial"),
        route=fixture.memory_route,
        session=fixture.session,
        scene=fixture.scene,
        persona=fixture.primary_persona,
        user_message="Good evening.",
        assistant_message="Good evening.",
    )
    invalid_visibility_rejected = await _expects_curator_failure(
        fixture=fixture,
        kind="invalid_visibility",
    )
    contradictory_write = await MemoryCurator().curate(
        provider=fixture.build_memory_provider(kind="write_without_candidates"),
        route=fixture.memory_route,
        session=fixture.session,
        scene=fixture.scene,
        persona=fixture.primary_persona,
        user_message=fixture.important_turn_user_message,
        assistant_message=fixture.important_turn_assistant_message,
    )
    checks = {
        "important_turn_creates_memory": important.write_memory and len(important.memories) == 1,
        "trivial_turn_skips_memory": trivial.write_memory is False and trivial.memories == [],
        "memory_visibility_required": important.memories[0].visibility.value == "player",
        "invalid_memory_visibility_rejected": invalid_visibility_rejected,
        "write_without_candidates_becomes_decline": (
            contradictory_write.write_memory is False and contradictory_write.memories == []
        ),
    }
    return CategoryResult(name="memory", passed=all(checks.values()), checks=checks)


async def _expects_curator_failure(*, fixture: EvalFixture, kind: str) -> bool:
    try:
        await MemoryCurator().curate(
            provider=fixture.build_memory_provider(kind=kind),
            route=fixture.memory_route,
            session=fixture.session,
            scene=fixture.scene,
            persona=fixture.primary_persona,
            user_message=fixture.important_turn_user_message,
            assistant_message=fixture.important_turn_assistant_message,
        )
    except MemoryCuratorOutputError:
        return True
    return False


def _draft_validation_result() -> CategoryResult:
    visible_texts = [
        "The Rose Gallery sits inside the Winter Palace.",
        "Iria the archivist guards the winter ledgers.",
        "The regent fears open daylight.",
    ]
    invented = validate_draft(
        draft="Duke Erran left a silver map with Iria before vanishing.",
        player_message="What happened here?",
        visible_texts=visible_texts,
    )
    harmless = validate_draft(
        draft="Iria gestures at the Rose Gallery while the regent stays unnamed.",
        player_message="I study the regent's portrait in the gallery.",
        visible_texts=visible_texts,
    )
    evaded = validate_draft(
        draft="The chandeliers glitter softly above the marble floor.",
        player_message="What did the regent promise you before dawn?",
        visible_texts=visible_texts,
    )
    answered = validate_draft(
        draft="The regent promised nothing; promises mean little before dawn.",
        player_message="What did the regent promise you before dawn?",
        visible_texts=visible_texts,
    )
    checks = {
        "invented_entity_flagged": "Duke Erran" in invented.unsupported_entities,
        "harmless_description_clean": harmless.flags == [],
        "evaded_question_flagged": evaded.evaded_action,
        "answered_question_clean": answered.evaded_action is False,
    }
    return CategoryResult(name="draft_validation", passed=all(checks.values()), checks=checks)


async def _provider_binding_result(fixture: EvalFixture) -> CategoryResult:
    """Every task runs on the session's bound provider; there is no escalation,
    fallback, or per-turn override (Task 2 routing collapse).

    Runtime checks (not just route-object inspection) run a full-stack orchestrator
    turn -- real CriticAgent + MemoryCurator, not stubs -- against task-routed
    recording fake providers, so the checks fail if the router ever lets a task
    slip to the wrong provider or if hidden persona/scene content leaks into a
    cloud request. Fakes only; no network or real model involved.
    """

    def route_for(task: ModelTask, provider: ModelProviderName) -> ModelRoute:
        return choose_route(
            task=task,
            session_provider=provider,
            local_model=fixture.local_route.model,
            cloud_model=fixture.cloud_route.model,
            local_max_tokens=fixture.local_route.max_tokens,
            cloud_max_tokens=fixture.cloud_route.max_tokens,
            local_temperature=fixture.local_route.temperature,
            cloud_temperature=fixture.cloud_route.temperature,
        )

    local_critic = route_for(ModelTask.CRITIC, ModelProviderName.LOCAL)
    cloud_critic = route_for(ModelTask.CRITIC, ModelProviderName.CLOUD)
    local_memory = route_for(ModelTask.MEMORY_EXTRACTION, ModelProviderName.LOCAL)
    cloud_memory = route_for(ModelTask.MEMORY_EXTRACTION, ModelProviderName.CLOUD)

    # local_session_never_calls_cloud: a local session's actor, critic, and memory
    # calls must never reach the cloud provider -- the cloud provider explodes if
    # touched at all.
    class _ExplodingProvider(LlmProvider):
        async def generate(self, request: LlmRequest) -> LlmResponse:
            raise AssertionError("cloud provider must never be called for a local session")

    local_orchestrator, local_fake, _ = fixture.build_full_stack_orchestrator(
        session_provider=ModelProviderName.LOCAL,
        actor_response_text="Local answer",
    )
    local_orchestrator.cloud_provider = _ExplodingProvider()
    local_orchestrator.generation_stage.cloud_provider = _ExplodingProvider()
    local_orchestrator.critique_stage.cloud_provider = _ExplodingProvider()
    local_orchestrator.memory_stage.cloud_provider = _ExplodingProvider()
    local_result = await local_orchestrator.run_turn(
        turn_input=TurnInput(session_id=fixture.session.id, message="What do I notice?")
    )
    local_never_calls_cloud = (
        local_result.outcome == TurnOutcome.SUCCESS
        and local_result.route.provider == ModelProviderName.LOCAL
        and len(local_fake.requests) >= 1
    )

    # cloud_session_runs_all_tasks_on_cloud + cloud_critic_prompt_carries_no_hidden_content:
    # a cloud session's actor, critic, and memory-extraction calls must all land on the
    # cloud provider (local provider explodes if touched), and none of the fixture's
    # hidden persona/scene strings may appear in any recorded cloud request.
    class _LocalExplodingProvider(LlmProvider):
        async def generate(self, request: LlmRequest) -> LlmResponse:
            raise AssertionError("local provider must never be called for a cloud session")

    cloud_orchestrator, _, cloud_fake = fixture.build_full_stack_orchestrator(
        session_provider=ModelProviderName.CLOUD,
        actor_response_text="Cloud answer",
    )
    cloud_orchestrator.provider = _LocalExplodingProvider()
    cloud_orchestrator.generation_stage.provider = _LocalExplodingProvider()
    cloud_orchestrator.critique_stage.provider = _LocalExplodingProvider()
    cloud_orchestrator.memory_stage.provider = _LocalExplodingProvider()
    cloud_result = await cloud_orchestrator.run_turn(
        turn_input=TurnInput(session_id=fixture.session.id, message="What do I notice?")
    )
    cloud_tasks = [request.metadata.get("task") for request in cloud_fake.requests]
    cloud_runs_all_tasks_on_cloud = (
        cloud_result.outcome == TurnOutcome.SUCCESS
        and cloud_result.route.provider == ModelProviderName.CLOUD
        and "critic" in cloud_tasks
        and "memory_extraction" in cloud_tasks
    )

    all_cloud_text = " ".join(
        message.content for request in cloud_fake.requests for message in request.messages
    )
    hidden_strings = (
        fixture.primary_persona_secret,
        fixture.primary_persona_private_description,
        fixture.primary_persona_forbidden_knowledge,
        fixture.scene_gm_only_text,
    )
    # private_description is asserted as defense-in-depth only: no prompt builder
    # references it today (structural absence, not an include_hidden gate) -- the
    # three load-bearing strings are secret / forbidden_knowledge / gm_private_summary.
    # Memory extraction has no include_hidden gate: its prompt omits hidden fields
    # structurally (MemoryCurator._build_context interpolates only ids/names/dialogue).
    cloud_critic_prompt_carries_no_hidden_content = all(
        hidden not in all_cloud_text for hidden in hidden_strings
    )

    checks = {
        "local_session_never_calls_cloud": local_never_calls_cloud,
        "cloud_session_runs_all_tasks_on_cloud": cloud_runs_all_tasks_on_cloud,
        "cloud_critic_prompt_carries_no_hidden_content": (
            cloud_critic_prompt_carries_no_hidden_content
        ),
        "structured_tasks_stay_greedy_on_both_providers": (
            local_critic.temperature == 0.0
            and local_memory.temperature == 0.0
            and cloud_critic.temperature == 0.0
            and cloud_memory.temperature == 0.0
        ),
    }
    return CategoryResult(name="provider_binding", passed=all(checks.values()), checks=checks)


def _containment_result() -> CategoryResult:
    """Output-side secret containment — the bake-off's confabulation failure mode."""
    secret = (
        "House Varn offered this daughter, and not the elder, because she was the one "
        "Duke Corvane could most afford to lose."
    )
    verbatim = scan_reply(f"In truth, {secret.lower()}", [secret])
    paraphrase = scan_reply(
        "House Varn sees her as the daughter they can most afford to lose; her elder "
        "sister carries the weight, so Corvane sent the one he could spare.",
        [secret],
    )
    deflection = scan_reply(
        "House Varn would not part with a treasure lightly, my lord.",
        [secret],
    )
    checks = {
        "verbatim_echo_redacted": verbatim.verbatim_redacted,
        "paraphrased_confabulation_flagged": bool(paraphrase.paraphrased_facts),
        "in_character_deflection_not_flagged": not deflection.flagged,
    }
    return CategoryResult(name="containment", passed=all(checks.values()), checks=checks)


class _ScriptedProvider(LlmProvider):
    """Returns scripted (text, finish_reason) pairs, one per call."""

    def __init__(self, responses: list[tuple[str, str]]) -> None:
        self.responses = responses
        self.calls = 0

    async def generate(self, request: LlmRequest) -> LlmResponse:
        text, finish_reason = self.responses[self.calls]
        self.calls += 1
        return LlmResponse(
            text=text, provider="scripted", model=request.model, finish_reason=finish_reason
        )


async def _structured_resilience_result() -> CategoryResult:
    """Truncated structured calls retry once instead of failing silently."""
    valid = '{"accepted": true, "issues": [], "repair_instruction": null}'
    route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=320,
        temperature=0.0,
        reason="critic stays local",
    )
    persona = build_eval_fixture().primary_persona
    scene = build_eval_fixture().scene

    truncated_then_valid = _ScriptedProvider([('{"accepted": fal', "length"), (valid, "stop")])
    recovered = await CriticAgent().evaluate(
        provider=truncated_then_valid,
        route=route,
        persona=persona,
        scene=scene,
        user_message="Hello.",
        draft="Good evening.",
        retrieved_chunks=[],
    )

    complete = _ScriptedProvider([(valid, "stop")])
    request = LlmRequest(messages=[], model="local-model", max_tokens=320, temperature=0.0)
    await generate_with_truncation_retry(complete, request)

    checks = {
        "truncated_structured_call_retries": truncated_then_valid.calls == 2 and recovered.accepted,
        "complete_structured_call_not_retried": complete.calls == 1,
    }
    return CategoryResult(name="structured_resilience", passed=all(checks.values()), checks=checks)


def _format_category(result: CategoryResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    checks = ", ".join(
        f"{name}={'ok' if passed else 'fail'}" for name, passed in sorted(result.checks.items())
    )
    return f"{status} {result.name}: {checks}"


def main() -> None:
    report = run_regressions()
    for result in report.results:
        print(_format_category(result))
    print(f"Overall: {'PASS' if report.passed else 'FAIL'} ({report.total_checks} checks)")
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
