from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from app.agents.memory_curator import MemoryCurator, MemoryCuratorOutputError
from app.evals.fixtures import EvalFixture, build_eval_fixture
from app.evals.memory_recall import evaluate_memory_recall
from app.evals.retrieval_quality import evaluate_retrieval_quality
from app.evals.role_consistency import evaluate_role_consistency
from app.llm.router import CloudMode, ModelProviderName, ModelTask, choose_route


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
        _memory_recall_result(fixture),
        _visibility_result(fixture),
        _role_consistency_result(fixture),
        asyncio.run(_memory_result(fixture)),
        _cloud_routing_result(fixture),
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
    invalid_write_rejected = await _expects_curator_failure(
        fixture=fixture,
        kind="write_without_candidates",
    )
    checks = {
        "important_turn_creates_memory": important.write_memory and len(important.memories) == 1,
        "trivial_turn_skips_memory": trivial.write_memory is False and trivial.memories == [],
        "memory_visibility_required": important.memories[0].visibility.value == "player",
        "invalid_memory_visibility_rejected": invalid_visibility_rejected,
        "write_without_candidates_rejected": invalid_write_rejected,
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


def _cloud_routing_result(fixture: EvalFixture) -> CategoryResult:
    off_route = choose_route(
        task=ModelTask.ACTOR_RESPONSE,
        cloud_mode=CloudMode.OFF,
        local_model=fixture.local_route.model,
        cloud_model=fixture.cloud_route.model,
        local_max_tokens=fixture.local_route.max_tokens,
        cloud_max_tokens=fixture.cloud_route.max_tokens,
        local_temperature=fixture.local_route.temperature,
        cloud_temperature=fixture.cloud_route.temperature,
        failed_local_attempts=0,
        retrieval_confidence=0.1,
        scene_complexity=1,
        user_requested_cloud=True,
    )
    ask_route = choose_route(
        task=ModelTask.ACTOR_RESPONSE,
        cloud_mode=CloudMode.ASK,
        local_model=fixture.local_route.model,
        cloud_model=fixture.cloud_route.model,
        local_max_tokens=fixture.local_route.max_tokens,
        cloud_max_tokens=fixture.cloud_route.max_tokens,
        local_temperature=fixture.local_route.temperature,
        cloud_temperature=fixture.cloud_route.temperature,
        failed_local_attempts=0,
        retrieval_confidence=0.1,
        scene_complexity=1,
    )
    auto_route = choose_route(
        task=ModelTask.ACTOR_RESPONSE,
        cloud_mode=CloudMode.AUTO,
        local_model=fixture.local_route.model,
        cloud_model=fixture.cloud_route.model,
        local_max_tokens=fixture.local_route.max_tokens,
        cloud_max_tokens=fixture.cloud_route.max_tokens,
        local_temperature=fixture.local_route.temperature,
        cloud_temperature=fixture.cloud_route.temperature,
        failed_local_attempts=0,
        retrieval_confidence=0.1,
        scene_complexity=1,
    )
    critic_route = choose_route(
        task=ModelTask.CRITIC,
        cloud_mode=CloudMode.AUTO,
        local_model=fixture.local_route.model,
        cloud_model=fixture.cloud_route.model,
        local_max_tokens=fixture.local_route.max_tokens,
        cloud_max_tokens=fixture.cloud_route.max_tokens,
        local_temperature=fixture.local_route.temperature,
        cloud_temperature=fixture.cloud_route.temperature,
        failed_local_attempts=2,
        retrieval_confidence=0.1,
        scene_complexity=5,
    )
    memory_route = choose_route(
        task=ModelTask.MEMORY_EXTRACTION,
        cloud_mode=CloudMode.AUTO,
        local_model=fixture.local_route.model,
        cloud_model=fixture.cloud_route.model,
        local_max_tokens=fixture.local_route.max_tokens,
        cloud_max_tokens=fixture.cloud_route.max_tokens,
        local_temperature=fixture.local_route.temperature,
        cloud_temperature=fixture.cloud_route.temperature,
        failed_local_attempts=2,
        retrieval_confidence=0.1,
        scene_complexity=5,
    )
    checks = {
        "cloud_mode_off_stays_local": off_route.provider == ModelProviderName.LOCAL,
        "cloud_mode_ask_requires_confirmation": ask_route.requires_user_confirmation,
        "cloud_mode_auto_allows_cloud": auto_route.provider == ModelProviderName.CLOUD,
        "critic_stays_local": critic_route.provider == ModelProviderName.LOCAL,
        "memory_extraction_stays_local": memory_route.provider == ModelProviderName.LOCAL,
    }
    return CategoryResult(name="cloud_routing", passed=all(checks.values()), checks=checks)


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
