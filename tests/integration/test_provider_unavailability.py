"""Provider-unavailability edge cases on the session's bound provider (docs/10 gap 3).

These pin the ACTUAL exception-handling seam at app/orchestration/turn_orchestrator.py
and app/orchestration/stages/{generation,critique}.py, simulated at the same boundary
the real OpenAI-compatible provider raises from (app/llm/openai_compatible.py): a
connection-refused/timed-out model server surfaces as ProviderUnavailableError /
ProviderTimeoutError (app/llm/provider.py), not a bare ConnectionError/TimeoutError.

Key finding pinned here (see the two "propagates_uncaught" tests below): only
EmptyProviderResponseError/TruncatedProviderResponseError are caught around the actor
generation call in TurnOrchestrator.run_turn. A raw ProviderTimeoutError/
ProviderUnavailableError from generation is NOT caught there -- it propagates out of
run_turn uncaught, so no controlled-failure turn is persisted for that attempt (the
player's message is lost for that specific call, though nothing else is corrupted:
session load/retrieval/routing are read-only, so the session remains fully resumable
and the next turn succeeds once the provider recovers). The API/CLI callers translate
the uncaught exception into an HTTP 503/504 (app/api/routes.py::_run_turn, already
covered by test_post_turn_returns_504_envelope_when_local_provider_times_out /
test_post_turn_returns_503_envelope_when_local_provider_unavailable in
test_api_turns.py) or a CLI exit 1 (app/cli.py::turn).

This is asymmetric with a provider failure during CRITIQUE: TurnCritiqueStage.run
wraps the critic's evaluate() call in a broad `except Exception`, so the exact same
exception type raised there fails CLOSED into a persisted CONTROLLED_FAILURE turn
(invariant #4) instead of crashing the turn. See
test_provider_error_mid_critique_fails_closed_to_persisted_controlled_failure.

This file only pins existing behavior with tests + comments (per task instructions);
no production code was changed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain import CriticResult, PersonaCard, SceneState, SessionState, TurnInput, TurnOutcome
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from app.llm.router import ModelProviderName
from app.memory import RecentDialogueStore
from app.orchestration.stages import CriticEvaluatingAgent
from app.orchestration.turn_orchestrator import (
    CONTROLLED_FAILURE_TEXT,
    TurnOrchestrator,
    TurnOrchestratorConfig,
)
from app.persistence import (
    DemoWorldRecord,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
    connect_sqlite,
    initialize_database,
)


class FakeLoader:
    def load_world(self, world_id: str) -> DemoWorldRecord:
        return DemoWorldRecord(
            id=world_id,
            name="Winter Palace Intrigue",
            default_scene_id="rose-gallery",
            persona_ids=["archivist"],
            scene_ids=["rose-gallery"],
        )

    def load_persona(self, persona_id: str) -> PersonaCard:
        return PersonaCard(
            id="archivist",
            name="Iria Vale",
            role="npc",
            public_description="A composed palace archivist.",
            speaking_style="Precise and dry.",
        )

    def load_scene(self, scene_id: str) -> SceneState:
        return SceneState(
            id="rose-gallery",
            title="Rose Gallery",
            location="Winter Palace",
            player_visible_summary="Courtiers drift between mirrors and roses.",
        )


class _SucceedingProvider(LlmProvider):
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            text="A plain, uneventful answer.",
            provider="fake",
            model=request.model,
            usage={"total_tokens": 10},
            finish_reason="stop",
        )


class _RaisesThenSucceedsProvider(LlmProvider):
    """Fails the first `fail_calls` generate() calls with `error`, then serves a normal
    reply -- simulates a local/cloud model server that is down/refused and later
    recovers."""

    def __init__(self, *, error: Exception, fail_calls: int = 1) -> None:
        self.error = error
        self.fail_calls = fail_calls
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if len(self.requests) <= self.fail_calls:
            raise self.error
        return LlmResponse(
            text="The provider is back online and answers plainly.",
            provider="fake",
            model=request.model,
            usage={"total_tokens": 12},
            finish_reason="stop",
        )


class _ExplodingProvider(LlmProvider):
    """A provider that must never be called -- proves no cross-provider fallback."""

    async def generate(self, request: LlmRequest) -> LlmResponse:
        raise AssertionError("this provider must never be called (no cross-provider fallback)")


class _AcceptingCritic:
    async def evaluate(self, **_: object) -> CriticResult:
        return CriticResult(accepted=True)

    def build_local_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        rejected_draft: str,
        issues: list[str],
        repair_instruction: str | None,
    ) -> list[LlmMessage]:
        # The deterministic draft validator (app/orchestration/draft_validator.py) can
        # flag a generic fake reply as "player action unaddressed" and force a repair
        # pass independent of the critic's own verdict -- harmless here, so this is a
        # no-op passthrough rather than an assertion failure.
        return list(actor_messages)


class _RaisingCritic:
    """Simulates the critic's own LLM call hitting the same kind of connection failure
    the generation stage can hit. Unlike TurnGenerationStage, TurnCritiqueStage.run
    catches any exception from evaluate() (fail-closed backstop, invariant #4)."""

    def __init__(self, *, error: Exception) -> None:
        self.error = error

    async def evaluate(self, **_: object) -> CriticResult:
        raise self.error

    def build_local_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        rejected_draft: str,
        issues: list[str],
        repair_instruction: str | None,
    ) -> list[LlmMessage]:
        raise AssertionError("repair should not run in this test")


def _build_orchestrator(
    tmp_path: Path,
    *,
    provider: LlmProvider,
    cloud_provider: LlmProvider | None = None,
    critic: CriticEvaluatingAgent | None = None,
    session_provider: ModelProviderName = ModelProviderName.LOCAL,
) -> TurnOrchestrator:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    session_repository.create_session(
        SessionState(
            id="demo-session",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
            provider=session_provider,
        )
    )
    turn_repository = SQLiteTurnRepository(connection)
    return TurnOrchestrator(
        loader=FakeLoader(),
        provider=provider,
        cloud_provider=cloud_provider,
        critic_agent=critic or _AcceptingCritic(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=RecentDialogueStore(turn_repository=turn_repository, recent_turns=8),
        config=TurnOrchestratorConfig(
            local_model="local-model",
            cloud_model="cloud-model",
            local_max_tokens=700,
            cloud_max_tokens=1000,
            local_temperature=0.75,
            cloud_temperature=0.65,
        ),
    )


@pytest.mark.asyncio
async def test_local_session_generation_timeout_propagates_uncaught_and_persists_no_turn(
    tmp_path: Path,
) -> None:
    """REAL FINDING (docs/10 gap 3): a ProviderTimeoutError raised from the actor
    generation call is not caught anywhere in TurnOrchestrator.run_turn, so it
    propagates to the caller instead of resolving to a persisted CONTROLLED_FAILURE
    turn. See module docstring for the full explanation and contrast with the
    mid-critique test below."""
    from app.llm.provider import ProviderTimeoutError

    provider = _RaisesThenSucceedsProvider(
        error=ProviderTimeoutError(provider="local", model="local-model", timeout_seconds=180.0),
        fail_calls=1,
    )
    orchestrator = _build_orchestrator(tmp_path, provider=provider)

    with pytest.raises(ProviderTimeoutError):
        await orchestrator.run_turn(
            turn_input=TurnInput(session_id="demo-session", message="Anyone home?")
        )

    # No controlled-failure row, no player message survives this attempt -- unlike
    # EmptyProviderResponseError/TruncatedProviderResponseError, which do persist.
    assert orchestrator.turn_repository.list_all_turns("demo-session") == []
    # Session load/retrieval/routing are read-only, so nothing else was corrupted by
    # the uncaught exception -- the session itself is still trivially resumable.
    assert orchestrator.resume_session("demo-session").id == "demo-session"

    # Provider "recovers" (second call succeeds): the next turn on the SAME session
    # completes normally and persists, proving resumability end to end.
    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Anyone home now?")
    )
    assert result.outcome == TurnOutcome.SUCCESS
    stored = orchestrator.turn_repository.list_all_turns("demo-session")
    assert len(stored) == 1
    assert stored[0].user_message == "Anyone home now?"


@pytest.mark.asyncio
async def test_local_session_connection_refused_propagates_uncaught_and_persists_no_turn(
    tmp_path: Path,
) -> None:
    """Same finding as the timeout test above, for the connection-refused variant
    (ProviderUnavailableError) -- the other exception app/llm/openai_compatible.py
    raises for a down/unreachable model server."""
    from app.llm.provider import ProviderUnavailableError

    provider = _RaisesThenSucceedsProvider(
        error=ProviderUnavailableError(
            provider="local", model="local-model", base_url="http://127.0.0.1:8080/v1"
        ),
        fail_calls=1,
    )
    orchestrator = _build_orchestrator(tmp_path, provider=provider)

    with pytest.raises(ProviderUnavailableError):
        await orchestrator.run_turn(
            turn_input=TurnInput(session_id="demo-session", message="Is the door locked?")
        )

    assert orchestrator.turn_repository.list_all_turns("demo-session") == []


@pytest.mark.asyncio
async def test_cloud_session_provider_error_propagates_uncaught_and_never_falls_back_to_local(
    tmp_path: Path,
) -> None:
    """Invariant #3 (session-bound provider, no per-turn escalation/fallback): a cloud
    session's turn must fail on its own bound cloud provider when that provider errors,
    never silently retry on local. Pins both halves: the local provider is never
    invoked (_ExplodingProvider would raise AssertionError if it were), and the
    failure surfaces the same way as the local-session case above -- uncaught, no
    persisted turn."""
    from app.llm.provider import ProviderUnavailableError

    local_provider = _ExplodingProvider()
    cloud_provider = _RaisesThenSucceedsProvider(
        error=ProviderUnavailableError(
            provider="cloud", model="cloud-model", base_url="https://api.example.com/v1"
        ),
        fail_calls=1,
    )
    orchestrator = _build_orchestrator(
        tmp_path,
        provider=local_provider,
        cloud_provider=cloud_provider,
        session_provider=ModelProviderName.CLOUD,
    )

    with pytest.raises(ProviderUnavailableError):
        await orchestrator.run_turn(
            turn_input=TurnInput(session_id="demo-session", message="What does the cloud know?")
        )

    assert orchestrator.turn_repository.list_all_turns("demo-session") == []
    assert len(cloud_provider.requests) == 1  # the bound provider was actually tried

    # Cloud recovers; the retry still never touches local.
    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Try again?")
    )
    assert result.outcome == TurnOutcome.SUCCESS
    assert result.route.provider == ModelProviderName.CLOUD


@pytest.mark.asyncio
async def test_provider_error_mid_critique_fails_closed_to_persisted_controlled_failure(
    tmp_path: Path,
) -> None:
    """Contrast with the generation-stage tests above: TurnCritiqueStage.run wraps the
    critic's evaluate() call in a broad `except Exception` (app/orchestration/stages/
    critique.py) specifically so any critic-side failure -- including the same
    ProviderUnavailableError a broken connection raises during generation -- fails
    CLOSED into a persisted CONTROLLED_FAILURE turn (invariant #4: critic failure is
    fail-closed) rather than crashing the turn uncaught. Same exception type,
    different stage, deliberately different outcome."""
    from app.llm.provider import ProviderUnavailableError

    provider = _SucceedingProvider()  # actor generation must succeed; only critique fails
    critic = _RaisingCritic(
        error=ProviderUnavailableError(
            provider="local", model="local-model", base_url="http://127.0.0.1:8080/v1"
        )
    )
    orchestrator = _build_orchestrator(tmp_path, provider=provider, critic=critic)

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Tell me a secret.")
    )

    assert result.outcome == TurnOutcome.CONTROLLED_FAILURE
    assert result.text == CONTROLLED_FAILURE_TEXT
    assert any("critic skipped" in warning for warning in result.warnings)
    stored = orchestrator.turn_repository.list_all_turns("demo-session")
    assert len(stored) == 1
    assert stored[0].outcome == TurnOutcome.CONTROLLED_FAILURE
    # Unlike the generation-stage failure above, the player's message DOES survive:
    # the turn is persisted with outcome=CONTROLLED_FAILURE per invariant #4/#8.
    assert stored[0].user_message == "Tell me a secret."
