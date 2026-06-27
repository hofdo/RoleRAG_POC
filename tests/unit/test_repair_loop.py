from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.domain import (
    CriticResult,
    CriticStatus,
    MemoryCandidate,
    PersonaCard,
    SceneState,
    SessionState,
    TurnInput,
    Visibility,
)
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from app.llm.router import CloudMode, ModelProviderName
from app.memory import MemoryEpisodeStore, RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator, TurnOrchestratorConfig
from app.persistence import (
    DemoWorldRecord,
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
    connect_sqlite,
    initialize_database,
)


class SequencedFakeProvider(LlmProvider):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            text=self.responses[len(self.requests) - 1],
            provider="fake",
            model=request.model,
            usage={"total_tokens": 15},
            finish_reason="stop",
        )


class FakeCritic:
    def __init__(
        self,
        results: list[CriticResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = results or [CriticResult(accepted=True)]
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def evaluate(self, **kwargs: Any) -> CriticResult:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results[len(self.calls) - 1]

    def build_local_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        rejected_draft: str,
        issues: list[str],
        repair_instruction: str | None,
    ) -> list[LlmMessage]:
        messages = list(actor_messages)
        messages.append(LlmMessage(role="assistant", content=rejected_draft))
        messages.append(
            LlmMessage(
                role="user",
                content=repair_instruction or ", ".join(issues) or "repair the response",
            )
        )
        return messages

    def build_cloud_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        issues: list[str],
    ) -> list[LlmMessage]:
        messages = list(actor_messages)
        messages.append(LlmMessage(role="user", content=", ".join(issues) or "repair the response"))
        return messages


class RecordingMemoryCurator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def consolidate(self, **_: Any) -> str:
        raise AssertionError("consolidate not expected in this test")

    async def curate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return type(
            "CuratorResult",
            (),
            {
                "write_memory": True,
                "memories": [
                    MemoryCandidate(
                        summary="The player pressed Iria for the truth.",
                        visibility=Visibility.PLAYER,
                        importance=3,
                        tags=["question"],
                        scene_id="rose-gallery",
                        actor_id="archivist",
                    )
                ],
                "reason": "This is likely to matter.",
            },
        )()


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
            private_description="She is quietly aiding the coup.",
            speaking_style="Precise and dry.",
            secrets=["She hides a cipher key in the gallery clock."],
            forbidden_knowledge=["The regent ordered the poisoning."],
        )

    def load_scene(self, scene_id: str) -> SceneState:
        return SceneState(
            id="rose-gallery",
            title="Rose Gallery",
            location="Winter Palace",
            player_visible_summary="Courtiers drift between mirrors and roses.",
            gm_private_summary="A spy waits behind the mirrored column.",
        )


def _build_orchestrator(
    tmp_path: Path,
    *,
    provider: SequencedFakeProvider,
    critic: FakeCritic,
    cloud_mode: CloudMode = CloudMode.ASK,
    cloud_provider: SequencedFakeProvider | None = None,
    memory_curator: RecordingMemoryCurator | None = None,
) -> tuple[TurnOrchestrator, SQLiteTurnRepository, SQLiteMemoryRepository]:
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
        )
    )
    turn_repository = SQLiteTurnRepository(connection)
    memory_repository = SQLiteMemoryRepository(connection)
    orchestrator = TurnOrchestrator(
        loader=FakeLoader(),
        provider=provider,
        cloud_provider=cloud_provider,
        critic_agent=critic,
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=RecentDialogueStore(turn_repository=turn_repository, recent_turns=8),
        memory_store=MemoryEpisodeStore(memory_repository=memory_repository),
        memory_curator=memory_curator,
        config=TurnOrchestratorConfig(
            local_model="local-model",
            cloud_model="cloud-model",
            local_max_tokens=700,
            cloud_max_tokens=1000,
            local_temperature=0.75,
            cloud_temperature=0.65,
            cloud_mode=cloud_mode,
        ),
    )
    return orchestrator, turn_repository, memory_repository


@pytest.mark.asyncio
async def test_orchestrator_persists_accepted_initial_draft(tmp_path: Path) -> None:
    orchestrator, turn_repository, _ = _build_orchestrator(
        tmp_path,
        provider=SequencedFakeProvider(["Initial accepted draft"]),
        critic=FakeCritic([CriticResult(accepted=True)]),
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Tell me the truth.")
    )

    turns = turn_repository.list_recent_turns("demo-session", 10)
    assert result.text == "Initial accepted draft"
    assert len(turns) == 1
    assert turns[0].assistant_message == "Initial accepted draft"


@pytest.mark.asyncio
async def test_orchestrator_repairs_rejected_draft_once_and_persists_only_final_response(
    tmp_path: Path,
) -> None:
    orchestrator, turn_repository, _ = _build_orchestrator(
        tmp_path,
        provider=SequencedFakeProvider(["Rejected draft", "Repaired draft"]),
        critic=FakeCritic(
            [
                CriticResult(
                    accepted=False,
                    issues=["secret leakage"],
                    repair_instruction="Remove hidden facts and answer directly.",
                ),
                CriticResult(accepted=True),
            ]
        ),
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Tell me the truth.")
    )

    turns = turn_repository.list_recent_turns("demo-session", 10)
    assert result.text == "Repaired draft"
    assert len(turns) == 1
    assert turns[0].assistant_message == "Repaired draft"


@pytest.mark.asyncio
async def test_orchestrator_memory_curator_receives_final_response_not_rejected_draft(
    tmp_path: Path,
) -> None:
    memory_curator = RecordingMemoryCurator()
    orchestrator, _, memory_repository = _build_orchestrator(
        tmp_path,
        provider=SequencedFakeProvider(["Rejected draft", "Repaired draft"]),
        critic=FakeCritic(
            [
                CriticResult(
                    accepted=False,
                    issues=["ignored user action"],
                    repair_instruction="Answer the player's action directly.",
                ),
                CriticResult(accepted=True),
            ]
        ),
        memory_curator=memory_curator,
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Tell me the truth.")
    )

    episodes = memory_repository.list_memories_for_session("demo-session")
    assert result.memory_written is True
    assert memory_curator.calls[0]["assistant_message"] == "Repaired draft"
    assert len(episodes) == 1


@pytest.mark.asyncio
async def test_orchestrator_does_not_use_cloud_repair_when_cloud_mode_is_off(
    tmp_path: Path,
) -> None:
    orchestrator, turn_repository, _ = _build_orchestrator(
        tmp_path,
        provider=SequencedFakeProvider(["Rejected draft", "Still rejected draft"]),
        critic=FakeCritic(
            [
                CriticResult(
                    accepted=False,
                    issues=["secret leakage"],
                    repair_instruction="Remove hidden facts.",
                ),
                CriticResult(
                    accepted=False,
                    issues=["secret leakage"],
                    repair_instruction="Remove hidden facts.",
                ),
            ]
        ),
        cloud_mode=CloudMode.OFF,
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Tell me the truth.")
    )

    assert result.route.provider == ModelProviderName.LOCAL
    assert "could not produce a response" in result.text.lower()
    dump = result.model_dump()
    assert dump.pop("stage_timings")
    assert dump == {
        "text": (
            "The system could not produce a response that passed validation. "
            "No memory or world state was changed."
        ),
        "route": {
            "provider": ModelProviderName.LOCAL,
            "model": "local-model",
            "max_tokens": 700,
            "temperature": 0.75,
            "reason": "cloud mode is off; cloud would have been used: local repair failed",
            "requires_user_confirmation": False,
        },
        "finish_reason": "stop",
        "memory_written": False,
        "critic_status": CriticStatus.REJECTED,
        "warnings": [
            "cloud actor skipped: cloud mode is off",
            "cloud actor skipped: cloud mode is off (local repair failed)",
        ],
        "retrieval": None,
    }
    assert turn_repository.count_turns("demo-session") == 0


@pytest.mark.asyncio
async def test_orchestrator_returns_warning_and_persists_latest_draft_when_critic_fails(
    tmp_path: Path,
) -> None:
    orchestrator, turn_repository, _ = _build_orchestrator(
        tmp_path,
        provider=SequencedFakeProvider(["Draft survives critic failure"]),
        critic=FakeCritic(error=ValueError("invalid critic output")),
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Tell me the truth.")
    )

    turns = turn_repository.list_recent_turns("demo-session", 10)
    assert result.text == "Draft survives critic failure"
    assert result.warnings == ["critic skipped: invalid critic output"]
    assert len(turns) == 1
    assert turns[0].assistant_message == "Draft survives critic failure"


@pytest.mark.asyncio
async def test_critic_status_is_accepted_for_accepted_initial_draft(tmp_path: Path) -> None:
    from app.domain import CriticStatus

    orchestrator, _, _ = _build_orchestrator(
        tmp_path,
        provider=SequencedFakeProvider(["A clean draft."]),
        critic=FakeCritic([CriticResult(accepted=True)]),
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Hello.")
    )

    assert result.critic_status == CriticStatus.ACCEPTED


@pytest.mark.asyncio
async def test_critic_status_is_repaired_after_successful_local_repair(tmp_path: Path) -> None:
    from app.domain import CriticStatus

    orchestrator, _, _ = _build_orchestrator(
        tmp_path,
        provider=SequencedFakeProvider(["A leaky draft.", "A repaired draft."]),
        critic=FakeCritic(
            [
                CriticResult(accepted=False, issues=["leak"], repair_instruction="Remove leak."),
                CriticResult(accepted=True),
            ]
        ),
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Hello.")
    )

    assert result.text == "A repaired draft."
    assert result.critic_status == CriticStatus.REPAIRED


@pytest.mark.asyncio
async def test_critic_status_is_rejected_when_repair_is_exhausted(tmp_path: Path) -> None:
    from app.domain import CriticStatus, TurnOutcome

    orchestrator, _, _ = _build_orchestrator(
        tmp_path,
        provider=SequencedFakeProvider(["A leaky draft.", "Still leaky."]),
        critic=FakeCritic(
            [
                CriticResult(accepted=False, issues=["leak"]),
                CriticResult(accepted=False, issues=["leak"]),
            ]
        ),
        cloud_mode=CloudMode.OFF,
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Hello.")
    )

    assert result.outcome == TurnOutcome.CONTROLLED_FAILURE
    assert result.critic_status == CriticStatus.REJECTED


@pytest.mark.asyncio
async def test_critic_status_is_skipped_when_critic_fails(tmp_path: Path) -> None:
    from app.domain import CriticStatus

    orchestrator, _, _ = _build_orchestrator(
        tmp_path,
        provider=SequencedFakeProvider(["Draft survives critic failure"]),
        critic=FakeCritic(error=ValueError("invalid critic output")),
    )

    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id="demo-session", message="Hello.")
    )

    assert result.critic_status == CriticStatus.SKIPPED
