from __future__ import annotations

from datetime import UTC, datetime

from app.domain import PersonaCard, SceneState, StoredTurn, TurnInput
from app.llm.router import ModelProviderName, ModelRoute
from app.orchestration.context_builder import build_actor_messages


def test_context_builder_includes_persona_scene_recent_events_and_user_message() -> None:
    persona = PersonaCard(
        id="archivist",
        name="Iria Vale",
        role="npc",
        public_description="A composed palace archivist.",
        private_description="She is quietly aiding the coup.",
        speaking_style="Precise and dry.",
        values=["order"],
        goals=["protect the archive"],
        secrets=["She forged one inventory ledger."],
        forbidden_knowledge=["The vault key sequence."],
    )
    scene = SceneState(
        id="rose-gallery",
        title="Rose Gallery",
        location="Winter Palace",
        current_time="late evening",
        player_visible_summary="Courtiers drift between mirrors and roses.",
        gm_private_summary="The regent's spy is already in the room.",
        recent_events=["A servant spilled wine near the west door."],
    )
    turn_input = TurnInput(
        session_id="demo-session",
        active_persona_id="archivist",
        message="What have you heard about the regent?",
    )

    messages = build_actor_messages(persona=persona, scene=scene, turn_input=turn_input)

    assert [message.role for message in messages] == ["system", "user"]
    assert "Iria Vale" in messages[0].content
    assert "A composed palace archivist." in messages[0].content
    assert "Precise and dry." in messages[0].content
    assert "Rose Gallery" in messages[0].content
    assert "Winter Palace" in messages[0].content
    assert "late evening" in messages[0].content
    assert "Courtiers drift between mirrors and roses." in messages[0].content
    assert "A servant spilled wine near the west door." in messages[0].content
    assert messages[1].content == "What have you heard about the regent?"


def test_context_builder_excludes_private_fields_by_default() -> None:
    persona = PersonaCard(
        id="archivist",
        name="Iria Vale",
        role="npc",
        public_description="A composed palace archivist.",
        private_description="She is quietly aiding the coup.",
        speaking_style="Precise and dry.",
        secrets=["She forged one inventory ledger."],
        forbidden_knowledge=["The vault key sequence."],
    )
    scene = SceneState(
        id="rose-gallery",
        title="Rose Gallery",
        location="Winter Palace",
        player_visible_summary="Courtiers drift between mirrors and roses.",
        gm_private_summary="The regent's spy is already in the room.",
    )
    turn_input = TurnInput(
        session_id="demo-session",
        active_persona_id="archivist",
        message="What have you heard about the regent?",
    )

    messages = build_actor_messages(persona=persona, scene=scene, turn_input=turn_input)
    system_prompt = messages[0].content

    assert "quietly aiding the coup" not in system_prompt
    assert "forged one inventory ledger" not in system_prompt
    assert "vault key sequence" not in system_prompt
    assert "regent's spy" not in system_prompt


def test_context_builder_includes_recent_turns_before_current_user_message() -> None:
    persona = PersonaCard(
        id="archivist",
        name="Iria Vale",
        role="npc",
        public_description="A composed palace archivist.",
        speaking_style="Precise and dry.",
    )
    scene = SceneState(
        id="rose-gallery",
        title="Rose Gallery",
        location="Winter Palace",
        player_visible_summary="Courtiers drift between mirrors and roses.",
    )
    route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=700,
        temperature=0.75,
        reason="default local route",
    )
    recent_turns = [
        StoredTurn(
            id=1,
            session_id="demo-session",
            turn_index=1,
            scene_id="rose-gallery",
            persona_id="archivist",
            user_message="Who was at the west door?",
            assistant_message="Only a servant and a guard.",
            route=route,
            created_at=datetime(2026, 5, 27, 11, 0, tzinfo=UTC),
        ),
    ]
    turn_input = TurnInput(
        session_id="demo-session",
        message="What have you heard about the regent?",
    )

    messages = build_actor_messages(
        persona=persona,
        scene=scene,
        turn_input=turn_input,
        recent_turns=recent_turns,
    )

    assert [message.role for message in messages] == ["system", "user", "assistant", "user"]
    assert [message.content for message in messages[1:]] == [
        "Who was at the west door?",
        "Only a servant and a guard.",
        "What have you heard about the regent?",
    ]


def test_context_builder_truncates_only_prior_dialogue_messages() -> None:
    persona = PersonaCard(
        id="archivist",
        name="Iria Vale",
        role="npc",
        public_description="A composed palace archivist.",
        speaking_style="Precise and dry.",
    )
    scene = SceneState(
        id="rose-gallery",
        title="Rose Gallery",
        location="Winter Palace",
        player_visible_summary="Courtiers drift between mirrors and roses.",
    )
    route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=500,
        temperature=0.75,
        reason="default local route",
    )
    full_user_message = "U" * 50
    full_assistant_message = "A" * 50
    recent_turns = [
        StoredTurn(
            id=1,
            session_id="demo-session",
            turn_index=1,
            scene_id="rose-gallery",
            persona_id="archivist",
            user_message=full_user_message,
            assistant_message=full_assistant_message,
            route=route,
            created_at=datetime(2026, 5, 27, 11, 0, tzinfo=UTC),
        )
    ]
    turn_input = TurnInput(
        session_id="demo-session",
        message="What have you heard about the regent?",
    )

    messages = build_actor_messages(
        persona=persona,
        scene=scene,
        turn_input=turn_input,
        recent_turns=recent_turns,
        recent_dialogue_max_message_chars=10,
    )

    assert messages[1].content.startswith("U" * 10)
    assert "Prior dialogue truncated for prompt budget" in messages[1].content
    assert messages[2].content.startswith("A" * 10)
    assert "Prior dialogue truncated for prompt budget" in messages[2].content
    assert messages[3].content == "What have you heard about the regent?"
    assert recent_turns[0].user_message == full_user_message
    assert recent_turns[0].assistant_message == full_assistant_message
