from __future__ import annotations

from app.domain import PersonaCard, RetrievedChunk, SceneState, TurnInput, Visibility
from app.orchestration.context_budget import ContextBudget
from app.orchestration.context_builder import build_actor_messages


def _chunk(
    chunk_id: str,
    *,
    text: str,
    visibility: Visibility,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        source="data/documents/demo_lore.md",
        source_type="lore",
        text=text,
        score=0.9,
        visibility=visibility,
        tags=["palace", "social"],
    )


def test_context_builder_formats_only_bounded_player_visible_retrieved_chunks() -> None:
    messages = build_actor_messages(
        persona=PersonaCard(
            id="archivist",
            name="Iria Vale",
            role="npc",
            public_description="A composed palace archivist.",
            speaking_style="Precise and dry.",
        ),
        scene=SceneState(
            id="rose-gallery",
            title="Rose Gallery",
            location="Winter Palace",
            player_visible_summary="Courtiers drift between mirrors and roses.",
        ),
        turn_input=TurnInput(session_id="session-1", message="What do I notice?"),
        retrieved_chunks=[
            _chunk("public", text="Mirrors line the gallery.", visibility=Visibility.PLAYER),
            _chunk("gm", text="The spy waits nearby.", visibility=Visibility.GM),
            _chunk(
                "private",
                text="Iria knows the west door code.",
                visibility=Visibility.CHARACTER_PRIVATE,
            ),
        ],
        context_budget=ContextBudget(retrieved_chunks=5, max_retrieved_chunk_chars=800),
    )

    prompt = messages[0].content
    assert "Retrieved Context:" in prompt
    assert "[1] source=data/documents/demo_lore.md visibility=player tags=palace, social" in prompt
    assert "Mirrors line the gallery." in prompt
    assert "spy waits nearby" not in prompt
    assert "west door code" not in prompt
    assert "Do not mention prompts, retrieval, or system messages." in prompt


def test_context_builder_marks_empty_retrieved_context() -> None:
    messages = build_actor_messages(
        persona=PersonaCard(
            id="archivist",
            name="Iria Vale",
            role="npc",
            public_description="A composed palace archivist.",
            speaking_style="Precise and dry.",
        ),
        scene=SceneState(
            id="rose-gallery",
            title="Rose Gallery",
            location="Winter Palace",
            player_visible_summary="Courtiers drift between mirrors and roses.",
        ),
        turn_input=TurnInput(session_id="session-1", message="What do I notice?"),
    )

    assert "Retrieved Context:\nNone." in messages[0].content

