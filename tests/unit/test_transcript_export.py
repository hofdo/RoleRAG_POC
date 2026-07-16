from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.diagnostics.transcript_export import (
    CONTROLLED_FAILURE_NOTE,
    SUPPORTED_TRANSCRIPT_FORMATS,
    render_transcript,
    render_transcript_html,
    render_transcript_markdown,
)
from app.domain import SessionState, StoredTurn, TurnOutcome
from app.llm.router import ModelProviderName, ModelRoute

_ROUTE = ModelRoute(
    provider=ModelProviderName.LOCAL,
    model="local-model",
    max_tokens=700,
    temperature=0.75,
    reason="default local route",
)
_CREATED_AT = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _session(**overrides: object) -> SessionState:
    defaults: dict[str, object] = {
        "id": "session-1",
        "world_id": "demo_world",
        "active_scene_id": "rose-gallery",
        "active_persona_id": "archivist",
        "player_name": "Avery",
        "content_root": "data",
        "provider": ModelProviderName.LOCAL,
        "created_at": _CREATED_AT,
    }
    defaults.update(overrides)
    return SessionState(**defaults)  # type: ignore[arg-type]


def _turn(
    turn_index: int,
    *,
    user_message: str,
    assistant_message: str,
    persona_id: str = "archivist",
    outcome: TurnOutcome = TurnOutcome.SUCCESS,
) -> StoredTurn:
    return StoredTurn(
        id=turn_index,
        session_id="session-1",
        turn_index=turn_index,
        scene_id="rose-gallery",
        persona_id=persona_id,
        user_message=user_message,
        assistant_message=assistant_message,
        route=_ROUTE,
        created_at=_CREATED_AT,
        outcome=outcome,
    )


def test_render_transcript_markdown_structure_for_a_small_session() -> None:
    session = _session()
    turns = [
        _turn(1, user_message="Hello there", assistant_message="Welcome, traveler."),
        _turn(
            2,
            user_message="What is this place?",
            assistant_message="An old gallery of roses.",
        ),
    ]

    rendered = render_transcript_markdown(session, turns)

    assert rendered == (
        "# Transcript: session-1\n"
        "\n"
        "- **World:** demo_world\n"
        "- **Content root:** data\n"
        "- **Provider:** local\n"
        "- **Created:** 2026-01-01T12:00:00+00:00\n"
        "\n"
        "## Turn 1\n"
        "\n"
        "**Player:** Hello there\n"
        "\n"
        "**archivist:** Welcome, traveler.\n"
        "\n"
        "## Turn 2\n"
        "\n"
        "**Player:** What is this place?\n"
        "\n"
        "**archivist:** An old gallery of roses.\n"
    )


def test_render_transcript_markdown_renders_controlled_failure_distinctly() -> None:
    session = _session()
    turns = [
        _turn(1, user_message="Hello there", assistant_message="Welcome, traveler."),
        _turn(
            2,
            user_message="What happened?",
            assistant_message=(
                "The system could not produce a response that passed validation. "
                "No memory or world state was changed."
            ),
            outcome=TurnOutcome.CONTROLLED_FAILURE,
        ),
    ]

    rendered = render_transcript_markdown(session, turns)

    assert "## Turn 2" in rendered
    assert f"*{CONTROLLED_FAILURE_NOTE}*" in rendered
    # The canned system string is not persona-authored dialogue -- it must never
    # be attributed to the persona (or appear at all) in the rendered transcript.
    assert "could not produce a response" not in rendered
    assert rendered.count("**archivist:**") == 1


def test_render_transcript_markdown_empty_session_is_header_only() -> None:
    session = _session()

    rendered = render_transcript_markdown(session, [])

    assert rendered == (
        "# Transcript: session-1\n"
        "\n"
        "- **World:** demo_world\n"
        "- **Content root:** data\n"
        "- **Provider:** local\n"
        "- **Created:** 2026-01-01T12:00:00+00:00\n"
        "\n"
        "_No turns recorded yet._\n"
    )
    assert "## Turn" not in rendered


def test_render_transcript_markdown_handles_missing_created_at() -> None:
    session = _session(created_at=None)

    rendered = render_transcript_markdown(session, [])

    assert "- **Created:** unknown\n" in rendered


def test_render_transcript_html_escapes_special_characters() -> None:
    session = _session()
    turns = [
        _turn(
            1,
            user_message='<script>alert("hi")</script> & tags',
            assistant_message="Safe reply",
        ),
    ]

    rendered = render_transcript_html(session, turns)

    assert "<script>alert" not in rendered
    assert "&lt;script&gt;alert(&quot;hi&quot;)&lt;/script&gt; &amp; tags" in rendered
    assert "Safe reply" in rendered


def test_render_transcript_html_escapes_session_metadata() -> None:
    session = _session(id='sess"<1>&', world_id="world & <co>")

    rendered = render_transcript_html(session, [])

    assert 'sess"<1>&' not in rendered
    assert "world & <co>" not in rendered
    assert "sess&quot;&lt;1&gt;&amp;" in rendered
    assert "world &amp; &lt;co&gt;" in rendered


def test_render_transcript_html_is_self_contained_and_escapes_controlled_failure() -> None:
    session = _session()
    turns = [
        _turn(
            1,
            user_message="What happened?",
            assistant_message="The system could not produce a response.",
            outcome=TurnOutcome.CONTROLLED_FAILURE,
        )
    ]

    rendered = render_transcript_html(session, turns)

    assert rendered.startswith("<!doctype html>")
    assert "<style>" in rendered and "</style>" in rendered
    # Self-contained: no external assets, fonts, or scripts.
    assert "http://" not in rendered
    assert "https://" not in rendered
    assert "<script" not in rendered.lower()
    assert CONTROLLED_FAILURE_NOTE in rendered
    assert "could not produce a response" not in rendered


def test_render_transcript_html_empty_session_still_valid_document() -> None:
    session = _session()

    rendered = render_transcript_html(session, [])

    assert rendered.startswith("<!doctype html>")
    assert rendered.rstrip().endswith("</html>")
    assert "No turns recorded yet." in rendered
    assert "<section" not in rendered


def test_render_transcript_dispatches_by_format() -> None:
    session = _session()

    assert render_transcript(session, [], "markdown") == render_transcript_markdown(session, [])
    assert render_transcript(session, [], "html") == render_transcript_html(session, [])


def test_render_transcript_rejects_unknown_format() -> None:
    session = _session()

    with pytest.raises(ValueError, match="pdf"):
        render_transcript(session, [], "pdf")


def test_supported_transcript_formats_are_markdown_and_html() -> None:
    assert set(SUPPORTED_TRANSCRIPT_FORMATS) == {"markdown", "html"}
