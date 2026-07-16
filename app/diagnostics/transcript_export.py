"""Transcript Exporter rendering (docs/SIDE_PROJECTS.md, Tier A).

Pure functions: ``SessionState`` + ``StoredTurn`` rows in (as already read by
``app.persistence.repositories`` -- this module does no I/O of its own), a
markdown or HTML string out. Kept separate from ``app.cli`` so the rendering is
unit-testable without going through Typer, matching the repo's "commands stay
thin" convention for routes/orchestration (CLAUDE.md invariant #6).

Visibility boundary (CLAUDE.md invariant #2): a ``StoredTurn`` only ever carries
what the player already saw -- ``user_message``, ``assistant_message``, its
``persona_id``/``scene_id`` attribution, and ``outcome``. This module renders
exactly those fields and nothing else: it never reads ``turn.diagnostics``
(retrieval candidates, stage timings, warnings) and never touches authored
hidden fields (persona ``secrets``/``forbidden_knowledge``, scene
``gm_private_summary``), because those never reach ``StoredTurn`` in the first
place.

A ``CONTROLLED_FAILURE`` turn's ``assistant_message`` is the canned system
string from ``app.orchestration.stages.repair.CONTROLLED_FAILURE_TEXT``, not
persona-authored dialogue -- rendering it under the persona's name would
misattribute a system message as something the character said. Both renderers
branch on the real ``turn.outcome`` enum and substitute a short, clearly
non-diegetic marker instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape as _escape

from app.domain.models import SessionState, StoredTurn, TurnOutcome

#: Rendered in place of a controlled-failure turn's assistant_message.
CONTROLLED_FAILURE_NOTE = "turn ended in controlled failure — no reply served"

#: Values accepted by ``render_transcript`` / the CLI's ``--format`` option.
SUPPORTED_TRANSCRIPT_FORMATS: tuple[str, ...] = ("markdown", "html")

# Kept modest and inline on purpose -- the HTML output must stay a single
# self-contained file with no external assets, fonts, or scripts.
_HTML_STYLE = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 46rem;
  margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; background: #fdfdfd; }
h1 { font-size: 1.5rem; margin-bottom: .25rem; }
dl.meta { display: grid; grid-template-columns: max-content 1fr; gap: .15rem .75rem;
  color: #444; margin: 0 0 1.5rem; }
dl.meta dt { font-weight: 600; }
dl.meta dd { margin: 0; }
p.empty { color: #666; font-style: italic; }
section.turn { margin: 0 0 1.25rem; padding-bottom: 1rem; border-bottom: 1px solid #ddd; }
section.turn h2 { font-size: .8rem; text-transform: uppercase; letter-spacing: .04em;
  color: #888; margin: 0 0 .5rem; font-weight: 600; }
p.player, p.persona { margin: .35rem 0; padding: .5rem .75rem; border-radius: 6px; }
p.player { background: #e8f0fe; }
p.persona { background: #f2f0e9; }
p.failure { font-style: italic; color: #8a3b3b; margin: .35rem 0; }
.speaker { font-weight: 600; margin-right: .35em; }
""".strip()


def _format_created_at(session: SessionState) -> str:
    return session.created_at.isoformat() if session.created_at is not None else "unknown"


def render_transcript_markdown(session: SessionState, turns: Sequence[StoredTurn]) -> str:
    """Render a session's turns as a markdown transcript."""
    lines: list[str] = [
        f"# Transcript: {session.id}",
        "",
        f"- **World:** {session.world_id}",
        f"- **Content root:** {session.content_root}",
        f"- **Provider:** {session.provider.value}",
        f"- **Created:** {_format_created_at(session)}",
    ]
    if not turns:
        lines += ["", "_No turns recorded yet._"]
    for turn in turns:
        lines += ["", f"## Turn {turn.turn_index}", "", f"**Player:** {turn.user_message}"]
        if turn.outcome is TurnOutcome.CONTROLLED_FAILURE:
            lines += ["", f"*{CONTROLLED_FAILURE_NOTE}*"]
        else:
            lines += ["", f"**{turn.persona_id}:** {turn.assistant_message}"]
    return "\n".join(lines) + "\n"


def _render_html_turn(turn: StoredTurn) -> str:
    if turn.outcome is TurnOutcome.CONTROLLED_FAILURE:
        reply_html = f'<p class="failure">{_escape(CONTROLLED_FAILURE_NOTE)}</p>'
    else:
        reply_html = (
            f'<p class="persona"><span class="speaker">{_escape(turn.persona_id)}:</span> '
            f"{_escape(turn.assistant_message)}</p>"
        )
    return (
        '<section class="turn">\n'
        f"<h2>Turn {turn.turn_index}</h2>\n"
        f'<p class="player"><span class="speaker">Player:</span> '
        f"{_escape(turn.user_message)}</p>\n"
        f"{reply_html}\n"
        "</section>"
    )


def render_transcript_html(session: SessionState, turns: Sequence[StoredTurn]) -> str:
    """Render a session's turns as one self-contained HTML file (inline CSS, no
    external assets/fonts/JS). Every piece of session/turn-originated text is
    passed through ``html.escape`` before interpolation."""
    turns_html = (
        "\n".join(_render_html_turn(turn) for turn in turns)
        if turns
        else '<p class="empty">No turns recorded yet.</p>'
    )
    title = f"Transcript: {_escape(session.id)}"
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f"<style>\n{_HTML_STYLE}\n</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{title}</h1>\n"
        '<dl class="meta">\n'
        f"<dt>World</dt><dd>{_escape(session.world_id)}</dd>\n"
        f"<dt>Content root</dt><dd>{_escape(session.content_root)}</dd>\n"
        f"<dt>Provider</dt><dd>{_escape(session.provider.value)}</dd>\n"
        f"<dt>Created</dt><dd>{_escape(_format_created_at(session))}</dd>\n"
        "</dl>\n"
        '<div class="turns">\n'
        f"{turns_html}\n"
        "</div>\n"
        "</body>\n"
        "</html>\n"
    )


def render_transcript(
    session: SessionState, turns: Sequence[StoredTurn], output_format: str
) -> str:
    """Dispatch to the renderer for ``output_format``.

    Callers (the CLI) are expected to validate ``output_format`` against
    ``SUPPORTED_TRANSCRIPT_FORMATS`` first and turn a bad value into a
    user-facing error; this raises ``ValueError`` rather than duplicating that
    presentation concern here.
    """
    if output_format == "markdown":
        return render_transcript_markdown(session, turns)
    if output_format == "html":
        return render_transcript_html(session, turns)
    raise ValueError(f"Unsupported transcript format: {output_format!r}")
