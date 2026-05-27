from __future__ import annotations

from collections.abc import Sequence

from app.domain import PersonaCard, SceneState, StoredTurn, TurnInput
from app.llm.provider import LlmMessage


def build_actor_messages(
    *,
    persona: PersonaCard,
    scene: SceneState,
    turn_input: TurnInput,
    recent_turns: Sequence[StoredTurn] = (),
) -> list[LlmMessage]:
    prompt_lines = [
        "You are roleplaying as the active character.",
        f"Persona name: {persona.name}",
        f"Persona role: {persona.role}",
        f"Public description: {persona.public_description}",
        f"Speaking style: {persona.speaking_style}",
    ]

    if persona.values:
        prompt_lines.append(f"Values: {', '.join(persona.values)}")
    if persona.goals:
        prompt_lines.append(f"Goals: {', '.join(persona.goals)}")

    prompt_lines.extend(
        [
            f"Scene title: {scene.title}",
            f"Location: {scene.location}",
        ]
    )

    if scene.current_time:
        prompt_lines.append(f"Current time: {scene.current_time}")

    prompt_lines.append(f"Visible scene summary: {scene.player_visible_summary}")

    if scene.recent_events:
        prompt_lines.append(f"Recent events: {'; '.join(scene.recent_events)}")

    prompt_lines.append("Respond in character using only the provided visible context.")

    messages = [LlmMessage(role="system", content="\n".join(prompt_lines))]

    for stored_turn in recent_turns:
        messages.append(LlmMessage(role="user", content=stored_turn.user_message))
        messages.append(LlmMessage(role="assistant", content=stored_turn.assistant_message))

    messages.append(LlmMessage(role="user", content=turn_input.message))
    return messages
