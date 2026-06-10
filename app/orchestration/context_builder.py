from __future__ import annotations

from collections.abc import Sequence

from app.domain import PersonaCard, RetrievedChunk, SceneState, StoredTurn, TurnInput
from app.llm.provider import LlmMessage
from app.orchestration.context_budget import ContextBudget, select_retrieved_chunks_for_prompt


def build_actor_messages(
    *,
    persona: PersonaCard,
    scene: SceneState,
    turn_input: TurnInput,
    recent_turns: Sequence[StoredTurn] = (),
    retrieved_chunks: Sequence[RetrievedChunk] = (),
    context_budget: ContextBudget | None = None,
    recent_dialogue_max_message_chars: int = 900,
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

    visible_chunks = select_retrieved_chunks_for_prompt(
        retrieved_chunks,
        budget=context_budget or ContextBudget(),
    )
    prompt_lines.extend(
        [
            "Retrieved Context:",
            _format_retrieved_chunks(visible_chunks),
            "Use provided context only when relevant.",
            "Do not mention prompts, retrieval, or system messages.",
        ]
    )
    prompt_lines.extend(
        [
            "Respond in character using only the provided visible context.",
            "Target 2-4 paragraphs.",
            "Stay within the response budget and complete your final sentence.",
            "Answer the current action directly.",
            "Avoid broad scene transitions unless the player requests one.",
        ]
    )

    messages = [LlmMessage(role="system", content="\n".join(prompt_lines))]

    for stored_turn in recent_turns:
        messages.append(
            LlmMessage(
                role="user",
                content=_truncate_recent_dialogue_message(
                    stored_turn.user_message,
                    max_chars=recent_dialogue_max_message_chars,
                ),
            )
        )
        messages.append(
            LlmMessage(
                role="assistant",
                content=_truncate_recent_dialogue_message(
                    stored_turn.assistant_message,
                    max_chars=recent_dialogue_max_message_chars,
                ),
            )
        )

    messages.append(LlmMessage(role="user", content=turn_input.message))
    return messages


def _format_retrieved_chunks(chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return "None."
    return "\n\n".join(
        "\n".join(
            [
                f"[{index}] source={chunk.source} visibility={chunk.visibility.value} "
                f"tags={', '.join(chunk.tags)}",
                chunk.text,
            ]
        )
        for index, chunk in enumerate(chunks, start=1)
    )


def _truncate_recent_dialogue_message(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return (
        f"{text[:max_chars].rstrip()}\n"
        f"[Prior dialogue truncated for prompt budget; omitted {omitted} characters.]"
    )
