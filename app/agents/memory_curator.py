from __future__ import annotations

from pydantic import ValidationError

from app.domain import MemoryCuratorResult, PersonaCard, SceneState, SessionState
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest
from app.llm.router import ModelRoute
from app.llm.structured_output import (
    StructuredOutputError,
    StructuredOutputParseError,
    parse_single_json_object,
)
from app.memory.consolidation import CONSOLIDATION_PROMPT
from app.memory.policies import MEMORY_EXTRACTION_PROMPT


class MemoryCuratorOutputError(StructuredOutputError):
    pass


class MemoryConsolidationError(RuntimeError):
    pass


class MemoryCurator:
    async def curate(
        self,
        *,
        provider: LlmProvider,
        route: ModelRoute,
        session: SessionState,
        scene: SceneState,
        persona: PersonaCard,
        user_message: str,
        assistant_message: str,
    ) -> MemoryCuratorResult:
        request = LlmRequest(
            messages=[
                LlmMessage(role="system", content=MEMORY_EXTRACTION_PROMPT),
                LlmMessage(
                    role="user",
                    content=self._build_context(
                        session=session,
                        scene=scene,
                        persona=persona,
                        user_message=user_message,
                        assistant_message=assistant_message,
                    ),
                ),
            ],
            model=route.model,
            max_tokens=route.max_tokens,
            temperature=route.temperature,
            response_format="json",
            response_schema=MemoryCuratorResult.model_json_schema(),
            metadata={"task": "memory_extraction", "session_id": session.id},
        )
        response = await provider.generate(request)
        try:
            payload = parse_single_json_object(response.text)
        except StructuredOutputParseError as exc:
            raise MemoryCuratorOutputError(
                "invalid structured output", category="parse", raw_text=response.text
            ) from exc
        try:
            return MemoryCuratorResult.model_validate(payload)
        except ValidationError as exc:
            raise MemoryCuratorOutputError(
                "invalid structured output", category="schema", raw_text=response.text
            ) from exc

    async def consolidate(
        self,
        *,
        provider: LlmProvider,
        route: ModelRoute,
        summaries: list[str],
    ) -> str:
        """Compress past memory summaries into one. Plain-text completion (no JSON)
        for reliability; the caller falls back to a deterministic roll-up on error."""
        request = LlmRequest(
            messages=[
                LlmMessage(role="system", content=CONSOLIDATION_PROMPT),
                LlmMessage(
                    role="user",
                    content="\n".join(f"- {summary}" for summary in summaries),
                ),
            ],
            model=route.model,
            max_tokens=route.max_tokens,
            temperature=route.temperature,
            metadata={"task": "memory_consolidation"},
        )
        response = await provider.generate(request)
        text = response.text.strip()
        if not text:
            raise MemoryConsolidationError("consolidation produced empty text")
        return text

    def _build_context(
        self,
        *,
        session: SessionState,
        scene: SceneState,
        persona: PersonaCard,
        user_message: str,
        assistant_message: str,
    ) -> str:
        return "\n".join(
            [
                f"Session id: {session.id}",
                f"Scene id: {scene.id}",
                f"Scene title: {scene.title}",
                f"Persona id: {persona.id}",
                f"Persona name: {persona.name}",
                "Completed turn:",
                f"User: {user_message}",
                f"Assistant: {assistant_message}",
            ]
        )
