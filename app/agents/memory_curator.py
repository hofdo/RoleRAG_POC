from __future__ import annotations

import json

from pydantic import ValidationError

from app.domain import MemoryCuratorResult, PersonaCard, SceneState, SessionState
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest
from app.llm.router import ModelRoute
from app.memory.policies import MEMORY_EXTRACTION_PROMPT


class MemoryCuratorOutputError(ValueError):
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
            metadata={"task": "memory_extraction", "session_id": session.id},
        )
        response = await provider.generate(request)
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise MemoryCuratorOutputError("invalid structured output") from exc
        try:
            return MemoryCuratorResult.model_validate(payload)
        except ValidationError as exc:
            raise MemoryCuratorOutputError("invalid structured output") from exc

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
