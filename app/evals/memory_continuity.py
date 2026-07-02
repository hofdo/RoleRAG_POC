from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from app.agents.memory_curator import MemoryCurator
from app.domain import MemoryEpisode, TurnInput, Visibility
from app.evals.fixtures import (
    AlwaysAcceptCritic,
    DeterministicKeywordEmbeddingProvider,
    EvalLoader,
    build_eval_fixture,
)
from app.llm.provider import LlmProvider, LlmRequest, LlmResponse
from app.memory import MemoryEpisodeStore, MemoryIndexer, RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator, TurnOrchestratorConfig
from app.persistence import (
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
)
from app.persistence.sqlite import connect_sqlite, initialize_database
from app.rag import ActorContextRetriever, InMemoryVectorStore, RagChunk, RagCollection, Retriever

_DIALOGUE_WINDOW = 4
_PROMISE_SUMMARY = "The player promised to return before dawn for the archive key."
_GM_ONLY_SUMMARY = "GM_ONLY_MEMORY: A loyalist spy listens behind the archive screen."
_PRIVATE_SUMMARY = "PRIVATE_MEMORY: Iria hid the archive cipher inside the gallery clock."
_EARLY_DIALOGUE_MARKER = "TURN_ONE_DIALOGUE_MARKER"
_FILLER_MESSAGE = "Good evening, Iria."
_ACTOR_RESPONSE = "Iria inclines her head, recalling the promise about the archive key."
_LATE_RECALL_QUESTION = "Do you remember my promise about the archive key before dawn?"


class MemoryContinuityResult(BaseModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)


class _TaskAwareFakeProvider(LlmProvider):
    def __init__(self) -> None:
        self.actor_requests: list[LlmRequest] = []
        self.memory_requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        if request.metadata.get("task") == "memory_extraction":
            self.memory_requests.append(request)
            text = self._memory_response(request)
        else:
            self.actor_requests.append(request)
            text = _ACTOR_RESPONSE
        return LlmResponse(text=text, provider="fake", model=request.model)

    def _memory_response(self, request: LlmRequest) -> str:
        completed_turn = request.messages[-1].content
        if _EARLY_DIALOGUE_MARKER in completed_turn:
            return _memory_payload(_PROMISE_SUMMARY, visibility=Visibility.PLAYER)
        if "GM_ONLY_TURN" in completed_turn:
            return _memory_payload(_GM_ONLY_SUMMARY, visibility=Visibility.GM)
        if "PRIVATE_TURN" in completed_turn:
            return _memory_payload(_PRIVATE_SUMMARY, visibility=Visibility.CHARACTER_PRIVATE)
        if "INVALID_VISIBILITY_TURN" in completed_turn:
            return json.dumps(
                {
                    "write_memory": True,
                    "memories": [
                        {
                            "summary": "Invalid visibility should be rejected.",
                            "visibility": "secret",
                            "importance": 4,
                            "tags": ["archive"],
                            "scene_id": "eval-rose-gallery",
                        }
                    ],
                    "reason": "Exercise structured output validation.",
                }
            )
        return json.dumps(
            {
                "write_memory": False,
                "memories": [],
                "reason": "Greeting-only exchange.",
            }
        )


def evaluate_memory_continuity() -> MemoryContinuityResult:
    return asyncio.run(_evaluate_memory_continuity())


async def _evaluate_memory_continuity() -> MemoryContinuityResult:
    fixture = build_eval_fixture()
    with tempfile.TemporaryDirectory(prefix="rolerag-memory-continuity-") as temp_dir:
        connection = connect_sqlite(Path(temp_dir) / "memory-continuity.db")
        initialize_database(connection)
        session_repository = SQLiteSessionRepository(connection)
        turn_repository = SQLiteTurnRepository(connection)
        memory_repository = SQLiteMemoryRepository(connection)
        memory_store = MemoryEpisodeStore(memory_repository=memory_repository)
        session_repository.create_session(fixture.session)

        embedding_provider = DeterministicKeywordEmbeddingProvider()
        live_vector_store = InMemoryVectorStore()
        live_indexer = MemoryIndexer(
            memory_store=memory_store,
            embedding_provider=embedding_provider,
            vector_store=live_vector_store,
        )
        provider = _TaskAwareFakeProvider()
        orchestrator = TurnOrchestrator(
            loader=EvalLoader(fixture),
            provider=provider,
            critic_agent=AlwaysAcceptCritic(),
            session_repository=session_repository,
            turn_repository=turn_repository,
            recent_dialogue_store=RecentDialogueStore(
                turn_repository=turn_repository,
                recent_turns=_DIALOGUE_WINDOW,
            ),
            memory_store=memory_store,
            memory_curator=MemoryCurator(),
            memory_indexer=live_indexer,
            actor_context_retriever=_build_actor_retriever(
                embedding_provider=embedding_provider,
                vector_store=live_vector_store,
            ),
            config=TurnOrchestratorConfig(
                local_model=fixture.local_route.model,
                cloud_model=fixture.cloud_route.model,
                local_max_tokens=fixture.local_route.max_tokens,
                cloud_max_tokens=fixture.cloud_route.max_tokens,
                local_temperature=fixture.local_route.temperature,
                cloud_temperature=fixture.cloud_route.temperature,
            ),
        )

        turn_results = []
        for message in _turn_messages():
            turn_results.append(
                await orchestrator.run_turn(
                    turn_input=TurnInput(session_id=fixture.session.id, message=message)
                )
            )

        persisted_memories = memory_store.list_memories_for_session(fixture.session.id)
        promise = _find_memory(persisted_memories, _PROMISE_SUMMARY)
        late_prompt = provider.actor_requests[-1]
        late_system_prompt = late_prompt.messages[0].content
        late_history = "\n".join(message.content for message in late_prompt.messages[1:-1])

        fresh_vector_store = InMemoryVectorStore()
        fresh_indexer = MemoryIndexer(
            memory_store=memory_store,
            embedding_provider=embedding_provider,
            vector_store=fresh_vector_store,
        )
        fresh_indexer.reindex_session(fixture.session.id)
        distractor_ids = _seed_scoped_distractors(
            vector_store=fresh_vector_store,
            embedding_provider=embedding_provider,
        )
        reindexed_chunks = _build_actor_retriever(
            embedding_provider=embedding_provider,
            vector_store=fresh_vector_store,
        ).retrieve_for_actor(
            query=_LATE_RECALL_QUESTION,
            world_id=fixture.world.id,
            session_id=fixture.session.id,
            persona_id=fixture.primary_persona.id,
            scene_id=fixture.scene.id,
            top_k=10,
        )
        persisted_turn_count = turn_repository.count_turns(fixture.session.id)
        connection.close()

    reindexed_ids = {chunk.id for chunk in reindexed_chunks}
    actor_prompts = provider.actor_requests
    plateau_requests = actor_prompts[8:15]
    all_actor_text = "\n".join(
        message.content for request in actor_prompts for message in request.messages
    )
    checks = {
        "sixteen_turns_complete": len(turn_results) == 16
        and persisted_turn_count == 16,
        "actor_history_never_exceeds_four_prior_turns": all(
            (len(request.messages) - 2) // 2 <= _DIALOGUE_WINDOW
            for request in actor_prompts
        ),
        "early_dialogue_markers_leave_recent_history": _EARLY_DIALOGUE_MARKER not in late_history,
        "actor_prompt_message_count_plateaus": len(
            {len(request.messages) for request in plateau_requests}
        )
        == 1,
        "actor_prompt_chars_plateau": len({_prompt_chars(request) for request in plateau_requests})
        == 1,
        "old_promise_recalled_from_durable_memory": (
            promise is not None
            and _PROMISE_SUMMARY in late_system_prompt
            and _PROMISE_SUMMARY not in late_history
        ),
        "hidden_memories_excluded_from_actor_prompts": (
            _GM_ONLY_SUMMARY not in all_actor_text and _PRIVATE_SUMMARY not in all_actor_text
        ),
        "greeting_only_turns_skip_memory": all(
            result.memory_written is False for result in turn_results[4:15]
        ),
        "invalid_visibility_skipped": any(
            warning.startswith("memory curation skipped:") for warning in turn_results[3].warnings
        )
        and len(persisted_memories) == 3,
        "fresh_reindex_recovers_sqlite_promise": (
            promise is not None and promise.id in reindexed_ids
        ),
        "reindexed_recovery_excludes_scoped_distractors": reindexed_ids.isdisjoint(distractor_ids),
    }
    return MemoryContinuityResult(passed=all(checks.values()), checks=checks)


def _turn_messages() -> list[str]:
    return [
        f"{_EARLY_DIALOGUE_MARKER}: I promise to return before dawn for the archive key.",
        "GM_ONLY_TURN: The hidden observer remains unknown to the player.",
        "PRIVATE_TURN: Iria keeps a private cipher location to herself.",
        "INVALID_VISIBILITY_TURN: Exercise rejected memory output.",
        *([_FILLER_MESSAGE] * 11),
        _LATE_RECALL_QUESTION,
    ]


def _memory_payload(summary: str, *, visibility: Visibility) -> str:
    return json.dumps(
        {
            "write_memory": True,
            "memories": [
                {
                    "summary": summary,
                    "visibility": visibility.value,
                    "importance": 4,
                    "tags": ["archive", "promise"],
                    "scene_id": "eval-rose-gallery",
                    "actor_id": "iria-vale",
                }
            ],
            "reason": "Durable continuity fixture.",
        }
    )


def _build_actor_retriever(
    *,
    embedding_provider: DeterministicKeywordEmbeddingProvider,
    vector_store: InMemoryVectorStore,
) -> ActorContextRetriever:
    return ActorContextRetriever(
        retriever=Retriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            default_top_k=5,
        )
    )


def _find_memory(memories: Sequence[MemoryEpisode], summary: str) -> MemoryEpisode | None:
    return next((memory for memory in memories if memory.summary == summary), None)


def _prompt_chars(request: LlmRequest) -> int:
    return sum(len(message.content) for message in request.messages)


def _seed_scoped_distractors(
    *,
    vector_store: InMemoryVectorStore,
    embedding_provider: DeterministicKeywordEmbeddingProvider,
) -> set[str]:
    chunks = [
        (
            RagCollection.SESSION_MEMORY,
            RagChunk(
                id="continuity-wrong-session",
                source="memory_episode:continuity-wrong-session",
                source_type=RagCollection.SESSION_MEMORY.value,
                text=_PROMISE_SUMMARY,
                visibility=Visibility.PLAYER,
                session_id="other-session",
            ),
        ),
        (
            RagCollection.CANON_LORE,
            RagChunk(
                id="continuity-wrong-world",
                source="continuity-wrong-world.md",
                source_type=RagCollection.CANON_LORE.value,
                text=_PROMISE_SUMMARY,
                visibility=Visibility.PLAYER,
                world_id="other-world",
            ),
        ),
        (
            RagCollection.PERSONA_MEMORY,
            RagChunk(
                id="continuity-wrong-persona",
                source="continuity-wrong-persona.md",
                source_type=RagCollection.PERSONA_MEMORY.value,
                text=_PROMISE_SUMMARY,
                visibility=Visibility.PLAYER,
                persona_id="other-persona",
            ),
        ),
    ]
    for collection, chunk in chunks:
        vector_store.ensure_collection(collection, embedding_provider.dimension)
        vector_store.upsert_chunks(collection, [chunk], [embedding_provider.embed_text(chunk.text)])
    return {chunk.id for _, chunk in chunks}
