from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.agents.critic_agent import CriticAgent
from app.agents.memory_curator import MemoryCurator
from app.domain import (
    CriticResult,
    MemoryEpisode,
    PersonaCard,
    RetrievedChunk,
    SceneState,
    SessionState,
    TurnInput,
)
from app.domain.visibility import Visibility
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from app.llm.router import ModelProviderName, ModelRoute
from app.memory import MemoryEpisodeStore, RecentDialogueStore
from app.orchestration.context_budget import ContextBudget
from app.orchestration.context_builder import build_actor_messages
from app.orchestration.turn_orchestrator import TurnOrchestrator, TurnOrchestratorConfig
from app.persistence import (
    DemoWorldRecord,
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
)
from app.persistence.sqlite import connect_sqlite, initialize_database
from app.rag import ActorContextRetriever, InMemoryVectorStore, RagChunk, RagCollection, Retriever
from app.rag.embeddings import EmbeddingProvider
from app.rag.retriever import build_retrieval_query


class DeterministicKeywordEmbeddingProvider(EmbeddingProvider):
    _KEYWORDS = (
        "gallery",
        "mirror",
        "archive",
        "dawn",
        "fountain",
        "spy",
        "clock",
        "west",
    )

    @property
    def dimension(self) -> int:
        return len(self._KEYWORDS)

    def embed_text(self, text: str) -> list[float]:
        normalized = text.lower()
        return [float(normalized.count(keyword)) for keyword in self._KEYWORDS]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class SequencedFakeProvider(LlmProvider):
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        index = len(self.requests) - 1
        text = self.responses[index] if index < len(self.responses) else self.responses[-1]
        return LlmResponse(
            text=text,
            provider="fake",
            model=request.model,
            usage={"total_tokens": 0},
            finish_reason="stop",
        )


class TaskRoutedRecordingProvider(LlmProvider):
    """Records every request and answers by ``request.metadata["task"]``.

    Used to run a *real* CriticAgent + MemoryCurator (not stubs) through the
    orchestrator so the provider-binding eval can assert, at the request-object
    level, which provider each task actually landed on and that no hidden
    fixture content ever appears in the recorded requests.
    """

    def __init__(
        self,
        *,
        actor_text: str,
        critic_text: str = '{"accepted": true, "issues": [], "repair_instruction": null}',
        memory_text: str = '{"write_memory": false, "memories": [], "reason": "nothing durable"}',
    ) -> None:
        self.actor_text = actor_text
        self.critic_text = critic_text
        self.memory_text = memory_text
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        task = request.metadata.get("task") if request.metadata else None
        if task == "critic":
            text = self.critic_text
        elif task == "memory_extraction":
            text = self.memory_text
        else:
            text = self.actor_text
        return LlmResponse(
            text=text,
            provider="fake",
            model=request.model,
            usage={"total_tokens": 0},
            finish_reason="stop",
        )


class NullMemoryIndexer:
    """Discards memory writes; the eval only needs the extraction call recorded."""

    def index_memories(self, memories: list[MemoryEpisode]) -> object:
        return None

    def unindex(self, memory_ids: Sequence[str]) -> None:
        raise AssertionError("unindex not expected in provider-binding eval")


class AlwaysAcceptCritic:
    async def evaluate(self, **_: object) -> CriticResult:
        return CriticResult(accepted=True)

    def build_local_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        rejected_draft: str,
        issues: list[str],
        repair_instruction: str | None,
    ) -> list[LlmMessage]:
        raise AssertionError("repair should not be used in eval fixtures")

    def build_cloud_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        issues: list[str],
    ) -> list[LlmMessage]:
        raise AssertionError("repair should not be used in eval fixtures")


class EvalLoader:
    def __init__(self, fixture: "EvalFixture") -> None:
        self.fixture = fixture

    def load_world(self, world_id: str) -> DemoWorldRecord:
        if world_id != self.fixture.world.id:
            raise ValueError(f"Unknown world: {world_id}")
        return self.fixture.world

    def load_persona(self, persona_id: str) -> PersonaCard:
        if persona_id == self.fixture.primary_persona.id:
            return self.fixture.primary_persona
        if persona_id == self.fixture.secondary_persona.id:
            return self.fixture.secondary_persona
        raise ValueError(f"Unknown persona: {persona_id}")

    def load_scene(self, scene_id: str) -> SceneState:
        if scene_id != self.fixture.scene.id:
            raise ValueError(f"Unknown scene: {scene_id}")
        return self.fixture.scene


@dataclass(frozen=True)
class EvalFixture:
    world: DemoWorldRecord
    scene: SceneState
    session: SessionState
    primary_persona: PersonaCard
    secondary_persona: PersonaCard
    canon_lore_chunk_id: str
    public_lore_chunk_id: str
    public_lore_text: str
    session_memory_chunk_id: str
    persona_memory_chunk_id: str
    persona_memory_text: str
    wrong_world_chunk_id: str
    wrong_world_text: str
    wrong_session_chunk_id: str
    wrong_session_text: str
    wrong_persona_chunk_id: str
    wrong_persona_text: str
    gm_only_chunk_id: str
    gm_only_lore_text: str
    character_private_chunk_id: str
    character_private_text: str
    relevant_memory_chunk_id: str
    relevant_memory_text: str
    irrelevant_memory_chunk_id: str
    irrelevant_memory_text: str
    scene_gm_only_text: str
    primary_persona_secret: str
    primary_persona_private_description: str
    primary_persona_forbidden_knowledge: str
    local_route: ModelRoute
    cloud_route: ModelRoute
    memory_route: ModelRoute
    embedding_provider: DeterministicKeywordEmbeddingProvider
    vector_store: InMemoryVectorStore
    important_turn_user_message: str
    important_turn_assistant_message: str

    def create_actor_context_retriever(self) -> ActorContextRetriever:
        return ActorContextRetriever(
            retriever=Retriever(
                embedding_provider=self.embedding_provider,
                vector_store=self.vector_store,
                default_top_k=3,
            )
        )

    def retrieve_actor_chunks(self, *, top_k: int = 3) -> list[RetrievedChunk]:
        return self.create_actor_context_retriever().retrieve_for_actor(
            query=self.build_retrieval_query(),
            world_id=self.world.id,
            session_id=self.session.id,
            persona_id=self.primary_persona.id,
            scene_id=self.scene.id,
            top_k=top_k,
        )

    def retrieve_actor_chunks_for_query(
        self,
        *,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        return self.create_actor_context_retriever().retrieve_for_actor(
            query=query,
            world_id=self.world.id,
            session_id=self.session.id,
            persona_id=self.primary_persona.id,
            scene_id=self.scene.id,
            top_k=top_k,
        )

    def build_actor_prompt(self) -> str:
        messages = build_actor_messages(
            persona=self.primary_persona,
            scene=self.scene,
            turn_input=TurnInput(session_id=self.session.id, message="What do I notice?"),
            retrieved_chunks=self.retrieve_actor_chunks(),
            context_budget=ContextBudget(retrieved_chunks=3, max_retrieved_chunk_chars=800),
        )
        return messages[0].content

    def build_retrieval_query(self) -> str:
        return build_retrieval_query(
            user_message=(
                "In the gallery, can I still trust your promise to keep the archive key "
                "ready before dawn?"
            ),
            scene=self.scene,
            persona=self.primary_persona,
            recent_turns=(),
        )

    def build_world_knowledge_query(self) -> str:
        return "\n".join(
            [
                "Scene: Rose Gallery",
                "Location: Winter Palace",
                "Question: What is publicly known about the mirrored columns and west door?",
            ]
        )

    def build_session_memory_query(self) -> str:
        return "What did I promise about the archive key before dawn in the gallery?"

    def build_persona_memory_query(self) -> str:
        return "What does Iria remember about the archive mirror before dawn?"

    def build_isolation_query(self) -> str:
        return "gallery mirror archive dawn fountain spy clock west"

    def build_memory_provider(self, *, kind: str) -> SequencedFakeProvider:
        responses = {
            "important": """
            {
              "write_memory": true,
              "memories": [
                {
                  "summary": "The player promised to return before dawn for the archive key.",
                  "visibility": "player",
                  "importance": 4,
                  "tags": ["promise", "archive", "dawn"],
                  "scene_id": "eval-rose-gallery",
                  "actor_id": "iria-vale"
                }
              ],
              "reason": "The promise should affect later access."
            }
            """,
            "trivial": """
            {
              "write_memory": false,
              "memories": [],
              "reason": "Greeting-only exchange."
            }
            """,
            "invalid_visibility": """
            {
              "write_memory": true,
              "memories": [
                {
                  "summary": "Bad visibility.",
                  "visibility": "secret",
                  "importance": 4,
                  "tags": [],
                  "scene_id": "eval-rose-gallery",
                  "actor_id": "iria-vale"
                }
              ],
              "reason": "Invalid."
            }
            """,
            "write_without_candidates": """
            {
              "write_memory": true,
              "memories": [],
              "reason": "Invalid shape."
            }
            """,
        }
        return SequencedFakeProvider([responses[kind]])

    def build_orchestrator(
        self,
        *,
        session_provider: ModelProviderName,
        actor_response_text: str,
    ) -> tuple[TurnOrchestrator, SequencedFakeProvider, SequencedFakeProvider]:
        temp_dir = Path(tempfile.mkdtemp(prefix="rolerag-evals-"))
        connection = connect_sqlite(temp_dir / "evals.db")
        initialize_database(connection)
        session_repository = SQLiteSessionRepository(connection)
        turn_repository = SQLiteTurnRepository(connection)
        session_repository.create_session(
            self.session.model_copy(update={"provider": session_provider})
        )
        local_provider = SequencedFakeProvider([actor_response_text])
        cloud_provider = SequencedFakeProvider(["Cloud answer"])
        orchestrator = TurnOrchestrator(
            loader=EvalLoader(self),
            provider=local_provider,
            cloud_provider=cloud_provider,
            critic_agent=AlwaysAcceptCritic(),
            session_repository=session_repository,
            turn_repository=turn_repository,
            recent_dialogue_store=RecentDialogueStore(
                turn_repository=turn_repository,
                recent_turns=8,
            ),
            actor_context_retriever=self.create_actor_context_retriever(),
            config=TurnOrchestratorConfig(
                local_model=self.local_route.model,
                cloud_model=self.cloud_route.model,
                local_max_tokens=self.local_route.max_tokens,
                cloud_max_tokens=self.cloud_route.max_tokens,
                local_temperature=self.local_route.temperature,
                cloud_temperature=self.cloud_route.temperature,
            ),
        )
        return orchestrator, local_provider, cloud_provider

    def build_full_stack_orchestrator(
        self,
        *,
        session_provider: ModelProviderName,
        actor_response_text: str,
    ) -> tuple[TurnOrchestrator, TaskRoutedRecordingProvider, TaskRoutedRecordingProvider]:
        """Orchestrator wired with a *real* CriticAgent + MemoryCurator (not stubs).

        Used by the provider-binding eval to prove -- at the request-object level --
        that critic and memory-extraction calls follow the session provider like the
        actor call does, and that hidden persona/scene fields never appear in any
        request sent to the recording providers. Both providers are fakes (task-routed
        by ``request.metadata["task"]``); no network or real model is involved.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="rolerag-evals-fullstack-"))
        connection = connect_sqlite(temp_dir / "evals.db")
        initialize_database(connection)
        session_repository = SQLiteSessionRepository(connection)
        turn_repository = SQLiteTurnRepository(connection)
        memory_repository = SQLiteMemoryRepository(connection)
        session_repository.create_session(
            self.session.model_copy(update={"provider": session_provider})
        )
        local_provider = TaskRoutedRecordingProvider(actor_text=actor_response_text)
        cloud_provider = TaskRoutedRecordingProvider(actor_text="Cloud answer")
        orchestrator = TurnOrchestrator(
            loader=EvalLoader(self),
            provider=local_provider,
            cloud_provider=cloud_provider,
            critic_agent=CriticAgent(),
            memory_store=MemoryEpisodeStore(memory_repository=memory_repository),
            memory_curator=MemoryCurator(),
            memory_indexer=NullMemoryIndexer(),
            session_repository=session_repository,
            turn_repository=turn_repository,
            recent_dialogue_store=RecentDialogueStore(
                turn_repository=turn_repository,
                recent_turns=8,
            ),
            actor_context_retriever=self.create_actor_context_retriever(),
            config=TurnOrchestratorConfig(
                local_model=self.local_route.model,
                cloud_model=self.cloud_route.model,
                local_max_tokens=self.local_route.max_tokens,
                cloud_max_tokens=self.cloud_route.max_tokens,
                local_temperature=self.local_route.temperature,
                cloud_temperature=self.cloud_route.temperature,
            ),
        )
        return orchestrator, local_provider, cloud_provider


def build_eval_fixture() -> EvalFixture:
    world = DemoWorldRecord(
        id="eval-winter-palace",
        name="Evaluation Winter Palace",
        default_scene_id="eval-rose-gallery",
        persona_ids=["iria-vale", "corvin-hale"],
        scene_ids=["eval-rose-gallery"],
    )
    scene_gm_only_text = "A loyalist listener watches from the south alcove."
    primary_persona_private_description = "She hides a dangerous debt to the old archive ring."
    primary_persona_secret = "She hid a cipher inside the gallery clock."
    primary_persona_forbidden_knowledge = "The archive key is hidden beneath the rose seal."
    scene = SceneState(
        id="eval-rose-gallery",
        title="Rose Gallery",
        location="Winter Palace",
        current_time="late evening",
        active_personas=["iria-vale", "corvin-hale"],
        player_visible_summary="Mirrored columns frame the winter roses and the guarded west door.",
        gm_private_summary=scene_gm_only_text,
        recent_events=["The regent departed moments ago."],
    )
    primary_persona = PersonaCard(
        id="iria-vale",
        name="Iria Vale",
        role="npc",
        public_description="A composed palace archivist who speaks with exacting calm.",
        private_description=primary_persona_private_description,
        speaking_style="Precise, dry, and quietly skeptical.",
        goals=["protect the archive", "test promises before trusting them"],
        secrets=[primary_persona_secret],
        forbidden_knowledge=[primary_persona_forbidden_knowledge],
    )
    secondary_persona = PersonaCard(
        id="corvin-hale",
        name="Corvin Hale",
        role="npc",
        public_description="A formal envoy who hides impatience behind court manners.",
        private_description="He reports directly to the regent's quiet network.",
        speaking_style="Formal, guarded, and slightly severe.",
    )
    session = SessionState(
        id="eval-session",
        world_id=world.id,
        active_scene_id=scene.id,
        active_persona_id=primary_persona.id,
        player_name="Avery",
    )

    local_route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=700,
        temperature=0.75,
        reason="default local route",
    )
    cloud_route = ModelRoute(
        provider=ModelProviderName.CLOUD,
        model="cloud-model",
        max_tokens=1000,
        temperature=0.65,
        reason="deterministic cloud route",
    )
    memory_route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=700,
        temperature=0.0,
        reason="memory extraction stays local",
    )

    embedding_provider = DeterministicKeywordEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    public_lore_text = (
        "The Rose Gallery's mirrored columns and guarded west door make secret meetings "
        "difficult to hide."
    )
    gm_only_lore_text = "A loyalist listener waits in the south alcove to overhear archive talk."
    character_private_text = "Iria alone knows the cipher is hidden inside the gallery clock."
    relevant_memory_text = (
        "The player promised to return before dawn for the archive key in the gallery."
    )
    persona_memory_text = "Iria remembers that the archive mirror must be checked before dawn."
    irrelevant_memory_text = "The player admired the fountain in the lower garden."
    wrong_world_text = "Another world's gallery mirror hides an archive clock beside the west door."
    wrong_session_text = "Another session promised to check the gallery archive before dawn."
    wrong_persona_text = "Corvin remembers the gallery archive mirror and west door."

    _seed_chunk(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        collection=RagCollection.CANON_LORE,
        chunk=RagChunk(
            id="eval-public-lore",
            source="eval-public-lore.md",
            source_type="lore",
            text=public_lore_text,
            visibility=Visibility.PLAYER,
            tags=["gallery", "archive"],
            world_id=world.id,
        ),
    )
    _seed_chunk(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        collection=RagCollection.CANON_LORE,
        chunk=RagChunk(
            id="eval-gm-lore",
            source="eval-gm-lore.md",
            source_type="lore",
            text=gm_only_lore_text,
            visibility=Visibility.GM,
            tags=["spy"],
            world_id=world.id,
        ),
    )
    _seed_chunk(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        collection=RagCollection.PERSONA_MEMORY,
        chunk=RagChunk(
            id="eval-persona-memory",
            source="eval-persona-memory.md",
            source_type="memory",
            text=persona_memory_text,
            visibility=Visibility.PLAYER,
            tags=["archive", "mirror", "dawn"],
            persona_id=primary_persona.id,
        ),
    )
    _seed_chunk(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        collection=RagCollection.PERSONA_MEMORY,
        chunk=RagChunk(
            id="eval-private-memory",
            source="eval-private-memory.md",
            source_type="memory",
            text=character_private_text,
            visibility=Visibility.CHARACTER_PRIVATE,
            tags=["clock", "cipher"],
            persona_id=primary_persona.id,
        ),
    )
    _seed_chunk(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        collection=RagCollection.SESSION_MEMORY,
        chunk=RagChunk(
            id="eval-relevant-memory",
            source="eval-relevant-memory.md",
            source_type="memory",
            text=relevant_memory_text,
            visibility=Visibility.PLAYER,
            tags=["archive", "dawn", "promise"],
            session_id=session.id,
        ),
    )
    _seed_chunk(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        collection=RagCollection.SESSION_MEMORY,
        chunk=RagChunk(
            id="eval-irrelevant-memory",
            source="eval-irrelevant-memory.md",
            source_type="memory",
            text=irrelevant_memory_text,
            visibility=Visibility.PLAYER,
            tags=["fountain", "garden"],
            session_id=session.id,
        ),
    )
    _seed_chunk(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        collection=RagCollection.CANON_LORE,
        chunk=RagChunk(
            id="eval-wrong-world-lore",
            source="eval-wrong-world-lore.md",
            source_type="lore",
            text=wrong_world_text,
            visibility=Visibility.PLAYER,
            tags=["gallery", "archive"],
            world_id="other-world",
        ),
    )
    _seed_chunk(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        collection=RagCollection.SESSION_MEMORY,
        chunk=RagChunk(
            id="eval-wrong-session-memory",
            source="eval-wrong-session-memory.md",
            source_type="memory",
            text=wrong_session_text,
            visibility=Visibility.PLAYER,
            tags=["gallery", "archive", "dawn"],
            session_id="other-session",
        ),
    )
    _seed_chunk(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        collection=RagCollection.PERSONA_MEMORY,
        chunk=RagChunk(
            id="eval-wrong-persona-memory",
            source="eval-wrong-persona-memory.md",
            source_type="memory",
            text=wrong_persona_text,
            visibility=Visibility.PLAYER,
            tags=["gallery", "archive", "mirror"],
            persona_id=secondary_persona.id,
        ),
    )

    return EvalFixture(
        world=world,
        scene=scene,
        session=session,
        primary_persona=primary_persona,
        secondary_persona=secondary_persona,
        canon_lore_chunk_id="eval-public-lore",
        public_lore_chunk_id="eval-public-lore",
        public_lore_text=public_lore_text,
        session_memory_chunk_id="eval-relevant-memory",
        persona_memory_chunk_id="eval-persona-memory",
        persona_memory_text=persona_memory_text,
        wrong_world_chunk_id="eval-wrong-world-lore",
        wrong_world_text=wrong_world_text,
        wrong_session_chunk_id="eval-wrong-session-memory",
        wrong_session_text=wrong_session_text,
        wrong_persona_chunk_id="eval-wrong-persona-memory",
        wrong_persona_text=wrong_persona_text,
        gm_only_chunk_id="eval-gm-lore",
        gm_only_lore_text=gm_only_lore_text,
        character_private_chunk_id="eval-private-memory",
        character_private_text=character_private_text,
        relevant_memory_chunk_id="eval-relevant-memory",
        relevant_memory_text=relevant_memory_text,
        irrelevant_memory_chunk_id="eval-irrelevant-memory",
        irrelevant_memory_text=irrelevant_memory_text,
        scene_gm_only_text=scene_gm_only_text,
        primary_persona_secret=primary_persona_secret,
        primary_persona_private_description=primary_persona_private_description,
        primary_persona_forbidden_knowledge=primary_persona_forbidden_knowledge,
        local_route=local_route,
        cloud_route=cloud_route,
        memory_route=memory_route,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        important_turn_user_message="I will return before dawn for the archive key.",
        important_turn_assistant_message="Then I will expect you before dawn at the archive door.",
    )


def _seed_chunk(
    *,
    vector_store: InMemoryVectorStore,
    embedding_provider: DeterministicKeywordEmbeddingProvider,
    collection: RagCollection,
    chunk: RagChunk,
) -> None:
    vector_store.ensure_collection(collection, embedding_provider.dimension)
    vector_store.replace_source(
        collection,
        chunk.source,
        [chunk],
        [embedding_provider.embed_text(chunk.text)],
    )
