from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from app.cli import app
from app.domain import (
    CriticResult,
    CriticStatus,
    MemoryCandidate,
    PersonaCard,
    RetrievalCandidateDiagnostic,
    RetrievedChunk,
    SceneState,
    SessionState,
    TurnDiagnostics,
    TurnRetrievalDiagnostics,
    Visibility,
)
from app.evals.fixtures import DeterministicKeywordEmbeddingProvider
from app.llm.provider import LlmProvider, LlmRequest, LlmResponse
from app.llm.router import ModelProviderName, ModelRoute
from app.persistence import (
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
)
from app.persistence.sqlite import connect_sqlite, initialize_database
from app.rag.diagnostics import ChunkRetrievalDiagnostic, RetrievalDiagnostics, RetrievalResult
from app.rag.models import RagChunk, RagCollection
from app.rag.vector_store import InMemoryVectorStore

runner = CliRunner()


class FakeProvider(LlmProvider):
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            text="I have heard enough to know the regent fears open daylight.",
            provider="fake",
            model=request.model,
            usage={"total_tokens": 15},
            finish_reason="stop",
        )


class FakeLoader:
    def load_world(self, world_id: str) -> object:
        if world_id != "demo_world":
            raise ValueError(f"Unknown world: {world_id}")
        return type(
            "WorldRecord",
            (),
            {
                "id": world_id,
                "default_scene_id": "rose-gallery",
                "persona_ids": ["archivist"],
                "scene_ids": ["rose-gallery"],
            },
        )()

    def load_persona(self, persona_id: str) -> PersonaCard:
        if persona_id != "archivist":
            raise ValueError(f"Unknown persona: {persona_id}")
        return PersonaCard(
            id="archivist",
            name="Iria Vale",
            role="npc",
            public_description="A composed palace archivist.",
            speaking_style="Precise and dry.",
        )

    def load_scene(self, scene_id: str) -> SceneState:
        if scene_id != "rose-gallery":
            raise ValueError(f"Unknown scene: {scene_id}")
        return SceneState(
            id="rose-gallery",
            title="Rose Gallery",
            location="Winter Palace",
            player_visible_summary="Courtiers drift between mirrors and roses.",
        )


class FakeEmbeddingProvider:
    dimension = 2

    def embed_text(self, text: str) -> list[float]:
        return [1.0, float(len(text))]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text))] for text in texts]


class RecordingVectorStore:
    def __init__(self) -> None:
        self.ensure_calls: list[tuple[RagCollection, int]] = []
        self.replace_calls: list[tuple[RagCollection, str, list[RagChunk], list[list[float]]]] = []
        self.upsert_calls: list[tuple[RagCollection, list[RagChunk], list[list[float]]]] = []
        self.drop_calls: list[RagCollection] = []
        self.delete_session_calls: list[tuple[RagCollection, str]] = []

    def ensure_collection(self, collection: RagCollection, vector_size: int) -> None:
        self.ensure_calls.append((collection, vector_size))

    def drop_collection(self, collection: RagCollection) -> None:
        self.drop_calls.append(collection)

    def delete_session_points(self, collection: RagCollection, session_id: str) -> None:
        self.delete_session_calls.append((collection, session_id))

    def replace_source(
        self,
        collection: RagCollection,
        source: str,
        chunks: list[RagChunk],
        vectors: list[list[float]],
    ) -> None:
        self.replace_calls.append((collection, source, chunks, vectors))

    def upsert_chunks(
        self,
        collection: RagCollection,
        chunks: list[RagChunk],
        vectors: list[list[float]],
    ) -> None:
        self.upsert_calls.append((collection, chunks, vectors))


def _seed_chunks(
    *,
    vector_store: InMemoryVectorStore,
    embedding_provider: DeterministicKeywordEmbeddingProvider,
    collection: RagCollection,
    chunks: list[RagChunk],
) -> None:
    vector_store.ensure_collection(collection, embedding_provider.dimension)
    vector_store.upsert_chunks(
        collection,
        chunks,
        embedding_provider.embed_batch([chunk.text for chunk in chunks]),
    )


def _write_scenario_pack(root: Path) -> None:
    (root / "worlds").mkdir(parents=True)
    (root / "personas").mkdir()
    (root / "scenes").mkdir()
    (root / "documents").mkdir()
    (root / "worlds" / "custom_world.json").write_text(
        json.dumps(
            {
                "id": "custom_world",
                "name": "Custom World",
                "default_scene_id": "custom-opening",
                "persona_ids": ["custom-narrator"],
                "scene_ids": ["custom-opening"],
            }
        ),
        encoding="utf-8",
    )
    (root / "personas" / "custom-narrator.json").write_text(
        json.dumps(
            {
                "id": "custom-narrator",
                "name": "Custom Narrator",
                "role": "narrator",
                "public_description": "A custom narrator.",
                "speaking_style": "Clear.",
            }
        ),
        encoding="utf-8",
    )
    (root / "scenes" / "custom_opening.json").write_text(
        json.dumps(
            {
                "id": "custom-opening",
                "title": "Custom Opening",
                "location": "Custom Hall",
                "player_visible_summary": "A custom hall waits.",
            }
        ),
        encoding="utf-8",
    )
    (root / "documents" / "lore.md").write_text(
        "# Custom Lore\n\nA custom banner hangs in the hall.",
        encoding="utf-8",
    )
    (root / "documents" / "manifest.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "path": "lore.md",
                        "visibility": Visibility.PLAYER.value,
                        "source_type": "lore",
                        "tags": ["custom"],
                        "world_id": "custom_world",
                        "scene_id": "custom-opening",
                        "persona_id": "custom-narrator",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


class FakeActorContextRetriever:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or []
        self.calls: list[dict[str, object]] = []

    def retrieve_for_actor(self, **kwargs: object) -> list[RetrievedChunk]:
        self.calls.append(kwargs)
        return self.chunks

    def retrieve_for_actor_with_diagnostics(self, **kwargs: object) -> RetrievalResult:
        self.calls.append(kwargs)
        return RetrievalResult(
            chunks=self.chunks,
            diagnostics=RetrievalDiagnostics(
                query=str(kwargs["query"]),
                selected=[
                    ChunkRetrievalDiagnostic(
                        id=chunk.id,
                        source=chunk.source,
                        source_type=chunk.source_type,
                        collection=RagCollection.SESSION_MEMORY,
                        visibility=chunk.visibility,
                        tags=chunk.tags,
                        original_score=chunk.score,
                        adjusted_score=chunk.score + 0.125,
                        applied_boosts={"collection": 0.08, "importance": 0.045},
                        selected_rank=index,
                    )
                    for index, chunk in enumerate(self.chunks, start=1)
                ],
            ),
        )


class FakeCritic:
    async def evaluate(self, **_: object) -> CriticResult:
        return CriticResult(accepted=True)

    def build_local_repair_messages(self, **_: object) -> list[object]:
        raise AssertionError("repair should not be used in this test")

    def build_cloud_repair_messages(self, **_: object) -> list[object]:
        raise AssertionError("repair should not be used in this test")


def test_cli_help_exits_successfully() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_cli_config_redacts_api_keys() -> None:
    result = runner.invoke(
        app,
        ["config"],
        env={
            "LOCAL_LLM_API_KEY": "super-secret-local",
            "CLOUD_LLM_API_KEY": "super-secret-cloud",
        },
    )

    assert result.exit_code == 0
    assert "super-secret-local" not in result.stdout
    assert "super-secret-cloud" not in result.stdout
    assert '"local_llm_api_key": "***"' in result.stdout
    assert '"cloud_llm_api_key": "***"' in result.stdout
    assert "cloud_llm_enabled" not in result.stdout


def test_cli_route_shows_local_route_by_default() -> None:
    result = runner.invoke(app, ["route", "--task", "actor_response"])

    assert result.exit_code == 0
    assert '"provider": "local"' in result.stdout


def test_cli_route_shows_cloud_route_when_provider_is_cloud() -> None:
    result = runner.invoke(
        app,
        ["route", "--task", "repair", "--provider", "cloud"],
    )

    assert result.exit_code == 0
    assert '"provider": "cloud"' in result.stdout
    assert '"reason": "session provider: cloud"' in result.stdout


def test_cli_route_defaults_to_local_provider() -> None:
    result = runner.invoke(
        app,
        ["route", "--task", "repair"],
    )

    assert result.exit_code == 0
    assert '"provider": "local"' in result.stdout
    assert '"reason": "session provider: local"' in result.stdout


def test_cli_start_session_and_turn_run_with_mocked_provider(tmp_path: Path) -> None:
    context_retriever = FakeActorContextRetriever()
    with (
        patch("app.cli._build_local_provider", return_value=FakeProvider()),
        patch("app.cli._build_critic_agent", return_value=FakeCritic()),
        patch("app.cli._build_file_loader", return_value=FakeLoader()),
        patch("app.cli._build_actor_context_retriever", return_value=context_retriever),
    ):
        start_result = runner.invoke(
            app,
            [
                "start-session",
                "--session-id",
                "demo-session",
                "--player-name",
                "Avery",
                "--skip-lore-ingest",
            ],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )
        turn_result = runner.invoke(
            app,
            [
                "turn",
                "--message",
                "What have you heard about the regent?",
                "--session-id",
                "demo-session",
            ],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db"), "CLOUD_MODE": "off"},
        )

    assert start_result.exit_code == 0
    assert json.loads(start_result.stdout)["id"] == "demo-session"
    assert turn_result.exit_code == 0
    assert "I have heard enough to know the regent fears open daylight." in turn_result.stdout
    assert len(context_retriever.calls) == 1


def test_cli_start_session_persists_custom_content_root(tmp_path: Path) -> None:
    pack_root = tmp_path / "pack"
    _write_scenario_pack(pack_root)
    database_path = tmp_path / "sessions.db"

    start_result = runner.invoke(
        app,
        [
            "start-session",
            "--session-id",
            "custom-session",
            "--content-root",
            str(pack_root),
            "--world-id",
            "custom_world",
            "--scene-id",
            "custom-opening",
            "--active-persona-id",
            "custom-narrator",
            "--skip-lore-ingest",
        ],
        env={"DATABASE_PATH": str(database_path)},
    )
    resume_result = runner.invoke(
        app,
        ["resume", "--session-id", "custom-session"],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert start_result.exit_code == 0
    assert json.loads(start_result.stdout)["content_root"] == str(pack_root)
    assert resume_result.exit_code == 0
    assert json.loads(resume_result.stdout)["content_root"] == str(pack_root)


def test_cli_start_session_auto_ingests_scenario_lore(tmp_path: Path) -> None:
    # start-session indexes the scenario's lore automatically, so the player does not have to
    # remember a separate ingest-scenario-lore step before the first turn.
    pack_root = tmp_path / "pack"
    _write_scenario_pack(pack_root)
    vector_store = RecordingVectorStore()

    with (
        patch("app.cli._build_embedding_provider", return_value=FakeEmbeddingProvider()),
        patch("app.cli._build_vector_store", return_value=vector_store),
    ):
        result = runner.invoke(
            app,
            [
                "start-session",
                "--session-id",
                "auto-ingest-session",
                "--content-root",
                str(pack_root),
                "--world-id",
                "custom_world",
                "--scene-id",
                "custom-opening",
                "--active-persona-id",
                "custom-narrator",
            ],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["id"] == "auto-ingest-session"
    # The scenario pack's lore document was indexed without a separate command.
    assert len(vector_store.replace_calls) == 1
    _, source, _, _ = vector_store.replace_calls[0]
    assert source.endswith("documents/lore.md")


def test_cli_resume_prints_session_metadata(tmp_path: Path) -> None:
    with (
        patch("app.cli._build_local_provider", return_value=FakeProvider()),
        patch("app.cli._build_critic_agent", return_value=FakeCritic()),
        patch("app.cli._build_file_loader", return_value=FakeLoader()),
    ):
        runner.invoke(
            app,
            [
                "start-session",
                "--session-id",
                "demo-session",
                "--player-name",
                "Avery",
                "--skip-lore-ingest",
            ],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )
        result = runner.invoke(
            app,
            ["resume", "--session-id", "demo-session"],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["id"] == "demo-session"
    assert payload["active_scene_id"] == "rose-gallery"


def test_cli_turn_fails_clearly_for_missing_session(tmp_path: Path) -> None:
    with (
        patch("app.cli._build_local_provider", return_value=FakeProvider()),
        patch("app.cli._build_critic_agent", return_value=FakeCritic()),
    ):
        result = runner.invoke(
            app,
            [
                "turn",
                "--message",
                "What have you heard about the regent?",
                "--session-id",
                "missing-session",
            ],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )

    assert result.exit_code == 1
    assert "missing-session" in result.stdout


def test_cli_ingest_uses_fake_embedding_provider_and_vector_store(tmp_path: Path) -> None:
    document = tmp_path / "demo_lore.md"
    document.write_text(
        "# Rose Gallery\n\nCourtiers drift between mirrors and roses.",
        encoding="utf-8",
    )
    vector_store = RecordingVectorStore()

    with (
        patch("app.cli._build_embedding_provider", return_value=FakeEmbeddingProvider()),
        patch("app.cli._build_vector_store", return_value=vector_store),
    ):
        result = runner.invoke(
            app,
            [
                "ingest",
                str(document),
                "--visibility",
                Visibility.PLAYER.value,
                "--source-type",
                "lore",
                "--world-id",
                "demo_world",
                "--tag",
                "palace",
            ],
        )

    assert result.exit_code == 0
    assert '"collection": "canon_lore"' in result.stdout
    assert '"chunk_count": 1' in result.stdout
    assert vector_store.ensure_calls == [(RagCollection.CANON_LORE, 2)]
    assert len(vector_store.replace_calls) == 1
    _, source, chunks, vectors = vector_store.replace_calls[0]
    assert source == str(document)
    assert len(chunks) == 1
    assert len(vectors) == 1
    assert chunks[0].visibility == Visibility.PLAYER
    assert chunks[0].tags == ["palace"]
    assert chunks[0].world_id == "demo_world"


def test_cli_ingest_scenario_lore_uses_manifest_metadata(tmp_path: Path) -> None:
    pack_root = tmp_path / "pack"
    _write_scenario_pack(pack_root)
    vector_store = RecordingVectorStore()

    with (
        patch("app.cli._build_embedding_provider", return_value=FakeEmbeddingProvider()),
        patch("app.cli._build_vector_store", return_value=vector_store),
    ):
        result = runner.invoke(
            app,
            ["ingest-scenario-lore", "--content-root", str(pack_root)],
        )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["total_chunk_count"] == 1
    assert payload["documents"][0]["source"].endswith("documents/lore.md")
    assert vector_store.ensure_calls == [(RagCollection.CANON_LORE, 2)]
    _, _, chunks, _ = vector_store.replace_calls[0]
    assert chunks[0].visibility == Visibility.PLAYER
    assert chunks[0].tags == ["custom"]
    assert chunks[0].world_id == "custom_world"
    assert chunks[0].scene_id == "custom-opening"
    assert chunks[0].persona_id == "custom-narrator"


def test_cli_turn_uses_in_memory_retrieval_and_excludes_hidden_or_isolated_chunks(
    tmp_path: Path,
) -> None:
    embedding_provider = DeterministicKeywordEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    _seed_chunks(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        collection=RagCollection.CANON_LORE,
        chunks=[
            RagChunk(
                id="lore-1",
                source="demo_lore.md",
                source_type="lore",
                text="The Rose Gallery has mirrored columns around the archive door.",
                visibility=Visibility.PLAYER,
                world_id="demo_world",
            ),
            RagChunk(
                id="wrong-world",
                source="other_lore.md",
                source_type="lore",
                text="Another gallery archive mirror hides a clock.",
                visibility=Visibility.PLAYER,
                world_id="other_world",
            ),
            RagChunk(
                id="gm-lore",
                source="gm_lore.md",
                source_type="lore",
                text="A spy waits behind the gallery mirror.",
                visibility=Visibility.GM,
                world_id="demo_world",
            ),
        ],
    )
    provider = FakeProvider()
    with (
        patch("app.cli._build_local_provider", return_value=provider),
        patch("app.cli._build_critic_agent", return_value=FakeCritic()),
        patch("app.cli._build_file_loader", return_value=FakeLoader()),
        patch("app.cli._build_embedding_provider", return_value=embedding_provider),
        patch("app.cli._build_vector_store", return_value=vector_store),
    ):
        runner.invoke(
            app,
            ["start-session", "--session-id", "demo-session", "--skip-lore-ingest"],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )
        result = runner.invoke(
            app,
            [
                "turn",
                "--session-id",
                "demo-session",
                "--message",
                "What do I notice about the regent near the gallery archive mirror?",
            ],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )

    assert result.exit_code == 0
    prompt = provider.requests[0].messages[0].content
    assert "The Rose Gallery has mirrored columns around the archive door." in prompt
    assert "Another gallery archive mirror hides a clock." not in prompt
    assert "A spy waits behind the gallery mirror." not in prompt


def test_cli_retrieve_debug_uses_in_memory_retrieval_and_omits_chunk_text(tmp_path: Path) -> None:
    embedding_provider = DeterministicKeywordEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    _seed_chunks(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        collection=RagCollection.SESSION_MEMORY,
        chunks=[
            RagChunk(
                id="memory-1",
                source="memory_episode:memory-1",
                source_type="session_memory",
                text="The player promised to return before dawn.",
                visibility=Visibility.PLAYER,
                tags=["promise", "dawn"],
                session_id="demo-session",
                importance=4,
            )
        ],
    )
    with (
        patch("app.cli._build_file_loader", return_value=FakeLoader()),
        patch("app.cli._build_embedding_provider", return_value=embedding_provider),
        patch("app.cli._build_vector_store", return_value=vector_store),
    ):
        runner.invoke(
            app,
            ["start-session", "--session-id", "demo-session", "--skip-lore-ingest"],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )
        result = runner.invoke(
            app,
            [
                "retrieve-debug",
                "--session-id",
                "demo-session",
                "--query",
                "What did I promise the archivist?",
            ],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["query"].endswith("What did I promise the archivist?")
    assert payload["selected"][0]["id"] == "memory-1"
    assert payload["selected"][0]["adjusted_score"] > payload["selected"][0]["original_score"]
    assert "text" not in result.stdout


def test_cli_turn_uses_stored_scenario_pack_world_for_retrieval(tmp_path: Path) -> None:
    pack_root = tmp_path / "pack"
    _write_scenario_pack(pack_root)
    embedding_provider = DeterministicKeywordEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    _seed_chunks(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        collection=RagCollection.CANON_LORE,
        chunks=[
            RagChunk(
                id="custom-lore",
                source="custom-lore.md",
                source_type="lore",
                text="The custom hall archive mirror opens before dawn.",
                visibility=Visibility.PLAYER,
                world_id="custom_world",
            ),
            RagChunk(
                id="demo-lore",
                source="demo-lore.md",
                source_type="lore",
                text="The demo hall archive mirror opens before dawn.",
                visibility=Visibility.PLAYER,
                world_id="demo_world",
            ),
        ],
    )
    provider = FakeProvider()
    database_path = tmp_path / "sessions.db"
    with (
        patch("app.cli._build_local_provider", return_value=provider),
        patch("app.cli._build_critic_agent", return_value=FakeCritic()),
        patch("app.cli._build_embedding_provider", return_value=embedding_provider),
        patch("app.cli._build_vector_store", return_value=vector_store),
    ):
        start_result = runner.invoke(
            app,
            [
                "start-session",
                "--session-id",
                "custom-session",
                "--content-root",
                str(pack_root),
                "--world-id",
                "custom_world",
                "--scene-id",
                "custom-opening",
                "--active-persona-id",
                "custom-narrator",
            ],
            env={"DATABASE_PATH": str(database_path)},
        )
        turn_result = runner.invoke(
            app,
            [
                "turn",
                "--session-id",
                "custom-session",
                "--message",
                "What opens before dawn?",
            ],
            env={"DATABASE_PATH": str(database_path)},
        )

    assert start_result.exit_code == 0
    assert turn_result.exit_code == 0
    prompt = provider.requests[0].messages[0].content
    assert "The custom hall archive mirror opens before dawn." in prompt
    assert "The demo hall archive mirror opens before dawn." not in prompt


def test_cli_reindex_memories_indexes_existing_sqlite_memories(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.db"
    connection = connect_sqlite(database_path)
    initialize_database(connection)
    SQLiteSessionRepository(connection).create_session(
        SessionState(
            id="demo-session",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    persisted = SQLiteMemoryRepository(connection).append_memories(
        session_id="demo-session",
        memories=[
            MemoryCandidate(
                summary="The archive key will be ready before dawn.",
                visibility=Visibility.PLAYER,
                importance=4,
                tags=["archive", "dawn"],
                scene_id="rose-gallery",
                actor_id="archivist",
            )
        ],
    )
    connection.close()
    vector_store = RecordingVectorStore()

    with (
        patch("app.cli._build_embedding_provider", return_value=FakeEmbeddingProvider()),
        patch("app.cli._build_vector_store", return_value=vector_store),
    ):
        result = runner.invoke(
            app,
            ["reindex-memories", "--session-id", "demo-session"],
            env={"DATABASE_PATH": str(database_path)},
        )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "indexed_count": 1,
        "session_id": "demo-session",
    }
    # This memory is PLAYER-visible, importance=4, with actor_id="archivist" -- it
    # clears PERSONA_MEMORY_IMPORTANCE_FLOOR, so it is also dual-written to
    # PERSONA_MEMORY (see app/memory/indexer.py:_index_persona_memories).
    assert (RagCollection.SESSION_MEMORY, 2) in vector_store.ensure_calls
    session_calls = [
        call for call in vector_store.upsert_calls if call[0] == RagCollection.SESSION_MEMORY
    ]
    assert len(session_calls) == 1
    assert session_calls[0][1][0].id == persisted[0].id


def test_cli_reindex_memories_fails_clearly_for_missing_session(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["reindex-memories", "--session-id", "missing-session"],
        env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
    )

    assert result.exit_code == 1
    assert "missing-session" in result.stdout


def _seed_session_with_turns(database_path: Path, *, session_id: str, turn_count: int) -> None:
    connection = connect_sqlite(database_path)
    initialize_database(connection)
    SQLiteSessionRepository(connection).create_session(
        SessionState(
            id=session_id,
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    turns = SQLiteTurnRepository(connection)
    for index in range(turn_count):
        turns.append_turn(
            session_id=session_id,
            scene_id="rose-gallery",
            persona_id="archivist",
            user_message=f"Question {index}",
            assistant_message=f"Answer {index}",
            route=ModelRoute(
                provider=ModelProviderName.LOCAL,
                model="local-model",
                max_tokens=256,
                temperature=0.7,
                reason="default local route",
            ),
        )
    connection.close()


def _seed_session_with_diagnostics_turn(database_path: Path, *, session_id: str) -> None:
    connection = connect_sqlite(database_path)
    initialize_database(connection)
    SQLiteSessionRepository(connection).create_session(
        SessionState(
            id=session_id,
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    turns = SQLiteTurnRepository(connection)
    stored = turns.append_turn(
        session_id=session_id,
        scene_id="rose-gallery",
        persona_id="archivist",
        user_message="What happened?",
        assistant_message="A great deal.",
        route=ModelRoute(
            provider=ModelProviderName.LOCAL,
            model="local-model",
            max_tokens=256,
            temperature=0.7,
            reason="default local route",
        ),
    )
    turns.update_turn_diagnostics(
        stored.id,
        TurnDiagnostics(
            retrieval=TurnRetrievalDiagnostics(
                query="What happened?",
                selected=[
                    RetrievalCandidateDiagnostic(
                        id="chunk-1",
                        source="rose-gallery",
                        source_type="scene",
                        collection="scene_lore",
                        visibility=Visibility.PLAYER,
                        tags=["history"],
                        original_score=0.9,
                        adjusted_score=0.95,
                        applied_boosts={"scene": 0.05},
                        selected_rank=1,
                    )
                ],
            ),
            stage_timings={"retrieval": 1.5, "generation": 2.0},
            critic_status=CriticStatus.ACCEPTED,
            finish_reason="stop",
            warnings=["clamped temperature"],
            memory_written=True,
        ),
    )
    connection.close()


def test_cli_turn_history_surfaces_diagnostics_after_persisted_turn(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.db"
    _seed_session_with_diagnostics_turn(database_path, session_id="demo-session")

    result = runner.invoke(
        app,
        ["turn-history", "--session-id", "demo-session"],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    diagnostics = payload["turns"][0]["diagnostics"]
    assert diagnostics is not None
    assert diagnostics["finish_reason"] == "stop"
    assert diagnostics["memory_written"] is True
    assert diagnostics["critic_status"] == "accepted"
    assert diagnostics["warnings"] == ["clamped temperature"]
    assert diagnostics["stage_timings"] == {"retrieval": 1.5, "generation": 2.0}
    retrieval = diagnostics["retrieval"]
    assert retrieval["query"] == "What happened?"
    assert retrieval["selected"][0]["id"] == "chunk-1"
    assert retrieval["selected"][0]["selected_rank"] == 1
    # Retrieval diagnostics are metadata-only and never carry chunk text.
    assert "chunk_text" not in result.stdout
    assert "text" not in retrieval["selected"][0]


def test_cli_turn_history_lists_all_turns(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.db"
    _seed_session_with_turns(database_path, session_id="demo-session", turn_count=3)

    result = runner.invoke(
        app,
        ["turn-history", "--session-id", "demo-session"],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["session_id"] == "demo-session"
    assert [turn["turn_index"] for turn in payload["turns"]] == [1, 2, 3]
    first = payload["turns"][0]
    assert first["scene_id"] == "rose-gallery"
    assert first["persona_id"] == "archivist"
    assert first["user_message"] == "Question 0"
    assert first["assistant_message"] == "Answer 0"
    assert first["route"] == {
        "provider": "local",
        "model": "local-model",
        "reason": "default local route",
    }
    assert "T" in first["created_at"]
    # Stored turns never carry diagnostics, so they must not appear in output.
    assert "rankings" not in result.stdout
    assert "stage_timings" not in result.stdout
    assert "critic_status" not in result.stdout


def test_cli_turn_history_filters_by_turn_index(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.db"
    _seed_session_with_turns(database_path, session_id="demo-session", turn_count=3)

    result = runner.invoke(
        app,
        ["turn-history", "--session-id", "demo-session", "--turn", "2"],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [turn["turn_index"] for turn in payload["turns"]] == [2]


def test_cli_turn_history_applies_limit(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.db"
    _seed_session_with_turns(database_path, session_id="demo-session", turn_count=4)

    result = runner.invoke(
        app,
        ["turn-history", "--session-id", "demo-session", "--limit", "2"],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [turn["turn_index"] for turn in payload["turns"]] == [1, 2]


def test_cli_turn_history_unknown_session(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["turn-history", "--session-id", "missing-session"],
        env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
    )

    assert result.exit_code == 1
    assert "Unknown session id: missing-session" in result.stdout


def test_cli_turn_history_empty_turns(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.db"
    _seed_session_with_turns(database_path, session_id="demo-session", turn_count=0)

    result = runner.invoke(
        app,
        ["turn-history", "--session-id", "demo-session"],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {"session_id": "demo-session", "turns": []}


def test_cli_reset_index_drops_all_collections_by_default() -> None:
    vector_store = RecordingVectorStore()

    with patch("app.cli._build_vector_store", return_value=vector_store):
        result = runner.invoke(app, ["reset-index", "--yes"])

    assert result.exit_code == 0
    assert set(vector_store.drop_calls) == {
        RagCollection.CANON_LORE,
        RagCollection.SESSION_MEMORY,
        RagCollection.PERSONA_MEMORY,
    }
    payload = json.loads(result.stdout)
    assert set(payload["dropped"]) == {
        RagCollection.CANON_LORE.value,
        RagCollection.SESSION_MEMORY.value,
        RagCollection.PERSONA_MEMORY.value,
    }


def test_cli_reset_index_drops_selected_collection() -> None:
    vector_store = RecordingVectorStore()

    with patch("app.cli._build_vector_store", return_value=vector_store):
        result = runner.invoke(
            app,
            ["reset-index", "--collection", "session_memory", "--yes"],
        )

    assert result.exit_code == 0
    assert vector_store.drop_calls == [RagCollection.SESSION_MEMORY]
    payload = json.loads(result.stdout)
    assert payload["dropped"] == [RagCollection.SESSION_MEMORY.value]


def test_cli_reset_index_rejects_unknown_collection() -> None:
    vector_store = RecordingVectorStore()

    with patch("app.cli._build_vector_store", return_value=vector_store):
        result = runner.invoke(
            app,
            ["reset-index", "--collection", "unknown_name", "--yes"],
        )

    assert result.exit_code == 1
    assert "unknown_name" in result.stdout
    assert vector_store.drop_calls == []


def test_delete_session_vectors_purges_both_session_and_persona_memory() -> None:
    # Controller decision (finding 4): PERSONA_MEMORY chunks retain their
    # originating session_id payload (cross-session persona memory dual-write --
    # see app/memory/indexer.py), so cleaning up only SESSION_MEMORY orphans them
    # in Qdrant forever. _delete_session_vectors must purge the session-scoped
    # points from BOTH collections. Exercised directly against the CLI helper
    # with a fake vector store (this is the "live Qdrant only" fallback the
    # controller decision calls out).
    from app.cli import _delete_session_vectors

    vector_store = RecordingVectorStore()

    with patch("app.cli._build_vector_store", return_value=vector_store):
        _delete_session_vectors("session-1")

    assert vector_store.delete_session_calls == [
        (RagCollection.SESSION_MEMORY, "session-1"),
        (RagCollection.PERSONA_MEMORY, "session-1"),
    ]


def test_cli_delete_session_purges_persona_memory_vectors(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.db"
    _seed_session_with_turns(database_path, session_id="demo-session", turn_count=1)
    vector_store = RecordingVectorStore()
    backup_dir = tmp_path / "backups"

    with (
        patch("app.cli._build_vector_store", return_value=vector_store),
        patch("app.cli._backup_database", return_value=backup_dir / "backup.db"),
    ):
        result = runner.invoke(
            app,
            ["delete-session", "--session-id", "demo-session", "--yes"],
            env={"DATABASE_PATH": str(database_path)},
        )

    assert result.exit_code == 0
    assert vector_store.delete_session_calls == [
        (RagCollection.SESSION_MEMORY, "demo-session"),
        (RagCollection.PERSONA_MEMORY, "demo-session"),
    ]


def test_cli_reset_db_purges_persona_memory_vectors_for_every_session(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "sessions.db"
    _seed_session_with_turns(database_path, session_id="session-a", turn_count=1)
    _seed_session_with_turns(database_path, session_id="session-b", turn_count=1)
    vector_store = RecordingVectorStore()
    backup_dir = tmp_path / "backups"

    with (
        patch("app.cli._build_vector_store", return_value=vector_store),
        patch("app.cli._backup_database", return_value=backup_dir / "backup.db"),
    ):
        result = runner.invoke(
            app,
            ["reset-db", "--yes"],
            env={"DATABASE_PATH": str(database_path)},
        )

    assert result.exit_code == 0
    calls_by_collection: dict[RagCollection, set[str]] = {}
    for collection, session_id in vector_store.delete_session_calls:
        calls_by_collection.setdefault(collection, set()).add(session_id)
    assert calls_by_collection[RagCollection.SESSION_MEMORY] == {"session-a", "session-b"}
    assert calls_by_collection[RagCollection.PERSONA_MEMORY] == {"session-a", "session-b"}
