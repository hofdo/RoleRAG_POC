from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from app.cli import app
from app.domain import (
    CriticResult,
    MemoryCandidate,
    PersonaCard,
    RetrievedChunk,
    SceneState,
    SessionState,
    Visibility,
)
from app.llm.provider import LlmProvider, LlmRequest, LlmResponse
from app.persistence import SQLiteMemoryRepository, SQLiteSessionRepository
from app.persistence.sqlite import connect_sqlite, initialize_database
from app.rag.diagnostics import ChunkRetrievalDiagnostic, RetrievalDiagnostics, RetrievalResult
from app.rag.models import RagChunk, RagCollection

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

    def ensure_collection(self, collection: RagCollection, vector_size: int) -> None:
        self.ensure_calls.append((collection, vector_size))

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


def test_cli_route_requires_confirmation_in_ask_mode() -> None:
    result = runner.invoke(
        app,
        ["route", "--task", "repair", "--failed-local-attempts", "2"],
        env={"CLOUD_MODE": "ask"},
    )

    assert result.exit_code == 0
    assert '"provider": "cloud"' in result.stdout
    assert '"requires_user_confirmation": true' in result.stdout


def test_cli_route_supports_explicit_cloud_request() -> None:
    result = runner.invoke(
        app,
        ["route", "--task", "actor_response", "--request-cloud"],
        env={"CLOUD_MODE": "ask"},
    )

    assert result.exit_code == 0
    assert '"provider": "cloud"' in result.stdout
    assert '"reason": "user requested cloud"' in result.stdout


def test_cli_route_forces_local_in_off_mode() -> None:
    result = runner.invoke(
        app,
        [
            "route",
            "--task",
            "repair",
            "--failed-local-attempts",
            "2",
        ],
        env={"CLOUD_MODE": "off"},
    )

    assert result.exit_code == 0
    assert '"provider": "local"' in result.stdout
    assert (
        '"reason": "cloud mode is off; cloud would have been used: local repair failed"'
        in result.stdout
    )


def test_cli_turn_warns_when_ask_mode_skips_cloud(tmp_path: Path) -> None:
    context_retriever = FakeActorContextRetriever(
        [
            RetrievedChunk(
                id="lore-1",
                source="demo_lore.md",
                source_type="lore",
                text="The Rose Gallery has mirrored columns.",
                score=0.1,
                visibility=Visibility.PLAYER,
            )
        ]
    )
    provider = FakeProvider()
    with (
        patch("app.cli._build_local_provider", return_value=provider),
        patch("app.cli._build_critic_agent", return_value=FakeCritic()),
        patch("app.cli._build_file_loader", return_value=FakeLoader()),
        patch("app.cli._build_actor_context_retriever", return_value=context_retriever),
    ):
        runner.invoke(
            app,
            ["start-session", "--session-id", "demo-session"],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )
        result = runner.invoke(
            app,
            [
                "turn",
                "--session-id",
                "demo-session",
                "--message",
                "What do I notice?",
                "--request-cloud",
            ],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db"), "CLOUD_MODE": "ask"},
        )

    assert result.exit_code == 0
    assert "Warning: cloud actor skipped: confirmation required" in result.stderr
    assert "I have heard enough to know the regent fears open daylight." in result.stdout


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
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
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


def test_cli_turn_uses_fake_retrieved_context_without_qdrant(tmp_path: Path) -> None:
    context_retriever = FakeActorContextRetriever(
        [
            RetrievedChunk(
                id="lore-1",
                source="demo_lore.md",
                source_type="lore",
                text="The Rose Gallery has mirrored columns.",
                score=0.91,
                visibility=Visibility.PLAYER,
            )
        ]
    )
    provider = FakeProvider()
    with (
        patch("app.cli._build_local_provider", return_value=provider),
        patch("app.cli._build_critic_agent", return_value=FakeCritic()),
        patch("app.cli._build_file_loader", return_value=FakeLoader()),
        patch("app.cli._build_actor_context_retriever", return_value=context_retriever),
    ):
        runner.invoke(
            app,
            ["start-session", "--session-id", "demo-session"],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )
        result = runner.invoke(
            app,
            ["turn", "--session-id", "demo-session", "--message", "What do I notice?"],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )

    assert result.exit_code == 0
    assert "The Rose Gallery has mirrored columns." in provider.requests[0].messages[0].content


def test_cli_retrieve_debug_uses_fake_retriever_and_omits_chunk_text(tmp_path: Path) -> None:
    context_retriever = FakeActorContextRetriever(
        [
            RetrievedChunk(
                id="memory-1",
                source="memory_episode:memory-1",
                source_type="session_memory",
                text="The player promised to return before dawn.",
                score=0.61,
                visibility=Visibility.PLAYER,
                tags=["promise", "dawn"],
                session_id="demo-session",
                importance=4,
            )
        ]
    )
    with (
        patch("app.cli._build_file_loader", return_value=FakeLoader()),
        patch("app.cli._build_actor_context_retriever", return_value=context_retriever),
    ):
        runner.invoke(
            app,
            ["start-session", "--session-id", "demo-session"],
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
    assert vector_store.ensure_calls == [(RagCollection.SESSION_MEMORY, 2)]
    assert vector_store.upsert_calls[0][1][0].id == persisted[0].id


def test_cli_reindex_memories_fails_clearly_for_missing_session(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["reindex-memories", "--session-id", "missing-session"],
        env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
    )

    assert result.exit_code == 1
    assert "missing-session" in result.stdout
