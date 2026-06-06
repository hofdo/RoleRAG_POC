from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.llm.router import CloudMode


def test_settings_use_llamacpp_friendly_local_defaults(tmp_path: Path) -> None:
    settings = Settings(_env_file=tmp_path / ".missing")  # type: ignore[call-arg]

    assert settings.database_path == "data/rolerag.db"
    assert settings.local_llm_base_url == "http://127.0.0.1:8080/v1"
    assert settings.local_llm_api_key == "local"
    assert settings.local_llm_model == "chatgpt-onnechan"
    assert settings.local_llm_max_tokens == 500
    assert settings.local_structured_max_tokens == 350
    assert settings.cloud_mode == CloudMode.ASK
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.rag_default_top_k == 5
    assert settings.rag_chunk_size_chars == 1000
    assert settings.rag_chunk_overlap_chars == 120
    assert settings.rag_max_retrieved_chunk_chars == 800
    assert settings.recent_dialogue_turns == 8
    assert settings.recent_dialogue_max_message_chars == 900
    assert settings.live_long_turn_count == 0
    assert settings.live_fail_on_structured_warnings is False
    assert "cloud_llm_enabled" not in settings.model_dump()


def test_settings_accept_valid_cloud_mode_override(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CLOUD_MODE=off\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.cloud_mode == CloudMode.OFF


def test_settings_ignore_removed_cloud_llm_enabled_variable(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CLOUD_LLM_ENABLED=true\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.cloud_mode == CloudMode.ASK
    assert "cloud_llm_enabled" not in settings.model_dump()


def test_settings_reject_invalid_cloud_mode(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CLOUD_MODE=invalid\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)  # type: ignore[call-arg]


def test_settings_accept_database_path_and_recent_turn_overrides(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATABASE_PATH=/tmp/test.db\n"
        "RECENT_DIALOGUE_TURNS=3\n"
        "RECENT_DIALOGUE_MAX_MESSAGE_CHARS=444\n"
        "RAG_CHUNK_SIZE_CHARS=333\n"
        "LOCAL_STRUCTURED_MAX_TOKENS=123\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.database_path == "/tmp/test.db"
    assert settings.recent_dialogue_turns == 3
    assert settings.recent_dialogue_max_message_chars == 444
    assert settings.rag_chunk_size_chars == 333
    assert settings.local_structured_max_tokens == 123


def test_settings_accept_live_smoke_overrides(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LIVE_LONG_TURN_COUNT=12\nLIVE_FAIL_ON_STRUCTURED_WARNINGS=1\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.live_long_turn_count == 12
    assert settings.live_fail_on_structured_warnings is True
