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
    assert settings.local_structured_max_tokens == 640
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
    assert settings.live_turn_count == 8
    assert settings.live_fail_on_structured_warnings is True
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


def test_settings_default_provider_timeouts_and_retries(tmp_path: Path) -> None:
    settings = Settings(_env_file=tmp_path / ".missing")  # type: ignore[call-arg]

    assert settings.local_llm_timeout_seconds == 180.0
    assert settings.local_llm_max_retries == 1
    assert settings.cloud_llm_timeout_seconds == 120.0
    assert settings.cloud_llm_max_retries == 1


def test_settings_accept_provider_timeout_overrides(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LOCAL_LLM_TIMEOUT_SECONDS=90\n"
        "LOCAL_LLM_MAX_RETRIES=2\n"
        "CLOUD_LLM_TIMEOUT_SECONDS=30\n"
        "CLOUD_LLM_MAX_RETRIES=0\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.local_llm_timeout_seconds == 90.0
    assert settings.local_llm_max_retries == 2
    assert settings.cloud_llm_timeout_seconds == 30.0
    assert settings.cloud_llm_max_retries == 0


def test_settings_default_structured_failure_log_dir_to_disabled(tmp_path: Path) -> None:
    settings = Settings(_env_file=tmp_path / ".missing")  # type: ignore[call-arg]

    assert settings.structured_output_failure_log_dir is None


def test_settings_treat_empty_structured_failure_log_dir_as_disabled(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("STRUCTURED_OUTPUT_FAILURE_LOG_DIR=\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.structured_output_failure_log_dir is None


def test_settings_accept_structured_failure_log_dir_override(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "STRUCTURED_OUTPUT_FAILURE_LOG_DIR=/tmp/failures\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.structured_output_failure_log_dir == Path("/tmp/failures")


def test_settings_accept_live_smoke_overrides(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LIVE_LONG_TURN_COUNT=12\nLIVE_FAIL_ON_STRUCTURED_WARNINGS=1\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.live_long_turn_count == 12
    assert settings.live_fail_on_structured_warnings is True


def test_settings_default_gating_to_always(tmp_path: Path) -> None:
    settings = Settings(_env_file=tmp_path / ".missing")  # type: ignore[call-arg]

    assert settings.critic_gating == "always"
    assert settings.curator_gating == "always"


def test_settings_accept_auto_gating_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRITIC_GATING", "auto")
    monkeypatch.setenv("CURATOR_GATING", "auto")

    settings = Settings(_env_file=tmp_path / ".missing")  # type: ignore[call-arg]

    assert settings.critic_gating == "auto"
    assert settings.curator_gating == "auto"


def test_settings_reject_invalid_gating_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRITIC_GATING", "sometimes")

    with pytest.raises(ValidationError):
        Settings(_env_file=tmp_path / ".missing")  # type: ignore[call-arg]


def test_settings_ranking_weight_defaults_mirror_ranking_constants(tmp_path: Path) -> None:
    from app.rag import ranking

    settings = Settings(_env_file=tmp_path / ".missing")  # type: ignore[call-arg]

    assert settings.rag_session_memory_weight == ranking.SESSION_MEMORY_WEIGHT
    assert settings.rag_persona_memory_weight == ranking.PERSONA_MEMORY_WEIGHT
    assert settings.rag_canon_lore_weight == ranking.CANON_LORE_WEIGHT
    assert settings.rag_session_id_match_boost == ranking.SESSION_ID_MATCH_BOOST
    assert settings.rag_scene_id_match_boost == ranking.SCENE_ID_MATCH_BOOST
    assert settings.rag_persona_id_match_boost == ranking.PERSONA_ID_MATCH_BOOST
    assert settings.rag_importance_step_boost == ranking.IMPORTANCE_STEP_BOOST
    assert settings.rag_lexical_match_step_boost == ranking.LEXICAL_MATCH_STEP_BOOST
    assert settings.rag_lexical_match_max_boost == ranking.LEXICAL_MATCH_MAX_BOOST
    assert settings.rag_recency_weight == 0.0


def test_settings_canon_and_index_floor_defaults(tmp_path: Path) -> None:
    settings = Settings(_env_file=tmp_path / ".missing")  # type: ignore[call-arg]

    assert settings.rag_index_importance_floor == 1
    assert settings.canon_importance_floor == 4
    assert settings.canon_max_items == 8
    assert settings.canon_max_chars == 900
    assert settings.session_memory_max_episodes == 0


def test_build_ranking_weights_defaults_match_default_ranking_weights(tmp_path: Path) -> None:
    from app.composition import build_ranking_weights
    from app.rag.ranking import DEFAULT_RANKING_WEIGHTS

    settings = Settings(_env_file=tmp_path / ".missing")  # type: ignore[call-arg]

    assert build_ranking_weights(settings) == DEFAULT_RANKING_WEIGHTS


def test_routing_threshold_defaults_mirror_module_constants(tmp_path: Path) -> None:
    from app.llm import router
    from app.orchestration.stages.generation import TRUNCATION_RETRY_BUDGET_MULTIPLIER

    settings = Settings(_env_file=tmp_path / ".missing")  # type: ignore[call-arg]

    assert settings.low_retrieval_confidence == router.LOW_RETRIEVAL_CONFIDENCE
    assert settings.high_scene_complexity == router.HIGH_SCENE_COMPLEXITY
    assert settings.truncation_retry_budget_multiplier == TRUNCATION_RETRY_BUDGET_MULTIPLIER
