from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.llm.router import CloudMode


def test_settings_use_llamacpp_friendly_local_defaults(tmp_path: Path) -> None:
    settings = Settings(_env_file=tmp_path / ".missing")

    assert settings.local_llm_base_url == "http://localhost:8080/v1"
    assert settings.local_llm_api_key == "local"
    assert settings.local_llm_model == "local-model"
    assert settings.cloud_mode == CloudMode.ASK


def test_settings_accept_valid_cloud_mode_override(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CLOUD_MODE=off\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.cloud_mode == CloudMode.OFF


def test_settings_reject_invalid_cloud_mode(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CLOUD_MODE=invalid\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)
