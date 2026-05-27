from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.llm.router import CloudMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"

    local_llm_base_url: str = "http://localhost:8080/v1"
    local_llm_api_key: str = "local"
    local_llm_model: str = "local-model"
    local_llm_max_tokens: int = 700
    local_llm_temperature: float = 0.75

    cloud_mode: CloudMode = CloudMode.ASK
    cloud_llm_enabled: bool = False
    cloud_llm_base_url: str = "https://api.openai.com/v1"
    cloud_llm_api_key: str = "replace_me"
    cloud_llm_model: str = "gpt-4.1-mini"
    cloud_llm_max_tokens: int = 1000
    cloud_llm_temperature: float = 0.65

    max_local_retries: int = Field(default=1, ge=0)


def get_settings() -> Settings:
    return Settings()
