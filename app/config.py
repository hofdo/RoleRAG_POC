from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.llm.router import CloudMode

_PLACEHOLDER_CLOUD_KEYS = {
    "",
    "replace_me",
    "changeme",
    "your_api_key_here",
    "your-api-key-here",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    database_path: str = "data/rolerag.db"
    content_root: Path = Path("data")

    local_llm_base_url: str = "http://127.0.0.1:8080/v1"
    local_llm_api_key: str = "local"
    local_llm_model: str = "chatgpt-onnechan"
    local_llm_max_tokens: int = 500
    local_structured_max_tokens: int = Field(default=350, ge=1)
    local_llm_temperature: float = 0.75
    local_llm_timeout_seconds: float = Field(default=180.0, gt=0)
    # One retry absorbs transient request stalls that otherwise kill a whole
    # session via a single 504 (2026-06-12 live acceptance, run #2).
    local_llm_max_retries: int = Field(default=1, ge=0)

    critic_gating: Literal["always", "auto"] = "always"
    curator_gating: Literal["always", "auto"] = "always"

    cloud_mode: CloudMode = CloudMode.ASK
    cloud_llm_base_url: str = "https://api.openai.com/v1"
    cloud_llm_api_key: str = "replace_me"
    cloud_llm_model: str = "gpt-4.1-mini"
    cloud_llm_max_tokens: int = 1000
    cloud_llm_temperature: float = 0.65
    cloud_llm_timeout_seconds: float = Field(default=120.0, gt=0)
    cloud_llm_max_retries: int = Field(default=1, ge=0)

    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_default_top_k: int = Field(default=5, ge=1)
    rag_chunk_size_chars: int = Field(default=1000, ge=1)
    rag_chunk_overlap_chars: int = Field(default=120, ge=0)
    rag_max_retrieved_chunk_chars: int = Field(default=800, ge=1)

    structured_output_failure_log_dir: Path | None = None

    recent_dialogue_turns: int = Field(default=8, ge=0)
    recent_dialogue_max_message_chars: int = Field(default=900, ge=1)
    live_turn_count: int = Field(default=8, ge=5, le=50)
    live_long_turn_count: int = Field(default=0, ge=0)
    live_fail_on_structured_warnings: bool = True

    @field_validator("structured_output_failure_log_dir", mode="before")
    @classmethod
    def _empty_structured_failure_log_dir_disables(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


def get_settings() -> Settings:
    return Settings()


def is_usable_cloud_api_key(value: str) -> bool:
    return value.strip().lower() not in _PLACEHOLDER_CLOUD_KEYS
