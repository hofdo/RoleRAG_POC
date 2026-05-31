from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.llm.router import CloudMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    database_path: str = "data/rolerag.db"

    local_llm_base_url: str = "http://localhost:8080/v1"
    local_llm_api_key: str = "local"
    local_llm_model: str = "local-model"
    local_llm_max_tokens: int = 700
    local_llm_temperature: float = 0.75

    cloud_mode: CloudMode = CloudMode.ASK
    cloud_llm_base_url: str = "https://api.openai.com/v1"
    cloud_llm_api_key: str = "replace_me"
    cloud_llm_model: str = "gpt-4.1-mini"
    cloud_llm_max_tokens: int = 1000
    cloud_llm_temperature: float = 0.65

    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_default_top_k: int = Field(default=5, ge=1)
    rag_chunk_size_chars: int = Field(default=1000, ge=1)
    rag_chunk_overlap_chars: int = Field(default=120, ge=0)
    rag_max_retrieved_chunk_chars: int = Field(default=800, ge=1)

    max_local_retries: int = Field(default=1, ge=0)
    recent_dialogue_turns: int = Field(default=8, ge=0)


def get_settings() -> Settings:
    return Settings()
