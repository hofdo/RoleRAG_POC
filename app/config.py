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

    # Retrieval reranking weights. Defaults mirror the canonical Final constants
    # in app/rag/ranking.py so behavior is unchanged unless explicitly tuned.
    rag_candidate_oversample_factor: int = Field(default=2, ge=1)
    rag_session_memory_weight: float = Field(default=0.08, ge=0.0)
    rag_persona_memory_weight: float = Field(default=0.04, ge=0.0)
    rag_canon_lore_weight: float = Field(default=0.0, ge=0.0)
    rag_session_id_match_boost: float = Field(default=0.02, ge=0.0)
    rag_scene_id_match_boost: float = Field(default=0.04, ge=0.0)
    rag_persona_id_match_boost: float = Field(default=0.03, ge=0.0)
    rag_importance_step_boost: float = Field(default=0.015, ge=0.0)
    rag_lexical_match_step_boost: float = Field(default=0.05, ge=0.0)
    rag_lexical_match_max_boost: float = Field(default=0.25, ge=0.0)
    # Recency boost weight; 0.0 keeps ranking byte-identical to pre-recency behavior.
    rag_recency_weight: float = Field(default=0.0, ge=0.0)
    # Memories below this importance are persisted to SQLite but not indexed for
    # retrieval. 1 indexes everything (no behavior change).
    rag_index_importance_floor: int = Field(default=1, ge=1, le=5)
    # Cosine threshold for semantic write-dedup of new memories. 1.0 disables the
    # semantic pass (lexical dedup still runs); lower it (e.g. 0.92) to drop
    # paraphrased near-duplicates the term-overlap check misses.
    rag_write_dedup_cosine_threshold: float = Field(default=1.0, ge=0.0, le=1.0)

    # Pinned session-canon block ("Standing facts") injected into the actor prompt.
    canon_importance_floor: int = Field(default=4, ge=1, le=5)
    canon_max_items: int = Field(default=8, ge=0)
    canon_max_chars: int = Field(default=900, ge=1)

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
