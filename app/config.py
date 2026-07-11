"""Central runtime configuration for the whole app.

``Settings`` is a pydantic-settings model loaded from environment variables and
``.env``; every non-comment key here mirrors one field, a contract enforced by
``tests/integration/test_documentation.py``. Values tune the LLM providers
(local/cloud base URLs, tokens, timeouts, retries), retrieval and ranking, the
session-canon block, and the critic/curator auto-gating risk thresholds; routing
does not read the threshold values because the session provider is fixed at
creation. ``get_settings()`` is the single construction point used by
``app.composition`` and the CLI.
"""

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
    # Critic/memory JSON budget. Verbose models (e.g. Qwen3.6) overflow a tight cap
    # and truncate mid-string, failing the parse; 640 fits their critic output. Bake-off
    # 2026-06: qwen35b had 6 critic-truncation failures at 350, 0 at 640.
    local_structured_max_tokens: int = Field(default=640, ge=1)
    local_llm_temperature: float = 0.75
    # Headroom for slow local models: a turn on a dense/large model can legitimately
    # take 1-4 min, and the app returns 504 if the provider call exceeds this. 300s
    # clears the bake-off's slow models; lower it for fast models if you want to fail fast.
    local_llm_timeout_seconds: float = Field(default=300.0, gt=0)
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
    # Recency boost weight, scaled by memory importance (recent + important ranks highest; lore
    # and unimportant memories are never lifted by recency). 0.0 keeps ranking byte-identical.
    # Opt-in: enable a small value (~0.02-0.04) and validate 50-turn recall via live-smoke before
    # relying on it -- offline evals do not fully exercise long-session recency reordering.
    rag_recency_weight: float = Field(default=0.0, ge=0.0)
    # Memories below this importance are persisted to SQLite but not indexed for
    # retrieval. 1 indexes everything (no behavior change).
    rag_index_importance_floor: int = Field(default=1, ge=1, le=5)
    # Cosine threshold for semantic write-dedup of new memories. 1.0 disables the
    # semantic pass (lexical dedup still runs); lower it (e.g. 0.92) to drop
    # paraphrased near-duplicates the term-overlap check misses.
    rag_write_dedup_cosine_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    # Cap on the number of session memories kept in the retrievable vector index
    # (lowest importance-then-recency evicted from the index; SQLite keeps all).
    # 0 = unbounded, and that is the deliberate default: enabling a hard cap measurably
    # REGRESSED 50-turn recall in testing (evicted memories stop being retrievable even
    # though dual-query + importance ranking keep the index usable unbounded at POC scale).
    # Only set a cap if a specific corpus-size problem appears, and re-run live-smoke recall.
    session_memory_max_episodes: int = Field(default=0, ge=0)
    # Memory consolidation ("sleep cycle"): once this many old, low-importance,
    # non-durable memories accumulate, roll them into one summary (LLM with a
    # deterministic fallback). 0 = disabled. Only memories at or below
    # memory_consolidation_max_importance are eligible.
    memory_consolidation_threshold: int = Field(default=0, ge=0)
    memory_consolidation_max_importance: int = Field(default=3, ge=1, le=5)
    # Minimum-age floor + batch-size cap on the consolidation candidate backlog
    # (app/memory/consolidation.py select_consolidatable). Age is rank-based (oldest
    # first among eligible memories) -- no wall-clock dependency. min_age=0 excludes
    # nothing (no rolling recent window); batch_cap=0 folds the whole eligible backlog
    # in one shot. Both 0 by default: byte-identical to pre-C2 selection even when
    # consolidation is enabled. Tune alongside memory_consolidation_threshold once
    # long-campaign consolidation is validated live (P2.2).
    memory_consolidation_min_age: int = Field(default=0, ge=0)
    memory_consolidation_batch_cap: int = Field(default=0, ge=0)
    # SSE: split the validated response into fragments of this many characters for
    # progressive client rendering. 0 = single frame (no behavior change). Never
    # exposes pre-validation tokens — fragments come from the already-validated text.
    sse_text_chunk_chars: int = Field(default=0, ge=0)

    # Pinned session-canon block ("Standing facts") injected into the actor prompt.
    canon_importance_floor: int = Field(default=4, ge=1, le=5)
    canon_max_items: int = Field(default=8, ge=0)
    canon_max_chars: int = Field(default=900, ge=1)

    # Critic/curator auto-gating risk thresholds. Defaults mirror the canonical
    # constants in app/llm/router.py and app/orchestration/stages/generation.py;
    # tuning these adjusts critic auto-gating and truncation retry without editing
    # source. Routing does not read these — the session provider is fixed at creation.
    low_retrieval_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    high_scene_complexity: int = Field(default=4, ge=1, le=5)
    truncation_retry_budget_multiplier: int = Field(default=2, ge=1)
    # Output-side containment backstop — a SECURITY parameter. Fraction of a hidden fact's
    # content words that must appear in the actor reply before it is flagged as a possible
    # paraphrase leak. Lower = more sensitive (more false positives). 0.6 caught the bake-off
    # leaks; the value is pinned by test_scan_reply_paraphrase_threshold_is_applied.
    containment_overlap_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # Opt-in context-ceiling preflight (#69). 0 disables entirely (byte-identical
    # default): no prompt-size estimate, no overflow warning, no behavior change.
    # Set to the local model's actual context window (e.g. 8192 for an 8K llama.cpp
    # profile) to get an early warning -- never a block -- when the estimated or
    # actual prompt+completion token count approaches it.
    model_context_window_tokens: int = Field(default=0, ge=0)
    # Fraction of model_context_window_tokens that triggers the preflight/post-hoc
    # warning. Only meaningful when model_context_window_tokens > 0.
    context_warn_ratio: float = Field(default=0.85, gt=0.0, le=1.0)

    structured_output_failure_log_dir: Path | None = None

    recent_dialogue_turns: int = Field(default=8, ge=0)
    recent_dialogue_max_message_chars: int = Field(default=900, ge=1)
    live_turn_count: int = Field(default=8, ge=5, le=100)
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
