from __future__ import annotations

from pydantic import BaseModel, Field

from app.evals.fixtures import EvalFixture


class RetrievalQualityResult(BaseModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    retrieved_ids: list[str] = Field(default_factory=list)


def evaluate_retrieval_quality(fixture: EvalFixture, *, top_k: int = 3) -> RetrievalQualityResult:
    chunks = fixture.retrieve_actor_chunks(top_k=top_k)
    retrieved_ids = [chunk.id for chunk in chunks]
    relevant_index = _index_of(retrieved_ids, fixture.relevant_memory_chunk_id)
    irrelevant_index = _index_of(retrieved_ids, fixture.irrelevant_memory_chunk_id)
    checks = {
        "public_lore_in_top_k": fixture.public_lore_chunk_id in retrieved_ids,
        "relevant_memory_in_top_k": fixture.relevant_memory_chunk_id in retrieved_ids,
        "irrelevant_memory_ranked_lower": (
            relevant_index is not None
            and irrelevant_index is not None
            and relevant_index < irrelevant_index
        ),
        "gm_only_excluded": fixture.gm_only_chunk_id not in retrieved_ids,
        "character_private_excluded": fixture.character_private_chunk_id not in retrieved_ids,
    }
    return RetrievalQualityResult(
        passed=all(checks.values()),
        checks=checks,
        retrieved_ids=retrieved_ids,
    )


def _index_of(values: list[str], target: str) -> int | None:
    try:
        return values.index(target)
    except ValueError:
        return None
