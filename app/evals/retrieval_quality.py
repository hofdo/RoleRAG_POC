from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain import RetrievedChunk
from app.evals.fixtures import EvalFixture


class RetrievalQualityResult(BaseModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)


def evaluate_retrieval_quality(fixture: EvalFixture, *, top_k: int = 5) -> RetrievalQualityResult:
    ranked_ids = [chunk.id for chunk in fixture.retrieve_actor_chunks(top_k=top_k)]
    relevant_index = _index_of(ranked_ids, fixture.relevant_memory_chunk_id)
    irrelevant_index = _index_of(ranked_ids, fixture.irrelevant_memory_chunk_id)
    world_chunks = fixture.retrieve_actor_chunks_for_query(
        query=fixture.build_world_knowledge_query(),
        top_k=top_k,
    )
    session_chunks = fixture.retrieve_actor_chunks_for_query(
        query=fixture.build_session_memory_query(),
        top_k=top_k,
    )
    persona_chunks = fixture.retrieve_actor_chunks_for_query(
        query=fixture.build_persona_memory_query(),
        top_k=top_k,
    )
    isolation_ids = {
        chunk.id
        for chunk in fixture.retrieve_actor_chunks_for_query(
            query=fixture.build_isolation_query(),
            top_k=top_k,
        )
    }
    checks = {
        "canon_lore_recalled": fixture.canon_lore_chunk_id in _ids(world_chunks),
        "session_memory_recalled": fixture.session_memory_chunk_id in _ids(session_chunks),
        "persona_memory_recalled": fixture.persona_memory_chunk_id in _ids(persona_chunks),
        "wrong_world_excluded": fixture.wrong_world_chunk_id not in isolation_ids,
        "wrong_session_excluded": fixture.wrong_session_chunk_id not in isolation_ids,
        "wrong_persona_excluded": fixture.wrong_persona_chunk_id not in isolation_ids,
        "gm_only_excluded": fixture.gm_only_chunk_id not in isolation_ids,
        "character_private_excluded": fixture.character_private_chunk_id not in isolation_ids,
        "relevant_memory_ranked_above_irrelevant": (
            relevant_index is not None
            and irrelevant_index is not None
            and relevant_index < irrelevant_index
        ),
    }
    return RetrievalQualityResult(passed=all(checks.values()), checks=checks)


def _ids(chunks: list[RetrievedChunk]) -> set[str]:
    return {chunk.id for chunk in chunks}


def _index_of(values: list[str], target: str) -> int | None:
    try:
        return values.index(target)
    except ValueError:
        return None
