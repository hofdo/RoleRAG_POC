from __future__ import annotations

from app.domain import RetrievedChunk, Visibility
from app.rag.diagnostics import ChunkRetrievalDiagnostic, RetrievalDiagnostics
from app.rag.models import RagCollection


def test_retrieval_diagnostics_serialize_without_chunk_text() -> None:
    diagnostics = RetrievalDiagnostics(
        query="What did I promise the archivist?",
        selected=[
            ChunkRetrievalDiagnostic(
                id="memory-1",
                source="memory_episode:memory-1",
                source_type="session_memory",
                collection=RagCollection.SESSION_MEMORY,
                visibility=Visibility.PLAYER,
                tags=["promise", "dawn"],
                original_score=0.61,
                adjusted_score=0.80,
                applied_boosts={"collection": 0.08, "importance": 0.045},
                selected_rank=1,
            )
        ],
    )

    payload = diagnostics.model_dump(mode="json")

    assert payload == {
        "query": "What did I promise the archivist?",
        "selected": [
            {
                "adjusted_score": 0.8,
                "applied_boosts": {"collection": 0.08, "importance": 0.045},
                "collection": "session_memory",
                "id": "memory-1",
                "original_score": 0.61,
                "selected_rank": 1,
                "source": "memory_episode:memory-1",
                "source_type": "session_memory",
                "tags": ["promise", "dawn"],
                "visibility": "player",
                # Lane B slice labels (docs/26 §3.4, #79): additive, no-slice defaults.
                "slice_score": None,
                "slice_matched_terms": [],
                "slice_guaranteed": False,
            }
        ],
        "rejected": [],
    }


def test_hidden_visibility_diagnostics_only_record_metadata() -> None:
    chunk = RetrievedChunk(
        id="private-1",
        source="memory_episode:private-1",
        source_type="session_memory",
        text="Hidden private note.",
        score=0.75,
        visibility=Visibility.CHARACTER_PRIVATE,
        tags=["secret"],
    )

    diagnostic = ChunkRetrievalDiagnostic(
        id=chunk.id,
        source=chunk.source,
        source_type=chunk.source_type,
        collection=RagCollection.SESSION_MEMORY,
        visibility=chunk.visibility,
        tags=chunk.tags,
        original_score=chunk.score,
        adjusted_score=chunk.score,
        applied_boosts={},
        selected_rank=1,
    )

    assert diagnostic.model_dump(mode="json")["visibility"] == "character_private"
    assert "text" not in diagnostic.model_dump(mode="json")
