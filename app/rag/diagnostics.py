from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.domain import RetrievedChunk
from app.domain.visibility import Visibility
from app.rag.models import RagCollection


class ChunkRetrievalDiagnostic(BaseModel):
    id: str
    source: str
    source_type: str
    collection: RagCollection
    visibility: Visibility
    tags: list[str] = Field(default_factory=list)
    original_score: float
    adjusted_score: float
    applied_boosts: dict[str, float] = Field(default_factory=dict)
    selected_rank: int | None = Field(default=None, ge=1)
    # Lane B lexical slice labels (docs/26 §3.4, #79). Additive, all default to the
    # no-slice values so quota-off diagnostics are byte-identical. slice_score is a
    # DEDICATED field (fix 3): it is the matched terms' summed session-pool IDF and
    # is NOT folded into applied_boosts, so the
    # adjusted_score == original_score + sum(applied_boosts) identity holds for
    # every chunk (an injected member is original 0.0 / boosts {} / slice_score>0).
    slice_score: float | None = None
    slice_matched_terms: list[str] = Field(default_factory=list)
    slice_guaranteed: bool = False


class RetrievalDiagnostics(BaseModel):
    query: str
    selected: list[ChunkRetrievalDiagnostic] = Field(default_factory=list)
    rejected: list[ChunkRetrievalDiagnostic] = Field(default_factory=list)


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    diagnostics: RetrievalDiagnostics
