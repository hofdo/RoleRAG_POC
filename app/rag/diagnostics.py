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
    selected_rank: int = Field(ge=1)


class RetrievalDiagnostics(BaseModel):
    query: str
    selected: list[ChunkRetrievalDiagnostic] = Field(default_factory=list)


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    diagnostics: RetrievalDiagnostics
