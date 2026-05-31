from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

try:
    from fastembed import TextEmbedding  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    TextEmbedding = None


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_text(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedEmbeddingProvider:
    def __init__(self, *, model_name: str) -> None:
        self._model_name = model_name
        self._model: TextEmbedding | None = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_text("dimension probe"))
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = [list(vector) for vector in self._get_model().embed(list(texts))]
        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])
        return vectors

    def _get_model(self) -> TextEmbedding:
        if TextEmbedding is None:
            raise ImportError("fastembed is required for FastEmbedEmbeddingProvider")
        if self._model is None:
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model
