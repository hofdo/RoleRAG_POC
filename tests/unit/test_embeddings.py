from __future__ import annotations

from collections.abc import Sequence

import pytest

import app.rag.embeddings as embeddings
from app.rag.embeddings import FastEmbedEmbeddingProvider


class _FakeModel:
    """Stands in for fastembed's TextEmbedding: records calls, returns fixed-dim vectors."""

    def __init__(self, *, dim: int = 3) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index)] * self.dim for index, _ in enumerate(texts, start=1)]


def _provider_with_fake(model: _FakeModel) -> FastEmbedEmbeddingProvider:
    provider = FastEmbedEmbeddingProvider(model_name="fake-model")
    provider._model = model  # bypass the lazy fastembed load
    return provider


def test_embed_batch_empty_short_circuits_without_touching_model() -> None:
    provider = FastEmbedEmbeddingProvider(model_name="fake-model")  # no model loaded
    assert provider.embed_batch([]) == []


def test_embed_batch_wraps_model_vectors_as_lists() -> None:
    model = _FakeModel(dim=3)
    provider = _provider_with_fake(model)

    vectors = provider.embed_batch(["a", "b"])

    assert vectors == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    assert all(isinstance(vector, list) for vector in vectors)
    assert model.calls == [["a", "b"]]


def test_embed_text_returns_the_single_vector() -> None:
    provider = _provider_with_fake(_FakeModel(dim=2))
    assert provider.embed_text("solo") == [1.0, 1.0]


def test_dimension_is_cached_from_batch_without_reprobing() -> None:
    model = _FakeModel(dim=4)
    provider = _provider_with_fake(model)

    provider.embed_batch(["x"])
    assert provider.dimension == 4
    assert len(model.calls) == 1  # dimension reused the batch result, did not re-embed


def test_dimension_probes_when_uncached() -> None:
    model = _FakeModel(dim=5)
    provider = _provider_with_fake(model)

    assert provider.dimension == 5
    assert model.calls == [["dimension probe"]]


def test_get_model_raises_when_fastembed_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings, "FastEmbedTextEmbedding", None)
    provider = FastEmbedEmbeddingProvider(model_name="fake-model")  # _model is None

    with pytest.raises(ImportError):
        provider.embed_batch(["x"])
