from typing import cast

from app.composition import build_embedding_provider, build_vector_store
from app.config import Settings
from app.rag.vector_store import QdrantVectorStore


def test_embedding_provider_is_cached_per_model() -> None:
    settings = Settings()
    assert build_embedding_provider(settings) is build_embedding_provider(settings)


def test_vector_store_is_cached_per_url() -> None:
    settings = Settings()
    assert build_vector_store(settings) is build_vector_store(settings)


def test_vector_store_wires_scalar_quantization_flag_from_settings() -> None:
    # P2.1 (docs/22): settings.qdrant_scalar_quantization must reach the constructed
    # QdrantVectorStore. The (url, scalar_quantization) cache key on _cached_vector_store
    # keeps this instance distinct from the default-flag instance other tests build.
    settings = Settings(qdrant_scalar_quantization=True)

    store = cast(QdrantVectorStore, build_vector_store(settings))

    assert store._scalar_quantization is True
