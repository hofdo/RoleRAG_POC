from app.composition import build_embedding_provider, build_vector_store
from app.config import Settings


def test_embedding_provider_is_cached_per_model() -> None:
    settings = Settings()
    assert build_embedding_provider(settings) is build_embedding_provider(settings)


def test_vector_store_is_cached_per_url() -> None:
    settings = Settings()
    assert build_vector_store(settings) is build_vector_store(settings)
