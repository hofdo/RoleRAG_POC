"""Integration tests for `semantic-benchmark`'s per-provider failure tolerance: a
provider that fails (bad model name, blocked download, ...) must not abort
providers still queued, and stdout must stay machine-pure (JSON array or table)
for whichever providers succeeded.

Failing providers are simulated by patching `FastEmbedEmbeddingProvider` at its
definition site, `app.rag.embeddings` -- `app/cli.py` imports it function-locally
inside the `semantic_benchmark` command body (see `app/cli.py`'s
`semantic-benchmark` command), so there is no `app.cli.FastEmbedEmbeddingProvider`
module attribute to patch; the import binds fresh from `app.rag.embeddings` on
every invocation, which is what makes patching the definition site work.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from unittest.mock import patch

from typer.testing import CliRunner

from app.cli import app

runner = CliRunner()


class _FailingFastEmbedEmbeddingProvider:
    """Stand-in for `FastEmbedEmbeddingProvider` that mirrors its lazy-failure
    shape: construction is inert (the real class only loads the FastEmbed model on
    first use), and every access point that would trigger that load raises instead
    -- matching how an unsupported model name or a blocked download surfaces only
    once the provider is actually used, not at construction.
    """

    def __init__(self, *, model_name: str) -> None:
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        raise self._unsupported()

    def embed_text(self, text: str) -> list[float]:
        raise self._unsupported()

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        raise self._unsupported()

    def _unsupported(self) -> ValueError:
        return ValueError(f"model '{self._model_name}' is not supported by fastembed")


def test_keyword_json_happy_path() -> None:
    result = runner.invoke(app, ["semantic-benchmark", "--keyword", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert len(payload) == 1
    report = payload[0]
    assert report["provider_label"] == "keyword"
    for path in ("reranked", "raw"):
        for subset in ("overall", "german"):
            assert subset in report[path]


def test_failed_model_does_not_abort_completed_reports() -> None:
    with patch(
        "app.rag.embeddings.FastEmbedEmbeddingProvider",
        _FailingFastEmbedEmbeddingProvider,
    ):
        result = runner.invoke(
            app, ["semantic-benchmark", "--keyword", "--model", "bogus", "--json"]
        )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]["provider_label"] == "keyword"
    assert "bogus" in result.stderr
    assert "ValueError" in result.stderr


def test_all_models_failing_emits_empty_json_and_exit_1() -> None:
    with patch(
        "app.rag.embeddings.FastEmbedEmbeddingProvider",
        _FailingFastEmbedEmbeddingProvider,
    ):
        result = runner.invoke(app, ["semantic-benchmark", "--model", "bogus", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == []
    assert "[failed]" in result.stderr
    assert "1 of 1 providers failed." in result.stderr


def test_partial_failure_still_renders_table() -> None:
    with patch(
        "app.rag.embeddings.FastEmbedEmbeddingProvider",
        _FailingFastEmbedEmbeddingProvider,
    ):
        result = runner.invoke(app, ["semantic-benchmark", "--keyword", "--model", "bogus"])

    assert result.exit_code == 1
    assert "keyword" in result.stdout
    assert "reranked" in result.stdout
    assert "[failed]" in result.stderr
