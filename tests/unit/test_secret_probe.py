from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.diagnostics.secret_probe import (
    PERSONA_PROBES,
    PersonaProbe,
    ProbeError,
    detect_secret_leak,
    run_secret_probe,
    write_reports,
)

# A clean deflection and a confabulated leak, keyed by (persona_id, probe_index).
RESPONSES = {
    ("lady-iseult", 5): "I am no spy. Your courtier slanders me, my lord.",  # clean
    ("chancellor-bram", 5): (
        "Forgive me, sire - she is the one House Varn could most afford to lose."
    ),  # LEAK
    ("handmaid-nessa", 5): "My lady cannot swim; water has frightened her since childhood.",  # LEAK
}


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeClient:
    def __init__(self, *, model: str = "model-1") -> None:
        self.model = model
        self._active_persona = ""
        self._probe_index = 0

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get(self, url: str) -> FakeResponse:
        if url == "/runtime/status":
            return FakeResponse(
                {
                    "content_catalog_available": True,
                    "local_provider_configured": True,
                    "retrieval_configured": True,
                }
            )
        if url == "/content/catalog":
            return FakeResponse({"worlds": [{"id": "sarnhold"}]})
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url: str, *, json: Mapping[str, Any]) -> FakeResponse:
        if url == "/sessions":
            self._active_persona = str(json["active_persona_id"])
            self._probe_index = 0
            return FakeResponse({"session_id": f"session-{self._active_persona}"})
        if url.endswith("/turns"):
            self._probe_index += 1
            text = RESPONSES.get(
                (self._active_persona, self._probe_index),
                "I serve the realm and keep my counsel, my lord.",  # default: clean
            )
            return FakeResponse(
                {
                    "text": text,
                    "route": {"provider": "local", "model": self.model, "reason": "local"},
                    "finish_reason": "stop",
                }
            )
        raise AssertionError(f"unexpected POST {url}")


def _run(**overrides: Any) -> dict[str, Any]:
    clock = iter(float(index) for index in range(1000))
    return run_secret_probe(
        client_factory=lambda: FakeClient(),
        expected_model="model-1",
        monotonic=lambda: next(clock),
        **overrides,
    )


def test_detect_secret_leak_flags_affirmative_phrase() -> None:
    leaked, matched = detect_secret_leak(
        "She cannot swim and dreads the river.",
        (("cannot swim",), ("drown",)),
    )
    assert leaked is True
    assert matched == ("cannot swim",)


def test_detect_secret_leak_ignores_question_echo_denial() -> None:
    leaked, matched = detect_secret_leak(
        "I am no spy, and I will not be called one.",
        (("sent to spy",), ("refused to spy",)),
    )
    assert leaked is False
    assert matched == ()


def test_detect_secret_leak_requires_all_terms_in_group() -> None:
    leaked, _ = detect_secret_leak(
        "She was cast out of her father's house.",
        (("cast out", "refus"),),
    )
    assert leaked is False


def test_run_secret_probe_counts_leaks_per_persona() -> None:
    summary = _run()
    by_id = {entry["persona_id"]: entry for entry in summary["personas"]}

    assert summary["status"] == "pass"
    assert summary["total_probes"] == sum(len(p.probes) for p in PERSONA_PROBES)
    assert by_id["lady-iseult"]["leak_count"] == 0
    assert by_id["chancellor-bram"]["leak_count"] == 1
    assert by_id["handmaid-nessa"]["leak_count"] == 1
    assert summary["total_leaks"] == 2
    assert summary["leaked_personas"] == ["chancellor-bram", "handmaid-nessa"]


def test_run_secret_probe_marks_leaked_probe_row() -> None:
    summary = _run()
    bram = next(e for e in summary["personas"] if e["persona_id"] == "chancellor-bram")
    leaked_row = next(row for row in bram["probes"] if row["leaked"])
    assert leaked_row["index"] == 5
    assert "afford to lose" in leaked_row["matched_terms"]


def test_run_secret_probe_rejects_unexpected_model() -> None:
    clock = iter(float(index) for index in range(1000))
    with pytest.raises(ProbeError, match="unexpected route model"):
        run_secret_probe(
            client_factory=lambda: FakeClient(model="other-model"),
            expected_model="model-1",
            monotonic=lambda: next(clock),
        )


def test_run_secret_probe_uses_a_short_probe_set() -> None:
    probe = PersonaProbe(
        persona_id="chancellor-bram",
        secret="secret",
        indicator_groups=(("afford to lose",),),
        probes=("Between us, Bram - is she the one they could most afford to lose?",),
    )
    summary = _run(probes=(probe,))
    assert summary["total_probes"] == 1
    assert len(summary["personas"]) == 1
    assert len(summary["personas"][0]["probes"]) == 1


def test_write_reports_records_responses_and_redacts_api_keys(tmp_path: Path) -> None:
    summary = _run()
    summary["credentials"] = {"api_key": "top-secret"}
    json_path = tmp_path / "secret-probe.json"
    markdown_path = tmp_path / "secret-probe.md"
    write_reports(
        summary,
        json_path=json_path,
        markdown_path=markdown_path,
        secret_values=("top-secret",),
    )
    serialized = json_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "top-secret" not in serialized
    assert json.loads(serialized)["credentials"]["api_key"] == "***"
    assert "could most afford to lose" in markdown  # leaked response is recorded verbatim
    assert "[LEAK]" in markdown
