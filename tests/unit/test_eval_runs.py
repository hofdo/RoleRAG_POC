from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.diagnostics.eval_runs import load_eval_run, load_eval_runs


def _write_checkpoint(run_dir: Path, payload: dict[str, Any], *, nested: bool = False) -> None:
    target = run_dir / "raw" if nested else run_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / "conversation-checkpoint.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_eval_runs_summarizes_quality_metrics(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path / "run-a",
        {
            "status": "pass",
            "persisted": {"turn_count": 50},
            "warning_counts": {"critic": 1, "memory": 2},
            "quality_metrics": {
                "callback_recall_misses": 3,
                "memory_extraction_misses": 1,
                "retrieval_selection_misses": 0,
                "latency": {"total_seconds": 120.0, "p50_seconds": 2.1, "p95_seconds": 8.4},
            },
        },
        nested=True,  # model_bakeoff layout: run-a/raw/conversation-checkpoint.json
    )

    base, runs = load_eval_runs(tmp_path)

    assert base == tmp_path
    assert len(runs) == 1
    run = runs[0]
    assert run["id"] == "run-a"
    assert run["status"] == "pass"
    assert run["turn_count"] == 50
    assert run["recall_misses"] == 3
    assert run["warning_total"] == 3
    assert run["p95_seconds"] == 8.4


def test_load_eval_runs_skips_incomplete_and_missing_base(tmp_path: Path) -> None:
    (tmp_path / "empty-run").mkdir()
    _write_checkpoint(tmp_path / "good-run", {"status": "pass"})

    _base, runs = load_eval_runs(tmp_path)
    assert [run["id"] for run in runs] == ["good-run"]
    # Defaults applied when metrics absent.
    assert runs[0]["recall_misses"] == 0
    assert runs[0]["turn_count"] == 0

    _missing_base, missing_runs = load_eval_runs(tmp_path / "does-not-exist")
    assert missing_runs == []


def test_load_eval_run_returns_payload_and_blocks_traversal(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path / "run-a",
        {"status": "pass", "quality_metrics": {"callback_recall_misses": 2}},
        nested=True,
    )
    payload = load_eval_run("run-a", tmp_path)
    assert payload is not None
    assert payload["quality_metrics"]["callback_recall_misses"] == 2
    assert load_eval_run("nope", tmp_path) is None
    # An untrusted run_id that escapes the results dir must be rejected.
    assert load_eval_run("../../etc", tmp_path) is None
