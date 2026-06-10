from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.diagnostics.model_comparison import (
    aligned_transcript,
    build_comparison,
    classify_outcome,
    write_comparison,
)


@pytest.mark.parametrize(
    ("deterministic", "small", "model_26b", "expected"),
    [
        (1, 0, 0, "application failure"),
        (0, 0, 0, "both passed"),
        (0, 1, 0, "small-model-only"),
        (0, 0, 1, "26B-only"),
        (0, 1, 1, "shared/model interaction"),
    ],
)
def test_classification(
    deterministic: int,
    small: int,
    model_26b: int,
    expected: str,
) -> None:
    assert classify_outcome(
        deterministic_exit=deterministic,
        small_exit=small,
        model_26b_exit=model_26b,
    ) == expected


def test_comparison_keeps_quality_findings_report_only() -> None:
    checkpoint = {
        "status": "pass",
        "warning_counts": {"critic": 2},
        "events": [{"recalled": False}],
        "quality_metrics": {"callback_recall_misses": 1},
    }
    comparison = build_comparison(
        deterministic_exit=0,
        small_exit=0,
        model_26b_exit=0,
        small=checkpoint,
        model_26b=checkpoint,
    )
    assert comparison["classification"] == "both passed"
    assert comparison["prose_quality_winner"] is None


def test_aligned_transcript_handles_partial_model_run() -> None:
    small = {
        "turns": [
            {"turn_index": 1, "prompt": "One", "response": "Small one"},
            {"turn_index": 2, "prompt": "Two", "response": "Small two"},
        ]
    }
    model_26b = {"turns": [{"turn_index": 1, "prompt": "One", "response": "Large one"}]}
    rows = aligned_transcript(small, model_26b)
    assert len(rows) == 2
    assert rows[1]["26b"]["available"] is False


def test_write_comparison_creates_all_artifacts(tmp_path: Path) -> None:
    checkpoint = {
        "status": "pass",
        "turns": [{"turn_index": 1, "prompt": "One", "response": "Answer"}],
        "warning_counts": {},
        "events": [],
        "quality_metrics": {},
    }
    for profile in ("small", "26b"):
        path = tmp_path / profile / "raw"
        path.mkdir(parents=True)
        (path / "conversation-checkpoint.json").write_text(json.dumps(checkpoint))
    write_comparison(
        output_dir=tmp_path,
        deterministic_exit=0,
        small_exit=0,
        model_26b_exit=0,
    )
    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / "comparison.md").exists()
    assert (tmp_path / "transcript-comparison.json").exists()
