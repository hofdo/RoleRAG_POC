from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.diagnostics.model_bakeoff import (
    RunSpec,
    build_report,
    build_row,
    parse_run_arg,
    render_markdown,
)


def _make_run(
    base: Path,
    label: str,
    *,
    checkpoint: dict[str, object] | None,
    secret: dict[str, object] | None,
    struct_fail_lines: int = 0,
) -> RunSpec:
    run_dir = base / label
    if checkpoint is not None:
        path = run_dir / "raw" / "conversation-checkpoint.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
    if secret is not None:
        path = run_dir / "secret-probe.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(secret), encoding="utf-8")
    if struct_fail_lines:
        path = run_dir / "raw" / "structured-failures" / "structured-output-failures.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps({"task": "critic", "n": i}) for i in range(struct_fail_lines)),
            encoding="utf-8",
        )
    return RunSpec(label=label, run_dir=run_dir)


def _checkpoint(**metrics: object) -> dict[str, object]:
    base: dict[str, object] = {
        "callback_recall_misses": 0,
        "retrieval_selection_misses": 0,
        "memory_extraction_misses": 0,
        "latency": {"p50_seconds": 12.0, "p95_seconds": 30.0},
        "finish_reason_distribution": {"stop": 49, "length": 1},
        "response_chars": [100, 200, 300],
    }
    base.update(metrics)
    return {"status": "pass", "quality_metrics": base}


@pytest.mark.parametrize(
    ("value", "label", "path"),
    [
        ("A:/tmp/run-a", "A", "/tmp/run-a"),
        ("gemma26b:/tmp/rolerag-bakeoff/A-gemma26b", "gemma26b", "/tmp/rolerag-bakeoff/A-gemma26b"),
    ],
)
def test_parse_run_arg(value: str, label: str, path: str) -> None:
    spec = parse_run_arg(value)
    assert spec.label == label
    assert str(spec.run_dir) == path


@pytest.mark.parametrize("value", ["nocolon", ":/tmp/dir", "label:"])
def test_parse_run_arg_rejects_malformed(value: str) -> None:
    with pytest.raises(ValueError, match="LABEL:DIR"):
        parse_run_arg(value)


def test_build_row_extracts_all_metrics(tmp_path: Path) -> None:
    spec = _make_run(
        tmp_path,
        "A",
        checkpoint=_checkpoint(callback_recall_misses=2, retrieval_selection_misses=1),
        secret={"total_leaks": 1, "total_probes": 15, "leaked_personas": ["chancellor-bram"]},
        struct_fail_lines=3,
    )
    row = build_row(spec)
    assert row["recall_status"] == "pass"
    assert row["recall_miss"] == 2
    assert row["select_miss"] == 1
    assert row["extract_miss"] == 0
    assert row["secret_leaks"] == 1
    assert row["p50_s"] == 12.0
    assert row["p95_s"] == 30.0
    assert row["finish_len"] == 1
    assert row["struct_fail"] == 3
    assert row["mean_resp_chars"] == 200.0
    assert row["checkpoint_available"] is True
    assert row["secret_probe_available"] is True


def test_build_row_tolerates_missing_artifacts(tmp_path: Path) -> None:
    spec = _make_run(tmp_path, "B", checkpoint=None, secret=None)
    row = build_row(spec)
    assert row["checkpoint_available"] is False
    assert row["secret_probe_available"] is False
    assert row["recall_miss"] is None
    assert row["secret_leaks"] is None
    assert row["struct_fail"] == 0
    assert row["finish_len"] == 0


def test_build_report_and_markdown_render_three_models(tmp_path: Path) -> None:
    specs = [
        _make_run(
            tmp_path,
            "A-gemma26b",
            checkpoint=_checkpoint(callback_recall_misses=0),
            secret={"total_leaks": 0, "total_probes": 15, "leaked_personas": []},
        ),
        _make_run(
            tmp_path,
            "B-qwen35b",
            checkpoint=_checkpoint(callback_recall_misses=3),
            secret={"total_leaks": 2, "total_probes": 15, "leaked_personas": ["handmaid-nessa"]},
            struct_fail_lines=5,
        ),
        _make_run(tmp_path, "C-qwen27b", checkpoint=None, secret=None),
    ]
    report = build_report(specs)
    assert len(report["rows"]) == 3

    markdown = render_markdown(report)
    assert "# Local Model Bake-Off" in markdown
    assert "A-gemma26b" in markdown
    assert "B-qwen35b" in markdown
    # C has no artifacts -> flagged as incomplete
    assert "## Incomplete runs" in markdown
    assert "C-qwen27b" in markdown
    # table contains the leak count for B
    assert "| B-qwen35b" in markdown
