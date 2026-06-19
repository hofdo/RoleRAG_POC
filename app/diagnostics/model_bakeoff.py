"""N-way aggregator for the local-model bake-off.

Reads, per model run directory, the recall checkpoint (`live_checkpoint`), the
secret-containment probe (`secret_probe`), and the structured-output failure log, then
emits a single side-by-side comparison table. Read-only: it consumes already-written
artifacts and never starts a server or model.

Sibling to the hardwired two-model `model_comparison.py`, which is left untouched.

Lower is better for every miss/leak/latency/struct/finish-length column. No winner is
declared automatically - prose quality and the leak/recall trade-off are a human call.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_REL = "raw/conversation-checkpoint.json"
SECRET_PROBE_REL = "secret-probe.json"
STRUCT_FAIL_REL = "raw/structured-failures/structured-output-failures.jsonl"

# (row_key, column_header) in display order.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("label", "label"),
    ("recall_status", "status"),
    ("recall_miss", "recall_miss"),
    ("select_miss", "select_miss"),
    ("extract_miss", "extract_miss"),
    ("secret_leaks", "secret_leaks"),
    ("p50_s", "p50_s"),
    ("p95_s", "p95_s"),
    ("finish_len", "finish_len"),
    ("struct_fail", "struct_fail"),
    ("mean_resp_chars", "mean_resp_chars"),
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_dir: Path


def parse_run_arg(value: str) -> RunSpec:
    """Parse a ``LABEL:DIR`` argument (DIR may itself contain no leading colon)."""
    label, separator, path = value.partition(":")
    if not separator or not label.strip() or not path.strip():
        raise ValueError(f"--run must be LABEL:DIR, got: {value!r}")
    return RunSpec(label=label.strip(), run_dir=Path(path.strip()))


def build_row(spec: RunSpec) -> dict[str, Any]:
    checkpoint = _load_json(spec.run_dir / CHECKPOINT_REL)
    secret = _load_json(spec.run_dir / SECRET_PROBE_REL)
    struct_fail = _count_lines(spec.run_dir / STRUCT_FAIL_REL)

    quality = _as_mapping(checkpoint.get("quality_metrics") if checkpoint else None)
    latency = _as_mapping(quality.get("latency"))
    finish = _as_mapping(quality.get("finish_reason_distribution"))
    response_chars = quality.get("response_chars")
    mean_resp_chars = (
        round(statistics.mean(response_chars), 1)
        if isinstance(response_chars, list) and response_chars
        else None
    )
    return {
        "label": spec.label,
        "run_dir": str(spec.run_dir),
        "checkpoint_available": checkpoint is not None,
        "secret_probe_available": secret is not None,
        "recall_status": checkpoint.get("status") if checkpoint else None,
        "recall_miss": quality.get("callback_recall_misses"),
        "select_miss": quality.get("retrieval_selection_misses"),
        "extract_miss": quality.get("memory_extraction_misses"),
        "secret_leaks": secret.get("total_leaks") if secret else None,
        "secret_probes": secret.get("total_probes") if secret else None,
        "leaked_personas": secret.get("leaked_personas") if secret else None,
        "p50_s": latency.get("p50_seconds"),
        "p95_s": latency.get("p95_seconds"),
        "finish_len": finish.get("length", 0),
        "struct_fail": struct_fail,
        "mean_resp_chars": mean_resp_chars,
    }


def build_report(specs: Sequence[RunSpec]) -> dict[str, Any]:
    rows = [build_row(spec) for spec in specs]
    return {
        "rows": rows,
        "columns": [header for _key, header in COLUMNS],
        "note": "lower is better for every miss/leak/latency/struct/finish_len column",
    }


def render_table(rows: Sequence[dict[str, Any]]) -> str:
    headers = [header for _key, header in COLUMNS]
    body = [[_cell(row.get(key)) for key, _header in COLUMNS] for row in rows]
    widths = [
        max(len(headers[col]), *(len(line[col]) for line in body)) if body else len(headers[col])
        for col in range(len(headers))
    ]

    def format_row(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[col]) for col, cell in enumerate(cells)) + " |"

    lines = [
        format_row(headers),
        "| " + " | ".join("-" * widths[col] for col in range(len(headers))) + " |",
        *(format_row(line) for line in body),
    ]
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    rows = report["rows"]
    lines = [
        "# Local Model Bake-Off",
        "",
        f"_{report['note']}._",
        "",
        render_table(rows),
        "",
    ]
    incomplete = [
        row["label"]
        for row in rows
        if not row["checkpoint_available"] or not row["secret_probe_available"]
    ]
    if incomplete:
        lines.extend(
            [
                "## Incomplete runs",
                "",
                *(
                    f"- {row['label']}: checkpoint={row['checkpoint_available']}, "
                    f"secret_probe={row['secret_probe_available']} ({row['run_dir']})"
                    for row in rows
                    if row["label"] in incomplete
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ",".join(str(item) for item in value) if value else "-"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate local-model bake-off run artifacts.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL:DIR",
        help="A model run as LABEL:DIR (repeatable).",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args()

    specs = [parse_run_arg(value) for value in args.run]
    report = build_report(specs)
    markdown = render_markdown(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown + "\n", encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
