"""Emit one TSV comparison row from a live-checkpoint JSON report.

Usage: ab_row.py <name> <run_status> <path-to-conversation-checkpoint.json>
Columns: config  extract_miss  select_miss  miss_ranks  recall_miss  status  total_s  mem_s

Tolerates an incomplete report (a run that hard-failed before quality_metrics
were computed): missing values render as "n/a".
"""

from __future__ import annotations

import json
import sys


def _fmt(value: object) -> str:
    return "n/a" if value is None else str(value)


def main() -> None:
    name, run_status, path = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = {}

    data = data if isinstance(data, dict) else {}
    metrics = data.get("quality_metrics") or {}
    latency = metrics.get("latency") or {}
    stage_means = metrics.get("stage_latency_means") or {}
    ranks = metrics.get("retrieval_miss_ranks") or []

    row = [
        name,
        _fmt(metrics.get("memory_extraction_misses")),
        _fmt(metrics.get("retrieval_selection_misses")),
        ",".join(str(rank) for rank in ranks) if ranks else "-",
        _fmt(metrics.get("callback_recall_misses")),
        str(data.get("status", run_status)),
        _fmt(latency.get("total_seconds")),
        _fmt(stage_means.get("memory")),
    ]
    print("\t".join(row))


if __name__ == "__main__":
    main()
