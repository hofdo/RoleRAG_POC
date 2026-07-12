"""Suggest recall/nDCG floor updates for tests/evals/test_semantic_benchmark_opt_in.py
from a `rolerag semantic-benchmark --json` artifact (docs/22 P0.4 follow-up).

Usage:
    python scripts/lib/suggest_floors.py <path-to-artifact.json>

The artifact is the JSON array `rolerag semantic-benchmark --json` (and
`scripts/semantic-benchmark.sh`) writes: one report per benchmarked provider, each
shaped like `dataclasses.asdict(SemanticBenchmarkReport)` from
`app/diagnostics/semantic_benchmark.py` --

    {
      "provider_label": str,
      "reranked": {"label": str, "per_query": [...], "overall": {agg}, "german": {agg}},
      "raw": {same shape as "reranked"},
    }

where each `agg` (`SemanticAggregate`) is `{"query_count", "recall_at_5",
"recall_at_10", "recall_at_5_strict", "ndcg_at_10", "mrr"}`.

For each report this prints a compact aggregate table (reranked/raw x
overall/german) and, for real (non-keyword) providers, paste-ready suggested
floor constants for tests/evals/test_semantic_benchmark_opt_in.py. The keyword
provider is a plumbing check only (deterministic keyword-count vectors, not a
real embedding model), so its numbers never produce floor suggestions.

This script only reads the artifact and prints to stdout; it never edits files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

# Provisional floors currently hardcoded in tests/evals/test_semantic_benchmark_opt_in.py
# (lines ~55-61, as of docs/22 P0.4). If a floor is ever hand-tuned outside of this
# helper's suggestions, update that file AND these constants together so future
# suggestions clamp against the right baseline.
_CURRENT_MIN_RECALL_AT_10 = 0.3  # tests/evals/test_semantic_benchmark_opt_in.py:55
_CURRENT_MIN_NDCG_AT_10 = 0.2  # tests/evals/test_semantic_benchmark_opt_in.py:56
_CURRENT_MIN_GERMAN_RECALL_AT_10 = 0.30  # tests/evals/test_semantic_benchmark_opt_in.py:61

# (SemanticAggregate field, table column header, format spec).
_AGGREGATE_COLUMNS = (
    ("query_count", "n", "{:d}"),
    ("recall_at_5", "recall@5", "{:.3f}"),
    ("recall_at_10", "recall@10", "{:.3f}"),
    ("recall_at_5_strict", "recall@5(strict)", "{:.3f}"),
    ("ndcg_at_10", "ndcg@10", "{:.3f}"),
    ("mrr", "mrr", "{:.3f}"),
)

# test_reranked_path_is_no_worse_than_raw_dense_on_aggregate_recall allows the reranked
# path's recall@10 to trail the raw dense path's by up to this much.
_ALLOWED_RERANKED_VS_RAW_RECALL_DROP = 0.05

# One rounding step (0.05) between consecutive suggested floors.
_FLOOR_STEP = 0.05


def suggest_floor(observed: float, current_provisional: float) -> float:
    """Round `observed` down to the nearest `_FLOOR_STEP`, then back off one more
    step so the suggestion has headroom under the observed run. Never suggests
    below `current_provisional` -- this is a floor *suggestion*, not licence to
    loosen whatever is already checked in.
    """
    stepped_down = math.floor(observed / _FLOOR_STEP + 1e-9) * _FLOOR_STEP - _FLOOR_STEP
    return max(current_provisional, stepped_down)


def _format_aggregate_table(report: dict[str, Any]) -> str:
    headers = ["path", "subset", *(label for _, label, _ in _AGGREGATE_COLUMNS)]
    rows: list[list[str]] = []
    for path_key in ("reranked", "raw"):
        for subset_key in ("overall", "german"):
            aggregate = report[path_key][subset_key]
            row = [path_key, subset_key]
            row.extend(fmt.format(aggregate[field]) for field, _, fmt in _AGGREGATE_COLUMNS)
            rows.append(row)

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) for col in range(len(headers))
    ]

    def _format_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[col]) for col, cell in enumerate(cells))

    lines = [
        _format_row(headers),
        "  ".join("-" * widths[col] for col in range(len(headers))),
        *(_format_row(row) for row in rows),
    ]
    return "\n".join(lines)


def _print_report(report: dict[str, Any]) -> None:
    provider_label = report["provider_label"]
    print(f"=== provider: {provider_label} ===")
    print(_format_aggregate_table(report))

    if provider_label == "keyword":
        print(
            "keyword provider = plumbing check only -- do NOT calibrate floors from "
            "these numbers (deterministic keyword-count vectors, not a real embedding "
            "model)."
        )
        print()
        return

    reranked_recall_10 = report["reranked"]["overall"]["recall_at_10"]
    reranked_ndcg_10 = report["reranked"]["overall"]["ndcg_at_10"]
    reranked_german_recall_10 = report["reranked"]["german"]["recall_at_10"]
    raw_recall_10 = report["raw"]["overall"]["recall_at_10"]

    suggested_recall_at_10 = suggest_floor(reranked_recall_10, _CURRENT_MIN_RECALL_AT_10)
    suggested_ndcg_at_10 = suggest_floor(reranked_ndcg_10, _CURRENT_MIN_NDCG_AT_10)
    suggested_german_recall_at_10 = suggest_floor(
        reranked_german_recall_10, _CURRENT_MIN_GERMAN_RECALL_AT_10
    )

    print("suggested floors for tests/evals/test_semantic_benchmark_opt_in.py:")
    print(f"_MIN_RECALL_AT_10 = {suggested_recall_at_10:.2f}")
    print(f"_MIN_NDCG_AT_10 = {suggested_ndcg_at_10:.2f}")
    print(f"_MIN_GERMAN_RECALL_AT_10 = {suggested_german_recall_at_10:.2f}")

    delta = reranked_recall_10 - raw_recall_10
    print(
        f"reranked-raw recall@10 delta (overall): {delta:+.3f} -- "
        "test_reranked_path_is_no_worse_than_raw_dense_on_aggregate_recall allows "
        f"down to -{_ALLOWED_RERANKED_VS_RAW_RECALL_DROP:.2f}"
    )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact_path",
        help="Path to a `rolerag semantic-benchmark --json` artifact file.",
    )
    args = parser.parse_args(argv)

    with open(args.artifact_path, encoding="utf-8") as handle:
        reports = json.load(handle)

    if not isinstance(reports, list):
        print(
            f"error: {args.artifact_path} does not contain a JSON array of reports",
            file=sys.stderr,
        )
        return 1

    if not reports:
        print("no reports in artifact (every provider may have failed) -- nothing to suggest.")
        return 0

    for report in reports:
        _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
