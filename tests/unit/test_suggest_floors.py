"""Tests for `scripts/lib/suggest_floors.py` (docs/22 P0.4 follow-up): the
floor-suggestion rounding/clamping rule, and the keyword-provider guard that
refuses to suggest floors from plumbing-check numbers. Invoked via subprocess
against synthetic `--json` artifact files, matching the
`tests/unit/test_local_model_profiles.py` precedent for testing `scripts/`
helpers as black boxes rather than importing them (`scripts/lib` is not a
package).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/lib/suggest_floors.py"


def _aggregate(
    *,
    query_count: int = 20,
    recall_at_5: float = 0.5,
    recall_at_10: float = 0.5,
    recall_at_5_strict: float = 0.4,
    ndcg_at_10: float = 0.5,
    mrr: float = 0.5,
) -> dict[str, Any]:
    return {
        "query_count": query_count,
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "recall_at_5_strict": recall_at_5_strict,
        "ndcg_at_10": ndcg_at_10,
        "mrr": mrr,
    }


def _path_report(
    *, label: str, overall: dict[str, Any], german: dict[str, Any]
) -> dict[str, Any]:
    return {"label": label, "per_query": [], "overall": overall, "german": german}


def _report(
    *, provider_label: str, reranked: dict[str, Any], raw: dict[str, Any]
) -> dict[str, Any]:
    return {"provider_label": provider_label, "reranked": reranked, "raw": raw}


def _run_suggest_floors(artifact_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(artifact_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_suggested_floors_round_down_and_clamp_to_the_current_provisional(
    tmp_path: Path,
) -> None:
    # reranked.overall.recall_at_10 = 0.72 -> rounds down to 0.70, backs off one more
    # 0.05 step -> 0.65 (clears the current 0.3 provisional, so 0.65 wins).
    # reranked.german.recall_at_10 = 0.31 -> rounds down to 0.30, backs off to 0.25,
    # which is below the current 0.30 provisional, so the suggestion clamps at 0.30.
    report = _report(
        provider_label="sentence-transformers/all-MiniLM-L6-v2",
        reranked=_path_report(
            label="reranked (production ActorContextRetriever)",
            overall=_aggregate(recall_at_10=0.72, ndcg_at_10=0.55),
            german=_aggregate(recall_at_10=0.31),
        ),
        raw=_path_report(
            label="raw dense (single-query, no rerank boosts)",
            overall=_aggregate(recall_at_10=0.70),
            german=_aggregate(recall_at_10=0.28),
        ),
    )
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps([report]), encoding="utf-8")

    result = _run_suggest_floors(artifact_path)

    assert result.returncode == 0, result.stderr
    assert "_MIN_RECALL_AT_10 = 0.65" in result.stdout
    assert "_MIN_GERMAN_RECALL_AT_10 = 0.30" in result.stdout
    assert "delta" in result.stdout
    assert "test_reranked_path_is_no_worse_than_raw_dense_on_aggregate_recall" in result.stdout
    assert "-0.05" in result.stdout


def test_keyword_only_artifact_prints_table_and_plumbing_tag_without_suggestions(
    tmp_path: Path,
) -> None:
    aggregate = _aggregate(recall_at_10=0.95, ndcg_at_10=0.9)
    report = _report(
        provider_label="keyword",
        reranked=_path_report(
            label="reranked (production ActorContextRetriever)",
            overall=aggregate,
            german=aggregate,
        ),
        raw=_path_report(
            label="raw dense (single-query, no rerank boosts)",
            overall=aggregate,
            german=aggregate,
        ),
    )
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps([report]), encoding="utf-8")

    result = _run_suggest_floors(artifact_path)

    assert result.returncode == 0, result.stderr
    assert "keyword" in result.stdout
    assert "recall@10" in result.stdout
    assert "plumbing check only" in result.stdout
    assert "_MIN_" not in result.stdout
