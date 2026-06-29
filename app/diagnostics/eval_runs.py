from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Read-only aggregation of eval run artifacts for the Eval dashboard. Each run lives in its own
# subdirectory under the results dir and carries a live-smoke `conversation-checkpoint.json`
# (model_bakeoff writes it at `raw/conversation-checkpoint.json`). We surface only the headline
# quality_metrics so the dashboard can chart recall/latency/warnings across runs.
#
# ponytail: the results dir is read from EVAL_RESULTS_DIR (default <repo>/eval-runs) rather than
# threaded through Settings — this is a dev-only diagnostics surface, not request-path config.

_CHECKPOINT_NAMES = ("raw/conversation-checkpoint.json", "conversation-checkpoint.json")


def default_results_dir() -> Path:
    override = os.environ.get("EVAL_RESULTS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "eval-runs"


def _find_checkpoint(run_dir: Path) -> Path | None:
    for name in _CHECKPOINT_NAMES:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    return None


def _summarize(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    quality_metrics = payload.get("quality_metrics") or {}
    latency = quality_metrics.get("latency") or {}
    warning_counts = payload.get("warning_counts") or {}
    persisted = payload.get("persisted") or {}
    return {
        "id": run_id,
        "status": str(payload.get("status", "unknown")),
        "turn_count": int(persisted.get("turn_count", 0)),
        "recall_misses": int(quality_metrics.get("callback_recall_misses", 0)),
        "extraction_misses": int(quality_metrics.get("memory_extraction_misses", 0)),
        "retrieval_misses": int(quality_metrics.get("retrieval_selection_misses", 0)),
        "total_seconds": float(latency.get("total_seconds", 0.0)),
        "p50_seconds": float(latency.get("p50_seconds", 0.0)),
        "p95_seconds": float(latency.get("p95_seconds", 0.0)),
        "warning_total": int(sum(int(value) for value in warning_counts.values())),
    }


def load_eval_runs(results_dir: Path | None = None) -> tuple[Path, list[dict[str, Any]]]:
    """Return (results_dir, run summaries sorted by run id). Missing dir → empty list."""
    base = results_dir if results_dir is not None else default_results_dir()
    runs: list[dict[str, Any]] = []
    if base.is_dir():
        for run_dir in sorted(path for path in base.iterdir() if path.is_dir()):
            checkpoint = _find_checkpoint(run_dir)
            if checkpoint is None:
                continue
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(payload, dict):
                runs.append(_summarize(run_dir.name, payload))
    return base, runs
