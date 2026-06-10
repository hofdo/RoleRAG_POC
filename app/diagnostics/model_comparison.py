from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def classify_outcome(*, deterministic_exit: int, small_exit: int, model_26b_exit: int) -> str:
    if deterministic_exit != 0:
        return "application failure"
    if small_exit == 0 and model_26b_exit == 0:
        return "both passed"
    if small_exit != 0 and model_26b_exit == 0:
        return "small-model-only"
    if small_exit == 0 and model_26b_exit != 0:
        return "26B-only"
    return "shared/model interaction"


def build_comparison(
    *,
    deterministic_exit: int,
    small_exit: int,
    model_26b_exit: int,
    small: Mapping[str, Any] | None,
    model_26b: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "classification": classify_outcome(
            deterministic_exit=deterministic_exit,
            small_exit=small_exit,
            model_26b_exit=model_26b_exit,
        ),
        "exit_codes": {
            "deterministic": deterministic_exit,
            "small": small_exit,
            "26b": model_26b_exit,
        },
        "models": {
            "small": _quality_summary(small),
            "26b": _quality_summary(model_26b),
        },
        "prose_quality_winner": None,
    }


def aligned_transcript(
    small: Mapping[str, Any] | None,
    model_26b: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    small_turns = _turns_by_index(small)
    model_26b_turns = _turns_by_index(model_26b)
    return [
        {
            "turn_index": turn_index,
            "prompt": (small_turns.get(turn_index) or model_26b_turns[turn_index]).get("prompt"),
            "small": _turn_summary(small_turns.get(turn_index)),
            "26b": _turn_summary(model_26b_turns.get(turn_index)),
        }
        for turn_index in sorted(set(small_turns) | set(model_26b_turns))
    ]


def write_comparison(
    *,
    output_dir: Path,
    deterministic_exit: int,
    small_exit: int,
    model_26b_exit: int,
) -> dict[str, Any]:
    small = _load_checkpoint(output_dir / "small/raw/conversation-checkpoint.json")
    model_26b = _load_checkpoint(output_dir / "26b/raw/conversation-checkpoint.json")
    comparison = build_comparison(
        deterministic_exit=deterministic_exit,
        small_exit=small_exit,
        model_26b_exit=model_26b_exit,
        small=small,
        model_26b=model_26b,
    )
    aligned = aligned_transcript(small, model_26b)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "transcript-comparison.json").write_text(
        json.dumps(aligned, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Local Model Comparison",
        "",
        f"- classification: {comparison['classification']}",
        f"- deterministic_exit: {deterministic_exit}",
        f"- small_exit: {small_exit}",
        f"- 26b_exit: {model_26b_exit}",
        "- prose_quality_winner: not declared",
        "",
        "## Report-Only Metrics",
        "",
        "```json",
        json.dumps(comparison["models"], indent=2, sort_keys=True),
        "```",
        "",
        "## Turn-Aligned Transcript",
        "",
    ]
    for row in aligned:
        lines.extend(
            [
                f"### Turn {row['turn_index']}",
                "",
                f"**Prompt:** {row['prompt']}",
                "",
                f"**Small:** {row['small'].get('response', '[missing]')}",
                "",
                f"**26B:** {row['26b'].get('response', '[missing]')}",
                "",
            ]
        )
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return comparison


def _quality_summary(checkpoint: Mapping[str, Any] | None) -> dict[str, Any]:
    if checkpoint is None:
        return {"available": False}
    return {
        "available": True,
        "status": checkpoint.get("status"),
        "warning_counts": checkpoint.get("warning_counts", {}),
        "events": checkpoint.get("events", []),
        "quality_metrics": checkpoint.get("quality_metrics", {}),
    }


def _turns_by_index(checkpoint: Mapping[str, Any] | None) -> dict[int, Mapping[str, Any]]:
    if checkpoint is None:
        return {}
    return {
        int(turn["turn_index"]): turn
        for turn in checkpoint.get("turns", [])
        if isinstance(turn, Mapping) and "turn_index" in turn
    }


def _turn_summary(turn: Mapping[str, Any] | None) -> dict[str, Any]:
    if turn is None:
        return {"available": False}
    return {
        "available": True,
        "response": turn.get("response"),
        "duration_seconds": turn.get("duration_seconds"),
        "response_chars": turn.get("response_chars"),
        "finish_reason": turn.get("finish_reason"),
        "warning_counts": turn.get("warning_counts"),
    }


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paired local-model live reports.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deterministic-exit", type=int, required=True)
    parser.add_argument("--small-exit", type=int, required=True)
    parser.add_argument("--model-26b-exit", type=int, required=True)
    args = parser.parse_args()
    comparison = write_comparison(
        output_dir=args.output_dir,
        deterministic_exit=args.deterministic_exit,
        small_exit=args.small_exit,
        model_26b_exit=args.model_26b_exit,
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
