"""Print a one-line progress heartbeat from an in-flight live-checkpoint report.

Usage: ab_progress.py <path-to-conversation-checkpoint.json> <total_turns>

The checkpoint writes this JSON incrementally (status "in_progress", with the
`turns` list growing one entry per completed turn), so polling it is the only
per-turn signal while live-smoke runs silently.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    path, total = sys.argv[1], sys.argv[2]
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        print("setting up (no checkpoint yet)")
        return

    turns = data.get("turns") or []
    parts = [f"turn {len(turns)}/{total}", f"status={data.get('status', '?')}"]
    if turns:
        last = turns[-1]
        duration = last.get("duration_seconds")
        if duration is not None:
            parts.append(f"last={duration}s")
        if last.get("finish_reason") and last.get("finish_reason") != "stop":
            parts.append(f"finish={last['finish_reason']}")
    events = data.get("event_inspections") or {}
    if events:
        parts.append(f"events_checked={len(events)}")
    print("  ".join(parts))


if __name__ == "__main__":
    main()
