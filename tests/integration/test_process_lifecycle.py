from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE_SCRIPT = REPOSITORY_ROOT / "scripts" / "lib" / "process-lifecycle.sh"


def _run_lifecycle_test(
    child_command: str,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    script = f"""
set -euo pipefail
source {LIFECYCLE_SCRIPT!s}
bash -c {child_command!r} &
pid=$!
sleep 0.1
stop_managed_process "$pid" test-child {timeout_seconds}
if kill -0 "$pid" >/dev/null 2>&1; then
  echo "child still running" >&2
  exit 1
fi
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_stop_managed_process_allows_graceful_sigterm() -> None:
    result = _run_lifecycle_test("trap 'exit 0' TERM; while true; do sleep 1; done", 2)

    assert result.returncode == 0, result.stderr
    assert "Stopping managed test-child process" in result.stdout
    assert "SIGKILL" not in result.stderr


def test_stop_managed_process_forces_process_that_ignores_sigterm() -> None:
    result = _run_lifecycle_test("trap '' TERM; while true; do sleep 1; done", 1)

    assert result.returncode == 0, result.stderr
    assert "sending SIGKILL" in result.stderr
