#!/usr/bin/env bash
# Aggregate the local-model bake-off into one comparison table.
#
# With no arguments, aggregates the three standard run directories under
# ${BAKEOFF_DIR:-/tmp/rolerag-bakeoff}. Otherwise passes its arguments straight through to
# `app.diagnostics.model_bakeoff` (each as `--run LABEL:DIR`), e.g.:
#   bash scripts/bakeoff.sh A:/tmp/run-a B:/tmp/run-b
#
# Each run directory is expected to contain raw/conversation-checkpoint.json (recall) and
# secret-probe.json (containment), produced by live-smoke.sh and secret-probe.sh.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
BAKEOFF_DIR="${BAKEOFF_DIR:-/tmp/rolerag-bakeoff}"

run_args=()
if [[ "$#" -gt 0 ]]; then
  for spec in "$@"; do
    run_args+=(--run "${spec}")
  done
else
  run_args=(
    --run "gemma26b:${BAKEOFF_DIR}/A-gemma26b"
    --run "qwen35b:${BAKEOFF_DIR}/B-qwen35b"
    --run "qwen27b:${BAKEOFF_DIR}/C-qwen27b"
  )
fi

cd "${SCRIPT_DIR}/.."
exec "${PYTHON}" -m app.diagnostics.model_bakeoff \
  "${run_args[@]}" \
  --json-out "${BAKEOFF_DIR}/bakeoff.json" \
  --markdown-out "${BAKEOFF_DIR}/bakeoff.md"
