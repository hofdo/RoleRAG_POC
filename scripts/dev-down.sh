#!/usr/bin/env bash
# Stop everything scripts/dev-up.sh started (and only that).
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${ROOT_DIR}/.dev-run"

source "${SCRIPT_DIR}/lib/process-lifecycle.sh"

stop_from_pid_file() {
  local pid_file="$1" label="$2"
  if [[ ! -f "${pid_file}" ]]; then
    echo "No managed ${label} (no pid file)."
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}")"
  stop_managed_process "${pid}" "${label}" 15
  rm -f "${pid_file}"
}

stop_from_pid_file "${RUN_DIR}/uvicorn.pid" "uvicorn"
stop_from_pid_file "${RUN_DIR}/llama-server.pid" "llama-server"

cd "${ROOT_DIR}"
if docker info >/dev/null 2>&1; then
  docker compose stop qdrant
fi

echo "RoleRAG stopped."
