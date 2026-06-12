#!/usr/bin/env bash
# One-command daily startup: Qdrant + local model + API/UI.
# Companion teardown: scripts/dev-down.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${ROOT_DIR}/.dev-run"
mkdir -p "${RUN_DIR}"

source "${SCRIPT_DIR}/lib/local-model-profile.sh"

cd "${ROOT_DIR}"

if [[ ! -f .env ]]; then
  echo "No .env found; using built-in defaults (create one with: cp .env.example .env)."
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "No .venv found. Create one first:  python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Start Docker Desktop first." >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://127.0.0.1:8080/v1}"
LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-chatgpt-onnechan}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
API_PORT="${DEV_API_PORT:-8000}"
LOCAL_LLM_ROOT_URL="${LOCAL_LLM_BASE_URL%/}"
LOCAL_LLM_ROOT_URL="${LOCAL_LLM_ROOT_URL%/v1}"
LOCAL_LLM_PORT="${LOCAL_LLM_ROOT_URL##*:}"
MODELS_URL="${LOCAL_LLM_BASE_URL%/}/models"

wait_for_url() {
  local label="$1" url="$2" attempts="${3:-60}"
  for attempt in $(seq 1 "${attempts}"); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "${label} ready."
      return 0
    fi
    sleep 1
  done
  echo "${label} did not become ready at ${url}" >&2
  return 1
}

echo "== Qdrant =="
docker compose up -d qdrant
wait_for_url "Qdrant" "${QDRANT_URL}/readyz" 60

echo "== Local model =="
if curl -fsS "${MODELS_URL}" 2>/dev/null | grep -q "${LOCAL_LLM_MODEL}"; then
  echo "Reusing local provider at ${LOCAL_LLM_BASE_URL}."
else
  if ! command -v llama-server >/dev/null 2>&1; then
    echo "llama-server not found on PATH and no provider serves ${LOCAL_LLM_MODEL}." >&2
    exit 1
  fi
  resolve_local_model_profile
  echo "Starting llama-server (profile: ${LOCAL_MODEL_PROFILE:-small})."
  nohup llama-server \
    -hf "${PROFILE_HF_MODEL}" \
    --host 127.0.0.1 --port "${LOCAL_LLM_PORT}" \
    --alias "${LOCAL_LLM_MODEL}" \
    "${PROFILE_LLAMA_ARGS[@]}" \
    > "${RUN_DIR}/llama-server.log" 2>&1 &
  echo $! > "${RUN_DIR}/llama-server.pid"
  wait_for_url "Local model" "${MODELS_URL}" 180
fi

echo "== API + UI =="
if curl -fsS "http://127.0.0.1:${API_PORT}/runtime/status" >/dev/null 2>&1; then
  echo "Reusing running API on port ${API_PORT}."
else
  nohup .venv/bin/uvicorn app.main:app --port "${API_PORT}" \
    > "${RUN_DIR}/uvicorn.log" 2>&1 &
  echo $! > "${RUN_DIR}/uvicorn.pid"
  wait_for_url "API" "http://127.0.0.1:${API_PORT}/runtime/status" 60
fi

echo
echo "RoleRAG is up:  http://127.0.0.1:${API_PORT}/play"
echo "Logs: ${RUN_DIR}/  |  Stop everything: scripts/dev-down.sh"
