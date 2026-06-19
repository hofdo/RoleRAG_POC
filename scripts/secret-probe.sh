#!/usr/bin/env bash
# Secret-containment probe driver for the "A Bride for Sarnhold" scenario.
#
# Stands up a disposable Qdrant + FastAPI against the Sarnhold content pack, reusing an
# already-running local llama-server (launch it manually with `--alias ${LOCAL_LLM_MODEL}`),
# ingests the scenario lore, then runs `app.diagnostics.secret_probe`. Writes results into
# the SAME RUN_DIR as the recall run WITHOUT clobbering its artifacts, so model_bakeoff.py
# can read both from one directory.
#
# Required env: RUN_DIR, LOCAL_LLM_MODEL (must equal the server's --alias).
# Optional env: LOCAL_LLM_TEMPERATURE, LOCAL_LLM_BASE_URL, LOCAL_LLM_API_KEY, PYTHON.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/process-lifecycle.sh"

PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to the bake-off run directory for this model}"
RAW_DIR="${RUN_DIR}/raw"
WORK_DIR="${RUN_DIR}/work"

CONTENT_ROOT="${CONTENT_ROOT:-data/scenarios/bride-for-sarnhold}"
DATABASE_PATH="${DATABASE_PATH:-${WORK_DIR}/rolerag-secret-probe.db}"
CLOUD_MODE="${CLOUD_MODE:-off}"

# Distinct port/container from live-smoke.sh so a recall run and a probe run never collide.
API_PORT="${SECRET_PROBE_API_PORT:-18081}"
QDRANT_PORT="${SECRET_PROBE_QDRANT_PORT:-6335}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:v1.18.1}"
QDRANT_CONTAINER="${SECRET_PROBE_QDRANT_CONTAINER:-rolerag-qdrant-secret-probe}"
QDRANT_URL="http://127.0.0.1:${QDRANT_PORT}"
API_BASE_URL="http://127.0.0.1:${API_PORT}"

LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://127.0.0.1:8080/v1}"
LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-local}"
LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:?set LOCAL_LLM_MODEL to the server alias such as gemma26b}"

LOCAL_LLM_V1_URL="${LOCAL_LLM_BASE_URL%/}"
MODELS_URL="${LOCAL_LLM_V1_URL}/models"

if [[ -z "${RUN_DIR}" || "${RUN_DIR}" == "/" ]]; then
  echo "Refusing to use unsafe RUN_DIR=${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${RAW_DIR}" "${WORK_DIR}"

API_PID=""
QDRANT_STARTED=0

cleanup() {
  local exit_code=$?
  if [[ -n "${API_PID}" ]]; then
    stop_managed_process "${API_PID}" "FastAPI" 10
  fi
  if [[ "${QDRANT_STARTED}" == "1" ]]; then
    docker rm -f "${QDRANT_CONTAINER}" > "${RAW_DIR}/qdrant-secret-probe-cleanup.out" 2>&1 || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

provider_curl() {
  local args=(-fsS)
  if [[ -n "${LOCAL_LLM_API_KEY}" ]]; then
    args+=(-H "Authorization: Bearer ${LOCAL_LLM_API_KEY}")
  fi
  curl "${args[@]}" "$@"
}

wait_for_url() {
  local label="$1" url="$2" attempts="${3:-90}"
  for attempt in $(seq 1 "${attempts}"); do
    if curl -fsS "${url}" >/dev/null; then
      echo "${label} ready at ${url}"
      return 0
    fi
    echo "waiting for ${label} at ${url} (${attempt}/${attempts})"
    sleep 1
  done
  echo "${label} did not become ready at ${url}" >&2
  return 1
}

require_running_provider() {
  local models_file="${RAW_DIR}/secret-probe-provider-models.json"
  if ! provider_curl -o "${models_file}" "${MODELS_URL}"; then
    echo "No local provider reachable at ${MODELS_URL}." >&2
    echo "Launch llama-server with --alias ${LOCAL_LLM_MODEL} before running this probe." >&2
    return 1
  fi
  "${PYTHON}" - "${models_file}" "${LOCAL_LLM_MODEL}" <<'PY'
import json, sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
ids: list[str] = []
if isinstance(payload, dict):
    data = payload.get("data")
    if isinstance(data, list):
        ids.extend(i["id"] for i in data if isinstance(i, dict) and isinstance(i.get("id"), str))
    models = payload.get("models")
    if isinstance(models, list):
        ids.extend(i for i in models if isinstance(i, str))
elif isinstance(payload, list):
    ids.extend(i for i in payload if isinstance(i, str))
if expected not in ids:
    print(f"provider does not expose required model {expected!r}; got {ids}", file=sys.stderr)
    print(f"Launch llama-server with --alias {expected}.", file=sys.stderr)
    sys.exit(1)
print(f"reusing local provider exposing {expected!r}")
PY
}

echo "== Require running local provider exposing ${LOCAL_LLM_MODEL} =="
require_running_provider

echo "== Start disposable Qdrant =="
docker rm -f "${QDRANT_CONTAINER}" > "${RAW_DIR}/qdrant-secret-probe-preremove.out" 2>&1 || true
docker run -d \
  --name "${QDRANT_CONTAINER}" \
  -p "127.0.0.1:${QDRANT_PORT}:6333" \
  -v "${WORK_DIR}/qdrant-secret-probe:/qdrant/storage" \
  "${QDRANT_IMAGE}" > "${RAW_DIR}/qdrant-secret-probe-start.out" 2>&1
QDRANT_STARTED=1
wait_for_url qdrant "${QDRANT_URL}/collections" 90

export APP_ENV="secret-probe"
export DATABASE_PATH
export CONTENT_ROOT
export QDRANT_URL
export LOCAL_LLM_BASE_URL
export LOCAL_LLM_API_KEY
export LOCAL_LLM_MODEL
export CLOUD_MODE
export CRITIC_GATING="${CRITIC_GATING:-always}"
export CURATOR_GATING="${CURATOR_GATING:-always}"
# Keep the probe run's structured failures separate from the recall run's log so the
# bake-off's struct_fail column stays attributed to the 50-turn recall benchmark.
export STRUCTURED_OUTPUT_FAILURE_LOG_DIR="${RAW_DIR}/structured-failures-secret-probe"

echo "== Ingest Sarnhold scenario lore =="
"${PYTHON}" -m app.cli ingest-scenario-lore --content-root "${CONTENT_ROOT}" \
  > "${RAW_DIR}/secret-probe-ingest.out" 2>&1

echo "== Start FastAPI on ${API_BASE_URL} =="
"${PYTHON}" -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port "${API_PORT}" \
  > "${RAW_DIR}/uvicorn-secret-probe.log" 2>&1 &
API_PID=$!
wait_for_url api "${API_BASE_URL}/runtime/status" 90

echo "== Run secret-containment probe =="
"${PYTHON}" -m app.diagnostics.secret_probe \
  --api-base-url "${API_BASE_URL}" \
  --expected-model "${LOCAL_LLM_MODEL}" \
  --json-report "${RUN_DIR}/secret-probe.json" \
  --markdown-report "${WORK_DIR}/secret-probe.md"

echo "Secret-probe complete: ${RUN_DIR}/secret-probe.json"
