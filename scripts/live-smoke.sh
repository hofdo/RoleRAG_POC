#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python}"
ARTIFACT_DIR="${LIVE_ARTIFACT_DIR:-/tmp/rolerag-live-test}"
RAW_DIR="${ARTIFACT_DIR}/raw"
WORK_DIR="${LIVE_WORK_DIR:-${ARTIFACT_DIR}/work}"
REPORT="${ARTIFACT_DIR}/report.md"

API_PORT="${API_PORT:-18080}"
QDRANT_PORT="${QDRANT_PORT:-6334}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:v1.18.1}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-rolerag-qdrant-live-smoke}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:${QDRANT_PORT}}"
API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:${API_PORT}}"

LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://127.0.0.1:8080/v1}"
LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-local}"
LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-${LIVE_LLM_MODEL:-chatgpt-onnechan}}"
CLOUD_MODE="${CLOUD_MODE:-off}"

LLAMA_CPP_SERVER_PATH="${LLAMA_CPP_SERVER_PATH:-}"
LLAMA_CPP_MODEL_PATH="${LLAMA_CPP_MODEL_PATH:-}"
LLAMA_CPP_HOST="${LLAMA_CPP_HOST:-127.0.0.1}"
LLAMA_CPP_PORT="${LLAMA_CPP_PORT:-8080}"
LLAMA_CPP_CTX_SIZE="${LLAMA_CPP_CTX_SIZE:-4096}"
LLAMA_CPP_N_GPU_LAYERS="${LLAMA_CPP_N_GPU_LAYERS:-}"
LLAMA_CPP_SERVER_ARGS="${LLAMA_CPP_SERVER_ARGS:-}"

DATABASE_PATH="${DATABASE_PATH:-${WORK_DIR}/rolerag-live.db}"
CONTENT_ROOT="${CONTENT_ROOT:-data}"
LIVE_SKIP_BROWSER="${LIVE_SKIP_BROWSER:-0}"
LIVE_LONG_TURN_COUNT="${LIVE_LONG_TURN_COUNT:-0}"
LIVE_FAIL_ON_STRUCTURED_WARNINGS="${LIVE_FAIL_ON_STRUCTURED_WARNINGS:-0}"

API_PID=""
LLAMA_CPP_PID=""
LLAMA_CPP_STARTED=0
QDRANT_STARTED=0

LOCAL_LLM_V1_URL="${LOCAL_LLM_BASE_URL%/}"
LOCAL_LLM_ROOT_URL="${LOCAL_LLM_V1_URL}"
if [[ "${LOCAL_LLM_ROOT_URL}" == */v1 ]]; then
  LOCAL_LLM_ROOT_URL="${LOCAL_LLM_ROOT_URL%/v1}"
fi
MODELS_URL="${LOCAL_LLM_V1_URL}/models"

if [[ -z "${ARTIFACT_DIR}" || "${ARTIFACT_DIR}" == "/" ]]; then
  echo "Refusing to use unsafe LIVE_ARTIFACT_DIR=${ARTIFACT_DIR}" >&2
  exit 1
fi

rm -rf "${ARTIFACT_DIR}"
mkdir -p "${RAW_DIR}" "${WORK_DIR}"

cat > "${REPORT}" <<EOF
# RoleRAG Live Smoke Report

- started_at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
- api_base_url: ${API_BASE_URL}
- qdrant_url: ${QDRANT_URL}
- local_llm_base_url: ${LOCAL_LLM_BASE_URL}
- local_llm_model: ${LOCAL_LLM_MODEL}
- database_path: ${DATABASE_PATH}
- cloud_mode: ${CLOUD_MODE}
- live_long_turn_count: ${LIVE_LONG_TURN_COUNT}
- live_fail_on_structured_warnings: ${LIVE_FAIL_ON_STRUCTURED_WARNINGS}

## Steps
EOF

cleanup() {
  local exit_code=$?
  if [[ -n "${API_PID}" ]]; then
    kill "${API_PID}" >/dev/null 2>&1 || true
    wait "${API_PID}" >/dev/null 2>&1 || true
  fi
  if [[ "${LLAMA_CPP_STARTED}" == "1" && -n "${LLAMA_CPP_PID}" ]]; then
    kill "${LLAMA_CPP_PID}" >/dev/null 2>&1 || true
    wait "${LLAMA_CPP_PID}" >/dev/null 2>&1 || true
  fi
  if [[ "${QDRANT_STARTED}" == "1" ]]; then
    docker rm -f "${QDRANT_CONTAINER}" > "${RAW_DIR}/qdrant-cleanup.out" 2>&1 || true
  fi
  {
    echo
    echo "## Completed"
    echo
    echo "- finished_at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "- exit_code: ${exit_code}"
  } >> "${REPORT}"
  exit "${exit_code}"
}
trap cleanup EXIT

run_step() {
  local name="$1"
  local title="$2"
  shift 2
  local output="${RAW_DIR}/${name}.out"

  echo "== ${title} =="
  if "$@" > "${output}" 2>&1; then
    printf -- "- PASS %s (\`raw/%s.out\`)\n" "${title}" "${name}" >> "${REPORT}"
  else
    local exit_code=$?
    printf -- "- FAIL %s (\`raw/%s.out\`, exit %s)\n" "${title}" "${name}" "${exit_code}" >> "${REPORT}"
    tail -n 80 "${output}" >&2 || true
    return "${exit_code}"
  fi
}

wait_for_url() {
  local label="$1"
  local url="$2"
  local attempts="${3:-60}"

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

provider_curl() {
  local args=(-fsS)
  if [[ -n "${LOCAL_LLM_API_KEY}" ]]; then
    args+=(-H "Authorization: Bearer ${LOCAL_LLM_API_KEY}")
  fi
  curl "${args[@]}" "$@"
}

fetch_provider_models() {
  local output="$1"
  provider_curl -o "${output}" "${MODELS_URL}"
}

fetch_provider_models_with_retry() {
  local output="$1"
  local attempts="${2:-10}"

  for attempt in $(seq 1 "${attempts}"); do
    if fetch_provider_models "${output}"; then
      return 0
    fi
    echo "waiting for local provider model list (${attempt}/${attempts})"
    sleep 1
  done
  return 1
}

model_file_contains_configured_model() {
  local models_file="$1"
  "${PYTHON}" - "${models_file}" "${LOCAL_LLM_MODEL}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]

ids: list[str] = []
if isinstance(payload, dict):
    data = payload.get("data")
    if isinstance(data, list):
        ids.extend(
            item["id"]
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        )
    models = payload.get("models")
    if isinstance(models, list):
        ids.extend(item for item in models if isinstance(item, str))
elif isinstance(payload, list):
    ids.extend(item for item in payload if isinstance(item, str))

if expected not in ids:
    print(f"expected model {expected!r} from /models, got: {ids}", file=sys.stderr)
    sys.exit(1)
print(f"found required model {expected!r}")
PY
}

wait_for_provider_health() {
  local attempts="${1:-90}"
  local root_health_url="${LOCAL_LLM_ROOT_URL}/health"
  local v1_health_url="${LOCAL_LLM_V1_URL}/health"

  for attempt in $(seq 1 "${attempts}"); do
    if provider_curl "${root_health_url}" >/dev/null 2>&1; then
      echo "local provider ready at ${root_health_url}"
      return 0
    fi
    if provider_curl "${v1_health_url}" >/dev/null 2>&1; then
      echo "local provider ready at ${v1_health_url}"
      return 0
    fi
    if [[ "${LLAMA_CPP_STARTED}" == "1" && -n "${LLAMA_CPP_PID}" ]] \
      && ! kill -0 "${LLAMA_CPP_PID}" >/dev/null 2>&1; then
      echo "managed llama.cpp exited before health became ready" >&2
      tail -n 80 "${RAW_DIR}/llama-server.log" >&2 || true
      return 1
    fi
    echo "waiting for local provider health (${attempt}/${attempts})"
    sleep 1
  done
  echo "local provider did not become healthy at ${root_health_url} or ${v1_health_url}" >&2
  return 1
}

resolve_llama_server_path() {
  if [[ "${LLAMA_CPP_SERVER_PATH}" == */* ]]; then
    if [[ ! -x "${LLAMA_CPP_SERVER_PATH}" ]]; then
      echo "LLAMA_CPP_SERVER_PATH is not executable: ${LLAMA_CPP_SERVER_PATH}" >&2
      return 1
    fi
    printf "%s\n" "${LLAMA_CPP_SERVER_PATH}"
    return 0
  fi
  if ! command -v "${LLAMA_CPP_SERVER_PATH}" >/dev/null 2>&1; then
    echo "LLAMA_CPP_SERVER_PATH was not found on PATH: ${LLAMA_CPP_SERVER_PATH}" >&2
    return 1
  fi
  command -v "${LLAMA_CPP_SERVER_PATH}"
}

print_redacted_command() {
  local rendered=()
  local redact_next=0
  local arg

  for arg in "$@"; do
    if [[ "${redact_next}" == "1" ]]; then
      rendered+=("***")
      redact_next=0
      continue
    fi
    if [[ "${arg}" == "--api-key" ]]; then
      rendered+=("${arg}")
      redact_next=1
      continue
    fi
    if [[ "${arg}" == --api-key=* ]]; then
      rendered+=("--api-key=***")
      continue
    fi
    if [[ -n "${LOCAL_LLM_API_KEY}" && "${arg}" == "${LOCAL_LLM_API_KEY}" ]]; then
      rendered+=("***")
      continue
    fi
    rendered+=("${arg}")
  done

  printf "%q " "${rendered[@]}"
  printf "\n"
}

start_managed_llama() {
  if [[ -z "${LLAMA_CPP_SERVER_PATH}" || -z "${LLAMA_CPP_MODEL_PATH}" ]]; then
    {
      echo "No existing local provider exposed LOCAL_LLM_MODEL=${LOCAL_LLM_MODEL} at ${MODELS_URL}."
      echo "Set both LLAMA_CPP_SERVER_PATH and LLAMA_CPP_MODEL_PATH to let this script start llama.cpp,"
      echo "or start an OpenAI-compatible local server that exposes the configured model."
    } >&2
    return 1
  fi

  if [[ ! -f "${LLAMA_CPP_MODEL_PATH}" ]]; then
    echo "LLAMA_CPP_MODEL_PATH does not exist: ${LLAMA_CPP_MODEL_PATH}" >&2
    return 1
  fi

  local server_path
  if ! server_path="$(resolve_llama_server_path)"; then
    return 1
  fi

  local cmd=(
    "${server_path}"
    -m "${LLAMA_CPP_MODEL_PATH}"
    --host "${LLAMA_CPP_HOST}"
    --port "${LLAMA_CPP_PORT}"
    --alias "${LOCAL_LLM_MODEL}"
    -c "${LLAMA_CPP_CTX_SIZE}"
  )

  if [[ -n "${LOCAL_LLM_API_KEY}" ]]; then
    cmd+=(--api-key "${LOCAL_LLM_API_KEY}")
  fi
  if [[ -n "${LLAMA_CPP_N_GPU_LAYERS}" ]]; then
    cmd+=(--n-gpu-layers "${LLAMA_CPP_N_GPU_LAYERS}")
  fi
  if [[ -n "${LLAMA_CPP_SERVER_ARGS}" ]]; then
    local extra_args=()
    read -r -a extra_args <<< "${LLAMA_CPP_SERVER_ARGS}"
    cmd+=("${extra_args[@]}")
  fi

  {
    echo "Starting managed llama.cpp:"
    print_redacted_command "${cmd[@]}"
    echo
  } > "${RAW_DIR}/llama-server.log"
  "${cmd[@]}" >> "${RAW_DIR}/llama-server.log" 2>&1 &
  LLAMA_CPP_PID=$!
  LLAMA_CPP_STARTED=1
  echo "${LLAMA_CPP_PID}" > "${RAW_DIR}/llama-server.pid"
  printf -- "- INFO started managed llama.cpp (\`raw/llama-server.log\`)\n" >> "${REPORT}"

  wait_for_provider_health 90 || return 1
  fetch_provider_models "${RAW_DIR}/local-provider-models.managed.json" || return 1
  model_file_contains_configured_model "${RAW_DIR}/local-provider-models.managed.json"
}

ensure_local_provider() {
  local initial_models="${RAW_DIR}/local-provider-models.initial.json"

  if fetch_provider_models_with_retry "${initial_models}" 10; then
    if model_file_contains_configured_model "${initial_models}"; then
      echo "Reusing existing local provider at ${LOCAL_LLM_BASE_URL}."
      printf -- "- INFO reused existing llama.cpp-compatible provider at \`%s\`\n" \
        "${LOCAL_LLM_BASE_URL}" >> "${REPORT}"
      return 0
    fi
    echo "Existing local provider did not expose ${LOCAL_LLM_MODEL}; trying managed llama.cpp startup."
  else
    echo "No existing local provider model list reachable at ${MODELS_URL}; trying managed llama.cpp startup."
  fi

  start_managed_llama
}

export APP_ENV="live-smoke"
export DATABASE_PATH
export CONTENT_ROOT
export QDRANT_URL
export LOCAL_LLM_BASE_URL
export LOCAL_LLM_API_KEY
export LOCAL_LLM_MODEL
export CLOUD_MODE
export LIVE_LONG_TURN_COUNT
export LIVE_FAIL_ON_STRUCTURED_WARNINGS
export PLAYWRIGHT_BASE_URL="${API_BASE_URL}"
export PLAYWRIGHT_OUTPUT_DIR="${ARTIFACT_DIR}/playwright-results"
export PLAYWRIGHT_HTML_REPORT="${ARTIFACT_DIR}/playwright-report"

run_step prerequisites "Check required local commands" bash -c \
  'command -v docker && command -v curl && command -v npm && command -v "$1"' _ "${PYTHON}"

run_step local-provider-startup "Ensure local provider model is available" ensure_local_provider

run_step local-provider-models "Fetch local provider model list" \
  fetch_provider_models "${RAW_DIR}/local-provider-models.json"

run_step local-provider-model "Require configured local model" \
  model_file_contains_configured_model "${RAW_DIR}/local-provider-models.json"

docker rm -f "${QDRANT_CONTAINER}" > "${RAW_DIR}/qdrant-preremove.out" 2>&1 || true
run_step qdrant-start "Start disposable Qdrant" docker run -d \
  --name "${QDRANT_CONTAINER}" \
  -p "127.0.0.1:${QDRANT_PORT}:6333" \
  -v "${WORK_DIR}/qdrant:/qdrant/storage" \
  "${QDRANT_IMAGE}"
QDRANT_STARTED=1

run_step qdrant-ready "Wait for disposable Qdrant" \
  wait_for_url qdrant "${QDRANT_URL}/collections" 90

run_step doctor "Run live doctor checks" \
  "${PYTHON}" -m app.cli doctor --check-qdrant --check-local-provider

run_step smoke-real-runtime "Run deterministic smoke with live dependency checks" \
  "${PYTHON}" -m app.cli smoke-run --real-runtime

run_step ingest-demo-lore "Ingest demo lore into disposable Qdrant" \
  "${PYTHON}" -m app.cli ingest data/documents/demo_lore.md \
  --visibility player \
  --source-type lore \
  --world-id demo_world \
  --tag palace

"${PYTHON}" -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port "${API_PORT}" \
  > "${RAW_DIR}/uvicorn.log" 2>&1 &
API_PID=$!

run_step api-ready "Wait for FastAPI" \
  wait_for_url api "${API_BASE_URL}/runtime/status" 90

run_step api-flow "Run live HTTP API flow" \
  "${PYTHON}" - "${API_BASE_URL}" "${LOCAL_LLM_MODEL}" "${RAW_DIR}/api-flow.json" <<'PY'
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

base_url = sys.argv[1]
expected_model = sys.argv[2]
output_path = Path(sys.argv[3])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def warning_counts(warnings: list[Any]) -> dict[str, int]:
    counts = {"critic": 0, "memory": 0, "other": 0}
    for warning in warnings:
        text = str(warning)
        if text.startswith("critic skipped:"):
            counts["critic"] += 1
        elif text.startswith("memory curation skipped:"):
            counts["memory"] += 1
        else:
            counts["other"] += 1
    return counts


with httpx.Client(base_url=base_url, timeout=180.0) as client:
    status = client.get("/runtime/status")
    status.raise_for_status()
    runtime_status: dict[str, Any] = status.json()
    require(runtime_status.get("content_catalog_available") is True, "content catalog unavailable")
    require(runtime_status.get("local_provider_configured") is True, "local provider not configured")

    catalog_response = client.get("/content/catalog")
    catalog_response.raise_for_status()
    catalog: dict[str, Any] = catalog_response.json()
    world_ids = {world.get("id") for world in catalog.get("worlds", []) if isinstance(world, dict)}
    require("demo_world" in world_ids, "demo_world missing from catalog")

    session_response = client.post(
        "/sessions",
        json={
            "world_id": "demo_world",
            "scene_id": "rose-gallery",
            "active_persona_id": "archivist",
            "player_name": "Live Smoke",
        },
    )
    session_response.raise_for_status()
    session: dict[str, Any] = session_response.json()
    session_id = str(session["session_id"])

    started = time.monotonic()
    turn_response = client.post(
        f"/sessions/{session_id}/turns",
        json={
            "message": "I request a quiet account of what the Rose Gallery knows about the regent.",
            "request_cloud": False,
        },
    )
    duration_seconds = time.monotonic() - started
    turn_response.raise_for_status()
    turn: dict[str, Any] = turn_response.json()
    route = turn.get("route", {})
    require(isinstance(turn.get("text"), str) and turn["text"].strip() != "", "empty actor text")
    require(route.get("provider") == "local", f"unexpected route provider: {route}")
    require(route.get("model") == expected_model, f"unexpected route model: {route}")

    lookup_response = client.get(f"/sessions/{session_id}")
    lookup_response.raise_for_status()
    lookup: dict[str, Any] = lookup_response.json()
    require(len(lookup.get("recent_turns", [])) == 1, "session lookup did not return one turn")

summary = {
    "runtime_status": runtime_status,
    "session": session,
    "turn": turn,
    "turn_diagnostics": {
        "turn_index": 1,
        "duration_seconds": round(duration_seconds, 3),
        "response_chars": len(turn.get("text", "")),
        "route": route,
        "memory_written": turn.get("memory_written"),
        "warning_counts": warning_counts(turn.get("warnings", [])),
    },
    "lookup": lookup,
}
output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(
    json.dumps(
        {
            "session_id": session_id,
            "route": route,
            "response_chars": len(turn.get("text", "")),
            "duration_seconds": round(duration_seconds, 3),
            "memory_written": turn.get("memory_written"),
            "warning_counts": warning_counts(turn.get("warnings", [])),
            "warnings": turn.get("warnings", []),
        },
        indent=2,
        sort_keys=True,
    )
)
PY

if [[ "${LIVE_SKIP_BROWSER}" == "1" ]]; then
  printf -- "- SKIP Playwright UI smoke (LIVE_SKIP_BROWSER=1)\n" >> "${REPORT}"
else
  run_step playwright "Run Playwright UI smoke" npm run test:live-ui -- --reporter=line
fi

if [[ "${LIVE_LONG_TURN_COUNT}" == "0" ]]; then
  printf -- "- SKIP Long-session Rose Gallery flow (LIVE_LONG_TURN_COUNT=0)\n" >> "${REPORT}"
else
  run_step long-session "Run ${LIVE_LONG_TURN_COUNT}-turn Rose Gallery flow" \
    "${PYTHON}" - \
    "${API_BASE_URL}" \
    "${LOCAL_LLM_MODEL}" \
    "${DATABASE_PATH}" \
    "${LIVE_LONG_TURN_COUNT}" \
    "${LIVE_FAIL_ON_STRUCTURED_WARNINGS}" \
    "${RAW_DIR}/long-session.json" <<'PY'
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import httpx

base_url = sys.argv[1]
expected_model = sys.argv[2]
database_path = Path(sys.argv[3])
turn_count = int(sys.argv[4])
fail_on_structured_warnings = sys.argv[5] == "1"
output_path = Path(sys.argv[6])

messages = [
    "I step into the Rose Gallery and ask Iria what first changed tonight.",
    "I lower my voice and ask which courtier she trusts least in this room.",
    "I promise to return before dawn if she can keep the archive door unbarred.",
    "I ask what the mirrors have shown her about the regent's messengers.",
    "I inspect the nearest rose arrangement and ask whether it hides a signal.",
    "I ask Iria to describe the safest path from here to the old archive.",
    "I tell her I noticed the servant by the west door and ask if that matters.",
    "I ask whether my promise before dawn changes what she is willing to risk.",
    "I request a private walk with Iria in the Rose Gallery.",
    "I ask what she has truly been afraid to say aloud tonight.",
    "I ask her to name the next small action I should take without alarming anyone.",
    "I thank her and ask what she wants me to remember when I leave.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def warning_counts(warnings: list[Any]) -> dict[str, int]:
    counts = {"critic": 0, "memory": 0, "other": 0}
    for warning in warnings:
        text = str(warning)
        if text.startswith("critic skipped:"):
            counts["critic"] += 1
        elif text.startswith("memory curation skipped:"):
            counts["memory"] += 1
        else:
            counts["other"] += 1
    return counts


require(turn_count == 12, "LIVE_LONG_TURN_COUNT currently supports only 0 or 12")

turns: list[dict[str, Any]] = []
with httpx.Client(base_url=base_url, timeout=240.0) as client:
    session_response = client.post(
        "/sessions",
        json={
            "world_id": "demo_world",
            "scene_id": "rose-gallery",
            "active_persona_id": "archivist",
            "player_name": "Long Live Smoke",
        },
    )
    session_response.raise_for_status()
    session = session_response.json()
    session_id = str(session["session_id"])

    for index, message in enumerate(messages, start=1):
        started = time.monotonic()
        turn_response = client.post(
            f"/sessions/{session_id}/turns",
            json={"message": message, "request_cloud": False},
        )
        duration_seconds = time.monotonic() - started
        turn_response.raise_for_status()
        turn = turn_response.json()
        route = turn.get("route", {})
        warnings = turn.get("warnings", [])
        counts = warning_counts(warnings)

        require(isinstance(turn.get("text"), str) and turn["text"].strip(), f"empty actor text on turn {index}")
        require(route.get("provider") == "local", f"unexpected route provider on turn {index}: {route}")
        require(route.get("model") == expected_model, f"unexpected route model on turn {index}: {route}")
        if fail_on_structured_warnings:
            require(
                counts["critic"] == 0 and counts["memory"] == 0,
                f"structured-output warnings on turn {index}: {warnings}",
            )

        turns.append(
            {
                "turn_index": index,
                "duration_seconds": round(duration_seconds, 3),
                "response_chars": len(turn.get("text", "")),
                "route": route,
                "memory_written": turn.get("memory_written"),
                "warning_counts": counts,
                "warnings": warnings,
            }
        )

    lookup_response = client.get(f"/sessions/{session_id}")
    lookup_response.raise_for_status()
    lookup = lookup_response.json()

with sqlite3.connect(database_path) as connection:
    persisted_turn_count = connection.execute(
        "SELECT COUNT(*) FROM turns WHERE session_id = ?",
        (session_id,),
    ).fetchone()[0]

require(persisted_turn_count == turn_count, f"persisted-turn count mismatch: {persisted_turn_count} != {turn_count}")

structured_warning_counts = {
    "critic": sum(turn["warning_counts"]["critic"] for turn in turns),
    "memory": sum(turn["warning_counts"]["memory"] for turn in turns),
}
summary = {
    "session": session,
    "turns": turns,
    "lookup_recent_turn_count": len(lookup.get("recent_turns", [])),
    "persisted_turn_count": persisted_turn_count,
    "structured_warning_counts": structured_warning_counts,
}
output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
fi

{
  echo
  echo "## Artifact Files"
  echo
  find "${ARTIFACT_DIR}" -maxdepth 3 -type f | sort | sed "s#${ARTIFACT_DIR}/#- #"
} >> "${REPORT}"
