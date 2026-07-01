#!/usr/bin/env bash
set -uo pipefail

PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${MODEL_COMPARE_ARTIFACT_DIR:-/tmp/rolerag-model-comparison}"
TURN_COUNT="${MODEL_COMPARE_TURN_COUNT:-20}"

if [[ "${TURN_COUNT}" != "20" && "${TURN_COUNT}" != "50" ]]; then
  echo "MODEL_COMPARE_TURN_COUNT must be 20 or 50" >&2
  exit 2
fi
if [[ -z "${OUTPUT_DIR}" || "${OUTPUT_DIR}" == "/" ]]; then
  echo "Refusing unsafe MODEL_COMPARE_ARTIFACT_DIR=${OUTPUT_DIR}" >&2
  exit 2
fi

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/deterministic"

run_recorded() {
  local name="$1"
  shift
  "$@" > "${OUTPUT_DIR}/deterministic/${name}.out" 2>&1
}

deterministic_exit=0
run_recorded ruff "${PYTHON}" -m ruff check . || deterministic_exit=1
run_recorded mypy "${PYTHON}" -m mypy . || deterministic_exit=1
run_recorded pytest "${PYTHON}" -m pytest || deterministic_exit=1
run_recorded frontend bash -c "cd frontend && npm test -- --watch=false --browsers=ChromeHeadless" || deterministic_exit=1
run_recorded regression "${PYTHON}" -m app.evals.regression_runner || deterministic_exit=1

run_model() {
  local profile="$1"
  local alias="$2"
  local artifact_dir="${OUTPUT_DIR}/${profile}"
  LOCAL_MODEL_PROFILE="${profile}" \
  LOCAL_LLM_MODEL="${alias}" \
  LIVE_ARTIFACT_DIR="${artifact_dir}" \
  LIVE_TURN_COUNT="${TURN_COUNT}" \
  LIVE_FAIL_ON_STRUCTURED_WARNINGS=0 \
  PYTHON="${PYTHON}" \
  bash scripts/live-smoke.sh
}

small_exit=0
model_26b_exit=0
run_model small rolerag-small || small_exit=$?
run_model 26b rolerag-26b || model_26b_exit=$?

comparison_exit=0
"${PYTHON}" -m app.diagnostics.model_comparison \
  --output-dir "${OUTPUT_DIR}" \
  --deterministic-exit "${deterministic_exit}" \
  --small-exit "${small_exit}" \
  --model-26b-exit "${model_26b_exit}" \
  > "${OUTPUT_DIR}/comparison.out" 2>&1 || comparison_exit=$?

if (( deterministic_exit != 0 || small_exit != 0 || model_26b_exit != 0 || comparison_exit != 0 )); then
  exit 1
fi
