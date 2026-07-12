#!/usr/bin/env bash
# One-command semantic-benchmark runner (docs/22 P0.4 follow-up): scores FastEmbed
# model(s) and/or the deterministic keyword provider on the graded semantic corpus,
# writes the --json artifact, prints the suggest_floors.py summary against it, and
# (opt-in) runs the -m semantic pytest tier against the first configured model.
# Wraps `rolerag semantic-benchmark` end to end; changes no application behavior.
#
# Configure via env (defaults shown):
#   MODELS="sentence-transformers/all-MiniLM-L6-v2"  # space-separated, one --model each.
#                                                     # MODELS="" means keyword-only (pair
#                                                     # with INCLUDE_KEYWORD=1).
#   TOP_K=10
#   INCLUDE_KEYWORD=0   # 1 also benchmarks the deterministic keyword provider (no downloads)
#   RUN_PYTEST=1        # 1 also runs the -m semantic opt-in tier (needs MODELS non-empty)
#   ARTIFACT_PATH=<repo>/docs/artifacts/semantic-benchmark-<today>.json
#
# Examples:
#   bash scripts/semantic-benchmark.sh
#   MODELS="" INCLUDE_KEYWORD=1 RUN_PYTEST=0 bash scripts/semantic-benchmark.sh   # keyword-only smoke, no downloads
#   MODELS="model-a model-b" bash scripts/semantic-benchmark.sh                  # compare two models
#
# Partial failure (some providers fail, e.g. a blocked download) is not fatal here:
# the artifact and suggest_floors.py summary are still produced for whichever
# providers succeeded. This script's own exit code is still non-zero if the CLI
# run or the opt-in pytest tier reported a failure, so CI/scripting can detect it.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Prefer the repo venv: macOS has no bare `python` on PATH by default.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${SCRIPT_DIR}/../.venv/bin/python" ]]; then
    PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

# No colon in the default-expansion below: MODELS="" (explicitly set, empty) means "no
# real models, keyword-only"; only a genuinely *unset* MODELS falls back to the default
# model name. Same reasoning for ARTIFACT_PATH further down.
MODELS="${MODELS-sentence-transformers/all-MiniLM-L6-v2}"
TOP_K="${TOP_K:-10}"
INCLUDE_KEYWORD="${INCLUDE_KEYWORD:-0}"
RUN_PYTEST="${RUN_PYTEST:-1}"
ARTIFACT_PATH="${ARTIFACT_PATH-${REPO_ROOT}/docs/artifacts/semantic-benchmark-$(date +%F).json}"

if [[ -z "${MODELS}" && "${INCLUDE_KEYWORD}" != "1" ]]; then
  echo "semantic-benchmark.sh: nothing to benchmark." >&2
  echo "  Set MODELS=\"<fastembed-model-name> ...\" and/or INCLUDE_KEYWORD=1." >&2
  echo "  (rolerag semantic-benchmark itself requires at least one --model or --keyword.)" >&2
  exit 1
fi

mkdir -p "$(dirname "${ARTIFACT_PATH}")"

args=(-m app.cli semantic-benchmark --top-k "${TOP_K}" --json)
# Intentional word-splitting: MODELS is a space-separated list of model names, one
# --model per word.
model_words=(${MODELS})
for model_name in "${model_words[@]}"; do
  args+=(--model "${model_name}")
done
if [[ "${INCLUDE_KEYWORD}" == "1" ]]; then
  args+=(--keyword)
fi

echo "artifact: ${ARTIFACT_PATH}"
echo "equivalent: ${PYTHON} ${args[*]} > ${ARTIFACT_PATH}"

status=0
"${PYTHON}" "${args[@]}" > "${ARTIFACT_PATH}" || status=$?

if [[ -s "${ARTIFACT_PATH}" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/lib/suggest_floors.py" "${ARTIFACT_PATH}"
fi

if [[ "${RUN_PYTEST}" == "1" && -n "${MODELS}" ]]; then
  ROLERAG_SEMANTIC_MODEL="${MODELS%% *}" "${PYTHON}" -m pytest -m semantic \
    tests/evals/test_semantic_benchmark_opt_in.py || status=$?
fi

exit "${status}"
