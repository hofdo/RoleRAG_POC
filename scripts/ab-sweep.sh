#!/usr/bin/env bash
# A/B sweep over RAG memory/retrieval knobs.
#
# Wraps scripts/live-smoke.sh: runs the live Rose Gallery checkpoint once per
# config (one axis off baseline = OFAT), each into its own isolated artifact
# dir, in report-only mode (a retrieval miss does not abort the run), then
# prints a comparison table built from each run's conversation-checkpoint.json
# quality_metrics. It changes no application behavior — pure orchestration.
#
# Prerequisites:
#   - Docker running (disposable Qdrant on :6334)
#   - llama-server on PATH; or already running with alias `chatgpt-onnechan`
#     so every run REUSES it (much faster than per-run managed startup)
#   - .venv installed; free host RAM (26B + KV must fit without swap)
#
# Configure via env (defaults shown) or edit CONFIGS below:
#   MODEL_PROFILE=26b TURNS=50 TOP_K=10 OUT=/tmp/rolerag-ab \
#   PYTHON=.venv/bin/python bash scripts/ab-sweep.sh
#
# Runtime: ~TURNS*~80s per run on 26B (50 turns ~= 70 min). Drop TURNS=38 (3
# story events) or 20 (1 event) for faster iteration.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_PROFILE="${MODEL_PROFILE:-26b}"
TURNS="${TURNS:-50}"
TOP_K="${TOP_K:-10}"
OUT="${OUT:-/tmp/rolerag-ab}"
POLL_SECONDS="${POLL_SECONDS:-20}"   # heartbeat interval while a run is in flight

# Each entry: "<name> <space-separated ENV=VALUE knob overrides>".
# Unset knobs fall back to application defaults (recency 0.0, floor 1,
# dedup threshold 1.0, canon floor 4). Keep it one-axis-at-a-time.
CONFIGS=(
  "baseline  RAG_RECENCY_WEIGHT=0.0 RAG_INDEX_IMPORTANCE_FLOOR=1 RAG_WRITE_DEDUP_COSINE_THRESHOLD=1.0 CANON_IMPORTANCE_FLOOR=4"
  "recency05 RAG_RECENCY_WEIGHT=0.05"
  "floor2    RAG_INDEX_IMPORTANCE_FLOOR=2"
  "dedup92   RAG_WRITE_DEDUP_COSINE_THRESHOLD=0.92"
  "canon3    CANON_IMPORTANCE_FLOOR=3"
)

if [[ ! -f scripts/live-smoke.sh ]]; then
  echo "scripts/live-smoke.sh not found — run from the repo root." >&2
  exit 1
fi

# Reuse an already-running llama-server: detect the model id it serves so the
# checkpoint's per-turn model check matches. live-smoke defaults to the
# 'chatgpt-onnechan' alias, which a `-hf`-launched server does not expose
# (it reports the full repo id). Override detection with MODEL_NAME=...
LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://127.0.0.1:8080/v1}"
if [[ -z "${MODEL_NAME:-}" ]]; then
  MODEL_NAME="$(curl -fsS "${LOCAL_LLM_BASE_URL}/models" 2>/dev/null \
    | "${PYTHON}" -c 'import sys, json; d = json.load(sys.stdin); items = d.get("data") or d.get("models") or []; it = items[0] if items else {}; print(it.get("id") or it.get("model") or it.get("name") or "")' \
      2>/dev/null || true)"
fi
MODEL_NAME="${MODEL_NAME:-chatgpt-onnechan}"

mkdir -p "${OUT}"
SUMMARY_TSV="${OUT}/summary.tsv"
printf 'config\textract_miss\tselect_miss\tmiss_ranks\trecall_miss\tstatus\ttotal_s\tmem_s\n' \
  > "${SUMMARY_TSV}"

echo "A/B sweep: profile=${MODEL_PROFILE} turns=${TURNS} top_k=${TOP_K} configs=${#CONFIGS[@]}"
echo "model: ${MODEL_NAME}  (at ${LOCAL_LLM_BASE_URL})"
echo "output: ${OUT}  (~$(( TURNS * 80 / 60 )) min/run on 26B)"

for entry in "${CONFIGS[@]}"; do
  # Split into "name" + a token array of NAME=VALUE knob overrides. Using an
  # explicit array (not unquoted word-splitting) is robust across shells, and
  # the ${arr[@]+...} guard keeps an empty override list safe under bash 3.2's
  # `set -u` (macOS default bash).
  read -r name rest <<< "${entry}"
  read -ra knobs <<< "${rest}"
  run_dir="${OUT}/${name}"
  json="${run_dir}/raw/conversation-checkpoint.json"
  log="${OUT}/${name}.log"
  echo ""
  echo "=== run: ${name} === ${rest}"

  # Run live-smoke in the background and poll the checkpoint's progress JSON,
  # because the checkpoint is silent on stdout for the whole run. Knob env vars
  # propagate to the FastAPI process and the checkpoint subprocess that
  # live-smoke.sh spawns (pydantic-settings reads them); a CONFIGS override
  # (later assignment) beats the defaults above.
  env \
      LOCAL_MODEL_PROFILE="${MODEL_PROFILE}" \
      LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL}" \
      LOCAL_LLM_MODEL="${MODEL_NAME}" \
      LIVE_TURN_COUNT="${TURNS}" \
      LIVE_FAIL_ON_STRUCTURED_WARNINGS=0 \
      LIVE_SKIP_BROWSER=1 \
      LIVE_ARTIFACT_DIR="${run_dir}" \
      RAG_DEFAULT_TOP_K="${TOP_K}" \
      PYTHON="${PYTHON}" \
      ${knobs[@]+"${knobs[@]}"} \
      bash scripts/live-smoke.sh > "${log}" 2>&1 &
  smoke_pid=$!

  last_print=""
  while kill -0 "${smoke_pid}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
    prog="$("${PYTHON}" scripts/lib/ab_progress.py "${json}" "${TURNS}" 2>/dev/null || echo 'setting up')"
    step="$(tail -n 1 "${log}" 2>/dev/null || true)"
    line="${prog} | ${step}"
    if [[ "${line}" != "${last_print}" ]]; then
      printf '  [%s] %s\n' "${name}" "${line}"
      last_print="${line}"
    fi
  done
  if wait "${smoke_pid}"; then run_status="ok"; else run_status="FAILED(exit $?)"; fi
  echo "  [${name}] status: ${run_status}  (log: ${log})"

  if ! "${PYTHON}" scripts/lib/ab_row.py "${name}" "${run_status}" "${json}" >> "${SUMMARY_TSV}"; then
    printf '%s\tn/a\tn/a\t-\tn/a\t%s\tn/a\tn/a\n' "${name}" "${run_status}" >> "${SUMMARY_TSV}"
  fi
done

echo ""
echo "=== A/B comparison (lower misses better; miss_ranks = how far off vs top-${TOP_K}) ==="
column -t -s "$(printf '\t')" "${SUMMARY_TSV}"
echo ""
echo "Per-run artifacts: ${OUT}/<name>/  (report.md, raw/conversation-checkpoint.json, <name>.log)"
echo "Pick the row with the lowest select_miss + recall_miss and acceptable total_s,"
echo "then put those knob values in your .env."
