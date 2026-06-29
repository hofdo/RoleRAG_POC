#!/usr/bin/env bash

SMALL_MODEL_HF="DavidAU/gemma-4-E4B-it-The-DECKARD-Expresso-Universe-HERETIC-UNCENSORED-Thinking-GGUF:Q8_0"
MODEL_26B_HF="HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M"
# Local MTP (multi-token-prediction / speculative-draft) build of the same 26B Balanced base.
# Files live on disk (not pulled via -hf); override the directory with MODEL_26B_MTP_DIR.
MODEL_26B_MTP_DIR_DEFAULT="${HOME}/models/gemma4-26b-qat-balanced-mtp"
MODEL_26B_MTP_MAIN="Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
MODEL_26B_MTP_DRAFT="mtp-gemma-4-26B-A4B-it.gguf"
LOCAL_MODEL_SEED_DEFAULT="424242"

resolve_local_model_profile() {
  local profile="${LOCAL_MODEL_PROFILE:-small}"
  local lib_dir
  lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROFILE_TEMPLATE_ARGS=()
  PROFILE_DRAFT_ARGS=()
  # Local main-model path (-m). Empty for HF profiles, which the launcher serves via -hf instead.
  PROFILE_MODEL_PATH=""
  local ctx_size="8192"
  case "${profile}" in
    small)
      PROFILE_HF_MODEL="${SMALL_MODEL_HF}"
      # The stock template hardcodes enable_thinking=true, which floods the
      # token budget with thought-channel output and starves grammar-constrained
      # JSON responses. The patched copy lets --chat-template-kwargs win.
      PROFILE_TEMPLATE_ARGS=(--chat-template-file "${lib_dir}/../templates/small-model.jinja")
      ;;
    26b)
      PROFILE_HF_MODEL="${MODEL_26B_HF}"
      ;;
    26b-mtp)
      # Same 26B Balanced base as `26b`, but served from local files with an MTP draft model
      # for speculative decoding (faster decode). Sampling stays app-controlled (the router sets
      # temperature per request), so the launch command carries no --temp/--top-k/etc.
      local mtp_dir="${MODEL_26B_MTP_DIR:-${MODEL_26B_MTP_DIR_DEFAULT}}"
      PROFILE_HF_MODEL=""
      PROFILE_MODEL_PATH="${mtp_dir}/${MODEL_26B_MTP_MAIN}"
      PROFILE_DRAFT_ARGS=(
        -md "${mtp_dir}/${MODEL_26B_MTP_DRAFT}"
        --spec-type draft-mtp
        --spec-draft-n-max 3
      )
      ctx_size="16384"
      ;;
    *)
      echo "LOCAL_MODEL_PROFILE must be small, 26b, or 26b-mtp, got: ${profile}" >&2
      return 1
      ;;
  esac

  PROFILE_LLAMA_ARGS=(
    --jinja
    --reasoning off
    -ngl all
    -c "${ctx_size}"
    -fa on
    --cache-type-k q8_0
    --cache-type-v q4_0
    --chat-template-kwargs '{"enable_thinking":false}'
    --seed "${LOCAL_MODEL_SEED:-${LOCAL_MODEL_SEED_DEFAULT}}"
    # Guarded expansion: bash 3.2 with `set -u` treats expanding an empty
    # array as an unbound-variable error.
    ${PROFILE_TEMPLATE_ARGS[@]+"${PROFILE_TEMPLATE_ARGS[@]}"}
    ${PROFILE_DRAFT_ARGS[@]+"${PROFILE_DRAFT_ARGS[@]}"}
  )
}

parse_extra_llama_args() {
  local serialized="${1:-}"
  PARSED_LLAMA_ARGS=()
  if [[ -z "${serialized}" ]]; then
    return 0
  fi
  while IFS= read -r argument; do
    PARSED_LLAMA_ARGS+=("${argument}")
  done < <(
    "${PYTHON:-python}" -c \
      'import shlex, sys; print(*shlex.split(sys.argv[1]), sep="\n")' \
      "${serialized}"
  )
}
