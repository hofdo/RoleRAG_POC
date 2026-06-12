#!/usr/bin/env bash

SMALL_MODEL_HF="DavidAU/gemma-4-E4B-it-The-DECKARD-Expresso-Universe-HERETIC-UNCENSORED-Thinking-GGUF:Q8_0"
MODEL_26B_HF="HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M"
LOCAL_MODEL_SEED_DEFAULT="424242"

resolve_local_model_profile() {
  local profile="${LOCAL_MODEL_PROFILE:-small}"
  local lib_dir
  lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROFILE_TEMPLATE_ARGS=()
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
    *)
      echo "LOCAL_MODEL_PROFILE must be small or 26b, got: ${profile}" >&2
      return 1
      ;;
  esac

  PROFILE_LLAMA_ARGS=(
    --jinja
    --reasoning off
    -ngl all
    -c 8192
    -fa on
    --cache-type-k q8_0
    --cache-type-v q4_0
    --chat-template-kwargs '{"enable_thinking":false}'
    --seed "${LOCAL_MODEL_SEED:-${LOCAL_MODEL_SEED_DEFAULT}}"
    # Guarded expansion: bash 3.2 with `set -u` treats expanding an empty
    # array as an unbound-variable error.
    ${PROFILE_TEMPLATE_ARGS[@]+"${PROFILE_TEMPLATE_ARGS[@]}"}
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
