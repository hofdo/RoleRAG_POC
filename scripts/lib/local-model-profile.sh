#!/usr/bin/env bash

SMALL_MODEL_HF="DavidAU/gemma-4-E4B-it-The-DECKARD-Expresso-Universe-HERETIC-UNCENSORED-Thinking-GGUF:Q8_0"
MODEL_26B_HF="HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M"
LOCAL_MODEL_SEED_DEFAULT="424242"

resolve_local_model_profile() {
  local profile="${LOCAL_MODEL_PROFILE:-small}"
  case "${profile}" in
    small)
      PROFILE_HF_MODEL="${SMALL_MODEL_HF}"
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
