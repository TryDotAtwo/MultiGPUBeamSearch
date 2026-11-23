#!/usr/bin/env bash
# Shared CLI arguments for the physical panel-v2 runner and its CPU parser check.
mgbs_vast_cnot_panel_cli_args() {
  if (( $# != 5 )); then
    echo "mgbs_vast_cnot_panel_cli_args expects run root, output, library, preflight path and preflight SHA" >&2
    return 64
  fi
  local run_root="$1"
  local output="$2"
  local library="$3"
  local handoff_preflight="$4"
  local handoff_preflight_sha256="$5"
  printf '%s\0' \
    --case panel \
    --backend nccl \
    --mode qhat \
    --engine rust \
    --require-rust-scorer \
    --library "$library" \
    --run-root "$run_root" \
    --output "$output" \
    --handoff-preflight "$handoff_preflight" \
    --handoff-preflight-sha256 "$handoff_preflight_sha256"
}

# Select a scorer artifact by manifest contract and always require strict Rust admission.
# The runner and CPU fixture call this same function.
mgbs_vast_cnot_scorer_environment() {
  if (( $# != 5 )); then
    echo "mgbs_vast_cnot_scorer_environment expects contract, two library paths and two SHA-256 values" >&2
    return 64
  fi
  local contract="$1"
  local dev_library="$2"
  local panel_library="$3"
  local dev_sha256="$4"
  local panel_sha256="$5"
  if [[ -z "$dev_library" || -z "$panel_library" || "$dev_library" == "$panel_library" ]]; then
    echo "dev-v1 and panel-v2 scorer libraries must be distinct" >&2
    return 64
  fi
  if [[ ! "$dev_sha256" =~ ^[0-9a-f]{64}$ || ! "$panel_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "both scorer libraries require lowercase SHA-256 pins" >&2
    return 64
  fi
  local selected_library selected_sha256
  case "$contract" in
    dev_v1)
      selected_library="$dev_library"
      selected_sha256="$dev_sha256"
      ;;
    panel_search_v2)
      selected_library="$panel_library"
      selected_sha256="$panel_sha256"
      ;;
    *)
      echo "unsupported CNOT scorer contract: $contract" >&2
      return 64
      ;;
  esac
  printf '%s\0' \
    MGBS_TQP_LIBRARY "$selected_library" \
    MGBS_CNOT_STRICT_BACKEND 1 \
    MGBS_CNOT_EXPECTED_TQP_LIBRARY_SHA256 "$selected_sha256"
}
