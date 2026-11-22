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
