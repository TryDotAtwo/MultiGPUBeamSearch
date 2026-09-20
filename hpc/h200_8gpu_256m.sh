#!/usr/bin/env bash
set -euo pipefail
export H200_WORLD_SIZE=8
export BEAM_WIDTH=256000000
exec bash "$(dirname "${BASH_SOURCE[0]}")/h200_dual_large_beam.sh"
