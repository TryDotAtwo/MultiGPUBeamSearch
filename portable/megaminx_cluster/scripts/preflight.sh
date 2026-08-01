#!/usr/bin/env bash
set -euo pipefail
exec python3 -m portable.megaminx_cluster.scripts.preflight_cli "$@"
