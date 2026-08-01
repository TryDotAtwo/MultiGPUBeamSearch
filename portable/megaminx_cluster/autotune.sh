#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [ -d "${SCRIPT_DIR}/portable/megaminx_cluster" ]; then ARCHIVE_ROOT="${SCRIPT_DIR}"; else ARCHIVE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"; fi
cd "${ARCHIVE_ROOT}"
exec python3 -m portable.megaminx_cluster.autotune_submit "$@"