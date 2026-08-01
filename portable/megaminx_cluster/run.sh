#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ARCHIVE_PARENT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
export PYTHONPATH="${ARCHIVE_PARENT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m portable.megaminx_cluster.submit "$@"
