#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [ -d "${SCRIPT_DIR}/portable/megaminx_cluster" ]; then ARCHIVE_ROOT="${SCRIPT_DIR}"; else ARCHIVE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"; fi
if [ -f "${ARCHIVE_ROOT}/publication.env" ]; then set -a; source "${ARCHIVE_ROOT}/publication.env"; set +a; fi
export PYTHONPATH="${ARCHIVE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [ "${1:-}" = "--publish-only" ]; then
  if [ "$#" -ne 2 ]; then echo "usage: ./run.sh --publish-only RUN_DIR" >&2; exit 2; fi
  : "${MEGAMINX_RESULTS_URL:?set MEGAMINX_RESULTS_URL in publication.env}"
  python3 -m portable.megaminx_cluster.scripts.verify_archive_payloads --archive-root "${ARCHIVE_ROOT}"
  exec python3 -m portable.megaminx_cluster.scripts.validate_and_publish --run-dir "$2" --url "${MEGAMINX_RESULTS_URL}" --poll
fi
exec python3 -m portable.megaminx_cluster.submit "$@"
