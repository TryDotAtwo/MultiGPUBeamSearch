#!/usr/bin/env bash
# Exact source/build/profile/cohort receipt is mandatory before any rank starts.
set -euo pipefail
repo=${CUBE4_REPO_DIR:-/workspace/complete_audit_20261005}
exec "${CUBE4_PYTHON:-python3}" "$repo/tools/cube4_admitted_launch.py" "$@"
