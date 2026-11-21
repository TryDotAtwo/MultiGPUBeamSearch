#!/usr/bin/env bash
set -euo pipefail

: "$VAST_SSH_PRIVATE_KEY"
: "$VAST_SSH_KNOWN_HOSTS"
: "$VAST_SSH_HOST"
: "$VAST_SSH_PORT"
: "$VAST_SSH_SCRIPT"
: "$VAST_REMOTE_ROOT"
: "$VAST_REMOTE_RESULTS_RELATIVE"
: "$VAST_REMOTE_RUNTIME_MAX_SECONDS"
: "$VAST_BUDGET_DEADLINE_EPOCH"
: "$VAST_STOP_RESERVE_SECONDS"
: "$VAST_RETRIEVAL_RESERVE_SECONDS"
: "$VAST_REMOTE_KEY_BLOB"
: "$VAST_REMOTE_ARCHIVE"
: "$VAST_REMOTE_ARCHIVE_SHA"
: "$VAST_REMOTE_MANIFEST_SHA"
: "$VAST_REMOTE_CHECKPOINT_SHA"
: "$VAST_SOURCE_SHA"
: "$VAST_INSTANCE_ID"
: "$VAST_GITHUB_RUN_ID"
: "$VAST_GITHUB_RUN_ATTEMPT"
[[ "$VAST_GITHUB_RUN_ID" =~ ^[0-9]+$ && "$VAST_GITHUB_RUN_ATTEMPT" =~ ^[0-9]+$ ]]
[[ "$VAST_REMOTE_RESULTS_RELATIVE" == "test_results/refactor_20261121/gpu-acceptance" ]]

ssh_opts=(
  -i "$VAST_SSH_PRIVATE_KEY"
  -p "$VAST_SSH_PORT"
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$VAST_SSH_KNOWN_HOSTS"
  -o ConnectTimeout=15
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=2
  -o ControlMaster=auto
  -o ControlPersist=5m
)
control_dir="$(mktemp -d /tmp/mgbs-ssh.XXXXXX)"
control_path="$control_dir/c-%C"
ssh_opts+=(-o ControlPath="$control_path")

cleanup() {
  local exit_code=$?
  trap - EXIT
  ssh "${ssh_opts[@]}" -O exit "root@$VAST_SSH_HOST" >/dev/null 2>&1 || true
  rm -rf -- "$control_dir"
  exit "$exit_code"
}
trap cleanup EXIT

if ! ssh "${ssh_opts[@]}" -o ControlMaster=yes -MNf "root@$VAST_SSH_HOST" >/dev/null 2>&1; then
  printf 'REMOTE_GPU_STATUS=not_started\n'
  exit 1
fi

if ! remote_runtime_seconds="$(python3 "$(dirname "$0")/vast_budget_deadline.py" --deadline-epoch "$VAST_BUDGET_DEADLINE_EPOCH" --stop-reserve-seconds "$VAST_STOP_RESERVE_SECONDS" --future-reserve-seconds "$VAST_RETRIEVAL_RESERVE_SECONDS" --max-seconds "$VAST_REMOTE_RUNTIME_MAX_SECONDS" --min-seconds 1200 --phase remote-build-and-run 2>/dev/null)"; then
  printf 'REMOTE_GPU_STATUS=not_started\n'
  exit 1
fi

printf -v remote_command '%q ' timeout --signal=TERM --kill-after=60s "$remote_runtime_seconds" bash -s -- "$VAST_REMOTE_ROOT/source" "$VAST_REMOTE_ROOT" "$VAST_REMOTE_KEY_BLOB" "$VAST_REMOTE_ARCHIVE" "$VAST_REMOTE_ARCHIVE_SHA" "$VAST_SOURCE_SHA" "$VAST_INSTANCE_ID" "$VAST_REMOTE_MANIFEST_SHA" "$VAST_REMOTE_CHECKPOINT_SHA" "$VAST_GITHUB_RUN_ID" "$VAST_GITHUB_RUN_ATTEMPT" "$VAST_REMOTE_RESULTS_RELATIVE"
remote_exit=0
if ssh "${ssh_opts[@]}" "root@$VAST_SSH_HOST" "$remote_command" < "$VAST_SSH_SCRIPT" >/dev/null 2>&1; then
  printf 'REMOTE_GPU_STATUS=passed\n'
else
  remote_exit=$?
  printf 'REMOTE_GPU_STATUS=failed\n'
  exit "$remote_exit"
fi
