#!/usr/bin/env bash
set -euo pipefail

: "$VAST_SSH_PRIVATE_KEY"
: "$VAST_SSH_KNOWN_HOSTS"
: "$VAST_SSH_HOST"
: "$VAST_SSH_PORT"
: "$VAST_SSH_LOG"
: "$VAST_SSH_RESULTS"
: "$VAST_SSH_SCRIPT"
: "$VAST_REMOTE_ROOT"
: "$VAST_REMOTE_RESULTS_RELATIVE"
: "$VAST_REMOTE_RUNTIME_MAX_SECONDS"
: "$VAST_BUDGET_DEADLINE_EPOCH"
: "$VAST_STOP_RESERVE_SECONDS"
: "$VAST_RETRIEVAL_RESERVE_SECONDS"
: "$VAST_RETRIEVAL_MAX_SECONDS"
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

# Keep one authenticated TCP connection alive while the remote EXIT trap
# removes this one-time key. Result retrieval then reuses the master socket.
ssh "${ssh_opts[@]}" -o ControlMaster=yes -MNf "root@$VAST_SSH_HOST"

# Refresh the remaining allowance after SSH setup, immediately before the
# remote build/search. Keep both result retrieval and the stop reserve intact.
remote_runtime_seconds="$(python3 "$(dirname "$0")/vast_budget_deadline.py" \
  --deadline-epoch "$VAST_BUDGET_DEADLINE_EPOCH" \
  --stop-reserve-seconds "$VAST_STOP_RESERVE_SECONDS" \
  --future-reserve-seconds "$VAST_RETRIEVAL_RESERVE_SECONDS" \
  --max-seconds "$VAST_REMOTE_RUNTIME_MAX_SECONDS" \
  --min-seconds 1200 --phase remote-build-and-run)"
printf -v remote_command '%q ' timeout --signal=TERM --kill-after=60s \
  "${remote_runtime_seconds}s" bash -s -- \
  "$VAST_REMOTE_ROOT/source" "$VAST_REMOTE_ROOT" "$VAST_REMOTE_KEY_BLOB" \
  "$VAST_REMOTE_ARCHIVE" "$VAST_REMOTE_ARCHIVE_SHA" "$VAST_SOURCE_SHA" \
  "$VAST_INSTANCE_ID" "$VAST_REMOTE_MANIFEST_SHA" "$VAST_REMOTE_CHECKPOINT_SHA" \
  "$VAST_GITHUB_RUN_ID" "$VAST_GITHUB_RUN_ATTEMPT" "$VAST_REMOTE_RESULTS_RELATIVE"
remote_exit=0
if ssh "${ssh_opts[@]}" "root@$VAST_SSH_HOST" "$remote_command" \
  < "$VAST_SSH_SCRIPT" > "$VAST_SSH_LOG" 2>&1; then
  remote_exit=0
else
  remote_exit=$?
fi

printf -v results_command 'tar -C %q -czf - %q' \
  "$VAST_REMOTE_ROOT" "$VAST_REMOTE_RESULTS_RELATIVE"
result_exit=0
if retrieval_seconds="$(python3 "$(dirname "$0")/vast_budget_deadline.py" \
  --deadline-epoch "$VAST_BUDGET_DEADLINE_EPOCH" \
  --stop-reserve-seconds "$VAST_STOP_RESERVE_SECONDS" \
  --future-reserve-seconds 0 \
  --max-seconds "$VAST_RETRIEVAL_MAX_SECONDS" \
  --min-seconds 1 --phase retrieval)"; then
  if timeout --signal=TERM --kill-after=30s "${retrieval_seconds}s" \
    ssh "${ssh_opts[@]}" "root@$VAST_SSH_HOST" "$results_command" > "$VAST_SSH_RESULTS"; then
    result_exit=0
  else
    result_exit=$?
    rm -f -- "$VAST_SSH_RESULTS"
    echo "Could not retrieve remote GPU test_results over the retained SSH master." >&2
  fi
else
  result_exit=$?
  rm -f -- "$VAST_SSH_RESULTS"
  echo "The absolute campaign deadline does not leave time for result retrieval and the stop reserve." >&2
fi

if [[ "$remote_exit" != 0 ]]; then
  tail -n 100 "$VAST_SSH_LOG"
  exit "$remote_exit"
fi
if [[ "$result_exit" != 0 ]]; then
  tail -n 100 "$VAST_SSH_LOG"
  exit "$result_exit"
fi
if [[ ! -s "$VAST_SSH_RESULTS" ]]; then
  echo "Retrieved remote GPU test_results archive is empty." >&2
  tail -n 100 "$VAST_SSH_LOG"
  exit 1
fi
cat "$VAST_SSH_LOG"
