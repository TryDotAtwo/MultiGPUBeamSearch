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
: "$VAST_REMOTE_RUNTIME_SECONDS"
: "$VAST_REMOTE_KEY_BLOB"
: "$VAST_REMOTE_ARCHIVE"
: "$VAST_REMOTE_ARCHIVE_SHA"
: "$VAST_SOURCE_SHA"

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

printf -v remote_command '%q ' timeout --signal=TERM --kill-after=60s \
  "${VAST_REMOTE_RUNTIME_SECONDS}s" bash -s -- \
  "$VAST_REMOTE_ROOT/source" "$VAST_REMOTE_ROOT" "$VAST_REMOTE_KEY_BLOB" \
  "$VAST_REMOTE_ARCHIVE" "$VAST_REMOTE_ARCHIVE_SHA" "$VAST_SOURCE_SHA"
remote_exit=0
if ssh "${ssh_opts[@]}" "root@$VAST_SSH_HOST" "$remote_command" \
  < "$VAST_SSH_SCRIPT" > "$VAST_SSH_LOG" 2>&1; then
  remote_exit=0
else
  remote_exit=$?
fi

printf -v results_command 'tar -C %q -czf - %q' \
  "$VAST_REMOTE_ROOT" "test_results/refactor_20261121/gpu-acceptance"
result_exit=0
if ssh "${ssh_opts[@]}" "root@$VAST_SSH_HOST" "$results_command" > "$VAST_SSH_RESULTS"; then
  result_exit=0
else
  result_exit=$?
  rm -f -- "$VAST_SSH_RESULTS"
  echo "Could not retrieve remote GPU test_results over the retained SSH master." >&2
fi

if [[ "$remote_exit" != 0 ]]; then
  tail -n 100 "$VAST_SSH_LOG"
  exit "$remote_exit"
fi
if [[ "$result_exit" != 0 ]]; then
  tail -n 100 "$VAST_SSH_LOG"
  exit "$result_exit"
fi
test -s "$VAST_SSH_RESULTS"
cat "$VAST_SSH_LOG"
