#!/usr/bin/env bash
set -euo pipefail

temp="${RUNNER_TEMP:?}"
evidence="$temp/cube555-a100-evidence"
rental_id="$temp/cube555-vast-id"
rental_attempt="$temp/cube555-vast-attempt"
label="${CUBE555_LEASE_LABEL:-cube555-a100-${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}}"
endpoint="https://console.vast.ai/api/v0/instances"
archive="cube555-standalone-linux-x86_64-py311.tar"
mkdir -p "$evidence"

list_instances() {
    curl --fail --silent --show-error --max-time 25 \
        -H "Authorization: Bearer ${VAST_API_KEY:?}" "$endpoint/"
}

find_lease() {
    jq -cer --arg label "$label" '
        if (.instances | type) != "array" then error("Invalid Vast response") end |
        [.instances[] | select(.label == $label)] | if length > 1
            then error("More than one instance matched this job") else .[0] // {} end'
}

cleanup() {
    if [[ ! -f "$rental_attempt" ]]; then
        return
    fi
    local row id response attempt
    for attempt in {1..5}; do
        if row="$(list_instances | find_lease)"; then
            id="$(jq -r '.id // empty' <<< "$row")"
            if [[ -z "$id" ]]; then
                jq -n --arg label "$label" --arg date "$(date -u +%FT%TZ)" \
                    --arg id "$(cat "$rental_id" 2>/dev/null || true)" \
                    '{label:$label, instance_id:$id, matching_instances:0, checked_at_utc:$date}' \
                    | tee "$evidence/cleanup.json"
                return
            fi
            if [[ -s "$rental_id" && "$(cat "$rental_id")" != "$id" ]]; then
                printf 'Lease %s differs from the recorded lease.\n' "$id" >&2
                return 1
            fi
            if response="$(curl --fail --silent --show-error --max-time 25 \
                -X DELETE -H "Authorization: Bearer ${VAST_API_KEY:?}" "$endpoint/$id/")"; then
                jq -n --arg id "$id" --arg at "$(date -u +%FT%TZ)" \
                    --argjson response "$response" \
                    '{instance_id:$id, destroy_submitted_at_utc:$at, response:$response}' \
                    > "$evidence/destroy.json"
            fi
        fi
        sleep 8
    done
    echo "Could not verify Vast removal for $label" >&2
    return 1
}

if [[ "${1:-}" == --cleanup ]]; then
    cleanup
    exit
fi

test -n "${VAST_API_KEY:?}"
key="$temp/cube555-ed25519"
ssh-keygen -q -t ed25519 -N "" -C "$label" -f "$key"
trap 'status=$?; trap - EXIT; cleanup || status=1; exit "$status"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

id="${CUBE555_EXISTING_INSTANCE_ID:?}"
row="$(list_instances | find_lease)"
jq -e --argjson id "$id" '
    select(.id == $id and .num_gpus == 2 and (.gpu_name | startswith("A100")) and .dph_total <= 2.5) |
    {id, gpu_name, num_gpus, gpu_ram, dph_total, start_date}
' <<< "$row" | tee "$evidence/offer.json"
printf '%s\n' "$id" > "$rental_id"
touch "$rental_attempt"
date -u --date="@$(jq -er '.start_date' <<< "$row")" +%FT%TZ > "$evidence/rental-started-utc.txt"
printf 'Vast lease %s (%s)\n' "$id" "$label"
jq -n --argjson id "$id" --arg label "$label" '{instance_id:$id,label:$label}' \
    > "$evidence/lease.json"

if [[ -n "${CUBE555_STALLED_RUN:-}" ]]; then
    gh_run="repos/$GITHUB_REPOSITORY/actions/runs/$CUBE555_STALLED_RUN"
    status="$(gh api "$gh_run" --jq .status)"
    if [[ "$status" != completed ]]; then
        gh api --method POST "$gh_run/cancel" >/dev/null || true
        for attempt in {1..15}; do
            status="$(gh api "$gh_run" --jq .status)"
            [[ "$status" == completed ]] && break
            sleep 8
        done
    fi
    [[ "$status" == completed ]] || { echo 'Previous transfer is still running.' >&2; exit 1; }
fi

request="$(jq -cn --arg ssh_key "$(cat "$key.pub")" '{ssh_key:$ssh_key}')"
curl --fail --silent --show-error --max-time 25 --retry 3 --retry-all-errors \
    --retry-delay 5 -X POST -H "Authorization: Bearer ${VAST_API_KEY}" \
    -H "Content-Type: application/json" --data-binary "$request" --output /dev/null \
    "$endpoint/$id/ssh"

deadline="$(( $(date +%s) + 1500 ))"
connected=false
while (( $(date +%s) < deadline )); do
    if row="$(list_instances | find_lease)" &&
        jq -e --argjson id "$id" '.id == $id and .actual_status == "running"' \
            >/dev/null <<< "$row"; then
        for transport in proxy direct; do
            if [[ "$transport" == proxy ]]; then
                host="$(jq -r '.ssh_host // empty' <<< "$row")"
                port="$(jq -r '.ssh_port // empty' <<< "$row")"
            else
                host="$(jq -r '.public_ipaddr // empty' <<< "$row")"
                port="$(jq -r '.ports["22/tcp"][0].HostPort // empty' <<< "$row")"
            fi
            [[ "$host" =~ ^[a-zA-Z0-9.-]+$ && "$port" =~ ^[0-9]+$ ]] || continue
            ssh_opts=(-i "$key" -p "$port" -o BatchMode=yes -o IdentitiesOnly=yes
                -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$temp/cube555-known-hosts"
                -o ConnectTimeout=12 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)
            if ssh "${ssh_opts[@]}" "root@$host" true; then
                connected=true
                printf '%s\n' "$transport" | tee "$evidence/ssh-transport.txt"
                break 2
            fi
        done
    fi
    sleep 15
done
if [[ "$connected" != true ]]; then
    echo "The A100 host did not open SSH within its startup window." >&2
    exit 1
fi

remote() { ssh "${ssh_opts[@]}" "root@$host" "$@"; }
remote nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv \
    | tee "$evidence/gpus.csv"
remote nvidia-smi topo -m | tee "$evidence/topology.txt"

remote_dir=/root/cube555-egress
remote "mkdir -p $remote_dir; if ! command -v unzip >/dev/null || ! command -v curl >/dev/null; then
    apt-get -qq update && apt-get -qq install -y --no-install-recommends unzip curl ca-certificates; fi"
echo "Downloading the checked offline archive on the GPU node"
location="$(curl --fail --silent --show-error --max-time 25 --dump-header - --output /dev/null \
    -H "Authorization: Bearer ${GH_TOKEN:?}" -H 'Accept: application/vnd.github+json' \
    "https://api.github.com/repos/$GITHUB_REPOSITORY/actions/artifacts/$CUBE555_BUNDLE_ARTIFACT/zip" \
    | awk 'tolower($1) == "location:" { sub(/\r$/, "", $2); print $2; exit }')"
[[ "$location" == https://* && "$location" != *'"'* && "$location" != *\\* ]] || {
    echo 'GitHub did not issue an HTTPS download URL.' >&2
    exit 1
}
printf 'url = "%s"\n' "$location" \
    | timeout 23m ssh "${ssh_opts[@]}" "root@$host" \
        "cd $remote_dir && curl --config - --proto '=https' --fail --silent --show-error \
        --max-time 1300 --write-out 'Downloaded %{size_download} bytes in %{time_total}s\n' \
        --output bundle.zip && unzip -oq bundle.zip '$archive' '$archive.sha256' && rm -- bundle.zip" \
    | tee "$evidence/transfer.txt"
unset location
date -u +%FT%TZ > "$evidence/test-started-utc.txt"

test_status=0
timeout --signal=TERM --kill-after=1m 170m ssh "${ssh_opts[@]}" "root@$host" \
    "cd $remote_dir && bash -s" < .github/scripts/cube555-a100-host.sh || test_status=$?
printf '%s\n' "$test_status" | tee "$evidence/test-exit-code.txt"
date -u +%FT%TZ > "$evidence/test-ended-utc.txt"

if ! remote "cd $remote_dir && tar -cf - evidence" | tar -xf - -C "$evidence"; then
    echo "Could not retrieve the complete evidence from the A100 host." >&2
    test_status=1
fi
exit "$test_status"
