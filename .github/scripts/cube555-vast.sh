#!/usr/bin/env bash
set -euo pipefail

temp="${RUNNER_TEMP:?}"
evidence="$temp/cube555-a100-evidence"
rental_id="$temp/cube555-vast-id"
rental_attempt="$temp/cube555-vast-attempt"
label="cube555-a100-${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}"
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
command -v vastai >/dev/null
key="$temp/cube555-ed25519"
ssh-keygen -q -t ed25519 -N "" -C "$label" -f "$key"
trap 'status=$?; trap - EXIT; cleanup || status=1; exit "$status"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

query='gpu_name in [A100_SXM4,A100_PCIE,A100_SXM4_80GB,A100_PCIE_80GB] num_gpus=2 verified=true rentable=true rented=false cuda_max_good>=12.8 cpu_ram>=64 disk_space>=65 inet_down>=200 reliability>=0.98 dph_total<=2.5'
finish_after="$(( $(date +%s) + 14400 ))"
vastai search offers "$query" --storage 65 -o dph_total --limit 30 --raw \
    --api-key "$VAST_API_KEY" | jq -er --argjson deadline "$finish_after" '
        map(select(
            .num_gpus == 2 and .end_date >= $deadline and
            .dph_total <= 2.5 and .inet_down_cost <= 0.05 and .inet_up_cost <= 0.05
        )) | first // error("No two-A100 offer fits the test budget") |
        {id, gpu_name, num_gpus, gpu_ram, cpu_ram, reliability,
         dph_total, inet_down, inet_down_cost, inet_up_cost, end_date}
    ' | tee "$evidence/offer.json"
offer_id="$(jq -er '.id' "$evidence/offer.json")"
touch "$rental_attempt"
date -u +%FT%TZ > "$evidence/rental-started-utc.txt"
vastai create instance "$offer_id" \
    --image 'nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7' \
    --disk 65 --ssh --direct --cancel-unavail --label "$label" \
    --raw --api-key "$VAST_API_KEY" > "$temp/cube555-create.json"
id="$(jq -er 'select(.success == true) | .new_contract | select(type == "number")' \
    "$temp/cube555-create.json")"
printf '%s\n' "$id" > "$rental_id"
printf 'Vast lease %s (%s)\n' "$id" "$label"
jq -n --argjson id "$id" --arg label "$label" '{instance_id:$id,label:$label}' \
    > "$evidence/lease.json"

request="$(jq -cn --arg ssh_key "$(cat "$key.pub")" '{ssh_key:$ssh_key}')"
curl --fail --silent --show-error --max-time 25 --retry 3 --retry-all-errors \
    --retry-delay 5 -X POST -H "Authorization: Bearer ${VAST_API_KEY}" \
    -H "Content-Type: application/json" --data-binary "$request" --output /dev/null \
    "$endpoint/$id/ssh"

deadline="$(( $(date +%s) + 1500 ))"
while (( $(date +%s) < deadline )); do
    if row="$(list_instances | find_lease)" &&
        jq -e --argjson id "$id" '.id == $id and .actual_status == "running"' \
            >/dev/null <<< "$row"; then
        host="$(jq -r '.public_ipaddr // empty' <<< "$row")"
        port="$(jq -r '.ports["22/tcp"][0].HostPort // empty' <<< "$row")"
        if [[ "$host" =~ ^[0-9.]+$ && "$port" =~ ^[0-9]+$ ]]; then
            ssh_opts=(-i "$key" -p "$port" -o BatchMode=yes -o IdentitiesOnly=yes
                -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$temp/cube555-known-hosts"
                -o ConnectTimeout=12 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)
            if ssh "${ssh_opts[@]}" "root@$host" true; then
                break
            fi
        fi
    fi
    sleep 15
done
if (( $(date +%s) >= deadline )); then
    echo "The A100 host did not open SSH within its startup window." >&2
    exit 1
fi

remote() { ssh "${ssh_opts[@]}" "root@$host" "$@"; }
remote nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv \
    | tee "$evidence/gpus.csv"
remote nvidia-smi topo -m | tee "$evidence/topology.txt"

remote_dir=/root/cube555-test
remote "mkdir -p $remote_dir"
echo "Staging the checked offline archive"
timeout 22m tar -C "$temp/cube555-bundle" -cf - "$archive" "$archive.sha256" \
    | timeout 22m ssh "${ssh_opts[@]}" "root@$host" "tar -xf - -C $remote_dir"
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
